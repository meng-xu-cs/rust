use rustc_hir::def::DefKind;
use rustc_middle::ty::{Instance, InstanceKind, TyCtxt};
use rustc_middle::{bug, ty};
use tracing::{info, warn};

use crate::solana::common::SolEnv;
use crate::solana::context::{SolContextBuilder, SolEntrypoint};

fn try_resolve_entry<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    sol: SolEnv,
) -> Option<SolEnv> {
    let def_id = instance.def_id();
    let body = tcx.instance_mir(instance.def);

    // the `entry` function has the following signature
    // pub fn entry<'info>(
    //     program_id: &Pubkey,
    //     accounts: &'info [AccountInfo<'info>],
    //     data: &[u8],
    // ) -> anchor_lang::solana_program::entrypoint::ProgramResult

    if tcx.def_path_str(def_id) != "entry" {
        info!("- skipped, function name is not 'entry'");
        return Some(sol);
    }
    if !instance.args.is_empty() {
        bug!("[assumption] expect `entry` to have no generics, found {}", instance.args.len());
    }
    if body.arg_count != 3 {
        bug!("[assumption] expect `entry` to have three arguments, found {}", body.arg_count);
    }

    // collect information
    warn!("- found the `entry` function");
    let mut builder = SolContextBuilder::new(tcx, sol);
    let (_, inst_ident, _) = builder.make_instance(instance);
    let (sol, context) = builder.build();

    // serialize the context to file
    let ctxt_file = sol.context_to_file(&context);

    // save the instruction to file
    let entrypoint = SolEntrypoint { function: inst_ident };
    sol.summary_to_file("entrypoint", &entrypoint);
    info!("- context saved at {}", ctxt_file.display());

    // resolved, consume the environment
    None
}

fn try_resolve_instruction<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>, _sol: SolEnv) {
    let body = tcx.instance_mir(instance.def);

    // check parameter types
    let ty_state = match body.args_iter().next() {
        None => {
            info!("- skipped, no parameters");
            return;
        }
        Some(arg) => {
            let decl = &body.local_decls[arg];
            match decl.ty.kind() {
                ty::Adt(adt_def, adt_ty_args) => {
                    // the first parameter of an instruction must be `anchor_lang::context::Context`
                    if tcx.def_path_str(adt_def.did()) != "anchor_lang::context::Context" {
                        info!("- skipped, first parameter is not Context");
                        return;
                    }

                    // extract the type that represents the state
                    // the definiton is `struct Context<'a, 'b, 'c, 'info, T: Bumps>`
                    if adt_ty_args.len() != 5 {
                        bug!(
                            "[assumption] anchor_lang::context::Context should take 5 type arguments, found: {}",
                            adt_ty_args.len()
                        );
                    }

                    // unpack the type arguments
                    let mut iter = adt_ty_args.iter();
                    iter.next().unwrap().expect_region();
                    iter.next().unwrap().expect_region();
                    iter.next().unwrap().expect_region();
                    iter.next().unwrap().expect_region();
                    iter.next().unwrap().expect_ty()
                }
                _ => {
                    info!("- skipped, first parameter is not an adt");
                    return;
                }
            }
        }
    };

    // assert that the state type must be user-defined
    let ty_state_def_id = match ty_state.kind() {
        ty::Adt(ty_def, ty_args) => {
            if !ty_def.is_struct() {
                bug!("[assumption] expect the state type to be a struct, found: {ty_state}");
            }
            if ty_args.iter().any(|t| t.as_region().map_or(true, |r| !r.is_erased())) {
                bug!("[assumption] expect erased lifetimes only, found: {ty_state}");
            }
            ty_def.did()
        }
        _ => bug!("[assumption] expect the state type to be an adt, found: {ty_state}"),
    };

    // try to list all impls
    warn!("[target state] {}", tcx.def_path_str(ty_state_def_id));
    for (trait_id, impl_ids) in tcx.all_local_trait_impls(()) {
        warn!("[trait] {}", tcx.def_path_str(trait_id));
        if tcx.def_path_str(trait_id) == "anchor_lang::Discriminator" {
            for impl_id in impl_ids {
                let impl_id = impl_id.to_def_id();
                let impl_self_ty = tcx.type_of(impl_id).instantiate_identity();
                warn!("  [impl type] for {} is {impl_self_ty}", tcx.def_path_str(impl_id));

                // try to obtain the type of the impl block
                match impl_self_ty.kind() {
                    ty::Adt(def, _) if def.did() == ty_state_def_id => {
                        warn!("  [impl block] for state {}", tcx.def_path_str(impl_id));
                        for item in tcx.associated_items(impl_id).in_definition_order() {
                            warn!("  [impl item] {}", tcx.def_path_str(item.def_id));
                        }
                    }
                    _ => {
                        warn!("  [impl mismatch] {ty_state} vs {impl_self_ty}");
                    }
                }
            }
        }
    }
}

pub(crate) fn phase_bootstrap<'tcx>(tcx: TyCtxt<'tcx>, sol: SolEnv, instance: Instance<'tcx>) {
    info!(
        "phase: {} on {} with source {}",
        sol.phase,
        sol.build_system,
        sol.src_file_name.display()
    );

    // debug marking
    let def_id = instance.def_id();
    let def_desc = tcx.def_path_str_with_args(def_id, instance.args);
    info!("processing {def_desc}");

    // skip source files that are not under the local source prefix
    if !sol.src_path_full.starts_with(&sol.source_dir) {
        info!("- skipped, not a local source file");
        return;
    }

    // skip non-local items (e.g., generics from external crates)
    if !def_id.is_local() {
        info!("- skipped, not a local definition");
        return;
    }

    // the `entry` function must be a `fn` item
    match instance.def {
        InstanceKind::Item(did) if tcx.def_kind(did) == DefKind::Fn => (),
        _ => {
            info!("- skipped, not a fn item");
            return;
        }
    };

    // first attempt to resolve the entry function
    match try_resolve_entry(tcx, instance, sol) {
        None => return, // resolved, nothing else to do
        Some(sol) => {
            // otherwise, try to resolve as an instruction
            try_resolve_instruction(tcx, instance, sol);
        }
    }
}
