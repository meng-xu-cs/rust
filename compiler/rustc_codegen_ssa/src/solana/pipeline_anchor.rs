//! This pass encapsulate all the transformations related to Solana's Anchor framework.
use std::fs;

use rustc_hir::def::DefKind;
use rustc_middle::ty::{Instance, InstanceKind, TyCtxt};
use rustc_middle::{bug, ty};
use tracing::{info, warn};

use crate::solana::common::SolEnv;
use crate::solana::context::{SolAnchorInstruction, SolContextBuilder, SolDeps, SolInstanceKind};

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

    // an instruction must be a `fn` item
    match instance.def {
        InstanceKind::Item(did) if tcx.def_kind(did) == DefKind::Fn => (),
        _ => {
            info!("- skipped, not a fn item");
            return;
        }
    };

    // reveal function definition
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
                    let last_ty_arg = iter.next().unwrap().expect_ty();
                    if !iter.next().is_none() {
                        bug!(
                            "[invariant] anchor_lang::context::Context takes more than 5 type arguments"
                        );
                    }
                    last_ty_arg
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
        ty::Adt(ty_state_def, _) => ty_state_def.did(),
        _ => bug!("[assumption] expect the state type to be an adt, found: {ty_state}"),
    };

    // collect information
    warn!("- found instruction {def_desc} with argument {ty_state}");

    let mut builder = SolContextBuilder::new(tcx, sol);
    let (_, inst_ident, _) = builder.make_instance(instance);
    let (sol, context) = builder.build();

    // serialize the context to file
    let ctxt_file = sol.context_to_file(&context);

    // save the instruction to file
    let instruction = SolAnchorInstruction {
        function: inst_ident,
        ty_state: SolContextBuilder::mk_ident_no_cache(tcx, ty_state_def_id),
    };
    sol.summary_to_file("instruction", &instruction);

    // done
    info!("- done with instruction {def_desc}, context saved at {}", ctxt_file.display());
}

pub(crate) fn phase_expansion<'tcx>(tcx: TyCtxt<'tcx>, sol: SolEnv, instance: Instance<'tcx>) {
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

    // load the deps
    let path = sol.prev_phase_output_dir().join("deps.json");
    let content = fs::read_to_string(path).unwrap_or_else(|e| {
        bug!("[invariant] failed to read previous phase output file deps.json: {e}")
    });
    let deps: SolDeps = serde_json::from_str(&content)
        .unwrap_or_else(|e| bug!("[invariant] failed to deserialize deps.json: {e}"));

    // derive the identifier
    let inst_ident = SolContextBuilder::mk_ident_no_cache(tcx, def_id);

    // compare with the dependencies
    let mut matched = false;
    for (dep_kind, dep_ident, dep_args, _) in &deps.fn_deps {
        if dep_ident != &inst_ident {
            continue;
        }

        // attempt to resolve the instance
        info!("- attempting dependency {def_desc}");
        let mut temp_builder = SolContextBuilder::new(tcx, sol.with_temporary_phase());

        // try to match the type arguments
        if dep_args.len() != instance.args.len() {
            info!("- skipped, type argument count mismatch");
            continue;
        }

        let mut ty_args_matched = true;
        for (dep_arg, ty_arg) in dep_args.iter().zip(instance.args.iter()) {
            let def_arg = temp_builder.mk_ty_arg(ty_arg);
            if dep_arg != &def_arg {
                ty_args_matched = false;
                break;
            }
        }
        if !ty_args_matched {
            info!("- skipped, type arguments mismatch");
            continue;
        }

        // try to match the instance kind
        match (dep_kind, instance.def) {
            (SolInstanceKind::Regular, InstanceKind::Item(_)) => (),

            (SolInstanceKind::DropEmpty, InstanceKind::DropGlue(_, None)) => (),
            (SolInstanceKind::DropGlued(dep_ty), InstanceKind::DropGlue(_, Some(ty))) => {
                if dep_ty.as_ref() != &temp_builder.mk_type(ty) {
                    info!("- skipped, drop glued type mismatch");
                    continue;
                }
            }

            (SolInstanceKind::VTableShim, InstanceKind::VTableShim { .. }) => (),
            (SolInstanceKind::ReifyShim, InstanceKind::ReifyShim { .. }) => (),
            (SolInstanceKind::ClosureOnceShim, InstanceKind::ClosureOnceShim { .. }) => (),
            (SolInstanceKind::FnPtrShim(dep_ty), InstanceKind::FnPtrShim(_, ty)) => {
                if dep_ty.as_ref() != &temp_builder.mk_type(ty) {
                    info!("- skipped, fn ptr shim type mismatch");
                    continue;
                }
            }
            (SolInstanceKind::FnPtrAddrShim(dep_ty), InstanceKind::FnPtrAddrShim(_, ty)) => {
                if dep_ty.as_ref() != &temp_builder.mk_type(ty) {
                    info!("- skipped, fn ptr addr shim type mismatch");
                    continue;
                }
            }
            (SolInstanceKind::CloneShim(dep_ty), InstanceKind::CloneShim(_, ty)) => {
                if dep_ty.as_ref() != &temp_builder.mk_type(ty) {
                    info!("- skipped, clone shim type mismatch");
                    continue;
                }
            }

            _ => {
                info!("- skipped, instance kind mismatch");
                continue;
            }
        }

        // end of the loop
        matched = true;
        break;
    }

    // exit early if no dependency matched
    if !matched {
        info!("- skipped, not in the dependency list");
        return;
    }

    // now resolve the context
    info!("- processing dependency {def_desc}");
    let mut builder = SolContextBuilder::new(tcx, sol);
    builder.make_instance(instance);

    let (sol, context) = builder.build();
    let ctxt_file = sol.context_to_file(&context);
    info!("- done with dependency {def_desc}, context saved at {}", ctxt_file.display());
}
