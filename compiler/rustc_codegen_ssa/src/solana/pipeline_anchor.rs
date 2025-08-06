//! This pass encapsulate all the transformations related to Solana's Anchor framework.
use std::fs;

use rustc_hir::def::DefKind;
use rustc_middle::ty::{Instance, InstanceKind, TyCtxt};
use rustc_middle::{bug, ty};
use tracing::{info, warn};

use crate::solana::common::SolEnv;
use crate::solana::context::{SolContextBuilder, SolDeps};

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
    if !matches!(ty_state.kind(), ty::Adt(_, _)) {
        bug!("[assumption] expect the state type to be an adt, found: {ty_state}");
    }

    // collect information
    warn!("- found instruction {def_desc} with argument {ty_state}");

    let mut builder = SolContextBuilder::new(tcx, sol);
    builder.make_instance(instance);
    let (sol, context) = builder.build();

    // serialize the information to file
    sol.serialize_to_file("context", &context);
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

    // an instruction must be a `fn` or `assoc fn` item
    match instance.def {
        InstanceKind::Item(did) if matches!(tcx.def_kind(did), DefKind::Fn | DefKind::AssocFn) => {}
        _ => {
            info!("- skipped, not a fn-alike item");
            return;
        }
    };

    let inst_desc = SolContextBuilder::mk_inst_desc(tcx, def_id, instance.args);

    // load the deps
    let path = sol.prev_phase_output_dir().join("deps.json");
    let content = fs::read_to_string(path).unwrap_or_else(|e| {
        bug!("[invariant] failed to read previous phase output file deps.json: {e}")
    });
    let deps: SolDeps = serde_json::from_str(&content)
        .unwrap_or_else(|e| bug!("[invariant] failed to deserialize deps.json: {e}"));

    // check if the current instance is a dependency
    let mut builder = SolContextBuilder::new(tcx, sol);
    let mut matched = false;
    for (dep_ident, dep_args, dep_desc) in &deps.fn_deps {
        if dep_desc != &inst_desc {
            continue;
        }

        // now build the context around this instance
        warn!("- found dependency {def_desc}");
        let (def_ident, def_args) = builder.make_instance(instance);

        // now also check if the ident and type arguments match
        if dep_ident != &def_ident {
            bug!("[invariant] dependency ident mismatch on {def_desc}");
        }
        if dep_args != &def_args {
            bug!("[invariant] dependency type arguments mismatch on {def_desc}");
        }

        // end of the loop
        matched = true;
        break;
    }

    // serialize the context
    if matched {
        let (sol, context) = builder.build();
        sol.serialize_to_file("context", &context);
    }
}
