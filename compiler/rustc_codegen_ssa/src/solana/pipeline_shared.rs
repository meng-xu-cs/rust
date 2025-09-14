use std::fs;

use rustc_middle::bug;
use rustc_middle::ty::{Instance, InstanceKind, TyCtxt};
use tracing::info;

use crate::solana::common::SolEnv;
use crate::solana::context::{SolContextBuilder, SolDeps, SolInstanceKind};

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
