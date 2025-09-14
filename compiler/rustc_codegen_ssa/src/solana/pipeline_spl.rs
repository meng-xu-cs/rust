use rustc_hir::def::DefKind;
use rustc_middle::bug;
use rustc_middle::mir::{Const, ConstValue, Operand, START_BLOCK, TerminatorKind};
use rustc_middle::ty::{self, Instance, InstanceKind, TyCtxt};
use tracing::{info, warn};

use crate::solana::common::SolEnv;
use crate::solana::context::{SolContextBuilder, SolSplTestEntrypoint};

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

    // the entrypoint of a test must be a "main" function
    if def_desc.as_str() != "main" {
        info!("- skipped, not the main function for rust unit tests");
        return;
    }

    // reveal function definition
    let body = tcx.instance_mir(instance.def);

    // expect zero parameters for the test function
    if body.arg_count != 0 {
        bug!("[invariant] expect a test function to have zero parameters");
    }

    // check the terminator of the first basic block
    let entry_block = body
        .basic_blocks
        .get(START_BLOCK)
        .unwrap_or_else(|| bug!("[invariant] missing entry block for main function"));

    let callee = match entry_block.terminator.as_ref().map(|t| &t.kind) {
        Some(TerminatorKind::Call { func, .. }) => func,
        _ => {
            info!("- skipped, entry block does not end with a function call");
            return;
        }
    };

    // try to derive the callee name
    let callee_ty = match callee {
        Operand::Constant(c) => match c.const_ {
            Const::Val(ConstValue::ZeroSized, const_ty) => const_ty,
            _ => {
                info!("- skipped, entry block calls a non-constant function");
                return;
            }
        },
        Operand::Copy(..) | Operand::Move(..) => {
            info!("- skipped, entry block ends with an indirect call");
            return;
        }
    };
    let callee_def_id = match callee_ty.kind() {
        ty::FnDef(callee_def_id, callee_ty_args) => {
            if !callee_ty_args.is_empty() {
                info!("- skipped, entry block calls a generic function");
                return;
            }
            callee_def_id
        }
        _ => {
            info!("- skipped, entry block calls a different function");
            return;
        }
    };
    let callee_desc = tcx.def_path_str_with_args(callee_def_id, &[]);
    if callee_desc != "test_main_static" {
        info!("- skipped, entry block not calling 'test_main_static'");
        return;
    }

    // collect information
    warn!("- found a test driver");
    let mut builder = SolContextBuilder::new(tcx, sol);
    let (_, inst_ident, _) = builder.make_instance(instance);
    let (sol, context) = builder.build();

    // serialize the context to file
    let ctxt_file = sol.context_to_file(&context);

    // save the metadata to file
    let entrypoint = SolSplTestEntrypoint { function: inst_ident };
    sol.summary_to_file("entrypoint", &entrypoint);

    // done
    info!("- done with unittest entrypoint, context saved at {}", ctxt_file.display());
}
