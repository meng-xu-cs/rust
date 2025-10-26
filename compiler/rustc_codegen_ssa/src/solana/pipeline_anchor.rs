use rustc_hir::def::DefKind;
use rustc_middle::bug;
use rustc_middle::ty::{Instance, InstanceKind, TyCtxt};
use tracing::{info, warn};

use crate::solana::common::SolEnv;
use crate::solana::context::{SolContextBuilder, SolEntrypoint};

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

    // the `entry` function has the following signature
    // pub fn entry<'info>(
    //     program_id: &Pubkey,
    //     accounts: &'info [AccountInfo<'info>],
    //     data: &[u8],
    // ) -> anchor_lang::solana_program::entrypoint::ProgramResult

    match instance.def {
        InstanceKind::Item(did) if tcx.def_kind(did) == DefKind::Fn => (),
        _ => {
            info!("- skipped, not a fn item");
            return;
        }
    };
    if tcx.def_path_str(def_id) != "entry" {
        info!("- skipped, function name is not 'entry'");
        return;
    }
    if !instance.args.is_empty() {
        bug!("[assumption] expect `entry` to have no generics, found {}", instance.args.len());
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
}
