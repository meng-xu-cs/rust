use rustc_hir::def::DefKind;
use rustc_middle::ty::{Instance, InstanceKind, TyCtxt};
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

    // the entrypoint of a test must be a function prefixed with "selftest_"
    if !def_desc.as_str().contains("selftest_") {
        info!("- skipped, not a selftest function");
        return;
    }

    // collect information
    warn!("- found a selftest function: {def_desc}");
    let mut builder = SolContextBuilder::new(tcx, sol);
    let (_, inst_ident, _) = builder.make_instance(instance);
    let (sol, context) = builder.build();

    // serialize the context to file
    let ctxt_file = sol.context_to_file(&context);

    // save the metadata to file
    let entrypoint = SolSplTestEntrypoint { function: inst_ident };
    sol.summary_to_file("selftest", &entrypoint);

    // done
    info!("- done with selftest {def_desc}, context saved at {}", ctxt_file.display());
}
