use rustc_hir::def::DefKind;
use rustc_middle::bug;
use rustc_middle::ty::{Instance, InstanceKind, TyCtxt};
use tracing::{info, warn};

use crate::solana::common::SolEnv;
use crate::solana::context::{SolContextBuilder, SolSplTestCase};

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

    // check attributes
    let mut has_attr_test = false;
    let mut has_attr_ignore = false;
    let mut has_attr_should_panic = false;
    for attr in tcx.get_all_attrs(def_id) {
        let name = match attr.name() {
            None => continue,
            Some(n) => n,
        };
        match name.as_str() {
            "test" => has_attr_test = true,
            "ignore" => has_attr_ignore = true,
            "should_panic" => has_attr_should_panic = true,
            _ => continue,
        }
    }

    if !has_attr_test {
        info!("- skipped, missing `test` attribute");
        return;
    }

    // reveal function definition
    let body = tcx.instance_mir(instance.def);

    // expect zero parameters for the test function
    if body.arg_count != 0 {
        bug!("[invariant] expect a test function to have zero parameters");
    }

    // check whether the test case is ignored
    if has_attr_ignore {
        info!("- skipped, `ignore = ...` attribute present");
        return;
    }

    // collect information
    warn!("- found test case {def_desc}");

    let mut builder = SolContextBuilder::new(tcx, sol);
    let (_, inst_ident, _) = builder.make_instance(instance);
    let (sol, context) = builder.build();

    // serialize the context to file
    let ctxt_file = sol.context_to_file(&context);

    // save the metadata to file
    let test_case = SolSplTestCase { function: inst_ident, expect_panic: has_attr_should_panic };
    sol.summary_to_file("test_case", &test_case);

    // done
    info!("- done with instruction {def_desc}, context saved at {}", ctxt_file.display());
}
