use std::fs::{self, OpenOptions};
use std::path::Path;

use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::mir::{Body, MirPhase, RuntimePhase};
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;

/// A complete dump of both the control-flow graph and the call graph of the compilation context
pub fn dump(tcx: TyCtxt<'_>, outdir: &Path) {
    // prepare directory
    fs::create_dir_all(outdir).expect("unable to create output directory");

    // extract the mir for each function
    let mut summary = CrateSummary {
        functions: Vec::new(),
    };
    for idx in tcx.mir_keys(()) {
        let body = tcx.optimized_mir(idx.to_def_id());
        summary.functions.push(FunctionSummary::process(tcx, body));
    }

    // dump output
    let symbol = tcx.crate_name(LOCAL_CRATE);
    let crate_name = symbol.as_str();
    let output = outdir.join(crate_name).with_extension("json");
    let _file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .expect("unable to create output file");
}

/// A struct containing serializable information about the entire crate
#[derive(Encodable)]
struct CrateSummary {
    functions: Vec<FunctionSummary>,
}

/// A struct containing serializable information about one function
#[derive(Encodable)]
struct FunctionSummary {
    id: DefId,
    name: Option<String>,
}

impl FunctionSummary {
    /// Process the mir body for one function
    fn process<'tcx>(tcx: TyCtxt<'_>, body: &Body<'tcx>) -> Self {
        // sanity check
        if !matches!(body.phase, MirPhase::Runtime(RuntimePhase::Optimized)) {
            panic!("MIR not optimized");
        }

        // get the basics
        let id = body.source.def_id();
        let name = tcx.opt_item_name(id).map(|s| s.to_string());

        // done
        FunctionSummary { id, name  }
    }
}