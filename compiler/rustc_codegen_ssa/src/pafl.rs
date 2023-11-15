use std::fs::{self, OpenOptions};
use std::path::Path;

use rustc_hir::def::DefKind;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::mir::{Body, MirPhase, RuntimePhase};
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;

/// A complete dump of both the control-flow graph and the call graph of the compilation context
pub fn dump(tcx: TyCtxt<'_>, outdir: &Path) {
    // prepare directory
    fs::create_dir_all(outdir).expect("unable to create output directory");

    // extract the mir for each function
    let mut summary = CrateSummary { functions: Vec::new() };
    for idx in tcx.mir_keys(()) {
        let def_id = idx.to_def_id();
        match tcx.def_kind(def_id) {
            // constants do not really have a function body
            DefKind::Const
            | DefKind::AssocConst
            | DefKind::AnonConst
            | DefKind::InlineConst
            | DefKind::Static(_) => {
                continue;
            }
            DefKind::Ctor(..)
            | DefKind::Fn
            | DefKind::AssocFn
            | DefKind::Closure
            | DefKind::Coroutine => (),
            dk => bug!("{:?} is not a MIR-ready node: {:?}", def_id, dk),
        };

        // sanity check
        let body = tcx.optimized_mir(def_id);
        let name = tcx.opt_item_name(def_id).map(|s| s.to_string());
        if !matches!(body.phase, MirPhase::Runtime(RuntimePhase::Optimized)) {
            bug!(
                "{:?} - {} MIR is at phase {:?}",
                def_id,
                name.as_ref().map_or("<unnamed>", |n| n.as_str()),
                body.phase,
            );
        }

        // handle function definition
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
        // get the basics
        let id = body.source.def_id();
        let name = tcx.opt_item_name(id).map(|s| s.to_string());

        // done
        FunctionSummary { id, name }
    }
}
