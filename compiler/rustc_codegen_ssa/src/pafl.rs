use std::path::Path;

use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::ty::TyCtxt;

/// A complete dump of both the control-flow graph and the call graph of the compilation context
pub fn dump(tcx: TyCtxt<'_>, _outdir: &Path) {
    info!("Codegen for crate: {}", tcx.crate_name(LOCAL_CRATE));
}