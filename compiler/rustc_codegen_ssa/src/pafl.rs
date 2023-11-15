use std::fs::{self, OpenOptions};
use std::path::Path;

use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::ty::TyCtxt;

/// A complete dump of both the control-flow graph and the call graph of the compilation context
pub fn dump(tcx: TyCtxt<'_>, outdir: &Path) {
    // prepare directory
    fs::create_dir_all(outdir).expect("unable to create output directory");

    // derive output
    let symbol = tcx.crate_name(LOCAL_CRATE);
    let crate_name = symbol.as_str();
    let output = outdir.join(crate_name).with_extension("json");
    let _file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .expect("unable to create output file");

    // dump output
}
