use std::env;
use std::path::PathBuf;

use rustc_middle::bug;
use rustc_middle::ty::TyCtxt;
use tracing::info;

const COMPONENT_NAME: &str = "solanalysis";

/// Entrypoint fo the solana-specific logic
pub(crate) fn solanalysis_entrypoint(tcx: TyCtxt<'_>) {
    // grab information from the environment variables
    let outdir = match env::var_os(COMPONENT_NAME.to_uppercase()) {
        None => return,
        Some(val) => PathBuf::from(val),
    };

    let prefix = match env::var_os(format!("{}_TARGET_PREFIX", COMPONENT_NAME.to_uppercase())) {
        None => bug!("unable to locate target prefix in environment variables"),
        Some(val) => PathBuf::from(val),
    };

    let src_path = match tcx.sess.local_crate_source_file() {
        None => bug!("unable to locate local crate source file"),
        Some(src) => match src.into_local_path() {
            None => bug!("local crate source file is not a local path"),
            Some(path) => path,
        },
    };

    // skip execution on source files that are not under the target prefix
    if !src_path.starts_with(&prefix) {
        return;
    }

    // now enters the actual workflow
    info!(
        "{COMPONENT_NAME} enabled for {} with output director: {}",
        src_path.display(),
        outdir.display()
    );
}
