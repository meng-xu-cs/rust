use std::env;
use std::path::PathBuf;

use rustc_middle::bug;
use rustc_middle::mir::mono::MonoItem;
use rustc_middle::ty::TyCtxt;
use rustc_span::FileNameDisplayPreference;
use tracing::info;

const COMPONENT_NAME: &str = "solanalysis";

/// Entrypoint fo the solana-specific logic
pub(crate) fn solanalysis_entrypoint(tcx: TyCtxt<'_>) {
    // grab information from the environment variables
    let _outdir = match env::var_os(COMPONENT_NAME.to_uppercase()) {
        None => return,
        Some(val) => PathBuf::from(val),
    };

    let prefix = match env::var_os(format!("{}_TARGET_PREFIX", COMPONENT_NAME.to_uppercase())) {
        None => bug!("unable to locate target prefix in environment variables"),
        Some(val) => PathBuf::from(val),
    };

    let (src_file_name, src_path_full) = match tcx.sess.local_crate_source_file() {
        None => bug!("unable to locate local crate source file"),
        Some(src) => {
            let file_name = src.to_path(FileNameDisplayPreference::Local).to_path_buf();
            let full_path = src.local_path_if_available().canonicalize().unwrap_or_else(|e| {
                bug!("failed to canonicalize source path for {}: {e}", file_name.display())
            });
            (file_name, full_path)
        }
    };

    // skip execution on source files that are not under the target prefix
    if !src_path_full.starts_with(&prefix) {
        return;
    }

    // now enters the actual workflow
    info!("running {COMPONENT_NAME} for {}", src_file_name.display());

    let partitions = tcx.collect_and_partition_mono_items(());
    for unit in partitions.codegen_units {
        info!("  -> codegen unit {}", unit.name());
        for (k, _v) in unit.items_in_deterministic_order(tcx) {
            match k {
                MonoItem::Fn(instance) => {
                    let id = instance.def_id();
                    info!("    - instance[{}]: {}", id.is_local(), tcx.def_path_str(id));
                }
                MonoItem::Static(id) => {
                    info!("    - static[{}]: {}", id.is_local(), tcx.def_path_str(id));
                }
                MonoItem::GlobalAsm(_) => {
                    bug!("[assumption] unexpected assembly item in codegen unit")
                }
            }
        }
        info!("  <- codegen unit {}", unit.name());
    }
}
