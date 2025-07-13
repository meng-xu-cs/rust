mod anchor;

use std::env;
use std::fmt::Display;
use std::path::PathBuf;

use rustc_middle::bug;
use rustc_middle::ty::TyCtxt;
use rustc_span::FileNameDisplayPreference;
use tracing::info;

const COMPONENT_NAME: &str = "solanalysis";

/// Known build systems
enum BuildSystem {
    Anchor,
}

impl Display for BuildSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Anchor => write!(f, "anchor"),
        }
    }
}

/// Entrypoint fo the solana-specific logic
pub(crate) fn entrypoint(tcx: TyCtxt<'_>) {
    // enable the component is explicitly enabled via environment variable
    let env_prefix = COMPONENT_NAME.to_uppercase();
    match env::var_os(&env_prefix) {
        None => return,
        Some(val) => {
            match val
                .into_string()
                .unwrap_or_else(|_| {
                    bug!("environment variable {env_prefix} is not a valid utf-8 string")
                })
                .as_str()
            {
                "0" | "false" | "no" | "off" => {
                    return;
                }

                "1" | "true" | "yes" | "on" => (),
                others => {
                    bug!("unexpected value for {env_prefix}: {others}");
                }
            }
        }
    };

    // grab information from the environment variables
    let build_system = match env::var(format!("{env_prefix}_BUILD_SYSTEM")) {
        Ok(val) => match val.as_str() {
            "anchor" => BuildSystem::Anchor,
            _ => bug!("unexpected build system: {val}"),
        },
        Err(e) => bug!("unable to locate build system in environment variables: {e}"),
    };
    let _output_dir = match env::var_os(format!("{env_prefix}_OUTPUT_DIR")) {
        None => bug!("unable to locate output directory in environment variables"),
        Some(val) => PathBuf::from(val),
    };
    let source_dir = match env::var_os(format!("{env_prefix}_SOURCE_DIR")) {
        None => bug!("unable to locate source directory in environment variables"),
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
    if !src_path_full.starts_with(&source_dir) {
        return;
    }

    // now enters the actual workflow
    info!("processing {} through {build_system}", src_file_name.display());
    match build_system {
        BuildSystem::Anchor => anchor::process_compilation(tcx),
    }
}
