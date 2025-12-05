use std::fmt::Display;
use std::path::PathBuf;
use std::{env, fs};

use rustc_middle::bug;
use rustc_middle::ty::TyCtxt;
use rustc_span::FileNameDisplayPreference;
use serde::Serialize;

/// The name of the component
pub(crate) const COMPONENT_NAME: &str = "nlai";

/// Context for nlai information collection
pub(crate) struct SolEnv {
    input_path: PathBuf,
    output_dir: PathBuf,
}

/// Obtain nlai context from environment variables
pub(crate) fn retrieve_env(tcx: TyCtxt<'_>) -> Option<SolEnv> {
    // enable the component is explicitly enabled via environment variable
    let env_prefix = COMPONENT_NAME.to_uppercase();
    match env::var_os(&env_prefix)?
        .into_string()
        .unwrap_or_else(|_| {
            bug!("[user-input] environment variable {env_prefix} is not a valid utf-8 string")
        })
        .as_str()
    {
        "0" | "false" | "no" | "off" => {
            return None;
        }

        "1" | "true" | "yes" | "on" => (),
        others => {
            bug!("[user-input] unexpected value for {env_prefix}: {others}");
        }
    };

    // grab information from the environment variables
    let output_dir = match env::var_os(format!("{env_prefix}_OUTPUT_DIR")) {
        None => bug!("[user-input] unable to locate output directory in environment variables"),
        Some(val) => PathBuf::from(val),
    };

    // retrieve the full input path
    let input_path = match tcx.sess.local_crate_source_file() {
        None => bug!("[invariant] unable to locate local crate source file"),
        Some(src) => src.local_path_if_available().canonicalize().unwrap_or_else(|e| {
            bug!(
                "[invariant] failed to canonicalize source path for {}: {e}",
                src.to_path(FileNameDisplayPreference::Local).display()
            )
        }),
    };

    // return the context
    Some(SolEnv { input_path, output_dir })
}

impl Display for SolEnv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{ input_path: {}, output_dir: {} }}",
            self.input_path.display(),
            self.output_dir.display()
        )
    }
}

impl SolEnv {
    /// Return the output directory for a fresh item
    fn fresh_output_dir(&self, prefix: &str) -> PathBuf {
        let mut counter = 0;
        loop {
            let subdir = self.output_dir.join(format!("{prefix}{counter}"));
            match fs::create_dir(&subdir) {
                Ok(()) => return subdir,
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    counter += 1;
                    continue;
                }
                Err(e) => {
                    bug!("[invariant] failed to create output directory {prefix}{counter}: {e}");
                }
            }
        }
    }

    /// Serialize data to a JSON file
    fn serialize_to_file<T: Serialize>(&self, dpx: &str, tag: &str, data: &T) -> PathBuf {
        let file_outdir = self.fresh_output_dir(dpx);
        let file_path = file_outdir.join(format!("{tag}.json"));
        if file_path.exists() {
            bug!("[invariant] file {file_path:?} already exists");
        }

        let json_data = serde_json::to_string_pretty(data)
            .unwrap_or_else(|e| bug!("[invariant] failed to serialize data to JSON: {e}"));
        fs::write(&file_path, json_data).unwrap_or_else(|e| {
            bug!("[invariant] failed to write JSON to file {}: {e}", file_path.display())
        });

        file_path
    }

    /// Serialize a module to a file
    pub(crate) fn serialize_module<T: Serialize>(&self, data: &T) -> PathBuf {
        self.serialize_to_file("f", "module", data)
    }
}
