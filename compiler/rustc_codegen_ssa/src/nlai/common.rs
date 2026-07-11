use std::fmt::Display;
use std::path::PathBuf;
use std::{env, fs};

use rustc_middle::bug;
use rustc_middle::ty::TyCtxt;
use serde::Serialize;

use super::context::{NLAI_ARTIFACT_PROTOCOL_VERSION, NLAI_IR_SCHEMA_VERSION, SolArtifactEnvelope};
use super::identity::producer_identity;
use super::{Activation, COMPONENT_NAME, activation_from_environment, schema};

/// Context for nlai information collection
pub(crate) struct SolEnv {
    input_path: PathBuf,
    output_dir: PathBuf,
}

/// Obtain nlai context from environment variables
pub(crate) fn retrieve_env(tcx: TyCtxt<'_>) -> Option<SolEnv> {
    // enable the component is explicitly enabled via environment variable
    let env_prefix = COMPONENT_NAME.to_uppercase();
    match activation_from_environment()? {
        Activation::Disabled => {
            return None;
        }
        Activation::Enabled => (),
    };
    // Fail at extractor activation rather than after THIR traversal if bootstrap omitted or
    // corrupted either compile-time identity or if the running executable cannot be measured.
    let _ = producer_identity();

    // grab information from the environment variables
    let output_dir = match env::var_os(format!("{env_prefix}_OUTPUT_DIR")) {
        None => bug!("[user-input] unable to locate output directory in environment variables"),
        Some(val) => PathBuf::from(val),
    };

    // retrieve the full input path
    let input_path = match tcx.sess.local_crate_source_file() {
        None => bug!("[invariant] unable to locate local crate source file"),
        Some(src) => {
            let local_path = src
                .local_path()
                .unwrap_or_else(|| bug!("[invariant] unable to get local path for local crate"));
            local_path.canonicalize().unwrap_or_else(|e| {
                bug!("[invariant] failed to canonicalize path for {}: {e}", local_path.display())
            })
        }
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

    /// Serialize a crate to a file
    pub(crate) fn serialize_crate<T: Serialize>(&self, data: &T) -> PathBuf {
        let envelope = artifact_envelope(data);
        self.serialize_to_file("f", "crate", &envelope)
    }

    /// Prepare source directory
    pub(crate) fn prepare_source_directory(&self) -> PathBuf {
        let path = self.fresh_output_dir("s");
        fs::create_dir_all(&path)
            .unwrap_or_else(|e| bug!("[invariant] failed to create source dir: {e}"));
        path
    }
}

fn artifact_envelope<T>(payload: T) -> SolArtifactEnvelope<T> {
    SolArtifactEnvelope {
        protocol_version: NLAI_ARTIFACT_PROTOCOL_VERSION,
        schema_version: NLAI_IR_SCHEMA_VERSION,
        schema_fingerprint: schema::fingerprint().to_owned(),
        payload,
    }
}

#[cfg(test)]
mod tests;
