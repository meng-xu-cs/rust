use std::fmt::Display;
use std::fs::File;
use std::path::PathBuf;
use std::{env, fs};

use rustc_middle::bug;
use rustc_middle::mir::Body;
use rustc_middle::mir::graphviz::write_mir_fn_graphviz;
use rustc_middle::mir::pretty::{PrettyPrintMirOptions, write_mir_fn};
use rustc_middle::ty::TyCtxt;
use rustc_span::FileNameDisplayPreference;
use serde::Serialize;

const COMPONENT_NAME: &str = "solanalysis";

/// Supported build systems
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum BuildSystem {
    Anchor,
}

impl Display for BuildSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Anchor => write!(f, "anchor"),
        }
    }
}

/// Phase of execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Phase {
    Bootstrap,
}

impl Display for Phase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bootstrap => write!(f, "bootstrap"),
        }
    }
}

/// Build context for Solana
#[derive(Clone)]
pub(crate) struct SolEnv {
    pub build_system: BuildSystem,
    pub phase: Phase,
    pub source_dir: PathBuf,
    pub src_file_name: PathBuf,
    pub src_path_full: PathBuf,
    pub output_dir: PathBuf,
}

/// Entrypoint fo the solana-specific logic
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
    let build_system = match env::var(format!("{env_prefix}_BUILD_SYSTEM")) {
        Ok(val) => match val.as_str() {
            "anchor" => BuildSystem::Anchor,
            _ => bug!("[user-input] unexpected build system: {val}"),
        },
        Err(e) => bug!("[user-input] unable to locate build system in environment variables: {e}"),
    };
    let phase = match env::var(format!("{env_prefix}_PHASE")) {
        Ok(val) => match val.as_str() {
            "bootstrap" => Phase::Bootstrap,
            _ => bug!("[user-input] unexpected phase: {val}"),
        },
        Err(e) => bug!("[user-input] unable to locate phase in environment variables: {e}"),
    };
    let output_dir = match env::var_os(format!("{env_prefix}_OUTPUT_DIR")) {
        None => bug!("[user-input] unable to locate output directory in environment variables"),
        Some(val) => PathBuf::from(val),
    };
    let source_dir = match env::var_os(format!("{env_prefix}_SOURCE_DIR")) {
        None => bug!("[user-input] unable to locate source directory in environment variables"),
        Some(val) => PathBuf::from(val),
    };

    // additional checks on the output directory for the phase
    let phase_output_dir = output_dir.join(phase.to_string());
    if !phase_output_dir.exists() || !phase_output_dir.is_dir() {
        bug!("[invariant] invalid output directory for phase {phase}: {}", output_dir.display());
    }

    // retrieve the source file name and its full path
    let (src_file_name, src_path_full) = match tcx.sess.local_crate_source_file() {
        None => bug!("[invariant] unable to locate local crate source file"),
        Some(src) => {
            let file_name = src.to_path(FileNameDisplayPreference::Local).to_path_buf();
            let full_path = src.local_path_if_available().canonicalize().unwrap_or_else(|e| {
                bug!(
                    "[invariant] failed to canonicalize source path for {}: {e}",
                    file_name.display()
                )
            });
            (file_name, full_path)
        }
    };

    // return the context
    Some(SolEnv { build_system, phase, source_dir, src_file_name, src_path_full, output_dir })
}

impl SolEnv {
    /// Return the output directory for the current phase
    fn phase_output_dir(&self) -> PathBuf {
        self.output_dir.join(self.phase.to_string())
    }

    /// Return the output directory for a new instance
    fn instance_output_dir(&self) -> PathBuf {
        let phase_output = self.phase_output_dir();
        let mut counter = 0;
        loop {
            let subdir = phase_output.join(counter.to_string());
            match fs::create_dir(&subdir) {
                Ok(()) => return subdir,
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    counter += 1;
                    continue;
                }
                Err(e) => {
                    bug!("[invariant] failed to create output directory {counter}: {e}");
                }
            }
        }
    }

    pub(crate) fn save_instance_info<'tcx>(&self, tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
        // prepare for output directory
        let instance_outdir = self.instance_output_dir();

        // dump the MIR to file
        let mut file_mir = File::create(instance_outdir.join("body.mir"))
            .unwrap_or_else(|e| bug!("[invariant] failed to create MIR file: {e}"));
        write_mir_fn(
            tcx,
            body,
            &mut |_, _| Ok(()),
            &mut file_mir,
            PrettyPrintMirOptions::from_cli(tcx),
        )
        .unwrap_or_else(|e| bug!("[invariant] failed to write MIR to file: {e}"));

        // dump the CFG to file
        let mut file_dot = File::create(instance_outdir.join("body.dot"))
            .unwrap_or_else(|e| bug!("[invariant] failed to create Dot file: {e}"));
        write_mir_fn_graphviz(tcx, body, false, &mut file_dot)
            .unwrap_or_else(|e| bug!("[invariant] failed to write Dot to file: {e}"));
    }

    pub(crate) fn serialize_to_file<T: Serialize>(&self, tag: &str, data: &T) {
        let file_path = self.phase_output_dir().join(format!("{tag}.json"));
        let json_data = serde_json::to_string_pretty(data)
            .unwrap_or_else(|e| bug!("[invariant] failed to serialize data to JSON: {e}"));
        fs::write(&file_path, json_data).unwrap_or_else(|e| {
            bug!("[invariant] failed to write JSON to file {}: {e}", file_path.display())
        });
    }
}
