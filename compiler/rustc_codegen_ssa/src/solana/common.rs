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
    Expansion(usize),
    Temporary,
}

impl Display for Phase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bootstrap => write!(f, "bootstrap"),
            Self::Expansion(n) => write!(f, "expansion{n}"),
            Self::Temporary => write!(f, "temporary"),
        }
    }
}

/// Build context for Solana
pub(crate) struct SolEnv {
    pub build_system: BuildSystem,
    pub phase: Phase,
    pub source_dir: PathBuf,
    pub src_file_name: PathBuf,
    pub src_path_full: PathBuf,
    pub output_dir: PathBuf,
}

/// Entrypoint to the solana-specific logic
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
        Ok(val) => {
            if val.as_str() == "bootstrap" {
                Phase::Bootstrap
            } else {
                match val.strip_prefix("expansion") {
                    None => bug!("[user-input] unexpected phase: {val}"),
                    Some(num) => {
                        let n: usize = num.parse().unwrap_or_else(|_| {
                            bug!("[user-input] unable to parse phase number from {val}")
                        });
                        Phase::Expansion(n)
                    }
                }
            }
        }
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

    /// Return the output directory for a fresh item
    fn fresh_output_dir(&self, prefix: &str) -> PathBuf {
        let phase_output = self.phase_output_dir();
        let mut counter = 0;
        loop {
            let subdir = phase_output.join(format!("{prefix}{counter}"));
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

    pub(crate) fn prev_phase_output_dir(&self) -> PathBuf {
        let phase = match self.phase {
            Phase::Bootstrap => bug!("[invariant] no phase before bootstrap"),
            Phase::Expansion(0) => Phase::Bootstrap,
            Phase::Expansion(n) => Phase::Expansion(n - 1),
            Phase::Temporary => bug!("[invariant] no phase before temporary"),
        };
        self.output_dir.join(phase.to_string())
    }

    pub(crate) fn save_instance_info<'tcx>(&self, tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
        // prepare for output directory
        let instance_outdir = self.fresh_output_dir("i");

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

    pub(crate) fn serialize_to_file<T: Serialize>(&self, tag: &str, data: &T) -> PathBuf {
        let file_outdir = self.fresh_output_dir("f");
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

    pub(crate) fn with_temporary_phase(&self) -> Self {
        let result = Self {
            build_system: self.build_system,
            phase: Phase::Temporary,
            source_dir: self.source_dir.clone(),
            src_file_name: self.src_file_name.clone(),
            src_path_full: self.src_path_full.clone(),
            output_dir: self.output_dir.clone(),
        };

        // re-create phase output directory
        let workdir = result.phase_output_dir();
        if workdir.exists() {
            fs::remove_dir_all(&workdir).unwrap_or_else(|e| {
                bug!("[invariant] failed to remove previous phase output directory: {e}")
            });
        }
        fs::create_dir(workdir).unwrap_or_else(|e| {
            bug!("[invariant] failed to create temporary phase output directory: {e}")
        });

        // return the updated env
        result
    }
}
