use std::fs;
use std::path::PathBuf;

use rustc_middle::bug;
use rustc_middle::mir::coverage::CoverageKind;
use rustc_middle::mir::{Body, Statement, StatementKind};
use rustc_middle::ty::{Instance, InstanceKind, TyCtxt};
use tempfile::tempdir;
use tracing::{info, warn};

use crate::solana::common::{BuildSystem, Phase, SolEnv};
use crate::solana::context::{SolContextBuilder, SolDeps, SolInstanceKind};

pub(crate) fn should_instrument<'tcx>(
    tcx: TyCtxt<'tcx>,
    path: PathBuf,
    instance: Instance<'tcx>,
) -> bool {
    // debug marking
    let def_id = instance.def_id();
    let def_desc = tcx.def_path_str_with_args(def_id, instance.args);
    info!("processing {def_desc}");

    // load the targets
    let content = fs::read_to_string(path)
        .unwrap_or_else(|e| bug!("[invariant] failed to read instrumentation targets: {e}"));
    let deps: SolDeps = serde_json::from_str(&content)
        .unwrap_or_else(|e| bug!("[invariant] failed to deserialize deps.json: {e}"));

    // derive the identifier
    let inst_ident = SolContextBuilder::mk_ident_no_cache(tcx, def_id);

    // create a dummy env for temporary phase
    let temp_dir = tempdir().expect("[invariant] failed to create temporary output directory");
    let env = SolEnv {
        build_system: BuildSystem::Anchor, // dummy
        phase: Phase::Temporary,
        source_dir: PathBuf::new(),
        src_file_name: PathBuf::new(),
        src_path_full: PathBuf::new(),
        output_dir: temp_dir.path().to_path_buf(),
    };

    // compare with the target list
    let mut matched = false;
    for (dep_kind, dep_ident, dep_args, _) in &deps.fn_deps {
        if dep_ident != &inst_ident {
            continue;
        }

        // attempt to resolve the instance
        info!("- attempting target {def_desc}");
        let mut temp_builder = SolContextBuilder::new(tcx, env.with_temporary_phase());

        // try to match the type arguments
        if dep_args.len() != instance.args.len() {
            info!("- skipped, type argument count mismatch");
            continue;
        }

        let mut ty_args_matched = true;
        for (dep_arg, ty_arg) in dep_args.iter().zip(instance.args.iter()) {
            let def_arg = temp_builder.mk_ty_arg(ty_arg);
            if dep_arg != &def_arg {
                ty_args_matched = false;
                break;
            }
        }
        if !ty_args_matched {
            info!("- skipped, type arguments mismatch");
            continue;
        }

        // try to match the instance kind
        match (dep_kind, instance.def) {
            (SolInstanceKind::Regular, InstanceKind::Item(_)) => (),

            (SolInstanceKind::DropEmpty, InstanceKind::DropGlue(_, None)) => (),
            (SolInstanceKind::DropGlued(dep_ty), InstanceKind::DropGlue(_, Some(ty))) => {
                if dep_ty.as_ref() != &temp_builder.mk_type(ty) {
                    info!("- skipped, drop glued type mismatch");
                    continue;
                }
            }

            (SolInstanceKind::VTableShim, InstanceKind::VTableShim { .. }) => (),
            (SolInstanceKind::ReifyShim, InstanceKind::ReifyShim { .. }) => (),
            (SolInstanceKind::ClosureOnceShim, InstanceKind::ClosureOnceShim { .. }) => (),
            (SolInstanceKind::FnPtrShim(dep_ty), InstanceKind::FnPtrShim(_, ty)) => {
                if dep_ty.as_ref() != &temp_builder.mk_type(ty) {
                    info!("- skipped, fn ptr shim type mismatch");
                    continue;
                }
            }
            (SolInstanceKind::FnPtrAddrShim(dep_ty), InstanceKind::FnPtrAddrShim(_, ty)) => {
                if dep_ty.as_ref() != &temp_builder.mk_type(ty) {
                    info!("- skipped, fn ptr addr shim type mismatch");
                    continue;
                }
            }
            (SolInstanceKind::CloneShim(dep_ty), InstanceKind::CloneShim(_, ty)) => {
                if dep_ty.as_ref() != &temp_builder.mk_type(ty) {
                    info!("- skipped, clone shim type mismatch");
                    continue;
                }
            }

            _ => {
                info!("- skipped, instance kind mismatch");
                continue;
            }
        }

        // end of the loop
        matched = true;
        break;
    }

    // exit early if no targets matched
    if !matched {
        info!("- skipped, not in the target list");
        return false;
    }

    // now we instrument
    warn!("- instrumenting target {def_desc}");
    true
}

/// Coverage kind for PC tracking
const COV_KIND_PC: u16 = 1;

/// Apply code coverage instrumentation to the given MIR body
pub(crate) fn codecov<'tcx>(
    _tcx: TyCtxt<'tcx>,
    _instance: Instance<'tcx>,
    mut body: Body<'tcx>,
) -> Body<'tcx> {
    // add coverage tracking at the beginning of each basic block
    for (i, block) in body.basic_blocks.as_mut_preserves_cfg().iter_enumerated_mut() {
        let cov = CoverageKind::SolMarker { kind: COV_KIND_PC, value: i.as_u32() };
        block.statements.insert(
            0,
            Statement {
                source_info: block.terminator().source_info,
                kind: StatementKind::Coverage(cov),
            },
        );
    }

    // done with the instrumentation
    body
}
