use rustc_middle::bug;
use rustc_middle::ty::{Instance, TyCtxt};

pub(crate) mod common;
pub(crate) mod context;
pub(crate) mod pipeline_anchor;
pub(crate) mod pipeline_shared;
pub(crate) mod pipeline_spl;

use common::{BuildSystem, Phase, retrieve_env};

/// Entrypoint for the solana-specific codegen logic
pub(crate) fn entrypoint<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) {
    let sol = match retrieve_env(tcx) {
        None => return,
        Some(ctxt) => ctxt,
    };

    // work on the monomorphised MIR
    match sol.phase {
        Phase::Bootstrap => match sol.build_system {
            BuildSystem::Spl => pipeline_spl::phase_bootstrap(tcx, sol, instance),
            BuildSystem::Anchor => pipeline_anchor::phase_bootstrap(tcx, sol, instance),
        },
        Phase::Expansion(_) => pipeline_shared::phase_expansion(tcx, sol, instance),
        Phase::Temporary => bug!("[invariant] unexpected temporary phase"),
    }
}
