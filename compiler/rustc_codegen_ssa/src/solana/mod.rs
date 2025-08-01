use rustc_middle::ty::{Instance, TyCtxt};

pub(crate) mod builtin;
pub(crate) mod common;
pub(crate) mod context;
pub(crate) mod pipeline_anchor;

/// Entrypoint for the solana-specific codegen logic
pub(crate) fn entrypoint<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) {
    let sol = match common::retrieve_env(tcx) {
        None => return,
        Some(ctxt) => ctxt,
    };

    // work on the monomorphised MIR
    match sol.build_system {
        common::BuildSystem::Anchor => match sol.phase {
            common::Phase::Bootstrap => {
                pipeline_anchor::phase_bootstrap(tcx, sol, instance);
            }
        },
    }
}
