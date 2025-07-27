use rustc_middle::mir::{MirPhase, RuntimePhase};
use rustc_middle::ty::{self, Instance, TyCtxt};

pub(crate) mod common;
pub(crate) mod ident;
pub(crate) mod typing;

pub(crate) mod pipeline_anchor;

/// Entrypoint for the solana-specific codegen logic
pub(crate) fn entrypoint<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) {
    let sol = match common::retrieve_context(tcx) {
        None => return,
        Some(ctxt) => ctxt,
    };

    // skip shims and other non-user defined items
    let instance_mir = tcx.instance_mir(instance.def);
    if !matches!(instance_mir.phase, MirPhase::Runtime(RuntimePhase::Optimized)) {
        return;
    }

    // convert the instance to monomorphised MIR
    let mut monomorphised_mir = instance.instantiate_mir_and_normalize_erasing_regions(
        tcx,
        instance_mir.typing_env(tcx),
        ty::EarlyBinder::bind(instance_mir.clone()),
    );

    // work on the monomorphised MIR
    match sol.build_system {
        common::BuildSystem::Anchor => match sol.phase {
            common::Phase::Bootstrap => {
                pipeline_anchor::phase_bootstrap(tcx, sol, &mut monomorphised_mir)
            }
        },
    }

    // FIXME: return the monomorphised MIR for further processing
}
