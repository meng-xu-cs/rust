use rustc_middle::ty::{self, Instance, TyCtxt};

pub(crate) mod common;
pub(crate) mod typing;

pub(crate) mod pipeline_anchor;

/// Entrypoint for the solana-specific codegen logic
pub(crate) fn entrypoint<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) {
    let sol = match common::retrieve_context(tcx) {
        None => return,
        Some(ctxt) => ctxt,
    };

    // convert the instance to monomorphised MIR
    let instance_mir = tcx.instance_mir(instance.def).clone();
    let mut monomorphised_mir = instance.instantiate_mir_and_normalize_erasing_regions(
        tcx,
        ty::TypingEnv::fully_monomorphized(),
        ty::EarlyBinder::bind(instance_mir),
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
