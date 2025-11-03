use rustc_middle::bug;
use rustc_middle::ty::{Instance, TyCtxt};

pub(crate) mod common;
pub(crate) mod context;
pub(crate) mod instrument;
pub(crate) mod pipeline_anchor;
pub(crate) mod pipeline_selftest;
pub(crate) mod pipeline_shared;
pub(crate) mod pipeline_spl;

use common::{BuildSystem, Phase, retrieve_command};

use crate::solana::common::SolCommand;

/// Entrypoint for the solana-specific codegen logic
pub(crate) fn entrypoint<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> bool {
    match retrieve_command(tcx) {
        None => false,
        Some(SolCommand::Analyze(env)) => {
            // work on the monomorphised MIR
            match env.phase {
                Phase::Bootstrap => match env.build_system {
                    BuildSystem::SelfTest => pipeline_selftest::phase_bootstrap(tcx, env, instance),
                    BuildSystem::Spl => pipeline_spl::phase_bootstrap(tcx, env, instance),
                    BuildSystem::Anchor => pipeline_anchor::phase_bootstrap(tcx, env, instance),
                },
                Phase::Expansion(_) => pipeline_shared::phase_expansion(tcx, env, instance),
                Phase::Temporary => bug!("[invariant] unexpected temporary phase"),
            }

            // indicate that we don't need to modify the instance
            false
        }
        Some(SolCommand::Instrument(extra)) => {
            instrument::should_instrument(tcx, extra.target, instance)
        }
    }
}
