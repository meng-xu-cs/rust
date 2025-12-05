pub(crate) mod common;
pub(crate) mod context;

use rustc_middle::ty::TyCtxt;
use tracing::warn;

use crate::nlai::common::{COMPONENT_NAME, retrieve_env};
use crate::nlai::context::Builder;

/// Entrypoint for nlai information collection
pub(crate) fn entrypoint<'tcx>(tcx: TyCtxt<'tcx>) {
    // retrieve the context
    let env = match retrieve_env(tcx) {
        None => return,
        Some(env) => env,
    };
    warn!("{COMPONENT_NAME} context: {env}");

    // build the module
    let builder = Builder::new(tcx);
    let module = builder.build();

    // emit the module to the output directory
    env.serialize_module(&module);
}
