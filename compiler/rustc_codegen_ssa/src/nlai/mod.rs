pub(crate) mod common;
// The synchronized schema deliberately uses plain `pub` so producer and consumer canonical bytes
// are identical. The enclosing module remains crate-private, so these items are not externally
// reachable despite their source-level visibility.
#[allow(unreachable_pub)]
pub(crate) mod context;
pub(crate) mod schema;

use rustc_middle::ty::TyCtxt;
use rustc_middle::ty::print::with_no_trimmed_paths;
use tracing::warn;

use crate::nlai::common::{COMPONENT_NAME, retrieve_env};
use crate::nlai::context::build;

/// Entrypoint for nlai information collection
pub(crate) fn entrypoint<'tcx>(tcx: TyCtxt<'tcx>) {
    // retrieve the context
    let env = match retrieve_env(tcx) {
        None => return,
        Some(env) => env,
    };
    let schema_fingerprint = schema::fingerprint();
    warn!("{COMPONENT_NAME} context: {env}; schema fingerprint: {schema_fingerprint}");

    // NLAI renders rustc types as analysis data, not as diagnostics. Keep the entire extraction
    // walk out of rustc's diagnostics-only path-trimming query: using that query without later
    // emitting a diagnostic is itself a compiler invariant violation.
    let krate = with_no_trimmed_paths!(build(tcx, env.prepare_source_directory()));

    // emit the crate to the output directory
    env.serialize_crate(&krate);
}
