pub(crate) mod common;
// The synchronized schema deliberately uses plain `pub` so producer and consumer canonical bytes
// are identical. The enclosing module remains crate-private, so these items are not externally
// reachable despite their source-level visibility.
#[allow(unreachable_pub)]
pub(crate) mod context;
pub(crate) mod identity;
pub(crate) mod schema;

use std::env;

use rustc_middle::bug;
use rustc_middle::ty::TyCtxt;
use rustc_middle::ty::print::with_no_trimmed_paths;
use tracing::warn;

use crate::nlai::common::retrieve_env;
use crate::nlai::context::build;

const COMPONENT_NAME: &str = "nlai";
const ACTIVATION_VARIABLE: &str = "NLAI";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Activation {
    Disabled,
    Enabled,
}

fn parse_activation(value: &str) -> Option<Activation> {
    match value {
        "0" | "false" | "no" | "off" => Some(Activation::Disabled),
        "1" | "true" | "yes" | "on" => Some(Activation::Enabled),
        _ => None,
    }
}

fn activation_from_environment() -> Option<Activation> {
    let value = env::var_os(ACTIVATION_VARIABLE)?.into_string().unwrap_or_else(|_| {
        bug!("[user-input] environment variable {ACTIVATION_VARIABLE} is not a valid utf-8 string")
    });
    Some(parse_activation(&value).unwrap_or_else(|| {
        bug!("[user-input] unexpected value for {ACTIVATION_VARIABLE}: {value}")
    }))
}

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
