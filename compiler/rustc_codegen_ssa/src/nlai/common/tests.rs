use std::collections::BTreeSet;
use std::path::PathBuf;

use serde_json::json;

use super::*;

#[test]
fn u1_2c2a_extractor_uses_canonical_compile_time_source_identity() {
    let Some(compiled) = rustc_session::nlai_rust_source_state_fingerprint() else {
        return;
    };
    // Plain source tarballs and ordinary non-NLAI compiler builds have no Git worktree to attest.
    // Their explicit sentinel remains visible through `rustc -vV`, while extractor activation
    // still rejects it. Exercise the compiled-identity invariant only for provenance-aware builds.
    if compiled == "unknown" {
        return;
    }
    let fingerprint = compiled_source_state_fingerprint();
    assert_eq!(fingerprint, compiled);
    assert_eq!(fingerprint.len(), 64);
    assert!(fingerprint.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()));
}

#[test]
fn u1_2b2_producer_envelope_binds_compiled_schema_identity() {
    let envelope = artifact_envelope("payload");

    assert_eq!(envelope.protocol_version, NLAI_ARTIFACT_PROTOCOL_VERSION);
    assert_eq!(envelope.schema_version, NLAI_IR_SCHEMA_VERSION);
    assert_eq!(envelope.schema_fingerprint, schema::fingerprint());
    assert_eq!(envelope.payload, "payload");
}

#[test]
fn u1_2b2_serialize_crate_emits_exact_envelope_shape() {
    let output = tempfile::tempdir().unwrap();
    let env =
        SolEnv { input_path: PathBuf::from("source.rs"), output_dir: output.path().to_path_buf() };

    let artifact = env.serialize_crate(&json!({ "sentinel": true }));
    let encoded = fs::read_to_string(artifact).unwrap();
    let value: serde_json::Value = serde_json::from_str(&encoded).unwrap();
    let object = value.as_object().unwrap();
    let keys = object.keys().map(String::as_str).collect::<BTreeSet<_>>();

    assert_eq!(
        keys,
        BTreeSet::from(["payload", "protocol_version", "schema_fingerprint", "schema_version",])
    );
    assert_eq!(object["protocol_version"], NLAI_ARTIFACT_PROTOCOL_VERSION);
    assert_eq!(object["schema_version"], NLAI_IR_SCHEMA_VERSION);
    assert_eq!(object["schema_fingerprint"], schema::fingerprint());
    assert_eq!(object["payload"], json!({ "sentinel": true }));
}
