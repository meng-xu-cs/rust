use std::collections::BTreeSet;
use std::path::PathBuf;

use serde_json::json;

use super::*;

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
