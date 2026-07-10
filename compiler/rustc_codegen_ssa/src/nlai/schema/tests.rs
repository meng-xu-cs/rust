use super::*;

const HEADER: &str = concat!(
    "/* --- BEGIN OF NLAI ARTIFACT PROTOCOL --- */\n",
    "pub const NLAI_ARTIFACT_PROTOCOL_VERSION: u32 = 1;\n",
    "pub const NLAI_SCHEMA_CANONICALIZATION: &str = \"nlai-schema-c14n-v1\";\n",
    "pub const NLAI_SCHEMA_FINGERPRINT_ALGORITHM: &str = \"blake3\";\n",
    "pub const NLAI_SCHEMA_FINGERPRINT_CONTEXT: &str = \"nlai.ir.schema.blake3.v1\";\n",
    "#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]\n",
    "#[serde(deny_unknown_fields)]\n",
    "pub struct SolArtifactEnvelope<T> {\n",
    "    pub protocol_version: u32,\n",
    "    pub schema_version: u32,\n",
    "    pub schema_fingerprint: String,\n",
    "    pub payload: T,\n",
    "}\n",
    "/* --- END OF NLAI ARTIFACT PROTOCOL --- */\n",
    "pub const NLAI_IR_SCHEMA_VERSION: u32 = 1;\n",
);

fn fixture() -> String {
    format!(
        "{SYNC_BEGIN}\n{HEADER}/// Documentation stays identity-bearing.  \n\tpub struct Demo {{\n    pub value: u8,\t\n}}\n{SYNC_END}\n"
    )
}

#[test]
fn u1_2b1_canonicalization_has_the_cross_repository_vector() {
    let lf = fixture();
    let crlf = lf.replace('\n', "\r\n");
    let canonical = canonical_schema_source(&lf).unwrap();
    assert_eq!(canonical_schema_source(&crlf).unwrap(), canonical);
    assert_eq!(
        canonical_schema_source(&format!("{SYNC_BEGIN}\n{canonical}{SYNC_END}\n")).unwrap(),
        canonical
    );
    assert_eq!(
        derived_fingerprint(NLAI_SCHEMA_FINGERPRINT_CONTEXT, canonical.as_bytes()),
        "fd9dde61f1fe23e9ad0538366fae204bb87a87efc3949d2f718808215e1667d2"
    );
    assert_eq!(
        version_basis_fingerprints(&canonical).unwrap(),
        (
            "a5d4f92b20b4bc01d61ff6c20886fe8a840ce6d367ab4ba7dc73a70ebd200721".to_owned(),
            "934a26a4189c24e116cd10a6e935668685efd3f2611c6e39060c7f23a47cbc73".to_owned(),
        )
    );
}

#[test]
fn u1_2b1_schema_markers_must_be_module_level_comments() {
    let raw_string_decoy =
        format!("pub const DECOY: &str = r#\"\n{SYNC_BEGIN}\n{HEADER}{SYNC_END}\n\"#;\n");
    let enclosing_comment_decoy =
        format!("/* enclosing comment\n{SYNC_BEGIN}\n{HEADER}{SYNC_END}\n*/\n");
    let macro_decoy =
        format!("macro_rules! decoy {{ () => {{\n{SYNC_BEGIN}\n{HEADER}{SYNC_END}\n}} }}\n");
    for source in [raw_string_decoy, enclosing_comment_decoy, macro_decoy] {
        assert!(
            canonical_schema_source(&source).is_err(),
            "accepted non-module marker decoy {source:?}"
        );
    }

    let valid = fixture().replacen(
        SYNC_END,
        &format!(
            "pub const MARKER_TEXT: &str = r#\"\n{SYNC_BEGIN}\n{SYNC_END}\n{PROTOCOL_BEGIN}\n{PROTOCOL_END}\n\"#;\n{SYNC_END}"
        ),
        1,
    );
    canonical_schema_source(&valid)
        .expect("literal marker text must not become a structural delimiter");
}

#[test]
fn u1_2b1_embedded_schema_fingerprint_is_exact_lower_hex() {
    assert_eq!(fingerprint().len(), 64);
    assert!(fingerprint().bytes().all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f')));
}

#[test]
fn u1_2b1_restricted_visibility_fails_closed() {
    for body in [
        "/// pub(crate) is forbidden even in comments.\n",
        "pub   (crate) struct Spaced;\n",
        "pub\t(crate) struct Tabbed;\n",
        "pub\n(crate) struct Split;\n",
        "pub /* hidden trivia */ (crate) struct Commented;\n",
    ] {
        let source = format!("{SYNC_BEGIN}\n{HEADER}{body}{SYNC_END}\n");
        assert!(canonical_schema_source(&source).is_err(), "accepted {body:?}");
    }
}

#[test]
fn u1_2b1_literal_whitespace_and_structural_metadata_are_not_textually_spoofable() {
    let valid = fixture();
    let literal_source = valid.replacen(
        SYNC_END,
        &format!("pub const MEANING_BEARING: &str = r#\"first  \nsecond\t\nthird\"#;\n{SYNC_END}"),
        1,
    );
    let canonical = canonical_schema_source(&literal_source).unwrap();
    assert!(canonical.contains("r#\"first  \nsecond\t\nthird\"#"));
    let changed =
        canonical_schema_source(&literal_source.replacen("first  \n", "first\n", 1)).unwrap();
    assert_ne!(
        derived_fingerprint(NLAI_SCHEMA_FINGERPRINT_CONTEXT, canonical.as_bytes()),
        derived_fingerprint(NLAI_SCHEMA_FINGERPRINT_CONTEXT, changed.as_bytes())
    );

    let decoy_constant = valid.replacen(
        "pub const NLAI_ARTIFACT_PROTOCOL_VERSION: u32 = 1;",
        "/*\npub const NLAI_ARTIFACT_PROTOCOL_VERSION: u32 = 99;\n*/\npub const NLAI_ARTIFACT_PROTOCOL_VERSION: u32 = 1;",
        1,
    );
    assert!(canonical_schema_source(&decoy_constant).is_ok());
    let raw_decoy_constant = valid.replacen(
        "pub const NLAI_ARTIFACT_PROTOCOL_VERSION: u32 = 1;",
        "pub const TEXTUAL_DECOY: &str = r#\"pub const NLAI_ARTIFACT_PROTOCOL_VERSION: u32 = 99;\"#;\npub const NLAI_ARTIFACT_PROTOCOL_VERSION: u32 = 1;",
        1,
    );
    assert!(canonical_schema_source(&raw_decoy_constant).is_ok());

    let bad_envelope = PROTOCOL_ENVELOPE_DEFINITION.replacen(
        "    pub payload: T,",
        "    pub payload: T,\n    pub extra: u8,",
        1,
    );
    let envelope_decoy = valid.replacen(
        PROTOCOL_ENVELOPE_DEFINITION,
        &format!("/*\n{PROTOCOL_ENVELOPE_DEFINITION}\n*/\n{bad_envelope}"),
        1,
    );
    assert!(canonical_schema_source(&envelope_decoy).is_err());
    let raw_envelope_decoy = valid.replacen(
        PROTOCOL_ENVELOPE_DEFINITION,
        &format!(
            "pub const ENVELOPE_DECOY: &str = r#\"{PROTOCOL_ENVELOPE_DEFINITION}\"#;\n{bad_envelope}"
        ),
        1,
    );
    assert!(canonical_schema_source(&raw_envelope_decoy).is_err());
}

#[test]
fn u1_2b1_protocol_partition_fails_closed() {
    let valid = fixture();
    let split_envelope = valid.replacen(PROTOCOL_END, "", 1).replacen(
        "pub struct SolArtifactEnvelope<T> {\n",
        &format!("pub struct SolArtifactEnvelope<T> {{\n{PROTOCOL_END}\n"),
        1,
    );
    for malformed in [
        valid.replacen(PROTOCOL_BEGIN, "/* broken protocol begin */", 1),
        valid.replacen(PROTOCOL_END, PROTOCOL_BEGIN, 1),
        valid.replacen(
            "/* --- END OF NLAI ARTIFACT PROTOCOL --- */\npub const NLAI_IR_SCHEMA_VERSION: u32 = 1;",
            "pub const NLAI_IR_SCHEMA_VERSION: u32 = 1;\n/* --- END OF NLAI ARTIFACT PROTOCOL --- */",
            1,
        ),
        valid.replacen(
            "pub const NLAI_ARTIFACT_PROTOCOL_VERSION: u32 = 1;",
            "pub const NLAI_ARTIFACT_PROTOCOL_VERSION: u32 = 1; // junk tail",
            1,
        ),
        valid.replacen(
            "pub struct SolArtifactEnvelope<T> {",
            "pub struct SolArtifactEnvelope<T> { // trailing text",
            1,
        ),
        split_envelope,
    ] {
        assert!(
            canonical_schema_source(&malformed).is_err(),
            "accepted {malformed:?}"
        );
    }

    let terminal_bare_cr = format!("{}\r", valid.strip_suffix('\n').expect("fixture ends in LF"));
    assert!(
        canonical_schema_source(&terminal_bare_cr).is_err(),
        "accepted terminal bare carriage return"
    );
}
