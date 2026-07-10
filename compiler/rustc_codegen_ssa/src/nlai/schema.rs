use std::ops::Range;
use std::sync::LazyLock;

use proc_macro2 as _;
use rustc_middle::bug;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::{
    Attribute, Expr, ExprLit, Fields, GenericParam, Item, ItemConst, ItemStruct, Lit, Meta, Type,
    Visibility,
};

use super::context::{
    NLAI_ARTIFACT_PROTOCOL_VERSION, NLAI_IR_SCHEMA_VERSION, NLAI_SCHEMA_CANONICALIZATION,
    NLAI_SCHEMA_FINGERPRINT_ALGORITHM, NLAI_SCHEMA_FINGERPRINT_CONTEXT, SolArtifactEnvelope,
};

const SYNC_BEGIN: &str = "/* --- BEGIN OF SYNC --- */";
const SYNC_END: &str = "/* --- END OF SYNC --- */";
const PROTOCOL_BEGIN: &str = "/* --- BEGIN OF NLAI ARTIFACT PROTOCOL --- */";
const PROTOCOL_END: &str = "/* --- END OF NLAI ARTIFACT PROTOCOL --- */";
const SUPPORTED_CANONICALIZATION: &str = "nlai-schema-c14n-v1";
const SUPPORTED_FINGERPRINT_ALGORITHM: &str = "blake3";
const SCHEMA_SOURCE: &str = include_str!("context.rs");
const PROTOCOL_VERSION_NAME: &str = "NLAI_ARTIFACT_PROTOCOL_VERSION";
const SCHEMA_VERSION_NAME: &str = "NLAI_IR_SCHEMA_VERSION";
const CANONICALIZATION_NAME: &str = "NLAI_SCHEMA_CANONICALIZATION";
const FINGERPRINT_ALGORITHM_NAME: &str = "NLAI_SCHEMA_FINGERPRINT_ALGORITHM";
const FINGERPRINT_CONTEXT_NAME: &str = "NLAI_SCHEMA_FINGERPRINT_CONTEXT";
const PROTOCOL_ENVELOPE_DEFINITION: &str = concat!(
    "#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]\n",
    "#[serde(deny_unknown_fields)]\n",
    "pub struct SolArtifactEnvelope<T> {\n",
    "    pub protocol_version: u32,\n",
    "    pub schema_version: u32,\n",
    "    pub schema_fingerprint: String,\n",
    "    pub payload: T,\n",
    "}",
);
#[cfg(test)]
const PROTOCOL_BASIS_CONTEXT: &str = "nlai.ir.artifact-protocol.version-basis.v1";
#[cfg(test)]
const SCHEMA_BASIS_CONTEXT: &str = "nlai.ir.schema.version-basis.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedU32Constant {
    value: u32,
    literal_range: Range<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedSchemaStructure {
    protocol_version: ParsedU32Constant,
    schema_version: ParsedU32Constant,
    canonicalization: String,
    fingerprint_algorithm: String,
    fingerprint_context: String,
}

#[derive(Debug, Clone, Copy)]
struct SchemaPartitions<'a> {
    protocol: &'a str,
    schema: &'a str,
}

static SCHEMA_FINGERPRINT: LazyLock<String> = LazyLock::new(|| {
    if NLAI_ARTIFACT_PROTOCOL_VERSION == 0 || NLAI_IR_SCHEMA_VERSION == 0 {
        bug!("[invariant] NLAI artifact protocol and IR schema versions must be nonzero");
    }
    // Keep the synchronized generic envelope type checked by the compiler before the U1.2b2 wire
    // cutover starts constructing it.
    let _ = std::mem::size_of::<SolArtifactEnvelope<()>>();
    if NLAI_SCHEMA_CANONICALIZATION != SUPPORTED_CANONICALIZATION {
        bug!(
            "[invariant] unsupported NLAI schema canonicalization {:?}; expected {:?}",
            NLAI_SCHEMA_CANONICALIZATION,
            SUPPORTED_CANONICALIZATION
        );
    }
    if NLAI_SCHEMA_FINGERPRINT_ALGORITHM != SUPPORTED_FINGERPRINT_ALGORITHM {
        bug!(
            "[invariant] unsupported NLAI schema fingerprint algorithm {:?}; expected {:?}",
            NLAI_SCHEMA_FINGERPRINT_ALGORITHM,
            SUPPORTED_FINGERPRINT_ALGORITHM
        );
    }
    if NLAI_SCHEMA_FINGERPRINT_CONTEXT.is_empty()
        || !NLAI_SCHEMA_FINGERPRINT_CONTEXT
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        bug!(
            "[invariant] NLAI schema fingerprint context must be nonempty ASCII [A-Za-z0-9._-], got {:?}",
            NLAI_SCHEMA_FINGERPRINT_CONTEXT
        );
    }
    let canonical = canonical_schema_source(SCHEMA_SOURCE)
        .unwrap_or_else(|error| bug!("[invariant] invalid embedded NLAI schema source: {error}"));
    derived_fingerprint(NLAI_SCHEMA_FINGERPRINT_CONTEXT, canonical.as_bytes())
});

/// Return the fingerprint of the schema source bytes compiled into this rustc.
pub(crate) fn fingerprint() -> &'static str {
    SCHEMA_FINGERPRINT.as_str()
}

/// Canonicalize the one schema section embedded in `context.rs`.
///
/// This implementation intentionally mirrors the consumer sync tool. The protocol accepts LF and
/// CRLF, rejects bare CR, preserves all other section bytes, and emits one LF per logical section
/// line. Whitespace inside Rust literals can affect semantics, so no in-section byte is trimmed.
/// The producer schema uses plain `pub` inside a crate-private module, so the consumer copies these
/// exact bytes and restricted visibility is forbidden rather than textually rewritten.
fn canonical_schema_source(content: &str) -> Result<String, String> {
    let lines = physical_lines(content)?;
    let mut normalized = String::new();
    for line in &lines {
        normalized.push_str(line);
        normalized.push('\n');
    }
    let begin = module_level_comment_line_ranges(&normalized, SYNC_BEGIN)?;
    let end = module_level_comment_line_ranges(&normalized, SYNC_END)?;
    if begin.len() != 1 || end.len() != 1 {
        return Err(format!(
            "synchronized schema requires exactly one module-level begin and end comment, found begin={}, end={}",
            begin.len(),
            end.len()
        ));
    }
    let (begin_start, begin_end) = begin[0];
    let (end_start, _) = end[0];
    if begin_start >= end_start || begin_end > end_start {
        return Err("synchronized schema end marker precedes its begin marker".to_owned());
    }

    let canonical = normalized[begin_end..end_start].to_owned();
    if canonical.lines().all(str::is_empty) {
        return Err("synchronized section is empty".to_owned());
    }
    validate_schema_partitions(&canonical)?;
    for (offset, _) in canonical.match_indices("pub") {
        let before = canonical[..offset].chars().next_back();
        let after = canonical[offset + "pub".len()..].chars().next();
        let is_identifier_continue =
            |character: char| character == '_' || character.is_alphanumeric();
        if before.is_some_and(is_identifier_continue) || after.is_some_and(is_identifier_continue) {
            continue;
        }

        let trivia_stripped =
            canonical[offset + "pub".len()..].trim_start_matches(char::is_whitespace);
        if trivia_stripped.starts_with('(') || trivia_stripped.starts_with('/') {
            let line_number = canonical[..offset].bytes().filter(|byte| *byte == b'\n').count() + 1;
            return Err(format!(
                "schema line {line_number}: restricted Rust visibility or ambiguous post-pub trivia is forbidden; use plain pub inside the crate-private producer module"
            ));
        }
    }
    Ok(canonical)
}

/// Locate exact-line block comments that are lexical comments between module items.
///
/// `syn` supplies the ranges occupied by actual top-level syntax. Scanning only the intervening
/// trivia then distinguishes a delimiter comment from identical bytes inside a literal, an item,
/// a macro token tree, or an enclosing nested block comment. This is deliberately stricter than a
/// line-prefix search because the synchronized bytes must denote the schema rustc compiles.
fn module_level_comment_line_ranges(
    source: &str,
    marker: &str,
) -> Result<Vec<(usize, usize)>, String> {
    let file = syn::parse_file(source).map_err(|error| {
        format!("schema container is not valid Rust syntax while locating {marker:?}: {error}")
    })?;
    let mut occupied = file
        .attrs
        .iter()
        .map(|attribute| attribute.span().byte_range())
        .chain(file.items.iter().map(|item| item.span().byte_range()))
        .filter(|range| range.start < range.end)
        .collect::<Vec<_>>();
    occupied.sort_by_key(|range| (range.start, range.end));

    let mut result = Vec::new();
    for (line_start, line_end) in exact_line_ranges(source, marker) {
        let comment_end = line_start + marker.len();
        if occupied.iter().any(|range| range.start < comment_end && line_start < range.end) {
            continue;
        }
        let gap_start = occupied
            .iter()
            .filter(|range| range.end <= line_start)
            .map(|range| range.end)
            .max()
            .unwrap_or(0);
        let gap_end = occupied
            .iter()
            .filter(|range| range.start >= comment_end)
            .map(|range| range.start)
            .min()
            .unwrap_or(source.len());
        if top_level_block_comments(&source[gap_start..gap_end], gap_start)?
            .iter()
            .any(|range| range.start == line_start && range.end == comment_end)
        {
            result.push((line_start, line_end));
        }
    }
    Ok(result)
}

fn top_level_block_comments(trivia: &str, base: usize) -> Result<Vec<Range<usize>>, String> {
    let bytes = trivia.as_bytes();
    let mut comments = Vec::new();
    let mut offset = 0;
    while offset < bytes.len() {
        if bytes[offset..].starts_with(b"//") {
            offset += 2;
            while offset < bytes.len() && bytes[offset] != b'\n' {
                offset += 1;
            }
            continue;
        }
        if !bytes[offset..].starts_with(b"/*") {
            offset += 1;
            continue;
        }

        let start = offset;
        let mut depth = 1_u32;
        offset += 2;
        while offset < bytes.len() && depth != 0 {
            if bytes[offset..].starts_with(b"/*") {
                depth = depth
                    .checked_add(1)
                    .ok_or_else(|| "block-comment nesting depth overflow".to_owned())?;
                offset += 2;
            } else if bytes[offset..].starts_with(b"*/") {
                depth -= 1;
                offset += 2;
            } else {
                offset += 1;
            }
        }
        if depth != 0 {
            return Err(
                "unterminated module-level block comment while locating schema markers".to_owned()
            );
        }
        comments.push(base + start..base + offset);
    }
    Ok(comments)
}

fn validate_schema_partitions(source: &str) -> Result<ParsedSchemaStructure, String> {
    let partitions = partition_schema(source)?;
    let structure = parse_schema_structure(partitions)?;
    if structure.protocol_version.value != NLAI_ARTIFACT_PROTOCOL_VERSION {
        return Err(format!(
            "parsed {PROTOCOL_VERSION_NAME}={} does not match compiled value {NLAI_ARTIFACT_PROTOCOL_VERSION}",
            structure.protocol_version.value
        ));
    }
    if structure.schema_version.value != NLAI_IR_SCHEMA_VERSION {
        return Err(format!(
            "parsed {SCHEMA_VERSION_NAME}={} does not match compiled value {NLAI_IR_SCHEMA_VERSION}",
            structure.schema_version.value
        ));
    }
    if structure.canonicalization != NLAI_SCHEMA_CANONICALIZATION {
        return Err(format!(
            "parsed {CANONICALIZATION_NAME}={:?} does not match compiled value {:?}",
            structure.canonicalization, NLAI_SCHEMA_CANONICALIZATION
        ));
    }
    if structure.fingerprint_algorithm != NLAI_SCHEMA_FINGERPRINT_ALGORITHM {
        return Err(format!(
            "parsed {FINGERPRINT_ALGORITHM_NAME}={:?} does not match compiled value {:?}",
            structure.fingerprint_algorithm, NLAI_SCHEMA_FINGERPRINT_ALGORITHM
        ));
    }
    if structure.fingerprint_context != NLAI_SCHEMA_FINGERPRINT_CONTEXT {
        return Err(format!(
            "parsed {FINGERPRINT_CONTEXT_NAME}={:?} does not match compiled value {:?}",
            structure.fingerprint_context, NLAI_SCHEMA_FINGERPRINT_CONTEXT
        ));
    }
    Ok(structure)
}

fn partition_schema(source: &str) -> Result<SchemaPartitions<'_>, String> {
    let begin = module_level_comment_line_ranges(source, PROTOCOL_BEGIN)?;
    let end = module_level_comment_line_ranges(source, PROTOCOL_END)?;
    if begin.len() != 1 || end.len() != 1 {
        return Err(format!(
            "protocol partition requires exactly one begin and end marker, found begin={}, end={}",
            begin.len(),
            end.len()
        ));
    }
    let (begin_start, begin_end) = begin[0];
    let (end_start, end_end) = end[0];
    if begin_end > end_start {
        return Err("protocol partition end marker precedes its begin marker".to_owned());
    }
    if !source[..begin_start].lines().all(str::is_empty) {
        return Err("protocol partition must be the first nonempty synchronized content".to_owned());
    }
    if source[begin_end..end_start].lines().all(str::is_empty)
        || source[end_end..].lines().all(str::is_empty)
    {
        return Err("protocol and payload-schema partitions must both be nonempty".to_owned());
    }
    Ok(SchemaPartitions { protocol: &source[..end_end], schema: &source[end_end..] })
}

fn exact_line_ranges(source: &str, needle: &str) -> Vec<(usize, usize)> {
    let mut offset = 0;
    let mut ranges = Vec::new();
    for line in source.split_inclusive('\n') {
        if line.strip_suffix('\n') == Some(needle) {
            ranges.push((offset, offset + line.len()));
        }
        offset += line.len();
    }
    ranges
}

fn parse_schema_structure(
    partitions: SchemaPartitions<'_>,
) -> Result<ParsedSchemaStructure, String> {
    let protocol = syn::parse_file(partitions.protocol).map_err(|error| {
        format!("artifact-protocol partition is not valid Rust syntax: {error}")
    })?;
    let schema = syn::parse_file(partitions.schema)
        .map_err(|error| format!("payload-schema partition is not valid Rust syntax: {error}"))?;

    let protocol_version = parse_u32_constant_item(
        unique_partitioned_const(&protocol, &schema, PROTOCOL_VERSION_NAME, true)?,
        partitions.protocol,
        PROTOCOL_VERSION_NAME,
    )?;
    let schema_version = parse_u32_constant_item(
        unique_partitioned_const(&protocol, &schema, SCHEMA_VERSION_NAME, false)?,
        partitions.schema,
        SCHEMA_VERSION_NAME,
    )?;
    let canonicalization = parse_string_constant_item(
        unique_partitioned_const(&protocol, &schema, CANONICALIZATION_NAME, true)?,
        partitions.protocol,
        CANONICALIZATION_NAME,
    )?;
    let fingerprint_algorithm = parse_string_constant_item(
        unique_partitioned_const(&protocol, &schema, FINGERPRINT_ALGORITHM_NAME, true)?,
        partitions.protocol,
        FINGERPRINT_ALGORITHM_NAME,
    )?;
    let fingerprint_context = parse_string_constant_item(
        unique_partitioned_const(&protocol, &schema, FINGERPRINT_CONTEXT_NAME, true)?,
        partitions.protocol,
        FINGERPRINT_CONTEXT_NAME,
    )?;
    validate_protocol_envelope(
        unique_partitioned_struct(&protocol, &schema, "SolArtifactEnvelope", true)?,
        partitions.protocol,
    )?;

    Ok(ParsedSchemaStructure {
        protocol_version,
        schema_version,
        canonicalization,
        fingerprint_algorithm,
        fingerprint_context,
    })
}

fn item_ident(item: &Item) -> Option<&syn::Ident> {
    match item {
        Item::Const(item) => Some(&item.ident),
        Item::Enum(item) => Some(&item.ident),
        Item::ExternCrate(item) => Some(&item.ident),
        Item::Fn(item) => Some(&item.sig.ident),
        Item::Macro(item) => item.ident.as_ref(),
        Item::Mod(item) => Some(&item.ident),
        Item::Static(item) => Some(&item.ident),
        Item::Struct(item) => Some(&item.ident),
        Item::Trait(item) => Some(&item.ident),
        Item::TraitAlias(item) => Some(&item.ident),
        Item::Type(item) => Some(&item.ident),
        Item::Union(item) => Some(&item.ident),
        _ => None,
    }
}

fn unique_partitioned_item<'a>(
    protocol: &'a syn::File,
    schema: &'a syn::File,
    name: &str,
    expected_protocol: bool,
) -> Result<&'a Item, String> {
    let protocol_matches: Vec<&Item> = protocol
        .items
        .iter()
        .filter(|item| item_ident(item).is_some_and(|ident| ident == name))
        .collect();
    let schema_matches: Vec<&Item> = schema
        .items
        .iter()
        .filter(|item| item_ident(item).is_some_and(|ident| ident == name))
        .collect();
    let expected_count =
        if expected_protocol { protocol_matches.len() } else { schema_matches.len() };
    if protocol_matches.len() + schema_matches.len() != 1 || expected_count != 1 {
        let expected_partition =
            if expected_protocol { "artifact-protocol" } else { "payload-schema" };
        return Err(format!(
            "the unique module-level Rust item {name} must be inside the {expected_partition} partition; found protocol={}, schema={}",
            protocol_matches.len(),
            schema_matches.len()
        ));
    }
    Ok(if expected_protocol { protocol_matches[0] } else { schema_matches[0] })
}

fn unique_partitioned_const<'a>(
    protocol: &'a syn::File,
    schema: &'a syn::File,
    name: &str,
    expected_protocol: bool,
) -> Result<&'a ItemConst, String> {
    match unique_partitioned_item(protocol, schema, name, expected_protocol)? {
        Item::Const(item) => Ok(item),
        _ => Err(format!("module-level item {name} must be a const declaration")),
    }
}

fn unique_partitioned_struct<'a>(
    protocol: &'a syn::File,
    schema: &'a syn::File,
    name: &str,
    expected_protocol: bool,
) -> Result<&'a ItemStruct, String> {
    match unique_partitioned_item(protocol, schema, name, expected_protocol)? {
        Item::Struct(item) => Ok(item),
        _ => Err(format!("module-level item {name} must be a struct declaration")),
    }
}

fn attributes_are_only_docs(attributes: &[Attribute]) -> bool {
    attributes.iter().all(|attribute| attribute.path().is_ident("doc"))
}

fn is_public(visibility: &Visibility) -> bool {
    matches!(visibility, Visibility::Public(_))
}

fn type_is_path_ident(ty: &Type, expected: &str) -> bool {
    let Type::Path(path) = ty else {
        return false;
    };
    path.qself.is_none()
        && path.path.leading_colon.is_none()
        && path.path.segments.len() == 1
        && path.path.segments[0].ident == expected
        && matches!(path.path.segments[0].arguments, syn::PathArguments::None)
}

fn type_is_str_reference(ty: &Type) -> bool {
    let Type::Reference(reference) = ty else {
        return false;
    };
    reference.lifetime.is_none()
        && reference.mutability.is_none()
        && type_is_path_ident(&reference.elem, "str")
}

fn validate_const_header(item: &ItemConst, name: &str) -> Result<(), String> {
    if !attributes_are_only_docs(&item.attrs)
        || !is_public(&item.vis)
        || !item.generics.params.is_empty()
        || item.generics.where_clause.is_some()
    {
        return Err(format!(
            "{name} must be one unconditional public module-level const; only documentation attributes are allowed"
        ));
    }
    Ok(())
}

fn exact_single_line_const_source<'a>(
    item: &ItemConst,
    partition_source: &'a str,
    name: &str,
) -> Result<&'a str, String> {
    let start = item.vis.span().byte_range().start;
    let end = item.semi_token.span.byte_range().end;
    let line_start = partition_source[..start].rfind('\n').map_or(0, |offset| offset + 1);
    let line_end =
        partition_source[end..].find('\n').map_or(partition_source.len(), |offset| end + offset);
    if line_start != start || line_end != end {
        return Err(format!(
            "{name} must occupy one exact declaration line without leading or trailing text"
        ));
    }
    partition_source
        .get(start..end)
        .ok_or_else(|| format!("{name} source span is outside its parsed partition"))
}

fn parse_u32_constant_item(
    item: &ItemConst,
    partition_source: &str,
    name: &str,
) -> Result<ParsedU32Constant, String> {
    validate_const_header(item, name)?;
    if !type_is_path_ident(&item.ty, "u32") {
        return Err(format!("{name} must have the exact type u32"));
    }
    let Expr::Lit(ExprLit { attrs, lit: Lit::Int(literal), .. }) = &*item.expr else {
        return Err(format!("{name} must be initialized by one canonical decimal integer literal"));
    };
    if !attrs.is_empty() || !literal.suffix().is_empty() {
        return Err(format!(
            "{name} must be initialized by one unsuffixed decimal integer literal"
        ));
    }
    let literal_range = literal.span().byte_range();
    let encoded = partition_source
        .get(literal_range.clone())
        .ok_or_else(|| format!("{name} literal span is outside its parsed partition"))?;
    if encoded.is_empty()
        || !encoded.bytes().all(|byte| byte.is_ascii_digit())
        || (encoded.len() > 1 && encoded.starts_with('0'))
    {
        return Err(format!(
            "{name} must use canonical unsuffixed decimal notation, got {encoded:?}"
        ));
    }
    let value = encoded
        .parse::<u32>()
        .map_err(|error| format!("invalid u32 value for {name}: {encoded:?}: {error}"))?;
    let declaration = exact_single_line_const_source(item, partition_source, name)?;
    let expected_declaration = format!("pub const {name}: u32 = {encoded};");
    if declaration != expected_declaration {
        return Err(format!(
            "{name} must use the exact declaration grammar {expected_declaration:?}"
        ));
    }
    Ok(ParsedU32Constant { value, literal_range })
}

fn parse_string_constant_item(
    item: &ItemConst,
    partition_source: &str,
    name: &str,
) -> Result<String, String> {
    validate_const_header(item, name)?;
    if !type_is_str_reference(&item.ty) {
        return Err(format!("{name} must have the exact type &str"));
    }
    let Expr::Lit(ExprLit { attrs, lit: Lit::Str(literal), .. }) = &*item.expr else {
        return Err(format!("{name} must be initialized by one string literal"));
    };
    if !attrs.is_empty() {
        return Err(format!("{name} string literal may not carry expression attributes"));
    }
    let value = literal.value();
    let declaration = exact_single_line_const_source(item, partition_source, name)?;
    let expected_declaration = format!("pub const {name}: &str = {value:?};");
    if declaration != expected_declaration {
        return Err(format!(
            "{name} must use the exact declaration grammar {expected_declaration:?}"
        ));
    }
    Ok(value)
}

fn path_is_single_ident(path: &syn::Path, expected: &str) -> bool {
    path.leading_colon.is_none()
        && path.segments.len() == 1
        && path.segments[0].ident == expected
        && matches!(path.segments[0].arguments, syn::PathArguments::None)
}

fn validate_protocol_envelope(item: &ItemStruct, partition_source: &str) -> Result<(), String> {
    if !is_public(&item.vis) || item.semi_token.is_some() || item.generics.where_clause.is_some() {
        return Err(
            "SolArtifactEnvelope must be an unconditional public named-field struct without a where clause"
                .to_owned(),
        );
    }
    let non_doc_attributes: Vec<&Attribute> =
        item.attrs.iter().filter(|attribute| !attribute.path().is_ident("doc")).collect();
    if non_doc_attributes.len() != 2
        || !non_doc_attributes[0].path().is_ident("derive")
        || !non_doc_attributes[1].path().is_ident("serde")
    {
        return Err("SolArtifactEnvelope must have exactly the supported derive and serde attributes in that order, besides documentation".to_owned());
    }
    let derives = non_doc_attributes[0]
        .parse_args_with(Punctuated::<syn::Path, syn::Token![,]>::parse_terminated)
        .map_err(|error| format!("invalid SolArtifactEnvelope derive attribute: {error}"))?;
    let expected_derives = [
        "Debug",
        "Clone",
        "PartialEq",
        "Eq",
        "PartialOrd",
        "Ord",
        "Hash",
        "Serialize",
        "Deserialize",
    ];
    if derives.len() != expected_derives.len()
        || derives
            .iter()
            .zip(expected_derives)
            .any(|(path, expected)| !path_is_single_ident(path, expected))
    {
        return Err(
            "SolArtifactEnvelope derive list does not match the supported protocol grammar"
                .to_owned(),
        );
    }
    let serde = non_doc_attributes[1]
        .parse_args_with(Punctuated::<Meta, syn::Token![,]>::parse_terminated)
        .map_err(|error| format!("invalid SolArtifactEnvelope serde attribute: {error}"))?;
    if serde.len() != 1
        || !matches!(serde.first(), Some(Meta::Path(path)) if path_is_single_ident(path, "deny_unknown_fields"))
    {
        return Err("SolArtifactEnvelope must use exactly #[serde(deny_unknown_fields)]".to_owned());
    }
    let item_range = non_doc_attributes[0].span().byte_range().start..item.span().byte_range().end;
    if partition_source.get(item_range) != Some(PROTOCOL_ENVELOPE_DEFINITION) {
        return Err(
            "the parsed SolArtifactEnvelope item must use the exact supported source grammar"
                .to_owned(),
        );
    }
    if item.generics.params.len() != 1 {
        return Err("SolArtifactEnvelope must declare exactly one type parameter T".to_owned());
    }
    let Some(GenericParam::Type(parameter)) = item.generics.params.first() else {
        return Err("SolArtifactEnvelope generic parameter must be the type parameter T".to_owned());
    };
    if parameter.ident != "T"
        || !parameter.attrs.is_empty()
        || parameter.colon_token.is_some()
        || !parameter.bounds.is_empty()
        || parameter.eq_token.is_some()
        || parameter.default.is_some()
    {
        return Err(
            "SolArtifactEnvelope generic parameter must be exactly T without bounds or defaults"
                .to_owned(),
        );
    }
    let Fields::Named(fields) = &item.fields else {
        return Err("SolArtifactEnvelope must use named fields".to_owned());
    };
    let expected_fields = [
        ("protocol_version", "u32"),
        ("schema_version", "u32"),
        ("schema_fingerprint", "String"),
        ("payload", "T"),
    ];
    if fields.named.len() != expected_fields.len() {
        return Err("SolArtifactEnvelope must contain exactly the four supported protocol fields"
            .to_owned());
    }
    for (field, (expected_name, expected_type)) in fields.named.iter().zip(expected_fields) {
        if !field.attrs.is_empty()
            || !is_public(&field.vis)
            || !matches!(field.mutability, syn::FieldMutability::None)
            || field.ident.as_ref().is_none_or(|ident| ident != expected_name)
            || !type_is_path_ident(&field.ty, expected_type)
        {
            return Err(format!(
                "SolArtifactEnvelope field {expected_name} must be an unconditional public {expected_type} field in protocol order"
            ));
        }
    }
    Ok(())
}

fn physical_lines(content: &str) -> Result<Vec<String>, String> {
    let mut lines: Vec<String> = content
        .split_inclusive('\n')
        .enumerate()
        .map(|(line_index, chunk)| {
            let line_number = line_index + 1;
            let (raw, terminated) =
                chunk.strip_suffix('\n').map_or((chunk, false), |raw| (raw, true));
            let raw = if terminated { raw.strip_suffix('\r').unwrap_or(raw) } else { raw };
            if raw.contains('\r') {
                return Err(format!(
                    "line {line_number}: bare carriage return is not a supported line ending"
                ));
            }
            Ok(raw.to_owned())
        })
        .collect::<Result<_, _>>()?;

    if content.is_empty() {
        lines.push(String::new());
    }

    Ok(lines)
}

fn derived_fingerprint(context: &str, bytes: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new_derive_key(context);
    hasher.update(bytes);
    hasher.finalize().to_hex().to_string()
}

#[cfg(test)]
fn version_basis_fingerprints(source: &str) -> Result<(String, String), String> {
    let partitions = partition_schema(source)?;
    let structure = validate_schema_partitions(source)?;
    let protocol =
        normalize_version_literal(partitions.protocol, structure.protocol_version.literal_range)?;
    let schema =
        normalize_version_literal(partitions.schema, structure.schema_version.literal_range)?;
    Ok((
        derived_fingerprint(PROTOCOL_BASIS_CONTEXT, protocol.as_bytes()),
        derived_fingerprint(SCHEMA_BASIS_CONTEXT, schema.as_bytes()),
    ))
}

#[cfg(test)]
fn normalize_version_literal(source: &str, literal_range: Range<usize>) -> Result<String, String> {
    if literal_range.start >= literal_range.end || source.get(literal_range.clone()).is_none() {
        return Err("version literal span is outside its parsed partition".to_owned());
    }
    let mut normalized = String::with_capacity(source.len());
    normalized.push_str(&source[..literal_range.start]);
    normalized.push('0');
    normalized.push_str(&source[literal_range.end..]);
    Ok(normalized)
}

#[cfg(test)]
mod tests;
