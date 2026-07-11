# NLAI Compiler Extractor

This is the provider-neutral guide for the `nlai` module in
`rustc_codegen_ssa`. Codex reads `AGENTS.md` directly. Claude Code reads the
local `CLAUDE.md`, which imports this file. Keep this file free of
provider-specific instructions unless a tool-specific compatibility note is
needed.

## Current Status

- This is a custom addition on the `nlai` branch of the Rust compiler fork.
- `rustc_codegen_ssa::nlai` is declared in `../lib.rs`.
- `codegen_crate()` calls `crate::nlai::entrypoint(tcx)` in `../base.rs` before
  monomorphization collection and normal codegen-unit partitioning.
- The compiler-side module currently consists of `mod.rs`, `common.rs`, `context.rs`, `identity.rs`,
  and `schema.rs`. `context.rs` is intentionally large and marked with
  `// ignore-tidy-filelength`.
- The synchronized output schema is in `context.rs` between
  `/* --- BEGIN OF SYNC --- */` and `/* --- END OF SYNC --- */`.
- The consumer crate is normally a sibling checkout at `../nlai`. Its `RUST_SRC`
  setting must resolve to this repository's canonical Git root before
  `cargo run -- dev sync`, extraction, or the rustc-suite harness proceeds.
  Sync writes the schema into the consumer's `src/rustc/ir.rs`.
- The default downstream analysis path in the sibling crate is still the local
  self-test pipeline: context construction, bundle display, control-flow
  ordering, and the basic abstract interpreter. The LLM-backed analysis code is
  present there but is not part of the default analysis run.

## Build Commands

This directory lives inside a Rust compiler checkout. Run compiler builds from
the canonical repository root (`<rust-root>`).

```bash
./x.py check compiler
./x.py build compiler
./x.py build compiler --stage 2
```

The local bootstrap config is `<rust-root>/bootstrap.toml`, normally using
`profile = "compiler"`. Rust's `dev` channel omits Git identity by default; NLAI builds must opt in
explicitly so consumer preflight can bind the binary to the canonical source HEAD:

```toml
[rust]
omit-git-hash = false
```

The consumer rejects a configured custom compiler whose canonical path is not
under the same `<rust-root>`. It also rejects missing, duplicate, `unknown`, malformed, or
nonmatching `commit-hash` metadata from `rustc -vV`. Rust bootstrap additionally fingerprints the
canonical commit, complete index, tracked worktree contents and executable/symlink state,
nonignored untracked inputs, and recursively initialized submodules. It compiles that fingerprint
into shared `rustc_session` state so separately built NLAI-capable codegen backends and the driver
observe one value, then exposes exactly one `nlai-source-state-fingerprint` version field; missing,
duplicate, `unknown`, or malformed values fail consumer preflight. See the consumer's U1.2
protocol documentation.
Extractor activation also validates the shared compiled commit and source-state values and hashes
the exact running rustc image under the derive-key context `nlai.rustc-executable.blake3.v1`.
Procfs hosts open the kernel's loaded-image link; macOS requires the opened handle and path to match
the vnode backing dyld's mapped main Mach-O header. Hosts without either primitive, including
Windows, fail closed when extraction is activated. Discovery, identity, stability, exact-length
read, and unsupported-host failures are fatal invariants. Rustc driver entry points force this
snapshot as the first normal-driver operation, before environment-configured logger setup, argument
processing, or input handling, and carry the resulting value through
`rustc_interface::Config` into the immutable `rustc_session::Session`. Artifact production reads
only that session value, so a separately loaded SSA backend cannot fall back to its own late lazy
snapshot; an embedding that reaches NLAI codegen without the explicit handoff fails loudly. A later
same-inode overwrite therefore cannot retag the measured launcher. Rustdoc's direct
`rustc_interface` paths do not emit these artifacts; rustc processes that it launches still cross
the rustc driver boundary. The launcher digest alone is not yet the complete producer-code closure:
the loaded driver image and any in-scope dynamic backend are tracked by U1.2c2b1b2.

## Activation

`entrypoint()` is gated by environment variables and returns immediately when
NLAI is unset or disabled.

- `NLAI=1`, `true`, `yes`, or `on` enables extraction.
- `NLAI=0`, `false`, `no`, or `off` disables extraction.
- `NLAI_OUTPUT_DIR=<path>` is required when extraction is enabled.

Invalid `NLAI` values and a missing `NLAI_OUTPUT_DIR` are treated as
`[user-input]` bugs. When enabled, `common.rs` creates fresh numbered
subdirectories under the output directory:

- `s<N>/` stores source-file snapshots.
- `f<N>/crate.json` stores `SolArtifactEnvelope<SolCrate>` with the exact protocol version,
  IR-schema version, and BLAKE3 fingerprint of the canonical schema bytes compiled into this rustc.
  The consumer validates that metadata before deserializing the raw payload.

`SolEnv` also records the canonical local crate input path for diagnostics.

## Architecture

- `mod.rs` owns the crate-visible entrypoint. It retrieves `SolEnv`, logs the active
  context, calls `build(tcx, env.prepare_source_directory())`, and serializes
  the returned crate.
- `common.rs` owns environment parsing and JSON/source output directory
  management, including construction of the versioned artifact envelope.
- `context.rs` owns extraction. It contains the builders, conversion logic, and
  all `Sol*` IR data types.
- `identity.rs` owns compile-time producer-identity validation and stable measurement of the exact
  running rustc executable.
- `schema.rs` structurally parses and fingerprints the synchronized schema source compiled into
  rustc, rejecting textual decoys and protocol-shape drift before extraction.

Two builders drive extraction:

- `BaseBuilder` handles crate-level data: module traversal, `DefId` to
  `SolIdent` mapping, spans, source-file caching, doc comments, and identifier
  descriptions.
- `ExecBuilder` handles each executable THIR body: expressions, statements,
  patterns, types, generics, ADT definitions, trait clauses, dynamic trait
  object clauses, static initializers, closures, upvars, local variables, and
  instruction indexes.

`build()` in `context.rs`:

1. Builds the module tree from `tcx.hir_root_module()`.
2. Wraps the root module in `SolMIR<SolModule>`.
3. Iterates `tcx.hir_body_owners()`.
4. Skips coroutine and coroutine-closure bodies.
5. Retrieves each THIR body.
6. Collects body generics, including const-generic types, while skipping
   rustc-injected closure type parameters.
7. Uses `ExecBuilder::mk_exec()` to build the body representation.
8. Flattens collected ADT, trait, dynamic-type, and static-initializer caches
   into a `SolBundle`.
9. Sorts identifier descriptions by path description and identifier.
10. Returns `SolCrate { root, bundles, id_desc }`.

## Output Schema

All synchronized `Sol*` types derive `Debug`, `Clone`, ordering/equality,
`Hash`, `Serialize`, and `Deserialize`. `SolIR` is the shared trait alias for
types that can appear inside `SolHIR<T>` or `SolMIR<T>`.

The synchronized section also defines protocol/schema versions, the canonicalization and BLAKE3
derive-key identifiers, and `SolArtifactEnvelope<T>`. A nested artifact-protocol partition owns the
protocol version, canonicalization/hash identifiers, and envelope grammar; the following payload
partition owns the IR-schema version and all `Sol*` payload definitions. Sync fingerprints the two
version-normalized partitions separately, so a protocol change must advance the protocol version
and a payload change must advance the schema version; version-only bumps fail. Its items use plain
`pub`; the enclosing producer modules remain crate-private, while the consumer copies the exact
canonical bytes without a textual visibility rewrite or consumer-local rustfmt transformation.
`nlai::context` therefore carries one narrowly scoped `unreachable_pub` allowance; do not broaden
the module's visibility or the allowance. Restricted visibility and ambiguous trivia after `pub`
are forbidden anywhere in the section.
The outer sync markers and nested protocol markers must be standalone lexical block comments
between module items. Identical marker text inside an enclosing comment, literal, item, or macro
token tree is identity-bearing input, never a delimiter.
Partition governance is intentionally positional: moving a payload declaration into the protocol
partition is itself a protocol-design change and must not be used to evade the payload-schema
version. The complete supported envelope item—including its derive/serde attributes, fields, and
closing brace—must remain wholly inside the protocol partition. The committed consumer schema is
the sync ratchet; sync refuses to recreate it when its framing, footer, metadata, canonical bytes,
or self-attestation are missing or inconsistent, and the consumer serializes validation and atomic
publication with a stable sidecar lock.
LF and CRLF are equivalent, but every other in-section byte—including whitespace inside literals
and at physical line ends—remains identity-bearing. Bare CR fails closed, comments remain
identity-bearing, and each logical section line ends in LF.
The generated consumer fingerprint constant is outside the hashed section to avoid self-reference.

Important root types:

- `SolCrate` contains the root module, executable bundles, and identifier
  descriptions.
- `SolBundle` contains generics, ADT definitions, trait definitions, dynamic
  type clauses, static initializer values, and one executable body.
- `SolExec` is one of `Function`, `Closure`, or `ConstEval`.

Important schema areas:

- `SolType` covers primitive types including `f16`/`f128`, pattern types,
  foreign types, ADTs, references, raw pointers, type parameters, tuples,
  slices, arrays, resolved function kinds, closures, function pointers,
  dynamic trait objects, and associated projections. Alias and opaque types are
  currently normalized to underlying forms when possible.
- `SolConst` and `SolValue` represent const parameters, unevaluated constants,
  concrete primitive/composite values, static ref/pointer values, null
  ref/pointer values, resolved function values, closures, and function
  pointers.
- `SolOp` represents THIR expression operations: scope/ascription markers,
  locals/upvars/consts/statics, operators, assignments, casts, field/index
  access, ADT construction, borrows, control flow, calls, let/match, blocks,
  literals, and closures.
- `SolPattern` and `SolPatRule` represent THIR patterns, including missing,
  wildcard, never, bind, variant, leaf, slice, array, deref, range, constant,
  and or-pattern rules.
- `SolSpan` references a cached source file by `StableSourceFileId` encoded as a
  `SolHash128` plus 1-based line and 0-based column bounds.
- `SolIdent` wraps `DefPathHash` as `{ krate: SolHash64, local: SolHash64 }`,
  giving stable identifiers across compilations.

## Source And Span Handling

Visible spans must stay within one source file. The first visible span for a
source file causes the full source text to be written to `s<N>/<hex-file-id>`.
Invisible spans are encoded as a zero file id and zero coordinates. Span
creation is also where identifier descriptions get source locations.

## Testing And Consumer Crate

Compiler-side extraction is normally tested through the sibling consumer
checkout (`<nlai-root>`, conventionally `../nlai` from this repository).

```bash
cd ../nlai
cargo run -- dev sync
cargo run -- dev check
cargo run -- dev check --summary
```

`dev check` uses a custom harness, not the `compiletest` crate as a library. It
walks `RUST_SRC/tests/{ui,mir-opt,codegen-llvm}`, parses rustc `//@`
directives, builds each selected test with NLAI enabled, loads the emitted JSON,
and runs the downstream analysis pipeline. Summary files are written under
`data/testing/summary-ui/`, `data/testing/summary-mir-opt/`, and
`data/testing/summary-codegen-llvm/`.

For one-off downstream runs:

```bash
cd ../nlai
cargo run -- cargo build --output .nlai -- <cargo build args>
cargo run -- cargo analyze --build-dir .nlai -v
```

## Conventions

- Use `bug!()` prefixes consistently:
  - `[user-input]` for invalid user/environment input.
  - `[invariant]` for compiler or extractor invariants.
  - `[unsupported]` for Rust features the extractor intentionally does not
    handle yet.
  - `[assumption]` for explicit assumptions that should be revisited if they
    fail.
- Use `tracing` for logging. `mod.rs` logs activation with `warn!`; detailed
  extraction tracing in `context.rs` uses `info!`.
- Prefer `BTreeMap` and `BTreeSet` for serialized or otherwise deterministic
  output ordering.
- Keep the synchronized schema block self-contained and serde-friendly. After
  changing it, run `cargo run -- dev sync` in `<nlai-root>` with `RUST_SRC`
  resolving to this canonical repository root.
- Do not edit `<nlai-root>/src/rustc/ir.rs` manually; it is
  generated from the synchronized block in this compiler fork.
