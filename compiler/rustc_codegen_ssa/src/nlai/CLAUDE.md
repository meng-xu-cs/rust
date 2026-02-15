# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the `nlai` module inside the Rust compiler's `rustc_codegen_ssa` crate. It is a custom addition (on the `nlai` branch) that hooks into the compilation pipeline to extract a structured, serializable IR representation of Rust crates. The module intercepts compilation at the codegen stage (`base.rs:692`) and serializes THIR (Typed High-Level IR) bodies, type information, and module structure into JSON files.

## Build Commands

This lives inside a fork of the Rust compiler. Build from the repo root (`/Users/mengxu/Sec3/rust`):

```bash
# Build the compiler (stage 1)
./x.py build compiler

# Build the compiler (stage 2, full)
./x.py build compiler --stage 2

# Check the compiler (faster, no codegen)
./x.py check compiler
```

The bootstrap config is in `bootstrap.toml` (profile = "compiler").

## Activating NLAI

The module is gated by environment variables:
- `NLAI=1` (or `true`/`yes`/`on`) — enables the extraction
- `NLAI_OUTPUT_DIR=<path>` — directory where JSON output is written

When disabled or unset, `entrypoint()` returns immediately with no overhead.

## Architecture

### Files

- **`mod.rs`** — Entrypoint (`entrypoint()`). Retrieves env config, calls `build()`, serializes the result.
- **`common.rs`** — `SolEnv` struct managing input/output paths and JSON serialization. Handles env var parsing (`NLAI`, `NLAI_OUTPUT_DIR`).
- **`context.rs`** (~4500 lines) — The core extraction logic and all `Sol*` data types.

### Key Types

Two builder structs drive extraction:

- **`BaseBuilder`** — Handles crate-level traversal: modules, identifiers (`DefId` → `SolIdent`), spans, source file caching, and doc comments.
- **`ExecBuilder`** — Handles per-body (function/closure/const) extraction: THIR expressions, patterns, types, generics, ADT defs, trait defs, closures, and static initializers.

The top-level `build()` function (context.rs:3307):
1. Builds the module tree via `BaseBuilder::mk_module()` (HIR traversal)
2. Iterates all `hir_body_owners` to extract THIR bodies via `ExecBuilder`
3. Returns a `SolCrate` containing the module tree and all execution bundles

### Output Schema (Sol* types)

All output types are defined after line 3522 in `context.rs` (marked `/* --- BEGIN OF SYNC --- */`). Key types:

- `SolCrate` — Root: module tree + bundles + identifier descriptions
- `SolBundle` — One per function/closure/const body: generics, ADT defs, trait defs, dyn types, static inits, and the executable
- `SolExec` — Either `FnDef`, `Closure`, or `CEval` (const evaluation)
- `SolExpr` / `SolOp` — Expression tree with ~60 operation variants
- `SolType` — Type representation (~30 variants covering all Rust types)
- `SolPattern` / `SolPatRule` — Pattern matching representation

All Sol* types implement `SolIR` (= `Debug + Clone + PartialEq + Eq + PartialOrd + Ord + Serialize + DeserializeOwned`).

### Identifiers

Items are identified by `SolIdent` which wraps `DefPathHash` as two hash fields (`krate: u64`, `local: u64`). This provides stable, unique identifiers across compilations.

### Source Caching

Source files are dumped to the output source directory keyed by `StableSourceFileId` (hex-encoded u128). Spans reference these via `SolSpan.file_id`.

## Testing

Testing of the NLAI component is done via the `nlai` crate, separately located in `/Users/mengxu/Sec3/nlai`.

Once the `nlai` crate is built, run self-tests with:

```bash
nlai dev check
```

By default, this uses rustc [compiletest](https://rustc-dev-guide.rust-lang.org/tests/compiletest.html) to check the robustness and feature-completeness of the NLAI component, with the goal of passing all test cases in `ui`, `mir-opt`, and `codegen-llvm`.

Consult the `/Users/mengxu/Sec3/nlai` crate for details of the testing logic, especially the code behind `nlai dev check`.

- **`--summary`** — Produces summary files of the overall testing status. If some test cases fail, it also produces replay commands. Run those commands to inspect failures directly.

## Conventions

- Error handling uses `bug!()` macro with prefixes: `[invariant]` for internal errors, `[user-input]` for bad env vars, `[unsupported]` for unhandled Rust features.
- Logging uses `tracing` (`warn!`, `info!`).
- Collections use `BTreeMap`/`BTreeSet` for deterministic ordering in output.
- All Sol* structs use `#[derive(Serialize, Deserialize)]` with serde.
