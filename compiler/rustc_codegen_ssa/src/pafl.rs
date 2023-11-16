use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

use serde::Serialize;

use rustc_hir::def::{CtorKind, DefKind};
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::mir::mono::MonoItem;
use rustc_middle::mir::{Body, MirPhase, RuntimePhase};
use rustc_middle::ty::{InstanceDef, TyCtxt};
use rustc_span::def_id::DefId;
use rustc_span::Symbol;

/// A complete dump of both the control-flow graph and the call graph of the compilation context
pub fn dump(tcx: TyCtxt<'_>, outdir: &Path) {
    // prepare directory
    fs::create_dir_all(outdir).expect("unable to create output directory");

    // extract the mir for each codegen unit
    let mut summary = CrateSummary { natives: Vec::new(), functions: Vec::new() };

    let (_, units) = tcx.collect_and_partition_mono_items(());
    for unit in units {
        let name = unit.name();
        for item in unit.items().keys() {
            let instance = match item {
                MonoItem::Fn(i) => i,
                MonoItem::Static(_) => continue,
                MonoItem::GlobalAsm(_) => bug!("unexpected assembly"),
            };

            // branch processing by instance type
            match &instance.def {
                InstanceDef::Item(id) => {
                    summary.functions.push(FunctionSummary::process(
                        tcx,
                        *id,
                        name,
                        tcx.optimized_mir(*id),
                    ));
                }
                InstanceDef::Intrinsic(id) => {
                    summary.natives.push(NativeSummary::process(
                        tcx,
                        *id,
                        name,
                        NativeKind::Intrinsic,
                    ));
                }
                InstanceDef::ClosureOnceShim { call_once: id, track_caller: _ } => {
                    summary.natives.push(NativeSummary::process(tcx, *id, name, NativeKind::Once));
                }
                InstanceDef::DropGlue(id, _) => {
                    summary.natives.push(NativeSummary::process(tcx, *id, name, NativeKind::Drop));
                }
                InstanceDef::CloneShim(id, _) => {
                    summary.natives.push(NativeSummary::process(tcx, *id, name, NativeKind::Clone));
                }
                InstanceDef::Virtual(id, index) => {
                    summary.natives.push(NativeSummary::process(
                        tcx,
                        *id,
                        name,
                        NativeKind::Virtual(*index),
                    ));
                }
                InstanceDef::VTableShim(id) => {
                    summary.natives.push(NativeSummary::process(
                        tcx,
                        *id,
                        name,
                        NativeKind::VTable,
                    ));
                }
                InstanceDef::FnPtrShim(id, _) => {
                    summary.natives.push(NativeSummary::process(tcx, *id, name, NativeKind::FnPtr));
                }
                InstanceDef::ReifyShim(..)
                | InstanceDef::FnPtrAddrShim(..)
                | InstanceDef::ThreadLocalShim(..) => {
                    bug!("unusual calls are not supported yet: {}", instance);
                }
            };
        }
    }

    // dump output
    let content =
        serde_json::to_string_pretty(&summary).expect("unexpected failure on JSON encoding");
    let symbol = tcx.crate_name(LOCAL_CRATE);
    let crate_name = symbol.as_str();
    let output = outdir.join(crate_name).with_extension("json");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .expect("unable to create output file");
    file.write_all(content.as_bytes()).expect("unexpected failure on outputting to file");
}

/// Identifier mimicking `DefId`
#[derive(Serialize)]
struct Ident {
    krate: usize,
    index: usize,
}

impl From<DefId> for Ident {
    fn from(id: DefId) -> Self {
        Self { krate: id.krate.as_usize(), index: id.index.as_usize() }
    }
}

/// A struct containing serializable information about the entire crate
#[derive(Serialize)]
struct CrateSummary {
    natives: Vec<NativeSummary>,
    functions: Vec<FunctionSummary>,
}

/// Kinds of native constructs
#[derive(Serialize)]
enum NativeKind {
    Intrinsic,
    Once,
    Drop,
    Clone,
    Virtual(usize),
    VTable,
    FnPtr,
}

/// A struct containing serializable information about one native function
#[derive(Serialize)]
struct NativeSummary {
    id: Ident,
    name: String,
    kind: NativeKind,
}

impl NativeSummary {
    /// Process an intrinsic instance
    fn process<'tcx>(_tcx: TyCtxt<'tcx>, id: DefId, name: Symbol, kind: NativeKind) -> Self {
        Self { id: id.into(), name: name.to_string(), kind }
    }
}

/// A struct containing serializable information about one user-defined function
#[derive(Serialize)]
struct FunctionSummary {
    id: Ident,
    name: String,
}

impl FunctionSummary {
    /// Process the mir body for one function
    fn process<'tcx>(tcx: TyCtxt<'tcx>, id: DefId, name: Symbol, body: &Body<'tcx>) -> Self {
        // sanity check
        let expected_phase = match tcx.def_kind(id) {
            DefKind::Ctor(_, CtorKind::Fn) => MirPhase::Built,
            DefKind::Fn | DefKind::AssocFn | DefKind::Closure | DefKind::Coroutine => {
                MirPhase::Runtime(RuntimePhase::Optimized)
            }
            kind => bug!("unexpected def_kind: {}", kind.descr(id)),
        };
        if body.phase != expected_phase {
            bug!(
                "MIR for '{}' with description {} is at an unexpected phase '{:?}'",
                name,
                tcx.def_descr(id),
                body.phase
            );
        }

        // done
        FunctionSummary { id: id.into(), name: name.to_string() }
    }
}
