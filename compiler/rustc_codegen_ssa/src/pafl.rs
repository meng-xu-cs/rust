use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

use serde::Serialize;

use rustc_hir::def::{CtorKind, DefKind};
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::mir::mono::MonoItem;
use rustc_middle::mir::{BasicBlock, BasicBlockData, MirPhase, RuntimePhase};
use rustc_middle::ty::{InstanceDef, TyCtxt};
use rustc_span::def_id::DefId;

/// A complete dump of both the control-flow graph and the call graph of the compilation context
pub fn dump(tcx: TyCtxt<'_>, outdir: &Path) {
    // prepare directory
    fs::create_dir_all(outdir).expect("unable to create output directory");

    // extract the mir for each codegen unit
    let mut summary = CrateSummary { natives: Vec::new(), functions: Vec::new() };

    let (_, units) = tcx.collect_and_partition_mono_items(());
    for unit in units {
        let hash = unit.name();
        trace!("processing code unit {}", hash);

        for item in unit.items().keys() {
            let instance = match item {
                MonoItem::Fn(i) => i,
                MonoItem::Static(_) => continue,
                MonoItem::GlobalAsm(_) => bug!("unexpected assembly"),
            };

            // branch processing by instance type
            match &instance.def {
                InstanceDef::Item(id) => {
                    summary.functions.push(FunctionSummary::process(tcx, *id));
                }
                InstanceDef::Intrinsic(id) => {
                    summary.natives.push(NativeSummary::process(tcx, *id, NativeKind::Intrinsic));
                }
                InstanceDef::ClosureOnceShim { call_once: id, track_caller: _ } => {
                    summary.natives.push(NativeSummary::process(tcx, *id, NativeKind::Once));
                }
                InstanceDef::DropGlue(id, _) => {
                    summary.natives.push(NativeSummary::process(tcx, *id, NativeKind::Drop));
                }
                InstanceDef::CloneShim(id, _) => {
                    summary.natives.push(NativeSummary::process(tcx, *id, NativeKind::Clone));
                }
                InstanceDef::Virtual(id, index) => {
                    summary.natives.push(NativeSummary::process(
                        tcx,
                        *id,
                        NativeKind::Virtual(*index),
                    ));
                }
                InstanceDef::VTableShim(id) => {
                    summary.natives.push(NativeSummary::process(tcx, *id, NativeKind::VTable));
                }
                InstanceDef::FnPtrShim(id, _) => {
                    summary.natives.push(NativeSummary::process(tcx, *id, NativeKind::FnPtr));
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
    path: String,
    kind: NativeKind,
}

impl NativeSummary {
    /// Process an intrinsic instance
    fn process<'tcx>(tcx: TyCtxt<'tcx>, id: DefId, kind: NativeKind) -> Self {
        let path = tcx.def_path(id).to_string_no_crate_verbose();
        Self { id: id.into(), path, kind }
    }
}

/// A struct containing serializable information about one user-defined function
#[derive(Serialize)]
struct FunctionSummary {
    id: Ident,
    path: String,
    blocks: Vec<BlockSummary>,
}

impl FunctionSummary {
    /// Process the mir body for one function
    fn process<'tcx>(tcx: TyCtxt<'tcx>, id: DefId) -> Self {
        let path = tcx.def_path(id).to_string_no_crate_verbose();
        let body = tcx.optimized_mir(id);

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
                "MIR for '{}' with description '{}' is at an unexpected phase '{:?}'",
                path,
                tcx.def_descr(id),
                body.phase
            );
        }

        // iterate over each basic blocks
        let mut blocks = vec![];
        for blk_id in body.basic_blocks.reverse_postorder() {
            let blk_data = body.basic_blocks.get(*blk_id).unwrap();
            blocks.push(BlockSummary::process(tcx, *blk_id, blk_data));
        }

        // done
        FunctionSummary { id: id.into(), path, blocks }
    }
}

/// A struct containing serializable information about a basic block
#[derive(Serialize)]
struct BlockSummary {
    index: usize,
}

impl BlockSummary {
    /// Process the mir for one basic block
    fn process<'tcx>(_tcx: TyCtxt<'tcx>, id: BasicBlock, _data: &BasicBlockData<'tcx>) -> Self {
        Self { index: id.as_usize() }
    }
}
