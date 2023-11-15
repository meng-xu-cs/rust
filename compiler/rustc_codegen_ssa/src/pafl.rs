use std::fs::{self, OpenOptions};
use std::path::Path;

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
                InstanceDef::Item(id) => summary.functions.push(FunctionSummary::process(
                    tcx,
                    *id,
                    name,
                    tcx.optimized_mir(*id),
                )),
                InstanceDef::Intrinsic(id) => summary.natives.push(NativeSummary::process(
                    tcx,
                    *id,
                    name,
                    NativeKind::Intrinsic,
                )),
                InstanceDef::ClosureOnceShim { call_once: id, track_caller: _ } => {
                    summary.natives.push(NativeSummary::process(tcx, *id, name, NativeKind::Once))
                }
                InstanceDef::DropGlue(id, _) => {
                    summary.natives.push(NativeSummary::process(tcx, *id, name, NativeKind::Drop))
                }
                InstanceDef::CloneShim(id, _) => {
                    summary.natives.push(NativeSummary::process(tcx, *id, name, NativeKind::Clone))
                }
                InstanceDef::VTableShim(..)
                | InstanceDef::ReifyShim(..)
                | InstanceDef::Virtual(..)
                | InstanceDef::FnPtrShim(..)
                | InstanceDef::FnPtrAddrShim(..)
                | InstanceDef::ThreadLocalShim(..) => (),
            };
        }
    }

    // dump output
    let symbol = tcx.crate_name(LOCAL_CRATE);
    let crate_name = symbol.as_str();
    let output = outdir.join(crate_name).with_extension("json");
    let _file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .expect("unable to create output file");
}

/// A struct containing serializable information about the entire crate
#[derive(Encodable)]
struct CrateSummary {
    natives: Vec<NativeSummary>,
    functions: Vec<FunctionSummary>,
}

/// A struct containing serializable information about one native function
#[derive(Encodable)]
enum NativeKind {
    Intrinsic,
    Once,
    Drop,
    Clone,
}

/// A struct containing serializable information about one native function
#[derive(Encodable)]
struct NativeSummary {
    id: DefId,
    name: String,
    kind: NativeKind,
}

impl NativeSummary {
    /// Process an intrinsic instance
    fn process<'tcx>(_tcx: TyCtxt<'tcx>, id: DefId, name: Symbol, kind: NativeKind) -> Self {
        Self { id, name: name.to_string(), kind }
    }
}

/// A struct containing serializable information about one user-defined function
#[derive(Encodable)]
struct FunctionSummary {
    id: DefId,
    name: String,
}

impl FunctionSummary {
    /// Process the mir body for one function
    fn process<'tcx>(_tcx: TyCtxt<'tcx>, id: DefId, name: Symbol, body: &Body<'tcx>) -> Self {
        // sanity check
        if !matches!(body.phase, MirPhase::Runtime(RuntimePhase::Optimized)) {
            bug!("MIR for {} is at phase {:?}", name, body.phase);
        }

        // done
        FunctionSummary { id, name: name.to_string() }
    }
}
