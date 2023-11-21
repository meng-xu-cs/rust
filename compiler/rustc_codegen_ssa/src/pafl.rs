use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

use rustc_middle::mir::graphviz::write_mir_fn_graphviz;
use serde::Serialize;

use rustc_hir::def::{CtorKind, DefKind};
use rustc_middle::mir::mono::MonoItem;
use rustc_middle::mir::{
    BasicBlock, BasicBlockData, Body, MirPhase, Operand, RuntimePhase, TerminatorKind, UnwindAction,
};
use rustc_middle::ty::{
    Const, ConstKind, GenericArgKind, GenericArgsRef, InstanceDef, RegionKind, TyCtxt, ValTree,
};
use rustc_span::def_id::{DefId, LOCAL_CRATE};

/// A complete dump of both the control-flow graph and the call graph of the compilation context
pub fn dump(tcx: TyCtxt<'_>, outdir: &Path) {
    // prepare directory layout
    fs::create_dir_all(outdir).expect("unable to create output directory");
    let path_meta = outdir.join("meta");
    fs::create_dir_all(&path_meta).expect("unable to create meta directory");
    let path_data = outdir.join("data");
    fs::create_dir_all(&path_data).expect("unable to create meta directory");

    // extract the mir for each codegen unit
    let mut summary = CrateSummary { functions: Vec::new() };

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

            // ignore codegen units not in the current crate
            let path = tcx.def_path(instance.def_id());
            if path.krate != LOCAL_CRATE {
                continue;
            }

            // create a place holder
            let index = loop {
                let mut count: usize = 0;
                for entry in fs::read_dir(&path_meta).expect("list meta directory") {
                    let _ = entry.expect("iterate meta directory entry");
                    count += 1;
                }
                match OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path_meta.join(count.to_string()))
                {
                    Ok(mut file) => {
                        let kind = match instance.def {
                            InstanceDef::Item(_) => "function",
                            InstanceDef::VTableShim(_) => "shim(vtable)",
                            InstanceDef::ReifyShim(_) => "shim(reify)",
                            InstanceDef::ThreadLocalShim(_) => "shim(tls)",
                            InstanceDef::Intrinsic(_) => "intrinsic",
                            InstanceDef::Virtual(_, _) => "virtual",
                            InstanceDef::FnPtrShim(_, _) => "shim(<fn>)",
                            InstanceDef::ClosureOnceShim { .. } => "shim(once)",
                            InstanceDef::DropGlue(_, _) => "shim(drop)",
                            InstanceDef::CloneShim(_, _) => "shim(clone)",
                            InstanceDef::FnPtrAddrShim(_, _) => "shim(&<fn>)",
                        };
                        let content = format!("[{}] {}", kind, path.to_string_no_crate_verbose());
                        file.write_all(content.as_bytes()).expect("save meta content");
                        break count;
                    }
                    Err(_) => continue,
                }
            };
            let data_prefix = path_data.join(index.to_string());

            // branch processing by instance type
            match &instance.def {
                InstanceDef::Item(id) => {
                    summary.functions.push(FunctionSummary::process(
                        tcx,
                        *id,
                        instance.args,
                        &data_prefix,
                    ));
                }
                InstanceDef::Intrinsic(..)
                | InstanceDef::ClosureOnceShim { .. }
                | InstanceDef::DropGlue(..)
                | InstanceDef::CloneShim(..)
                | InstanceDef::Virtual(..)
                | InstanceDef::VTableShim(..)
                | InstanceDef::FnPtrShim(..)
                | InstanceDef::ReifyShim(..)
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
    functions: Vec<FunctionSummary>,
}

/// A struct containing serializable information about a type
#[derive(Serialize)]
enum GenericSummary {
    Lifetime,
    Type,
    Const(ConstSummary),
}

impl GenericSummary {
    /// Process the generic arguments
    fn process<'tcx>(tcx: TyCtxt<'tcx>, args: GenericArgsRef<'tcx>) -> Vec<Self> {
        let mut generics = vec![];
        for arg in args {
            let sub = match arg.unpack() {
                GenericArgKind::Lifetime(region) => {
                    if !matches!(region.kind(), RegionKind::ReErased) {
                        bug!("lifetime not erased yet");
                    }
                    GenericSummary::Lifetime
                }
                GenericArgKind::Type(..) => GenericSummary::Type,
                GenericArgKind::Const(item) => {
                    GenericSummary::Const(ConstSummary::process(tcx, item))
                }
            };
            generics.push(sub);
        }
        generics
    }
}

/// A struct containing serializable information about a const
#[derive(Serialize)]
enum ConstSummary {
    Param { index: u32, name: String },
    Value(ValueTree),
}

impl ConstSummary {
    /// Process the constant
    fn process<'tcx>(_tcx: TyCtxt<'tcx>, item: Const<'tcx>) -> Self {
        match item.kind() {
            ConstKind::Param(param) => {
                Self::Param { index: param.index, name: param.name.to_string() }
            }
            ConstKind::Value(value) => Self::Value(ValueTree::process(value)),
            _ => bug!("unrecognized constant: {:?}", item),
        }
    }
}

#[derive(Serialize)]
enum ValueTree {
    Scalar { bit: usize, val: u128 },
    Struct(Vec<ValueTree>),
}

impl ValueTree {
    fn process<'tcx>(tree: ValTree<'tcx>) -> Self {
        match tree {
            ValTree::Leaf(scalar) => Self::Scalar {
                bit: scalar.size().bits_usize(),
                val: scalar.to_bits(scalar.size()).expect("scalar value"),
            },
            ValTree::Branch(items) => {
                let mut subs = vec![];
                for item in items {
                    subs.push(ValueTree::process(*item));
                }
                Self::Struct(subs)
            }
        }
    }
}

/// A struct containing serializable information about one user-defined function
#[derive(Serialize)]
struct FunctionSummary {
    id: Ident,
    path: String,
    generics: Vec<GenericSummary>,
    blocks: Vec<BlockSummary>,
}

impl FunctionSummary {
    /// Process the mir body for one function
    fn process<'tcx>(
        tcx: TyCtxt<'tcx>,
        id: DefId,
        generic_args: GenericArgsRef<'tcx>,
        prefix: &Path,
    ) -> Self {
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

        // handle the generics
        let generics = GenericSummary::process(tcx, generic_args);

        // iterate over each basic blocks
        let mut blocks = vec![];
        for blk_id in body.basic_blocks.reverse_postorder() {
            let blk_data = body.basic_blocks.get(*blk_id).unwrap();
            blocks.push(BlockSummary::process(tcx, *blk_id, blk_data, body, prefix));
        }

        // done
        FunctionSummary { id: id.into(), path, generics, blocks }
    }
}

/// Identifier mimicking `DefId`
#[derive(Serialize)]
struct BlkId {
    index: usize,
}

impl From<BasicBlock> for BlkId {
    fn from(id: BasicBlock) -> Self {
        Self { index: id.as_usize() }
    }
}

/// Kinds of callee
#[derive(Serialize)]
struct Callee {
    id: Ident,
    generics: Vec<GenericSummary>,
}

impl Callee {
    /// Resolve the call target
    fn process<'tcx>(
        tcx: TyCtxt<'tcx>,
        callee: &Operand<'tcx>,
        bid: BasicBlock,
        body: &Body<'tcx>,
        prefix: &Path,
    ) -> Self {
        match callee.const_fn_def() {
            None => {
                // dump the control flow graph if requested
                match std::env::var_os("PAFL_DEBUG") {
                    None => (),
                    Some(v) => {
                        if v.to_str().map_or(false, |s| s == "1") {
                            // dump the cfg
                            let dot_path = prefix.with_extension(format!("{}.dot", bid.as_usize()));
                            let mut dot_file = OpenOptions::new()
                                .write(true)
                                .create_new(true)
                                .open(&dot_path)
                                .expect("unable to create dot file");
                            write_mir_fn_graphviz(tcx, body, false, &mut dot_file)
                                .expect("failed to create dot file");
                        } else {
                            bug!("invalid value for environment variable PAFL_CFG");
                        }
                    }
                }

                // panic on indirect calls
                bug!("unable to handle the indirect calls in function: {:?}", body.span);
            }
            Some((def_id, generic_args)) => {
                Self { id: def_id.into(), generics: GenericSummary::process(tcx, generic_args) }
            }
        }
    }
}

/// How unwind will work
#[derive(Serialize)]
enum UnwindRoute {
    Resume,
    Terminate,
    Unreachable,
    Cleanup(BlkId),
}

impl From<&UnwindAction> for UnwindRoute {
    fn from(action: &UnwindAction) -> Self {
        match action {
            UnwindAction::Continue => Self::Resume,
            UnwindAction::Unreachable => Self::Unreachable,
            UnwindAction::Terminate(..) => Self::Terminate,
            UnwindAction::Cleanup(blk) => Self::Cleanup((*blk).into()),
        }
    }
}

/// Kinds of terminator instructions
#[derive(Serialize)]
enum TermKind {
    Unreachable,
    Goto(BlkId),
    Switch(Vec<BlkId>),
    Return,
    UnwindResume,
    UnwindFinish,
    Assert { target: BlkId, unwind: UnwindRoute },
    Drop { target: BlkId, unwind: UnwindRoute },
    Call { callee: Callee, target: Option<BlkId>, unwind: UnwindRoute },
}

/// A struct containing serializable information about a basic block
#[derive(Serialize)]
struct BlockSummary {
    id: BlkId,
    term: TermKind,
}

impl BlockSummary {
    /// Process the mir for one basic block
    fn process<'tcx>(
        tcx: TyCtxt<'tcx>,
        id: BasicBlock,
        data: &BasicBlockData<'tcx>,
        body: &Body<'tcx>,
        prefix: &Path,
    ) -> Self {
        let term = data.terminator();

        // match by the terminator
        let kind = match &term.kind {
            // basics
            TerminatorKind::Goto { target } => TermKind::Goto((*target).into()),
            TerminatorKind::SwitchInt { discr: _, targets } => {
                TermKind::Switch(targets.all_targets().iter().map(|b| (*b).into()).collect())
            }
            TerminatorKind::Unreachable => TermKind::Unreachable,
            TerminatorKind::Return => TermKind::Return,
            // call (which may unwind)
            TerminatorKind::Call {
                func,
                args: _,
                destination: _,
                target,
                unwind,
                call_source: _,
                fn_span: _,
            } => TermKind::Call {
                callee: Callee::process(tcx, func, id, body, prefix),
                target: target.as_ref().map(|t| (*t).into()),
                unwind: unwind.into(),
            },
            TerminatorKind::Drop { place: _, target, unwind, replace: _ } => {
                TermKind::Drop { target: (*target).into(), unwind: unwind.into() }
            }
            TerminatorKind::Assert { cond: _, expected: _, msg: _, target, unwind } => {
                TermKind::Assert { target: (*target).into(), unwind: unwind.into() }
            }
            // unwinding
            TerminatorKind::UnwindResume => TermKind::UnwindResume,
            TerminatorKind::UnwindTerminate(..) => TermKind::UnwindFinish,
            // imaginary
            TerminatorKind::FalseEdge { real_target, imaginary_target: _ }
            | TerminatorKind::FalseUnwind { real_target, unwind: _ } => {
                TermKind::Goto((*real_target).into())
            }
            // coroutine
            TerminatorKind::Yield { .. } | TerminatorKind::CoroutineDrop => {
                bug!("unexpected coroutine")
            }
            // assembly
            TerminatorKind::InlineAsm { .. } => bug!("unexpected inline assembly"),
        };

        // done
        Self { id: id.into(), term: kind }
    }
}
