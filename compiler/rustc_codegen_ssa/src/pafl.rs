use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use rustc_middle::mir::graphviz::write_mir_graphviz;
use rustc_span::Span;
use rustc_type_ir::Mutability;
use serde::Serialize;

use rustc_hir::def::{CtorKind, DefKind};
use rustc_middle::mir::mono::MonoItem;
use rustc_middle::mir::{
    BasicBlock, BasicBlockData, MirPhase, Operand, RuntimePhase, TerminatorKind, UnwindAction,
};
use rustc_middle::ty::{
    self, Const, ConstKind, ExistentialPredicate, FloatTy, GenericArgKind, GenericArgsRef,
    Instance, InstanceDef, IntTy, ParamEnv, Ty, TyCtxt, UintTy, ValTree,
};
use rustc_span::def_id::{DefId, LOCAL_CRATE};
use rustc_target::spec::abi::Abi;

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

/// Constant value or aggregates
#[derive(Serialize)]
enum ValueTree {
    Scalar { bit: usize, val: u128 },
    Struct(Vec<ValueTree>),
}

/// Serializable information about a Rust const
#[derive(Serialize)]
enum PaflConst {
    Param { index: u32, name: String },
    Value(ValueTree),
}

/// Serializable information about a Rust type
#[derive(Serialize)]
enum PaflType {
    Bool,
    Char,
    Isize,
    I8,
    I16,
    I32,
    I64,
    I128,
    Usize,
    U8,
    U16,
    U32,
    U64,
    U128,
    F32,
    F64,
    Str,
    Param { index: u32, name: String },
    Adt { id: Ident, generics: Vec<PaflGeneric> },
    Alias { id: Ident, generics: Vec<PaflGeneric> },
    Foreign(Ident),
    FnPtr(Vec<PaflType>, Box<PaflType>),
    FnDef { id: Ident, krate: Option<String>, path: String, generics: Vec<PaflGeneric> },
    Closure { id: Ident, krate: Option<String>, path: String, generics: Vec<PaflGeneric> },
    Dynamic(Vec<Ident>),
    ImmRef(Box<PaflType>),
    MutRef(Box<PaflType>),
    Slice(Box<PaflType>),
    Array(Box<PaflType>, PaflConst),
    Tuple(Vec<PaflType>),
}

/// Serializable information about a Rust generic argument
#[derive(Serialize)]
enum PaflGeneric {
    Lifetime,
    Type(PaflType),
    Const(PaflConst),
}

/// Callee of a call instruction
#[derive(Serialize)]
enum Callee {
    Local(PaflFunction),
    Cycle {
        id: Ident,
        path: String,
        generics: Vec<PaflGeneric>,
    },
    Virtual {
        id: Ident,
        krate: Option<String>,
        path: String,
        generics: Vec<PaflGeneric>,
        offset: usize,
    },
    Foreign {
        id: Ident,
        krate: String,
        path: String,
        generics: Vec<PaflGeneric>,
    },
    Intrinsic {
        id: Ident,
        path: String,
        generics: Vec<PaflGeneric>,
    },
    Unresolved {
        id: Ident,
        krate: Option<String>,
        path: String,
        generics: Vec<PaflGeneric>,
    },
}

/// Identifier mimicking `BasicBlock`
#[derive(Serialize)]
struct BlkId {
    index: usize,
}

impl From<BasicBlock> for BlkId {
    fn from(id: BasicBlock) -> Self {
        Self { index: id.as_usize() }
    }
}

/// How unwind might work
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

/// Kinds of terminator instructions of a basic block
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

/// Serializable information about a basic block
#[derive(Serialize)]
struct PaflBlock {
    id: BlkId,
    term: TermKind,
}

/// Serializable information about a user-defined function
#[derive(Serialize)]
struct PaflFunction {
    id: Ident,
    path: String,
    generics: Vec<PaflGeneric>,
    blocks: Vec<PaflBlock>,
}

/// Serializable information about the entire crate
#[derive(Serialize)]
struct PaflCrate {
    functions: Vec<PaflFunction>,
}

/// Helper for dumping path-AFL related information
struct PaflDump<'tcx> {
    /// context provider
    tcx: TyCtxt<'tcx>,
    /// parameter env
    param_env: ParamEnv<'tcx>,
    /// verbosity
    verbose: bool,
    /// path to meta directory
    path_meta: PathBuf,
    /// path to data directory
    path_data: PathBuf,
    /// path to the data file
    path_prefix: PathBuf,
    /// stack
    stack: Vec<Instance<'tcx>>,
}

impl<'tcx> PaflDump<'tcx> {
    /// Process a value tree
    fn process_vtree(&self, tree: ValTree<'tcx>) -> ValueTree {
        match tree {
            ValTree::Leaf(scalar) => ValueTree::Scalar {
                bit: scalar.size().bits_usize(),
                val: scalar.to_bits(scalar.size()).expect("scalar value"),
            },
            ValTree::Branch(items) => {
                let mut subs = vec![];
                for item in items {
                    subs.push(self.process_vtree(*item));
                }
                ValueTree::Struct(subs)
            }
        }
    }

    /// Process a constant
    fn process_const(&self, item: Const<'tcx>) -> PaflConst {
        match item.kind() {
            ConstKind::Param(param) => {
                PaflConst::Param { index: param.index, name: param.name.to_string() }
            }
            ConstKind::Value(value) => PaflConst::Value(self.process_vtree(value)),
            _ => bug!("unrecognized constant: {:?}", item),
        }
    }

    /// Process the type
    fn process_type(&self, item: Ty<'tcx>) -> PaflType {
        match item.kind() {
            ty::Bool => PaflType::Bool,
            ty::Char => PaflType::Char,
            ty::Int(IntTy::Isize) => PaflType::Isize,
            ty::Int(IntTy::I8) => PaflType::I8,
            ty::Int(IntTy::I16) => PaflType::I16,
            ty::Int(IntTy::I32) => PaflType::I32,
            ty::Int(IntTy::I64) => PaflType::I64,
            ty::Int(IntTy::I128) => PaflType::I128,
            ty::Uint(UintTy::Usize) => PaflType::Usize,
            ty::Uint(UintTy::U8) => PaflType::U8,
            ty::Uint(UintTy::U16) => PaflType::U16,
            ty::Uint(UintTy::U32) => PaflType::U32,
            ty::Uint(UintTy::U64) => PaflType::U64,
            ty::Uint(UintTy::U128) => PaflType::U128,
            ty::Float(FloatTy::F32) => PaflType::F32,
            ty::Float(FloatTy::F64) => PaflType::F64,
            ty::Str => PaflType::Str,
            ty::Param(p) => PaflType::Param { index: p.index, name: p.name.to_string() },
            ty::Adt(def, args) => {
                PaflType::Adt { id: def.did().into(), generics: self.process_generics(*args) }
            }
            ty::Alias(_, alias) => PaflType::Alias {
                id: alias.def_id.into(),
                generics: self.process_generics(alias.args),
            },
            ty::Foreign(def_id) => PaflType::Foreign((*def_id).into()),
            ty::FnPtr(binder) => {
                if !matches!(binder.abi(), Abi::Rust | Abi::RustCall) {
                    bug!("fn ptr not following the RustCall ABI: {}", binder.abi());
                }
                if binder.c_variadic() {
                    bug!("variadic not supported yet");
                }

                let mut inputs = vec![];
                for item in binder.inputs().iter() {
                    let ty = *item.skip_binder();
                    inputs.push(self.process_type(ty));
                }
                let output = self.process_type(binder.output().skip_binder());
                PaflType::FnPtr(inputs, output.into())
            }
            ty::FnDef(def_id, args) => {
                let krate = if def_id.is_local() {
                    None
                } else {
                    Some(self.tcx.crate_name(def_id.krate).to_string())
                };
                PaflType::FnDef {
                    id: (*def_id).into(),
                    krate,
                    path: self.tcx.def_path(*def_id).to_string_no_crate_verbose(),
                    generics: self.process_generics(*args),
                }
            }
            ty::Closure(def_id, args) => {
                let krate = if def_id.is_local() {
                    None
                } else {
                    Some(self.tcx.crate_name(def_id.krate).to_string())
                };
                PaflType::Closure {
                    id: (*def_id).into(),
                    krate,
                    path: self.tcx.def_path(*def_id).to_string_no_crate_verbose(),
                    generics: self.process_generics(*args),
                }
            }
            ty::Ref(_region, sub, mutability) => {
                let converted = self.process_type(*sub);
                match mutability {
                    Mutability::Not => PaflType::ImmRef(converted.into()),
                    Mutability::Mut => PaflType::MutRef(converted.into()),
                }
            }
            ty::Slice(sub) => PaflType::Slice(self.process_type(*sub).into()),
            ty::Array(sub, len) => {
                PaflType::Array(self.process_type(*sub).into(), self.process_const(*len))
            }
            ty::Tuple(elems) => {
                PaflType::Tuple(elems.iter().map(|e| self.process_type(e)).collect())
            }
            ty::Dynamic(binders, _region, _) => {
                let mut traits = vec![];
                for binder in *binders {
                    let predicate = binder.skip_binder();
                    let def_id = match predicate {
                        ExistentialPredicate::Trait(r) => r.def_id,
                        ExistentialPredicate::Projection(r) => r.def_id,
                        ExistentialPredicate::AutoTrait(r) => r,
                    };
                    traits.push(def_id.into());
                }
                PaflType::Dynamic(traits)
            }
            _ => bug!("unrecognized type: {:?}", item),
        }
    }

    /// Process the generic arguments
    fn process_generics(&self, args: GenericArgsRef<'tcx>) -> Vec<PaflGeneric> {
        let mut generics = vec![];
        for arg in args {
            let sub = match arg.unpack() {
                GenericArgKind::Lifetime(_region) => PaflGeneric::Lifetime,
                GenericArgKind::Type(item) => PaflGeneric::Type(self.process_type(item)),
                GenericArgKind::Const(item) => PaflGeneric::Const(self.process_const(item)),
            };
            generics.push(sub);
        }
        generics
    }

    /// Resolve the call target
    fn process_callee(&self, callee: &Operand<'tcx>, span: Span) -> Callee {
        match callee.const_fn_def() {
            None => bug!("unable to handle the indirect call: {:?}", span),
            Some((def_id, generic_args)) => {
                // resolve trait targets, if possible
                match Instance::resolve(self.tcx, self.param_env, def_id, generic_args)
                    .expect("resolution failure")
                {
                    None => {
                        let krate = if def_id.is_local() {
                            None
                        } else {
                            Some(self.tcx.crate_name(def_id.krate).to_string())
                        };
                        Callee::Unresolved {
                            id: def_id.into(),
                            krate,
                            path: self.tcx.def_path(def_id).to_string_no_crate_verbose(),
                            generics: self.process_generics(generic_args),
                        }
                    }
                    Some(resolved) => match resolved.def {
                        InstanceDef::Item(item_id) => {
                            if item_id.is_local() {
                                match PaflDump::process_instance(
                                    self.tcx,
                                    resolved,
                                    self.verbose,
                                    &self.path_meta,
                                    &self.path_data,
                                    &self.stack,
                                ) {
                                    None => Callee::Cycle {
                                        id: item_id.into(),
                                        path: self
                                            .tcx
                                            .def_path(item_id)
                                            .to_string_no_crate_verbose(),
                                        generics: self.process_generics(generic_args),
                                    },
                                    Some(func) => Callee::Local(func),
                                }
                            } else {
                                Callee::Foreign {
                                    id: item_id.into(),
                                    krate: self.tcx.crate_name(item_id.krate).to_string(),
                                    path: self.tcx.def_path(item_id).to_string_no_crate_verbose(),
                                    generics: self.process_generics(resolved.args),
                                }
                            }
                        }
                        InstanceDef::Virtual(virtual_id, offset) => {
                            let krate = if virtual_id.is_local() {
                                None
                            } else {
                                Some(self.tcx.crate_name(virtual_id.krate).to_string())
                            };
                            Callee::Virtual {
                                id: virtual_id.into(),
                                krate,
                                path: self.tcx.def_path(virtual_id).to_string_no_crate_verbose(),
                                generics: self.process_generics(resolved.args),
                                offset,
                            }
                        }
                        InstanceDef::Intrinsic(intrinsic_id) => Callee::Intrinsic {
                            id: intrinsic_id.into(),
                            path: self.tcx.def_path(intrinsic_id).to_string_no_crate_verbose(),
                            generics: self.process_generics(resolved.args),
                        },
                        InstanceDef::ClosureOnceShim { .. }
                        | InstanceDef::DropGlue(..)
                        | InstanceDef::CloneShim(..)
                        | InstanceDef::VTableShim(..)
                        | InstanceDef::FnPtrShim(..)
                        | InstanceDef::ReifyShim(..)
                        | InstanceDef::FnPtrAddrShim(..)
                        | InstanceDef::ThreadLocalShim(..) => {
                            bug!("unusual calls are not supported yet: {}", resolved);
                        }
                    },
                }
            }
        }
    }

    /// Process the mir for one basic block
    fn process_block(&self, id: BasicBlock, data: &BasicBlockData<'tcx>) -> PaflBlock {
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
                callee: self.process_callee(func, term.source_info.span),
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
        PaflBlock { id: id.into(), term: kind }
    }

    /// Process the mir body for one function
    fn process_function(&self, id: DefId, generic_args: GenericArgsRef<'tcx>) -> PaflFunction {
        let path = self.tcx.def_path(id).to_string_no_crate_verbose();
        let body = self.tcx.optimized_mir(id);

        // sanity check
        let expected_phase = match self.tcx.def_kind(id) {
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
                self.tcx.def_descr(id),
                body.phase
            );
        }

        // handle the generics
        let generics = self.process_generics(generic_args);

        // dump the control flow graph if requested
        match std::env::var_os("PAFL_CFG") {
            None => (),
            Some(v) => {
                if v.to_str().map_or(false, |s| s == path.as_str()) {
                    // dump the cfg
                    let dot_path = self.path_prefix.with_extension("dot");
                    let mut dot_file = OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&dot_path)
                        .expect("unable to create dot file");
                    write_mir_graphviz(self.tcx, Some(id), &mut dot_file)
                        .expect("failed to create dot file");
                }
            }
        }

        // iterate over each basic blocks
        let mut blocks = vec![];
        for blk_id in body.basic_blocks.reverse_postorder() {
            let blk_data = body.basic_blocks.get(*blk_id).unwrap();
            blocks.push(self.process_block(*blk_id, blk_data));
        }

        // done
        PaflFunction { id: id.into(), path, generics, blocks }
    }

    /// Process a codegen instance
    fn process_instance(
        tcx: TyCtxt<'tcx>,
        instance: Instance<'tcx>,
        verbose: bool,
        path_meta: &Path,
        path_data: &Path,
        stack: &[Instance<'tcx>],
    ) -> Option<PaflFunction> {
        // avoid recursion
        if stack.iter().any(|i| *i == instance) {
            return None;
        }

        // verbose mode
        let path = tcx.def_path(instance.def_id());
        if verbose {
            println!("Processing {}", path.to_string_no_crate_verbose());
        }

        // normalize and check consistency
        let param_env = tcx.param_env_reveal_all_normalized(instance.def_id());
        let normalized_ty = instance.ty(tcx, param_env);
        match normalized_ty.kind() {
            ty::FnDef(ty_def_id, ty_def_args) | ty::Closure(ty_def_id, ty_def_args) => {
                if *ty_def_id != instance.def_id() {
                    bug!("normalized type def_id mismatch");
                }
                if ty_def_args.len() != instance.args.len() {
                    bug!("normalized type generics mismatch");
                }
                for (t1, t2) in ty_def_args.iter().zip(instance.args.iter()) {
                    if t1 != t2 {
                        bug!("normalized type generics mismatch");
                    }
                }
            }
            _ => bug!("normalized type is neither function nor closure"),
        }

        // create a place holder
        let index = loop {
            let mut count: usize = 0;
            for entry in fs::read_dir(path_meta).expect("list meta directory") {
                let _ = entry.expect("iterate meta directory entry");
                count += 1;
            }
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path_meta.join(count.to_string()))
            {
                Ok(mut file) => {
                    let content = format!("{}", path.to_string_no_crate_verbose(),);
                    file.write_all(content.as_bytes()).expect("save meta content");
                    break count;
                }
                Err(_) => continue,
            }
        };
        let path_prefix = path_data.join(index.to_string());

        // construct the dumper
        let mut new_stack: Vec<_> = stack.iter().cloned().collect();
        new_stack.push(instance);

        let dumper = PaflDump {
            tcx,
            param_env,
            verbose,
            path_meta: path_meta.to_path_buf(),
            path_data: path_data.to_path_buf(),
            path_prefix,
            stack: new_stack,
        };

        // branch processing by instance type
        let fun_id = match &instance.def {
            InstanceDef::Item(id) => *id,
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
                bug!("invalid top-level instance: {}", instance);
            }
        };
        Some(dumper.process_function(fun_id, instance.args))
    }
}

/// A complete dump of both the control-flow graph and the call graph of the compilation context
pub fn dump(tcx: TyCtxt<'_>, outdir: &Path) {
    // prepare directory layout
    fs::create_dir_all(outdir).expect("unable to create output directory");
    let path_meta = outdir.join("meta");
    fs::create_dir_all(&path_meta).expect("unable to create meta directory");
    let path_data = outdir.join("data");
    fs::create_dir_all(&path_data).expect("unable to create meta directory");

    // verbosity
    let verbose = std::env::var_os("PAFL_VERBOSE")
        .and_then(|v| v.into_string().ok())
        .map_or(false, |v| v.as_str() == "1");

    // extract the mir for each codegen unit
    let mut summary = PaflCrate { functions: Vec::new() };

    let (_, units) = tcx.collect_and_partition_mono_items(());
    for unit in units {
        for item in unit.items().keys() {
            let instance = match item {
                MonoItem::Fn(i) => *i,
                MonoItem::Static(_) => continue,
                MonoItem::GlobalAsm(_) => bug!("unexpected assembly"),
            };

            // ignore codegen units not in the current crate
            if !instance.def_id().is_local() {
                continue;
            }

            // process it and save the result to summary
            match PaflDump::process_instance(tcx, instance, verbose, &path_meta, &path_data, &[]) {
                None => bug!("cannot have a recursive call when call stack is empty"),
                Some(converted) => {
                    summary.functions.push(converted);
                }
            }
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
