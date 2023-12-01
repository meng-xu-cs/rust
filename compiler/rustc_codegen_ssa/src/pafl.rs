use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use rustc_data_structures::fx::FxHashMap;
use rustc_middle::mir::graphviz::write_mir_fn_graphviz;
use rustc_middle::mir::mono::MonoItem;
use rustc_middle::mir::{BasicBlock, BasicBlockData, Body, Operand, TerminatorKind, UnwindAction};
use rustc_middle::ty::{
    self, Const, ConstKind, EarlyBinder, ExistentialPredicate, FloatTy, GenericArgKind,
    GenericArgsRef, Instance, InstanceDef, IntTy, ParamEnv, Ty, TyCtxt, UintTy, ValTree,
};
use rustc_span::def_id::{DefId, LOCAL_CRATE};
use rustc_span::Span;
use rustc_target::spec::abi::Abi;
use rustc_type_ir::Mutability;

use serde::Serialize;

/// Native functions
#[derive(Serialize)]
enum Native {
    TLSWith,
}

impl Native {
    const BUILTINS: [Native; 1] = [Self::TLSWith];

    /// unlock the pattern triple
    fn pattern(&self) -> (&str, &str, &str) {
        match self {
            Self::TLSWith => ("std", "::thread::local::", "with"),
        }
    }

    /// probe whether a def_id is a native built-in
    fn probe(tcx: TyCtxt<'_>, id: DefId) -> Option<Self> {
        if id.is_local() {
            return None;
        }

        let krate = tcx.crate_name(id.krate).to_string();
        let path = tcx.def_path(id).to_string_no_crate_verbose();

        for item in Self::BUILTINS {
            let (k, prefix, suffix) = item.pattern();
            if &krate != k {
                continue;
            }
            match path.as_str().strip_prefix(prefix).and_then(|s| s.strip_suffix(suffix)) {
                None => continue,
                Some(_) => return Some(item),
            }
        }
        None
    }
}

/// Identifier mimicking `DefId`
#[derive(Serialize, Clone)]
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
#[derive(Serialize, Clone)]
enum ValueTree {
    Scalar { bit: usize, val: u128 },
    Struct(Vec<ValueTree>),
}

/// Serializable information about a Rust const
#[derive(Serialize, Clone)]
enum PaflConst {
    Param { index: u32, name: String },
    Value(ValueTree),
}

/// Serializable information about a Rust type
#[derive(Serialize, Clone)]
enum PaflType {
    Never,
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
    Adt(TyInstKey),
    Alias(TyInstKey),
    Opaque(Ident),
    FnPtr(Vec<PaflType>, Box<PaflType>),
    FnDef(FnInstKey),
    Closure(FnInstKey),
    Dynamic(Vec<Ident>),
    ImmRef(Box<PaflType>),
    MutRef(Box<PaflType>),
    ImmPtr(Box<PaflType>),
    MutPtr(Box<PaflType>),
    Slice(Box<PaflType>),
    Array(Box<PaflType>, PaflConst),
    Tuple(Vec<PaflType>),
}

/// Serializable information about a Rust generic argument
#[derive(Serialize, Clone)]
enum PaflGeneric {
    Lifetime,
    Type(PaflType),
    Const(PaflConst),
}

/// Identifier for type instance
#[derive(Serialize, Clone)]
struct TyInstKey {
    krate: Option<String>,
    index: usize,
    path: String,
    generics: Vec<PaflGeneric>,
}

/// Identifier for function instance
#[derive(Serialize, Clone)]
struct FnInstKey {
    krate: Option<String>,
    index: usize,
    path: String,
    generics: Vec<PaflGeneric>,
}

/// Kind of a call instruction
#[derive(Serialize)]
enum CallKind {
    Direct,
    Bridge,
    Virtual(usize),
    Builtin(Native),
    Intrinsic,
}

/// Callee of a call instruction
#[derive(Serialize)]
struct CallSite {
    inst: FnInstKey,
    kind: CallKind,
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
    Call { site: CallSite, target: Option<BlkId>, unwind: UnwindRoute },
}

/// Serializable information about a basic block
#[derive(Serialize)]
struct PaflBlock {
    id: BlkId,
    term: TermKind,
}

/// Serializable information about a user-defined function
#[derive(Serialize)]
struct PaflCFG {
    blocks: Vec<PaflBlock>,
}

/// Serializable information about a user-defined function
#[derive(Serialize)]
enum FnBody {
    Defined(PaflCFG),
    Bridged(PaflCFG),
    Skipped,
    Intrinsic,
}

/// Serializable information about a user-defined function
#[derive(Serialize)]
struct PaflFunction {
    inst: FnInstKey,
    body: FnBody,
}

/// Serializable information about the entire crate
#[derive(Serialize)]
struct PaflCrate {
    functions: Vec<PaflFunction>,
}

/// Helper for dumping path-AFL related information
struct PaflDump<'sum, 'tcx> {
    /// context provider
    tcx: TyCtxt<'tcx>,
    /// parameter environment
    param_env: ParamEnv<'tcx>,
    /// verbosity
    verbose: bool,
    /// path to meta directory
    path_meta: PathBuf,
    /// path to data directory
    path_data: PathBuf,
    /// path to the data file
    path_prefix: PathBuf,
    /// call stack
    stack: &'sum mut Vec<Instance<'tcx>>,
    /// information cache
    cache: &'sum mut FxHashMap<Instance<'tcx>, FnInstKey>,
    /// summary repository
    summary: &'sum mut Vec<PaflFunction>,
}

impl<'sum, 'tcx> PaflDump<'sum, 'tcx> {
    /// initialize the context for information dumping
    fn initialize(&self, instance: Instance<'tcx>) {
        // normalize and check consistency
        let normalized_ty = instance.ty(self.tcx, self.param_env);
        match normalized_ty.kind() {
            ty::FnDef(ty_def_id, ty_def_args) | ty::Closure(ty_def_id, ty_def_args) => {
                if *ty_def_id != instance.def_id() {
                    bug!("normalized type def_id mismatch");
                }
                if ty_def_args.len() != instance.args.len() {
                    bug!("normalized type generics length mismatch");
                }
                for (t1, t2) in ty_def_args.iter().zip(instance.args.iter()) {
                    if t1 != t2 {
                        bug!("normalized type generics content mismatch");
                    }
                }
            }
            _ => bug!("normalized type is neither function nor closure"),
        }
    }
}

impl<'sum, 'tcx> PaflDump<'sum, 'tcx> {
    /// Resolve an instantiation to a ty key
    fn resolve_ty_key(&self, id: DefId, args: GenericArgsRef<'tcx>) -> TyInstKey {
        let krate =
            if id.is_local() { None } else { Some(self.tcx.crate_name(id.krate).to_string()) };
        TyInstKey {
            krate,
            index: id.index.as_usize(),
            path: self.tcx.def_path(id).to_string_no_crate_verbose(),
            generics: self.process_generics(args),
        }
    }

    /// Resolve an instantiation to a fn key
    fn resolve_fn_key(&self, id: DefId, args: GenericArgsRef<'tcx>) -> FnInstKey {
        let krate =
            if id.is_local() { None } else { Some(self.tcx.crate_name(id.krate).to_string()) };
        FnInstKey {
            krate,
            index: id.index.as_usize(),
            path: self.tcx.def_path(id).to_string_no_crate_verbose(),
            generics: self.process_generics(args),
        }
    }
}

impl<'sum, 'tcx> PaflDump<'sum, 'tcx> {
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
            ty::Never => PaflType::Never,
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
            ty::Adt(def, args) => PaflType::Adt(self.resolve_ty_key(def.did(), args)),
            ty::Alias(_, alias) => PaflType::Alias(self.resolve_ty_key(alias.def_id, alias.args)),
            ty::Foreign(def_id) => PaflType::Opaque((*def_id).into()),
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
            ty::FnDef(def_id, args) => PaflType::FnDef(self.resolve_fn_key(*def_id, *args)),
            ty::Closure(def_id, args) => PaflType::Closure(self.resolve_fn_key(*def_id, *args)),
            ty::Ref(_region, sub, mutability) => {
                let converted = self.process_type(*sub);
                match mutability {
                    Mutability::Not => PaflType::ImmRef(converted.into()),
                    Mutability::Mut => PaflType::MutRef(converted.into()),
                }
            }
            ty::RawPtr(ty_and_mut) => {
                let converted = self.process_type(ty_and_mut.ty);
                match ty_and_mut.mutbl {
                    Mutability::Not => PaflType::ImmPtr(converted.into()),
                    Mutability::Mut => PaflType::MutPtr(converted.into()),
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
    fn process_callsite(&mut self, callee: &Operand<'tcx>, span: Span) -> CallSite {
        // extract def_id and generic arguments for callee
        let cty = match callee.constant() {
            None => bug!("callee is not a constant: {:?}", span),
            Some(c) => c.const_.ty(),
        };
        let (def_id, generic_args) = match cty.kind() {
            ty::Closure(def_id, generic_args) | ty::FnDef(def_id, generic_args) => {
                (*def_id, *generic_args)
            }
            _ => bug!("callee is not a function or closure: {:?}", span),
        };

        // test if we should skip this function
        if let Some(native) = Native::probe(self.tcx, def_id) {
            let inst = self.resolve_fn_key(def_id, generic_args);
            return CallSite { inst, kind: CallKind::Builtin(native) };
        }

        if self.verbose {
            print!(
                "{} - resolving: {}{}",
                "  ".repeat(self.stack.len()),
                self.tcx.crate_name(def_id.krate).to_string(),
                self.tcx.def_path(def_id).to_string_no_crate_verbose(),
            );
        }

        // resolve trait targets, if possible
        let resolved = Instance::expect_resolve(self.tcx, self.param_env, def_id, generic_args);
        let call_site = match resolved.def {
            InstanceDef::Item(_) => {
                if self.verbose {
                    println!(" ~> direct");
                }
                let inst = PaflDump::summarize_instance(
                    self.tcx,
                    self.param_env,
                    resolved,
                    self.verbose,
                    &self.path_meta,
                    &self.path_data,
                    self.stack,
                    self.cache,
                    self.summary,
                );
                CallSite { inst, kind: CallKind::Direct }
            }
            InstanceDef::ClosureOnceShim { .. } => {
                if self.verbose {
                    println!(" ~> closure");
                }

                // extract the actual callee
                assert_eq!(resolved.args.len(), 2);
                let unwrapped = match resolved.args.get(0).unwrap().expect_ty().kind() {
                    ty::Closure(closure_id, closure_args) => {
                        Instance::new(*closure_id, *closure_args)
                    }
                    _ => bug!("expect closure"),
                };

                // handle the actual callee
                let inst = PaflDump::summarize_instance(
                    self.tcx,
                    self.param_env,
                    unwrapped,
                    self.verbose,
                    &self.path_meta,
                    &self.path_data,
                    self.stack,
                    self.cache,
                    self.summary,
                );
                CallSite { inst, kind: CallKind::Direct }
            }
            InstanceDef::FnPtrShim(shim_id, _) => {
                if self.verbose {
                    println!(" ~> indirect");
                }

                // extract the actual callee
                let body = self.tcx.instance_mir(resolved.def).clone();
                let instantiated = resolved.instantiate_mir_and_normalize_erasing_regions(
                    self.tcx,
                    self.param_env,
                    EarlyBinder::bind(body),
                );

                let shim_crate = self.tcx.crate_name(shim_id.krate).to_string();
                let shim_path = self.tcx.def_path(shim_id).to_string_no_crate_verbose();

                let fn_ty = match shim_crate.as_str() {
                    "core" => match shim_path.as_str() {
                        "::ops::function::FnOnce::call_once" => {
                            let args: Vec<_> = instantiated.args_iter().collect();
                            assert_eq!(args.len(), 2);

                            let arg0 = *args.get(0).unwrap();
                            instantiated.local_decls.get(arg0).unwrap().ty
                        }
                        "::ops::function::Fn::call" => {
                            let args: Vec<_> = instantiated.args_iter().collect();
                            assert_eq!(args.len(), 2);

                            let arg0 = *args.get(0).unwrap();
                            match instantiated.local_decls.get(arg0).unwrap().ty.kind() {
                                ty::Ref(_, t, Mutability::Not) => *t,
                                _ => bug!("invalid argument type for call"),
                            }
                        }
                        "::ops::function::FnMut::call_mut" => {
                            let args: Vec<_> = instantiated.args_iter().collect();
                            assert_eq!(args.len(), 2);

                            let arg0 = *args.get(0).unwrap();
                            match instantiated.local_decls.get(arg0).unwrap().ty.kind() {
                                ty::Ref(_, t, Mutability::Mut) => *t,
                                _ => bug!("invalid argument type for call_mut"),
                            }
                        }
                        _ => bug!("unrecognized fn ptr shim: {}{}", shim_crate, shim_path),
                    },
                    _ => bug!("unrecognized fn ptr shim: {}{}", shim_crate, shim_path),
                };

                let unwrapped = match fn_ty.kind() {
                    ty::Closure(fn_def_id, fn_generic_args)
                    | ty::FnDef(fn_def_id, fn_generic_args) => {
                        Instance::new(*fn_def_id, *fn_generic_args)
                    }
                    _ => bug!(
                        "{}{} into neither a function nor closure: {:?}",
                        shim_crate,
                        shim_path,
                        span
                    ),
                };

                // handle the actual callee
                let inst = PaflDump::summarize_instance(
                    self.tcx,
                    self.param_env,
                    unwrapped,
                    self.verbose,
                    &self.path_meta,
                    &self.path_data,
                    self.stack,
                    self.cache,
                    self.summary,
                );
                CallSite { inst, kind: CallKind::Direct }
            }
            InstanceDef::DropGlue(_, _) | InstanceDef::CloneShim(_, _) => {
                if self.verbose {
                    println!(" ~> bridge");
                }
                let inst = PaflDump::summarize_instance(
                    self.tcx,
                    self.param_env,
                    resolved,
                    self.verbose,
                    &self.path_meta,
                    &self.path_data,
                    self.stack,
                    self.cache,
                    self.summary,
                );
                CallSite { inst, kind: CallKind::Bridge }
            }
            InstanceDef::Virtual(virtual_id, offset) => {
                if self.verbose {
                    println!(" ~> virtual#{}", offset);
                }
                let inst = self.resolve_fn_key(virtual_id, resolved.args);
                CallSite { inst, kind: CallKind::Virtual(offset) }
            }
            InstanceDef::Intrinsic(intrinsic_id) => {
                if self.verbose {
                    println!(" ~> intrinsic");
                }
                let inst: FnInstKey = self.resolve_fn_key(intrinsic_id, resolved.args);
                CallSite { inst, kind: CallKind::Intrinsic }
            }
            InstanceDef::VTableShim(..)
            | InstanceDef::ReifyShim(..)
            | InstanceDef::FnPtrAddrShim(..)
            | InstanceDef::ThreadLocalShim(..) => {
                bug!("unusual calls are not supported yet: {}", resolved);
            }
        };

        // done with the resolution
        call_site
    }

    /// Process the mir for one basic block
    fn process_block(&mut self, id: BasicBlock, data: &BasicBlockData<'tcx>) -> PaflBlock {
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
                site: self.process_callsite(func, term.source_info.span),
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
    fn process_cfg(&mut self, id: DefId, body: &Body<'tcx>) -> PaflCFG {
        let path = self.tcx.def_path(id).to_string_no_crate_verbose();

        // dump the control flow graph if requested
        match std::env::var_os("PAFL_CFG") {
            None => (),
            Some(v) => {
                if v.to_str().map_or(false, |s| s == path.as_str()) {
                    let dot_path = self.path_prefix.with_extension("dot");
                    let mut dot_file = OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&dot_path)
                        .expect("unable to create dot file");
                    write_mir_fn_graphviz(self.tcx, body, false, &mut dot_file)
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
        PaflCFG { blocks }
    }

    /// Process a codegen instance
    fn summarize_instance(
        tcx: TyCtxt<'tcx>,
        param_env: ParamEnv<'tcx>,
        instance: Instance<'tcx>,
        verbose: bool,
        path_meta: &Path,
        path_data: &Path,
        stack: &'sum mut Vec<Instance<'tcx>>,
        cache: &'sum mut FxHashMap<Instance<'tcx>, FnInstKey>,
        summary: &'sum mut Vec<PaflFunction>,
    ) -> FnInstKey {
        // check if we have seen the instance
        if let Some(cached) = cache.get(&instance) {
            return cached.clone();
        }

        let id = instance.def_id();
        let path = tcx.def_path(id);
        let depth = stack.len();

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

        // construct the worker
        let mut dumper = PaflDump {
            tcx,
            param_env,
            verbose,
            path_meta: path_meta.to_path_buf(),
            path_data: path_data.to_path_buf(),
            path_prefix,
            stack,
            cache,
            summary,
        };

        // derive the inst key and mark it in the cache
        let inst = dumper.resolve_fn_key(id, instance.args);

        // mark beginning of processing
        if verbose {
            println!(
                "{}[->] {}{}",
                "  ".repeat(depth),
                inst.krate.as_ref().map_or("@", |s| s.as_str()),
                inst.path
            );
        }

        // normalize, check consistency, and initialize
        dumper.initialize(instance);
        dumper.stack.push(instance);
        dumper.cache.insert(instance, inst.clone());

        // branch processing by instance type
        let body = match &instance.def {
            InstanceDef::Item(id) => {
                if dumper.tcx.is_mir_available(*id) {
                    let body = dumper.tcx.instance_mir(instance.def).clone();
                    let instantiated = instance.instantiate_mir_and_normalize_erasing_regions(
                        dumper.tcx,
                        dumper.param_env,
                        EarlyBinder::bind(body),
                    );
                    let cfg = dumper.process_cfg(*id, &instantiated);
                    FnBody::Defined(cfg)
                } else {
                    FnBody::Skipped
                }
            }
            InstanceDef::ClosureOnceShim { call_once: id, track_caller: _ }
            | InstanceDef::DropGlue(id, _)
            | InstanceDef::CloneShim(id, _)
            | InstanceDef::FnPtrShim(id, _) => {
                let body = dumper.tcx.instance_mir(instance.def).clone();
                let instantiated = instance.instantiate_mir_and_normalize_erasing_regions(
                    dumper.tcx,
                    dumper.param_env,
                    EarlyBinder::bind(body),
                );
                let cfg = dumper.process_cfg(*id, &instantiated);
                FnBody::Bridged(cfg)
            }
            InstanceDef::Intrinsic(..) => FnBody::Intrinsic,
            // not supported
            InstanceDef::Virtual(..)
            | InstanceDef::VTableShim(..)
            | InstanceDef::ReifyShim(..)
            | InstanceDef::FnPtrAddrShim(..)
            | InstanceDef::ThreadLocalShim(..) => {
                bug!("unexpected instance type: {}", instance);
            }
        };

        if dumper.stack.pop().map_or(true, |v| v != instance) {
            bug!("unbalanced stack");
        }
        dumper.summary.push(PaflFunction { inst: inst.clone(), body });

        // mark end of processing
        if verbose {
            println!(
                "{}[<-] {}{}",
                "  ".repeat(depth),
                inst.krate.as_ref().map_or("@", |s| s.as_str()),
                inst.path
            );
        }

        // return the instantiation key
        inst
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
    let mut cache = FxHashMap::default();
    let mut summary = PaflCrate { functions: Vec::new() };

    let (_, units) = tcx.collect_and_partition_mono_items(());
    for unit in units {
        for item in unit.items().keys() {
            // filter
            let instance = match item {
                MonoItem::Fn(i) => *i,
                MonoItem::Static(_) => continue,
                MonoItem::GlobalAsm(_) => bug!("unexpected assembly"),
            };
            if !instance.def_id().is_local() {
                continue;
            }

            // process it and save the result to summary
            let mut stack = vec![];
            PaflDump::summarize_instance(
                tcx,
                ParamEnv::reveal_all(),
                instance,
                verbose,
                &path_meta,
                &path_data,
                &mut stack,
                &mut cache,
                &mut summary.functions,
            );
            if !stack.is_empty() {
                bug!("unbalanced call stack");
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
