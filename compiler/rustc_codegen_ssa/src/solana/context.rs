use std::collections::BTreeMap;
use std::fmt::Display;

use regex::Regex;
use rustc_abi::Size;
use rustc_ast::Mutability;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::bug;
use rustc_middle::mir::interpret::{AllocId, AllocRange, GlobalAlloc, Scalar};
use rustc_middle::mir::{
    AggregateKind, BinOp, BorrowKind, CastKind, Const as OpConst, ConstValue, MirPhase,
    NonDivergingIntrinsic, NullOp, Operand, Place, PlaceElem, RawPtrKind, RuntimePhase, Rvalue,
    Statement, StatementKind, Terminator, TerminatorKind, UnOp, UnwindAction,
};
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::{
    self, AdtDef, AdtKind, Const as TyConst, ConstKind as TyConstKind, EarlyBinder,
    ExistentialPredicate, FloatTy, GenericArg, GenericArgKind, GenericArgsRef, Instance,
    InstanceKind, IntTy, Ty, TyCtxt, TypingEnv, UintTy,
};
use rustc_span::DUMMY_SP;
use rustc_span::def_id::DefId;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::solana::common::{Phase, SolEnv};

/// A builder for creating a Solana context
pub(crate) struct SolContextBuilder<'tcx> {
    /// compiler context
    tcx: TyCtxt<'tcx>,

    /// solana environment
    sol: SolEnv,

    /// All built-in functions
    builtin_funcs: BTreeMap<SolBuiltinFunc, Regex>,

    /// All built-in datatypes
    builtin_types: BTreeMap<SolBuiltinType, Regex>,

    /// depth of the current context
    depth: Depth,

    /// a cache of id to identifier mappings
    id_cache: FxHashMap<DefId, (SolIdent, SolPathDesc)>,

    /// a cache of alloc id to global slot mappings
    alloc_map: FxHashMap<AllocId, SolGlobalSlot>,

    /// collected definitions of datatypes
    ty_defs: BTreeMap<SolIdent, BTreeMap<Vec<SolGenericArg>, Option<SolTyDef>>>,

    /// collected definitions of functions
    fn_defs: BTreeMap<
        SolIdent,
        BTreeMap<Vec<SolGenericArg>, BTreeMap<SolInstanceKind, Option<SolFnDef>>>,
    >,

    /// collected definitions of global variables
    globals: BTreeMap<SolGlobalSlot, SolGlobalObject>,

    /// functions without MIRs available
    dep_fns: BTreeMap<
        SolIdent,
        BTreeMap<Vec<SolGenericArg>, BTreeMap<SolInstanceKind, SolPathDescWithArgs>>,
    >,
}

impl<'tcx> SolContextBuilder<'tcx> {
    /// Create a new context builder
    pub(crate) fn new(tcx: TyCtxt<'tcx>, sol: SolEnv) -> Self {
        Self {
            tcx,
            sol,
            builtin_funcs: SolBuiltinFunc::all()
                .into_iter()
                .map(|item| {
                    let regex = item.regex();
                    (item, regex)
                })
                .collect(),
            builtin_types: SolBuiltinType::all()
                .into_iter()
                .map(|item| {
                    let regex = item.regex();
                    (item, regex)
                })
                .collect(),
            depth: Depth::new(),
            id_cache: FxHashMap::default(),
            alloc_map: FxHashMap::default(),
            ty_defs: BTreeMap::new(),
            fn_defs: BTreeMap::new(),
            globals: BTreeMap::new(),
            dep_fns: BTreeMap::new(),
        }
    }

    /// Whether we are in the dry-run mode
    fn is_dry_run(&self) -> bool {
        matches!(self.sol.phase, Phase::Temporary)
    }

    /// Create an identifier without caching it
    pub(crate) fn mk_ident_no_cache(tcx: TyCtxt<'tcx>, def_id: DefId) -> SolIdent {
        // now construct the identifier
        let def_path_hash = tcx.def_path_hash(def_id);
        SolIdent {
            krate: SolHash64(def_path_hash.stable_crate_id().as_u64()),
            local: SolHash64(def_path_hash.local_hash().as_u64()),
        }
    }

    /// Create an identifier
    fn mk_ident(&mut self, def_id: DefId) -> SolIdent {
        // check cache first
        if let Some((ident, _)) = self.id_cache.get(&def_id) {
            return ident.clone();
        }

        // now construct the identifier
        let def_path_hash = self.tcx.def_path_hash(def_id);
        let ident = SolIdent {
            krate: SolHash64(def_path_hash.stable_crate_id().as_u64()),
            local: SolHash64(def_path_hash.local_hash().as_u64()),
        };

        // insert into the cache
        let desc = SolPathDesc(self.tcx.def_path_debug_str(def_id));
        self.id_cache.insert(def_id, (ident.clone(), desc));

        // return ident
        ident
    }

    /// Create a fully qualified instance description
    fn mk_path_desc_with_args(
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        ty_args: GenericArgsRef<'tcx>,
    ) -> SolPathDescWithArgs {
        let path_str = tcx.def_path_str_with_args(def_id, ty_args);
        let path_str_with_crate = if def_id.is_local() {
            let krate = tcx.crate_name(LOCAL_CRATE).to_ident_string();
            format!("{krate}::{path_str}")
        } else {
            path_str
        };
        SolPathDescWithArgs(path_str_with_crate)
    }

    /// Create a type
    pub(crate) fn mk_type(&mut self, ty: Ty<'tcx>) -> SolType {
        // mark start
        self.depth.push();
        info!("{}-> type {ty}", self.depth);

        // force normalize the type due to lazy normalization (this is needed for at least associated types)
        let normalized_ty =
            self.tcx.normalize_erasing_regions(TypingEnv::fully_monomorphized(), ty);
        info!("{}-- normalized to {normalized_ty}", self.depth);

        // switch by the type kind
        let result = match *normalized_ty.kind() {
            ty::Never => SolType::Never,
            ty::Bool => SolType::Bool,
            ty::Char => SolType::Char,
            ty::Int(int_ty) => SolType::Int(int_ty.into()),
            ty::Uint(uint_ty) => SolType::Int(uint_ty.into()),
            ty::Float(float_ty) => SolType::Float(float_ty.into()),
            ty::Str => SolType::Str,
            ty::Array(elem_ty, size_const) => {
                let elem = self.mk_type(elem_ty);
                let size = self.mk_ty_const(size_const);
                SolType::Array(Box::new(elem), Box::new(size))
            }
            ty::Tuple(tys) => {
                let elems = tys.iter().map(|ty| self.mk_type(ty)).collect();
                SolType::Tuple(elems)
            }
            ty::Adt(adt_def, generics) => {
                let (ident, ty_args) = self.make_type_adt(adt_def, generics);
                SolType::Adt(ident, ty_args)
            }
            ty::Slice(elem_ty) => {
                let elem = self.mk_type(elem_ty);
                SolType::Slice(Box::new(elem))
            }
            ty::Ref(_, ref_ty, mutability) => {
                let elem = self.mk_type(ref_ty);
                match mutability {
                    Mutability::Not => SolType::ImmRef(Box::new(elem)),
                    Mutability::Mut => SolType::MutRef(Box::new(elem)),
                }
            }
            ty::RawPtr(ref_ty, mutability) => {
                let elem = self.mk_type(ref_ty);
                match mutability {
                    Mutability::Not => SolType::ImmPtr(Box::new(elem)),
                    Mutability::Mut => SolType::MutPtr(Box::new(elem)),
                }
            }
            ty::FnDef(def_id, generics) => {
                let (kind, ident, ty_args) = self.make_type_function(def_id, generics);
                SolType::Function(kind, ident, ty_args)
            }
            ty::Closure(def_id, generics) => {
                let (kind, ident, ty_args) = self.make_type_closure(def_id, generics);
                SolType::Closure(kind, ident, ty_args)
            }
            ty::FnPtr(sig, _header) => {
                let sig = self.tcx.instantiate_bound_regions_with_erased(sig);
                let inputs = sig.inputs().iter().map(|ty| self.mk_type(*ty)).collect();
                let output = self.mk_type(sig.output());
                SolType::FnPtr(inputs, Box::new(output))
            }
            ty::Dynamic(predicates, _, _) => {
                let mut trait_refs = vec![];
                for pred_bounded in predicates {
                    let pred_instiated =
                        self.tcx.instantiate_bound_regions_with_erased(pred_bounded);
                    match pred_instiated {
                        ExistentialPredicate::Trait(trait_ref) => {
                            let trait_ident = self.mk_ident(trait_ref.def_id);
                            let trait_ty_args = trait_ref
                                .args
                                .iter()
                                .map(|ty_arg| self.mk_ty_arg(ty_arg))
                                .collect();
                            trait_refs.push((trait_ident, trait_ty_args));
                        }
                        ExistentialPredicate::AutoTrait(auto_trait_id) => {
                            let trait_ident = self.mk_ident(auto_trait_id);
                            trait_refs.push((trait_ident, vec![]));
                        }
                        ExistentialPredicate::Projection(..) => {
                            bug!("[unsupported] unsupported dynamic type: {ty}");
                        }
                    }
                }
                SolType::Dynamic(trait_refs)
            }
            ty::Pat(..) => {
                bug!("[assumption] unexpected pattern type: {ty}");
            }
            ty::Coroutine(..) | ty::CoroutineClosure(..) | ty::CoroutineWitness(..) => {
                bug!("[assumption] unexpected coroutine-related type: {ty}");
            }
            ty::Foreign(..) => {
                bug!("[assumption] unexpected foreign type {ty}");
            }
            ty::Alias(..) => {
                bug!("[invariant] unexpected alias type {ty}");
            }
            ty::Param(..) | ty::Bound(..) | ty::UnsafeBinder(..) => {
                bug!("[invariant] unexpected generic type: {ty}");
            }
            ty::Infer(..) | ty::Placeholder(..) | ty::Error(..) => {
                bug!("[invariant] unexpected type used in type analysis: {ty}");
            }
        };

        // mark end
        info!("{}<- type {ty}", self.depth);
        self.depth.pop();

        // return the result
        result
    }

    fn make_type_adt(
        &mut self,
        adt_def: AdtDef<'tcx>,
        generics: GenericArgsRef<'tcx>,
    ) -> (SolIdent, Vec<SolGenericArg>) {
        let def_id = adt_def.did();
        let def_desc = Self::mk_path_desc_with_args(self.tcx, def_id, generics);

        // locate the key of the definition
        let ident = self.mk_ident(adt_def.did());
        let ty_args: Vec<_> = generics.iter().map(|ty_arg| self.mk_ty_arg(ty_arg)).collect();

        // in dry-run mode, do not analyze the defs
        if self.is_dry_run() {
            return (ident, ty_args);
        }

        // check if this is a builtin datatype
        for (builtin, regex) in self.builtin_types.iter() {
            if regex.is_match(&def_desc.0) {
                info!("{}-- builtin datatype {builtin:?}: {def_desc}", self.depth);
                return (ident, ty_args);
            }
        }

        // if already defined or is being defined, return the key
        if self.ty_defs.get(&ident).map_or(false, |inner| inner.contains_key(&ty_args)) {
            return (ident, ty_args);
        }

        // mark start
        self.depth.push();
        info!("{}-> adt definition", self.depth);

        // first update the entry to mark that type definition in progress
        self.ty_defs.entry(ident.clone()).or_default().insert(ty_args.clone(), None);

        // now create the type definition
        let ty_def = match adt_def.adt_kind() {
            AdtKind::Struct => {
                if adt_def.variants().len() != 1 {
                    bug!("[invariant] struct {def_desc} has multiple variants");
                }
                let fields = adt_def
                    .non_enum_variant()
                    .fields
                    .iter_enumerated()
                    .map(|(field_idx, field_def)| {
                        let field_index = SolFieldIndex(field_idx.index());
                        let field_name = SolFieldName(field_def.name.to_ident_string());
                        let field_ty = self.mk_type(field_def.ty(self.tcx, generics));
                        SolField { index: field_index, name: field_name, ty: field_ty }
                    })
                    .collect();
                SolTyDef::Struct { fields }
            }
            AdtKind::Union => {
                if adt_def.variants().len() != 1 {
                    bug!("[invariant] union {def_desc} has multiple variants");
                }
                let fields = adt_def
                    .non_enum_variant()
                    .fields
                    .iter_enumerated()
                    .map(|(field_idx, field_def)| {
                        let field_index = SolFieldIndex(field_idx.index());
                        let field_name = SolFieldName(field_def.name.to_ident_string());
                        let field_ty = self.mk_type(field_def.ty(self.tcx, generics));
                        SolField { index: field_index, name: field_name, ty: field_ty }
                    })
                    .collect();
                SolTyDef::Union { fields }
            }
            AdtKind::Enum => {
                let variants = adt_def
                    .variants()
                    .iter_enumerated()
                    .map(|(variant_idx, variant_def)| {
                        let variant_index = SolVariantIndex(variant_idx.index());
                        let variant_name = SolVariantName(variant_def.name.to_ident_string());
                        let fields = variant_def
                            .fields
                            .iter_enumerated()
                            .map(|(field_idx, field_def)| {
                                let field_index = SolFieldIndex(field_idx.index());
                                let field_name = SolFieldName(field_def.name.to_ident_string());
                                let field_ty = self.mk_type(field_def.ty(self.tcx, generics));
                                SolField { index: field_index, name: field_name, ty: field_ty }
                            })
                            .collect();
                        SolVariant { index: variant_index, name: variant_name, fields }
                    })
                    .collect();
                SolTyDef::Enum { variants }
            }
        };

        // update the type definition
        self.ty_defs.entry(ident.clone()).or_default().insert(ty_args.clone(), Some(ty_def));

        // mark end
        info!("{}<- adt definition", self.depth);
        self.depth.pop();

        // return the key to the lookup table
        (ident, ty_args)
    }

    fn make_type_function(
        &mut self,
        def_id: DefId,
        generics: GenericArgsRef<'tcx>,
    ) -> (SolInstanceKind, SolIdent, Vec<SolGenericArg>) {
        // resolve the function instance
        let instance = match Instance::try_resolve(
            self.tcx,
            TypingEnv::fully_monomorphized(),
            def_id,
            generics,
        ) {
            Ok(Some(resolved)) => resolved,
            Ok(None) => {
                bug!(
                    "[assumption] unresolved function: {}",
                    self.tcx.def_path_str_with_args(def_id, generics)
                );
            }
            Err(_) => {
                bug!(
                    "[invariant] failed to resolve function {}",
                    self.tcx.def_path_str_with_args(def_id, generics)
                );
            }
        };

        // return the key to the lookup table
        self.make_instance(instance)
    }

    fn make_type_closure(
        &mut self,
        def_id: DefId,
        generics: GenericArgsRef<'tcx>,
    ) -> (SolInstanceKind, SolIdent, Vec<SolGenericArg>) {
        // resolve the closure instance
        let closure_ty_args = generics.as_closure();
        let instance =
            Instance::resolve_closure(self.tcx, def_id, generics, closure_ty_args.kind());

        // return the key to the lookup table
        self.make_instance(instance)
    }

    /// Create a generic argument
    pub(crate) fn mk_ty_arg(&mut self, ty_arg: GenericArg<'tcx>) -> SolGenericArg {
        match ty_arg.kind() {
            GenericArgKind::Type(ty) => SolGenericArg::Type(self.mk_type(ty)),
            GenericArgKind::Const(ty_const) => SolGenericArg::Const(self.mk_ty_const(ty_const)),
            GenericArgKind::Lifetime(_) => SolGenericArg::Lifetime,
        }
    }

    /// Create a constant known in the type system
    fn mk_ty_const(&mut self, ty_const: TyConst<'tcx>) -> SolTyConst {
        match ty_const.kind() {
            TyConstKind::Value(val) => {
                let const_ty = self.mk_type(val.ty);
                let const_val = if val.valtree.is_zst() {
                    SolConst::ZeroSized
                } else {
                    match val.valtree.try_to_scalar() {
                        None => bug!("[unsupported] non-scalar type constant {ty_const:?}"),
                        Some(Scalar::Int(scalar)) => SolConst::Scalar(SolScalar {
                            bits: scalar.size().bits_usize(),
                            value: scalar.to_bits_unchecked(),
                        }),
                        Some(Scalar::Ptr(..)) => {
                            bug!("[invariant] unexpected pointer type constant {ty_const:?}");
                        }
                    }
                };
                SolTyConst::Simple { ty: const_ty, val: const_val }
            }
            TyConstKind::Param(..)
            | TyConstKind::Infer(..)
            | TyConstKind::Bound(..)
            | TyConstKind::Placeholder(..)
            | TyConstKind::Unevaluated(..) => {
                bug!("[assumption] uninstantiated type constant {ty_const:?}");
            }
            TyConstKind::Expr(expr) => {
                bug!("[unsupported] expression type constant {expr:?}");
            }
            TyConstKind::Error(_) => {
                bug!("[invariant] error type constant");
            }
        }
    }

    /// Create a constant known in the value system
    fn mk_val_const(&mut self, val_const: ConstValue) -> SolConst {
        match val_const {
            ConstValue::Scalar(Scalar::Int(scalar)) => SolConst::Scalar(SolScalar {
                bits: scalar.size().bits_usize(),
                value: scalar.to_bits_unchecked(),
            }),
            ConstValue::Scalar(Scalar::Ptr(ptr, _ /* pointer size */)) => {
                let (prov, offset) = ptr.prov_and_relative_offset();
                SolConst::Pointer {
                    origin: self.make_global(prov.alloc_id()),
                    offset: SolOffset(offset.bytes_usize()),
                }
            }
            ConstValue::ZeroSized => SolConst::ZeroSized,
            ConstValue::Slice { alloc_id, meta } => {
                let origin = self.make_global(alloc_id);
                let length = meta as usize;
                SolConst::Slice { origin, length }
            }
            ConstValue::Indirect { alloc_id, offset } => {
                let origin = self.make_global(alloc_id);
                let offset = SolOffset(offset.bytes_usize());
                SolConst::Indirect { origin, offset }
            }
        }
    }

    /// Create a constant known in the operation system
    fn mk_op_const(&mut self, op_const: OpConst<'tcx>) -> SolOpConst {
        match op_const {
            OpConst::Ty(ty, ty_const) => {
                let const_ty = self.mk_type(ty);
                let ty_const_val = self.mk_ty_const(ty_const);
                let const_val = match ty_const_val {
                    SolTyConst::Simple { ty: const_ty_inner, val: const_val } => {
                        if const_ty != const_ty_inner {
                            bug!(
                                "[invariant] type constant does not match its value type: {op_const}"
                            );
                        }
                        const_val
                    }
                };
                SolOpConst::Type(const_ty, const_val)
            }
            OpConst::Val(val_const, ty) => {
                let const_val = self.mk_val_const(val_const);
                let const_ty = self.mk_type(ty);
                SolOpConst::Value(const_ty, const_val)
            }
            OpConst::Unevaluated(unevaluated, ty) => {
                let val_const = self
                    .tcx
                    .const_eval_resolve(TypingEnv::fully_monomorphized(), unevaluated, DUMMY_SP)
                    .unwrap_or_else(|e| {
                        bug!("[invariant] unable to resolve unevaluated constant {op_const}: {e:?}")
                    });
                let const_val = self.mk_val_const(val_const);
                let const_ty = self.mk_type(ty);
                SolOpConst::Value(const_ty, const_val)
            }
        }
    }

    /// Create a projection operation
    fn mk_projection(&mut self, proj: PlaceElem<'tcx>) -> SolProjection {
        match proj {
            PlaceElem::Deref => SolProjection::Deref,
            PlaceElem::Field(field, ty) => {
                let field_index = SolFieldIndex(field.index());
                let field_ty = self.mk_type(ty);
                SolProjection::Field { field: field_index, ty: field_ty }
            }
            PlaceElem::Index(local) => {
                let local_slot = SolLocalSlot(local.index());
                SolProjection::VariableIndex { local: local_slot }
            }
            PlaceElem::ConstantIndex { offset, min_length, from_end } => {
                SolProjection::ConstantIndex {
                    offset: offset as usize,
                    min_length: min_length as usize,
                    from_end,
                }
            }
            PlaceElem::Subslice { from, to, from_end } => {
                SolProjection::Subslice { from: from as usize, to: to as usize, from_end }
            }
            PlaceElem::Downcast(_, variant) => {
                let variant_index = SolVariantIndex(variant.index());
                SolProjection::Downcast { variant: variant_index }
            }
            PlaceElem::Subtype(ty) => SolProjection::Subtype(self.mk_type(ty)),
            PlaceElem::OpaqueCast(ty) => {
                bug!("[invariant] unexpected opaque cast in place conversion: {ty}");
            }
            PlaceElem::UnwrapUnsafeBinder(ty) => {
                bug!("[invariant] unexpected unwrap unsafe binder in place conversion: {ty}");
            }
        }
    }

    /// Create a place
    fn mk_place(&mut self, place: Place<'tcx>) -> SolPlace {
        let local_slot = SolLocalSlot(place.local.index());
        let projection = place.projection.iter().map(|proj| self.mk_projection(proj)).collect();
        SolPlace { local: local_slot, projection }
    }

    /// Create an operand
    fn mk_operand(&mut self, operand: &Operand<'tcx>) -> SolOperand {
        match operand {
            Operand::Copy(place) => SolOperand::Copy(self.mk_place(*place)),
            Operand::Move(place) => SolOperand::Move(self.mk_place(*place)),
            Operand::Constant(op_const) => SolOperand::Constant(self.mk_op_const(op_const.const_)),
        }
    }

    /// Create an expression
    fn mk_expr(&mut self, rvalue: &Rvalue<'tcx>) -> SolExpr {
        match rvalue {
            Rvalue::Use(operand) => SolExpr::Use(self.mk_operand(operand)),
            Rvalue::Repeat(operand, count_const) => {
                let elem = self.mk_operand(operand);
                let count = self.mk_ty_const(*count_const);
                SolExpr::Repeat(elem, count)
            }
            Rvalue::Ref(_, borrow_kind, place) => {
                let converted_place = self.mk_place(*place);
                match borrow_kind {
                    BorrowKind::Shared => SolExpr::BorrowImm(converted_place),
                    BorrowKind::Mut { .. } => SolExpr::BorrowMut(converted_place),
                    BorrowKind::Fake(..) => bug!("[invariant] fake borrow not expected"),
                }
            }
            Rvalue::RawPtr(raw_ptr_kind, place) => {
                let converted_place = self.mk_place(*place);
                match raw_ptr_kind {
                    RawPtrKind::Const => SolExpr::PointerImm(converted_place),
                    RawPtrKind::Mut => SolExpr::PointerMut(converted_place),
                    RawPtrKind::FakeForPtrMetadata => bug!("[invariant] fake raw ptr not expected"),
                }
            }
            Rvalue::Len(place) => SolExpr::Length(self.mk_place(*place)),
            Rvalue::Cast(cast_kind, operand, ty) => {
                let opcode = match cast_kind {
                    CastKind::IntToInt => SolOpcodeCast::IntToInt,
                    CastKind::FloatToFloat => SolOpcodeCast::FloatToFloat,
                    CastKind::IntToFloat => SolOpcodeCast::IntToFloat,
                    CastKind::FloatToInt => SolOpcodeCast::FloatToInt,
                    CastKind::Transmute | CastKind::PtrToPtr | CastKind::FnPtrToPtr => {
                        SolOpcodeCast::Reinterpret
                    }
                    CastKind::PointerCoercion(coercion_type, _) => match coercion_type {
                        PointerCoercion::MutToConstPointer
                        | PointerCoercion::ArrayToPointer
                        | PointerCoercion::UnsafeFnPointer
                        | PointerCoercion::Unsize => SolOpcodeCast::Reinterpret,
                        PointerCoercion::ReifyFnPointer | PointerCoercion::ClosureFnPointer(_) => {
                            SolOpcodeCast::ReifyFnPtr
                        }
                    },
                    CastKind::PointerExposeProvenance => SolOpcodeCast::PtrToAddr,
                    CastKind::PointerWithExposedProvenance => SolOpcodeCast::AddrToPtr,
                };
                SolExpr::Cast { opcode, place: self.mk_operand(operand), ty: self.mk_type(*ty) }
            }
            Rvalue::ShallowInitBox(operand, ty) => SolExpr::Cast {
                opcode: SolOpcodeCast::Reinterpret,
                place: self.mk_operand(operand),
                ty: self.mk_type(*ty),
            },
            Rvalue::NullaryOp(op, ty) => {
                let opcode = match op {
                    NullOp::SizeOf => SolOpcodeNullary::SizeOf,
                    NullOp::AlignOf => SolOpcodeNullary::AlignOf,
                    NullOp::OffsetOf(indices) => SolOpcodeNullary::OffsetOf(
                        indices
                            .iter()
                            .map(|(variant_idx, field_idx)| {
                                (
                                    SolVariantIndex(variant_idx.index()),
                                    SolFieldIndex(field_idx.index()),
                                )
                            })
                            .collect(),
                    ),
                    NullOp::UbChecks => SolOpcodeNullary::UbCheckEnabled,
                    NullOp::ContractChecks => SolOpcodeNullary::ContractCheckEnabled,
                };
                SolExpr::OpNullary { opcode, ty: self.mk_type(*ty) }
            }
            Rvalue::UnaryOp(op, operand) => {
                let opcode = match op {
                    UnOp::Not => SolOpcodeUnary::Not,
                    UnOp::Neg => SolOpcodeUnary::Neg,
                    UnOp::PtrMetadata => SolOpcodeUnary::PtrMetadata,
                };
                SolExpr::OpUnary { opcode, val: self.mk_operand(operand) }
            }
            Rvalue::BinaryOp(op, operand_pair) => {
                let (v1, v2) = operand_pair.as_ref();
                let opcode = match op {
                    // arithmetic
                    BinOp::Add | BinOp::AddUnchecked => SolOpcodeBinary::Add(false),
                    BinOp::Sub | BinOp::SubUnchecked => SolOpcodeBinary::Sub(false),
                    BinOp::Mul | BinOp::MulUnchecked => SolOpcodeBinary::Mul(false),
                    BinOp::Div => SolOpcodeBinary::Div(false),
                    BinOp::Rem => SolOpcodeBinary::Rem(false),
                    BinOp::AddWithOverflow => SolOpcodeBinary::Add(true),
                    BinOp::SubWithOverflow => SolOpcodeBinary::Sub(true),
                    BinOp::MulWithOverflow => SolOpcodeBinary::Mul(true),
                    // bitwise
                    BinOp::BitAnd => SolOpcodeBinary::BitAnd,
                    BinOp::BitOr => SolOpcodeBinary::BitOr,
                    BinOp::BitXor => SolOpcodeBinary::BitXor,
                    // shift
                    BinOp::Shl | BinOp::ShlUnchecked => SolOpcodeBinary::Shl,
                    BinOp::Shr | BinOp::ShrUnchecked => SolOpcodeBinary::Shr,
                    // comparison
                    BinOp::Eq => SolOpcodeBinary::Eq,
                    BinOp::Ne => SolOpcodeBinary::Ne,
                    BinOp::Lt => SolOpcodeBinary::Lt,
                    BinOp::Le => SolOpcodeBinary::Le,
                    BinOp::Gt => SolOpcodeBinary::Gt,
                    BinOp::Ge => SolOpcodeBinary::Ge,
                    BinOp::Cmp => SolOpcodeBinary::Cmp,
                    // pointer
                    BinOp::Offset => SolOpcodeBinary::Offset,
                };
                SolExpr::OpBinary { opcode, v1: self.mk_operand(v1), v2: self.mk_operand(v2) }
            }
            Rvalue::Discriminant(place) => SolExpr::Discriminant(self.mk_place(*place)),
            Rvalue::Aggregate(kind, operands) => {
                let opcode = match kind.as_ref() {
                    AggregateKind::Tuple => SolOpcodeAggregate::Tuple,
                    AggregateKind::Array(ty) => SolOpcodeAggregate::Array(self.mk_type(*ty)),
                    AggregateKind::Adt(def_id, variant_idx, generics, _, union_field_idx) => {
                        let adt_def = self.tcx.adt_def(def_id);
                        let (ident, ty_args) = self.make_type_adt(adt_def, generics);
                        match adt_def.adt_kind() {
                            AdtKind::Struct => {
                                if variant_idx.index() != 0 {
                                    bug!(
                                        "[invariant] unexpected non-zero variant index for struct pack: {}",
                                        variant_idx.index()
                                    );
                                }
                                if union_field_idx.is_some() {
                                    bug!(
                                        "[invariant] unexpected union field index for struct pack"
                                    );
                                }
                                SolOpcodeAggregate::Struct { ty: SolType::Adt(ident, ty_args) }
                            }
                            AdtKind::Union => {
                                if variant_idx.index() != 0 {
                                    bug!(
                                        "[invariant] unexpected non-zero variant index for union pack: {}",
                                        variant_idx.index()
                                    );
                                }
                                match union_field_idx {
                                    None => {
                                        bug!(
                                            "[invariant] unexpected missing union field index for union pack"
                                        );
                                    }
                                    Some(field_idx) => SolOpcodeAggregate::Union {
                                        ty: SolType::Adt(ident, ty_args),
                                        field: SolFieldIndex(field_idx.index()),
                                    },
                                }
                            }
                            ty::AdtKind::Enum => {
                                if union_field_idx.is_some() {
                                    bug!(
                                        "[invariant] unexpected union field index for struct pack"
                                    );
                                }
                                SolOpcodeAggregate::Enum {
                                    ty: SolType::Adt(ident, ty_args),
                                    variant: SolVariantIndex(variant_idx.index()),
                                }
                            }
                        }
                    }
                    AggregateKind::RawPtr(ty, mutability) => {
                        let pointee_ty = self.mk_type(*ty);
                        match mutability {
                            Mutability::Not => SolOpcodeAggregate::ImmPtr { pointee_ty },
                            Mutability::Mut => SolOpcodeAggregate::MutPtr { pointee_ty },
                        }
                    }
                    AggregateKind::Closure(def_id, generics) => {
                        let (kind, ident, ty_args) = self.make_type_closure(*def_id, *generics);
                        SolOpcodeAggregate::Closure(kind, ident, ty_args)
                    }
                    AggregateKind::Coroutine(..) | AggregateKind::CoroutineClosure(..) => {
                        bug!("[invariant] unexpected coroutine in aggregate conversion");
                    }
                };
                let vals = operands
                    .iter_enumerated()
                    .map(|(field_idx, operand)| {
                        (SolFieldIndex(field_idx.index()), self.mk_operand(operand))
                    })
                    .collect();
                SolExpr::Aggregate { opcode, vals }
            }
            Rvalue::CopyForDeref(place) => SolExpr::Load(self.mk_place(*place)),
            Rvalue::ThreadLocalRef(..) => {
                bug!("[assumption] unexpected thread-local reference in rvalue conversion");
            }
            Rvalue::WrapUnsafeBinder(..) => {
                bug!("[invariant] unexpected wrap unsafe binder in rvalue conversion");
            }
        }
    }

    /// Create an unwind action
    fn mk_unwind_action(&mut self, action: UnwindAction) -> SolUnwindAction {
        match action {
            UnwindAction::Continue => SolUnwindAction::Continue,
            UnwindAction::Unreachable => SolUnwindAction::Unreachable,
            UnwindAction::Terminate(_) => SolUnwindAction::Terminate,
            UnwindAction::Cleanup(block) => SolUnwindAction::Cleanup(SolBlockId(block.index())),
        }
    }

    /// Create a statement
    fn mk_statement(&mut self, stmt: &Statement<'tcx>) -> SolStatement {
        // mark start
        self.depth.push();
        info!("{}-> statement {}", self.depth, stmt.kind.name());

        let converted = match &stmt.kind {
            // assign
            StatementKind::Assign(assign) => {
                let (place, rvalue) = assign.as_ref();
                SolStatement::Assign { lhs: self.mk_place(*place), rhs: self.mk_expr(rvalue) }
            }
            StatementKind::SetDiscriminant { place, variant_index } => {
                SolStatement::SetDiscriminant {
                    place: self.mk_place(**place),
                    variant: SolVariantIndex(variant_index.index()),
                }
            }
            // storage
            StatementKind::Deinit(place) => SolStatement::Deinit(self.mk_place(**place)),
            StatementKind::StorageLive(local_idx) => {
                SolStatement::StorageLive(SolLocalSlot(local_idx.index()))
            }
            StatementKind::StorageDead(local_idx) => {
                SolStatement::StorageDead(SolLocalSlot(local_idx.index()))
            }
            StatementKind::PlaceMention(place) => SolStatement::Deinit(self.mk_place(**place)),
            // intrinsic
            StatementKind::Intrinsic(intrinsic) => match intrinsic.as_ref() {
                NonDivergingIntrinsic::Assume(operand) => {
                    SolStatement::Assume(self.mk_operand(operand))
                }
                NonDivergingIntrinsic::CopyNonOverlapping(details) => {
                    SolStatement::CopyNonoverlapping {
                        src: self.mk_operand(&details.src),
                        dst: self.mk_operand(&details.dst),
                        count: self.mk_operand(&details.count),
                    }
                }
            },
            // no-op
            StatementKind::Nop
            | StatementKind::ConstEvalCounter
            | StatementKind::FakeRead(..)
            | StatementKind::AscribeUserType(..)
            | StatementKind::BackwardIncompatibleDropHint { .. } => SolStatement::Nop,
            // should not appear
            StatementKind::Retag(..) => {
                bug!("[invariant] unexpected retag statement {stmt:?}");
            }
            StatementKind::Coverage(..) => {
                bug!("[invariant] unexpected coverage statement {stmt:?}");
            }
        };

        // mark end
        info!("{}<- statement {}", self.depth, stmt.kind.name());
        self.depth.pop();

        // return the converted statement
        converted
    }

    /// Create a terminator
    fn mk_terminator(&mut self, term: &Terminator<'tcx>) -> SolTerminator {
        // mark start
        self.depth.push();
        info!("{}-> terminator {}", self.depth, term.kind.name());

        let converted = match &term.kind {
            TerminatorKind::Unreachable => SolTerminator::Unreachable,
            TerminatorKind::Return => SolTerminator::Return,
            TerminatorKind::Goto { target } => {
                SolTerminator::Goto { target: SolBlockId(target.index()) }
            }
            TerminatorKind::SwitchInt { discr, targets } => {
                let mut target_pairs = vec![];
                for &value in targets.all_values() {
                    target_pairs.push((
                        value.get(),
                        SolBlockId(targets.target_for_value(value.get()).index()),
                    ));
                }
                SolTerminator::Switch {
                    cond: self.mk_operand(discr),
                    targets: target_pairs,
                    otherwise: SolBlockId(targets.otherwise().index()),
                }
            }
            TerminatorKind::Drop { place, target, unwind, replace: _, drop, async_fut } => {
                if drop.is_some() || async_fut.is_some() {
                    bug!("[invariant] unexpected async features in drop terminator");
                }
                SolTerminator::Drop {
                    place: self.mk_place(*place),
                    target: SolBlockId(target.index()),
                    unwind: self.mk_unwind_action(*unwind),
                }
            }
            TerminatorKind::Assert { cond, expected, msg: _, target, unwind } => {
                SolTerminator::Assert {
                    cond: self.mk_operand(cond),
                    expected: *expected,
                    target: SolBlockId(target.index()),
                    unwind: self.mk_unwind_action(*unwind),
                }
            }
            TerminatorKind::Call {
                func,
                args,
                destination,
                target,
                unwind,
                call_source: _,
                fn_span: _,
            } => {
                let converted_func = self.mk_operand(func);
                let converted_args = args.iter().map(|arg| self.mk_operand(&arg.node)).collect();
                let converted_dest = self.mk_place(*destination);
                let converted_target = target.map(|block| SolBlockId(block.index()));
                SolTerminator::Call {
                    func: converted_func,
                    args: converted_args,
                    dest: converted_dest,
                    target: converted_target,
                    unwind: self.mk_unwind_action(*unwind),
                }
            }
            TerminatorKind::TailCall { func, args, fn_span: _ } => {
                let converted_func = self.mk_operand(func);
                let converted_args = args.iter().map(|arg| self.mk_operand(&arg.node)).collect();
                SolTerminator::TailCall { func: converted_func, args: converted_args }
            }
            TerminatorKind::UnwindResume => SolTerminator::UnwindResume,
            TerminatorKind::UnwindTerminate(_) => SolTerminator::UnwindTerminate,
            TerminatorKind::InlineAsm { .. } => {
                bug!("[unsupported] inline assembly {term:?}");
            }
            TerminatorKind::Yield { .. } | TerminatorKind::CoroutineDrop => {
                bug!("[assumption] unexpected async-related terminator {term:?}");
            }
            TerminatorKind::FalseEdge { .. } | TerminatorKind::FalseUnwind { .. } => {
                bug!("[invariant] unexpected false terminator {term:?}");
            }
        };

        // mark end
        info!("{}<- terminator {}", self.depth, term.kind.name());
        self.depth.pop();

        // return the converted statement
        converted
    }

    /// Create a function definition
    pub(crate) fn make_instance(
        &mut self,
        instance: Instance<'tcx>,
    ) -> (SolInstanceKind, SolIdent, Vec<SolGenericArg>) {
        let def_id = instance.def_id();
        let def_desc = Self::mk_path_desc_with_args(self.tcx, def_id, instance.args);

        // locate the key of the definition
        let ident = self.mk_ident(def_id);
        let ty_args: Vec<_> = instance.args.iter().map(|ty_arg| self.mk_ty_arg(ty_arg)).collect();

        // derive the kind of the instance
        let kind = match instance.def {
            InstanceKind::Item(_) => {
                // check if this is a builtin
                for (builtin, regex) in self.builtin_funcs.iter() {
                    if regex.is_match(&def_desc.0) {
                        info!("{}-- builtin {builtin:?}: {def_desc}", self.depth);
                        return (SolInstanceKind::Builtin(builtin.clone()), ident, ty_args);
                    }
                }

                // not a builtin, continue processing
                SolInstanceKind::Regular
            }

            // drop operation
            InstanceKind::DropGlue(_, None) => SolInstanceKind::DropEmpty,
            InstanceKind::DropGlue(_, Some(drop_ty)) => {
                SolInstanceKind::DropGlued(Box::new(self.mk_type(drop_ty)))
            }

            // shims
            InstanceKind::VTableShim(_) => SolInstanceKind::VTableShim,
            InstanceKind::ReifyShim(_, _) => SolInstanceKind::ReifyShim,
            InstanceKind::FnPtrShim(_, fn_ty) => {
                SolInstanceKind::FnPtrShim(Box::new(self.mk_type(fn_ty)))
            }
            InstanceKind::FnPtrAddrShim(_, fn_ty) => {
                SolInstanceKind::FnPtrAddrShim(Box::new(self.mk_type(fn_ty)))
            }
            InstanceKind::CloneShim(_, clone_ty) => {
                SolInstanceKind::CloneShim(Box::new(self.mk_type(clone_ty)))
            }
            InstanceKind::ClosureOnceShim { call_once: _, track_caller: _ } => {
                SolInstanceKind::ClosureOnceShim
            }

            // instance that does not have a MIR body
            InstanceKind::Intrinsic(_) => {
                info!("{}-- intrinsic: {def_desc}", self.depth);
                return (SolInstanceKind::Intrinsic, ident, ty_args);
            }
            InstanceKind::Virtual(_, idx) => {
                info!("{}-- virtual: {def_desc}", self.depth);
                return (SolInstanceKind::Virtual(idx), ident, ty_args);
            }

            // unexpected instance kind
            InstanceKind::ConstructCoroutineInClosureShim { .. }
            | InstanceKind::ThreadLocalShim(..)
            | InstanceKind::FutureDropPollShim(..)
            | InstanceKind::AsyncDropGlueCtorShim(..)
            | InstanceKind::AsyncDropGlue(..) => {
                bug!("[invariant] unexpected instance kind: {def_desc}");
            }
        };

        // skip if we are in dry-run mode
        if self.is_dry_run() {
            return (kind, ident, ty_args);
        }

        // if already defined or is being defined, return the key
        if self
            .fn_defs
            .get(&ident)
            .and_then(|inner| inner.get(&ty_args))
            .map_or(false, |inner| inner.contains_key(&kind))
        {
            return (kind, ident, ty_args);
        }

        // convert the instance to monomorphised MIR
        let instance_mir = match kind {
            SolInstanceKind::Regular => {
                if !self.tcx.is_mir_available(def_id) {
                    info!("{}-- external dependency {def_desc} of kind {kind:#?}", self.depth);
                    let deps_inner = self
                        .dep_fns
                        .entry(ident.clone())
                        .or_default()
                        .entry(ty_args.clone())
                        .or_default();
                    if !deps_inner.contains_key(&kind) {
                        deps_inner.insert(
                            kind.clone(),
                            Self::mk_path_desc_with_args(self.tcx, def_id, instance.args),
                        );
                    }
                    return (kind, ident, ty_args);
                }

                // this is a regular function definition
                let body = self.tcx.instance_mir(instance.def).clone();
                if body.phase != MirPhase::Runtime(RuntimePhase::Optimized) {
                    bug!("converted instance is not runtime optimized: {def_desc}");
                }
                body
            }

            SolInstanceKind::DropEmpty
            | SolInstanceKind::DropGlued(_)
            | SolInstanceKind::VTableShim
            | SolInstanceKind::ReifyShim
            | SolInstanceKind::FnPtrShim(_)
            | SolInstanceKind::FnPtrAddrShim(_)
            | SolInstanceKind::CloneShim(_)
            | SolInstanceKind::ClosureOnceShim => self.tcx.mir_shims(instance.def).clone(),

            // already handled before
            SolInstanceKind::Builtin(_)
            | SolInstanceKind::Intrinsic
            | SolInstanceKind::Virtual(_) => {
                bug!("[invariant] unexpected instance kind");
            }
        };

        // mark start
        self.depth.push();
        warn!("{}-> function {def_desc}", self.depth);

        // first update the entry to mark that instance definition in progress
        self.fn_defs
            .entry(ident.clone())
            .or_default()
            .entry(ty_args.clone())
            .or_default()
            .insert(kind.clone(), None);

        // NOTE: lazy normalization seems to be in effect here, at least associated types
        // are not easily resolved after this transformation.
        let body = instance.instantiate_mir_and_normalize_erasing_regions(
            self.tcx,
            instance_mir.typing_env(self.tcx),
            EarlyBinder::bind(instance_mir),
        );

        // dump the instance info
        self.sol.save_instance_info(self.tcx, &body);

        // convert function signatures
        let mut args = vec![];
        for arg_idx in body.args_iter() {
            let decl = &body.local_decls[arg_idx];
            args.push(self.mk_type(decl.ty));
        }
        let ret_ty = self.mk_type(body.return_ty());

        // convert function body
        let mut locals = vec![];
        for local_idx in body.vars_and_temps_iter() {
            let decl = &body.local_decls[local_idx];
            locals.push(self.mk_type(decl.ty));
        }

        let mut blocks = vec![];
        for (block_id, block_data) in body.basic_blocks.iter_enumerated() {
            info!("{}-- basic block {}", self.depth, block_id.index());

            // block data
            let statements =
                block_data.statements.iter().map(|stmt| self.mk_statement(stmt)).collect();
            let terminator = self.mk_terminator(block_data.terminator());

            // add to block list
            let block = SolBasicBlock { id: SolBlockId(block_id.index()), statements, terminator };
            blocks.push(block);
        }

        // update the function definition lookup table
        let fn_def = SolFnDef { args, ret_ty, locals, blocks };
        self.fn_defs
            .entry(ident.clone())
            .or_default()
            .entry(ty_args.clone())
            .or_default()
            .insert(kind.clone(), Some(fn_def));

        // mark end
        warn!("{}<- function {def_desc}", self.depth);
        self.depth.pop();

        // return the key to the lookup table
        (kind, ident, ty_args)
    }

    /// Create a global variable definition
    fn make_global(&mut self, alloc_id: AllocId) -> SolGlobalSlot {
        // if already defined or is being defined, return the key
        if let Some(slot) = self.alloc_map.get(&alloc_id) {
            return slot.clone();
        }

        // now analyze the global variable
        let global = match self.tcx.global_alloc(alloc_id) {
            GlobalAlloc::Function { instance } => {
                let (kind, ident, ty_args) = self.make_instance(instance);
                SolGlobalObject::Function(kind, ident, ty_args)
            }
            GlobalAlloc::Memory(memory) => {
                let memory = memory.inner();

                let bytes = memory
                    .get_bytes_unchecked(AllocRange { start: Size::ZERO, size: memory.size() })
                    .to_vec();

                let mut prov = BTreeMap::new();
                for (offset, ptr) in memory.provenance().ptrs().iter() {
                    prov.insert(SolOffset(offset.bytes_usize()), self.make_global(ptr.alloc_id()));
                }

                let align = SolAlign(memory.align.bytes_usize());
                match memory.mutability {
                    Mutability::Not => SolGlobalObject::ImmData { bytes, provenance: prov, align },
                    Mutability::Mut => SolGlobalObject::MutData { bytes, provenance: prov, align },
                }
            }
            GlobalAlloc::Static(def_id) => {
                let memory = self
                    .tcx
                    .eval_static_initializer(def_id)
                    .unwrap_or_else(|_| {
                        bug!(
                            "[invariant] unable to evaluate static initializer for {}",
                            self.tcx.def_path_str(def_id)
                        )
                    })
                    .inner();

                let bytes = memory
                    .get_bytes_unchecked(AllocRange { start: Size::ZERO, size: memory.size() })
                    .to_vec();

                let mut prov = BTreeMap::new();
                for (offset, ptr) in memory.provenance().ptrs().iter() {
                    prov.insert(SolOffset(offset.bytes_usize()), self.make_global(ptr.alloc_id()));
                }

                let align = SolAlign(memory.align.bytes_usize());
                match memory.mutability {
                    Mutability::Not => SolGlobalObject::ImmData { bytes, provenance: prov, align },
                    Mutability::Mut => SolGlobalObject::MutData { bytes, provenance: prov, align },
                }
            }
            GlobalAlloc::VTable(..) => bug!("[unsupported] vtable in global variable"),
            GlobalAlloc::TypeId { .. } => bug!("[unsupported] type id in global variable"),
        };

        // update the global variable lookup table
        let key = SolGlobalSlot(self.globals.len());
        self.globals.insert(key.clone(), global);
        self.alloc_map.insert(alloc_id, key.clone());

        // return the key
        key
    }

    /// Build the context
    pub(crate) fn build(self) -> (SolEnv, SolContext) {
        // check we are balanced on stack
        if self.depth.level != 0 {
            bug!("[invariant] depth stack is not balanced");
        }

        // unpack the fields
        let krate = SolCrateName(self.tcx.crate_name(LOCAL_CRATE).to_ident_string());

        let mut ty_defs = vec![];
        for (ident, l1) in self.ty_defs.into_iter() {
            for (mono, def) in l1.into_iter() {
                match def {
                    None => bug!("[invariant] missing type definition"),
                    Some(val) => ty_defs.push((ident.clone(), mono, val)),
                }
            }
        }

        let mut fn_defs = vec![];
        for (ident, l1) in self.fn_defs.into_iter() {
            for (mono, l2) in l1.into_iter() {
                for (kind, def) in l2.into_iter() {
                    match def {
                        None => bug!("[invariant] missing function definition"),
                        Some(val) => fn_defs.push((kind, ident.clone(), mono.clone(), val)),
                    }
                }
            }
        }

        let mut globals = vec![];
        for (slot, global) in self.globals.into_iter() {
            globals.push((slot, global));
        }

        let mut dep_fns = vec![];
        for (ident, l1) in self.dep_fns.into_iter() {
            for (mono, l2) in l1.into_iter() {
                for (kind, desc) in l2.into_iter() {
                    dep_fns.push((kind, ident.clone(), mono.clone(), desc));
                }
            }
        }

        (self.sol, SolContext { krate, ty_defs, fn_defs, globals, dep_fns })
    }
}

/* --- BEGIN OF SYNC --- */

/*
 * Context
 */

/// A complete Solana context
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolContext {
    pub(crate) krate: SolCrateName,
    pub(crate) ty_defs: Vec<(SolIdent, Vec<SolGenericArg>, SolTyDef)>,
    pub(crate) fn_defs: Vec<(SolInstanceKind, SolIdent, Vec<SolGenericArg>, SolFnDef)>,
    pub(crate) globals: Vec<(SolGlobalSlot, SolGlobalObject)>,
    pub(crate) dep_fns: Vec<(SolInstanceKind, SolIdent, Vec<SolGenericArg>, SolPathDescWithArgs)>,
}

/// Dependencies
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolDeps {
    pub(crate) fn_deps: Vec<(SolInstanceKind, SolIdent, Vec<SolGenericArg>, SolPathDescWithArgs)>,
}

/*
 * Naming
 */

/// An identifier in the Solana context
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolIdent {
    pub(crate) krate: SolHash64,
    pub(crate) local: SolHash64,
}

/// A 64-bit hash
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolHash64(pub(crate) u64);

/// A description of a definition path
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolPathDesc(pub(crate) String);

/// A description of an instance path with generic arguments
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolPathDescWithArgs(pub(crate) String);

/// A crate name
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolCrateName(pub(crate) String);

/// A field name
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolFieldName(pub(crate) String);

/// A field index
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolFieldIndex(pub(crate) usize);

/// A variant name
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolVariantName(pub(crate) String);

/// A variant index
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolVariantIndex(pub(crate) usize);

/// A slot number for locals
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolLocalSlot(pub(crate) usize);

/// A basic block index
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolBlockId(pub(crate) usize);

/// A location id for global values
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolGlobalSlot(pub(crate) usize);

/*
 * Typing
 */

/// Primitive integer types
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolTypeInt {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    I128,
    U128,
    Isize,
    Usize,
}

/// Primitive floating-point types
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolTypeFloat {
    F16,
    F32,
    F64,
    F128,
}

/// All supported types in the Solana context
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolType {
    Never,
    Bool,
    Char,
    Int(SolTypeInt),
    Float(SolTypeFloat),
    Str,
    Array(Box<SolType>, Box<SolTyConst>),
    Tuple(Vec<SolType>),
    Adt(SolIdent, Vec<SolGenericArg>),
    Slice(Box<SolType>),
    ImmRef(Box<SolType>),
    MutRef(Box<SolType>),
    ImmPtr(Box<SolType>),
    MutPtr(Box<SolType>),
    Function(SolInstanceKind, SolIdent, Vec<SolGenericArg>),
    Closure(SolInstanceKind, SolIdent, Vec<SolGenericArg>),
    FnPtr(Vec<SolType>, Box<SolType>),
    Dynamic(Vec<(SolIdent, Vec<SolGenericArg>)>),
}

/// User-defined type, i.e., an algebraic data type (ADT)
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolTyDef {
    Struct { fields: Vec<SolField> },
    Union { fields: Vec<SolField> },
    Enum { variants: Vec<SolVariant> },
}

/// A field definition in an ADT
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolField {
    pub(crate) index: SolFieldIndex,
    pub(crate) name: SolFieldName,
    pub(crate) ty: SolType,
}

/// A field definition in an ADT
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolVariant {
    pub(crate) index: SolVariantIndex,
    pub(crate) name: SolVariantName,
    pub(crate) fields: Vec<SolField>,
}

/// A constant known in the type system
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolTyConst {
    Simple { ty: SolType, val: SolConst },
}

/// A generic argument
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolGenericArg {
    Type(SolType),
    Const(SolTyConst),
    Lifetime,
}

/*
 * Value
 */

/// A place
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolPlace {
    pub(crate) local: SolLocalSlot,
    pub(crate) projection: Vec<SolProjection>,
}

/// A projection operation
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolProjection {
    Deref,
    Field { field: SolFieldIndex, ty: SolType },
    VariableIndex { local: SolLocalSlot },
    ConstantIndex { offset: usize, min_length: usize, from_end: bool },
    Subslice { from: usize, to: usize, from_end: bool },
    Downcast { variant: SolVariantIndex },
    Subtype(SolType),
}

/// A constant known in the operation system
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolOpConst {
    Type(SolType, SolConst),
    Value(SolType, SolConst),
}

/// An operand used in expressions
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolOperand {
    Copy(SolPlace),
    Move(SolPlace),
    Constant(SolOpConst),
}

/*
 * Expression
 */

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolExpr {
    Use(SolOperand),
    Repeat(SolOperand, SolTyConst),
    BorrowImm(SolPlace),
    BorrowMut(SolPlace),
    PointerImm(SolPlace),
    PointerMut(SolPlace),
    Length(SolPlace),
    Cast { opcode: SolOpcodeCast, place: SolOperand, ty: SolType },
    OpNullary { opcode: SolOpcodeNullary, ty: SolType },
    OpUnary { opcode: SolOpcodeUnary, val: SolOperand },
    OpBinary { opcode: SolOpcodeBinary, v1: SolOperand, v2: SolOperand },
    Discriminant(SolPlace),
    Aggregate { opcode: SolOpcodeAggregate, vals: Vec<(SolFieldIndex, SolOperand)> },
    Load(SolPlace),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolOpcodeCast {
    IntToInt,
    FloatToFloat,
    IntToFloat,
    FloatToInt,
    Reinterpret,
    ReifyFnPtr,
    PtrToAddr,
    AddrToPtr,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolOpcodeNullary {
    SizeOf,
    AlignOf,
    OffsetOf(Vec<(SolVariantIndex, SolFieldIndex)>),
    UbCheckEnabled,
    ContractCheckEnabled,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolOpcodeUnary {
    Not,
    Neg,
    PtrMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolOpcodeBinary {
    Add(bool),
    Sub(bool),
    Mul(bool),
    Div(bool),
    Rem(bool),
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Cmp,
    Offset,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolOpcodeAggregate {
    Tuple,
    Array(SolType /* element type */),
    Struct { ty: SolType },
    Union { ty: SolType, field: SolFieldIndex },
    Enum { ty: SolType, variant: SolVariantIndex },
    ImmPtr { pointee_ty: SolType },
    MutPtr { pointee_ty: SolType },
    Closure(SolInstanceKind, SolIdent, Vec<SolGenericArg>),
}

/*
 * Control-flow
 */

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolUnwindAction {
    Continue,
    Unreachable,
    Terminate,
    Cleanup(SolBlockId),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolStatement {
    Nop,
    Deinit(SolPlace),
    StorageLive(SolLocalSlot),
    StorageDead(SolLocalSlot),
    PlaceMention(SolPlace),
    Assign { lhs: SolPlace, rhs: SolExpr },
    Assume(SolOperand),
    CopyNonoverlapping { src: SolOperand, dst: SolOperand, count: SolOperand },
    SetDiscriminant { place: SolPlace, variant: SolVariantIndex },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolTerminator {
    Unreachable,
    Return,
    Goto {
        target: SolBlockId,
    },
    Switch {
        cond: SolOperand,
        targets: Vec<(u128, SolBlockId)>,
        otherwise: SolBlockId,
    },
    Drop {
        place: SolPlace,
        target: SolBlockId,
        unwind: SolUnwindAction,
    },
    Assert {
        cond: SolOperand,
        expected: bool,
        target: SolBlockId,
        unwind: SolUnwindAction,
    },
    Call {
        func: SolOperand,
        args: Vec<SolOperand>,
        dest: SolPlace,
        target: Option<SolBlockId>,
        unwind: SolUnwindAction,
    },
    TailCall {
        func: SolOperand,
        args: Vec<SolOperand>,
    },
    UnwindResume,
    UnwindTerminate,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolBasicBlock {
    pub(crate) id: SolBlockId,
    pub(crate) statements: Vec<SolStatement>,
    pub(crate) terminator: SolTerminator,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolFnDef {
    pub(crate) args: Vec<SolType>,
    pub(crate) ret_ty: SolType,
    pub(crate) locals: Vec<SolType>,
    pub(crate) blocks: Vec<SolBasicBlock>,
}

/*
 * Global
 */

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolGlobalObject {
    ImmData { bytes: Vec<u8>, provenance: BTreeMap<SolOffset, SolGlobalSlot>, align: SolAlign },
    MutData { bytes: Vec<u8>, provenance: BTreeMap<SolOffset, SolGlobalSlot>, align: SolAlign },
    Function(SolInstanceKind, SolIdent, Vec<SolGenericArg>),
}

/*
 * Shared
 */

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolOffset(pub(crate) usize);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolAlign(pub(crate) usize);

/// A scalar value
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolScalar {
    pub(crate) bits: usize,
    pub(crate) value: u128,
}

/// A constant, used by both the type system and the value system
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolConst {
    ZeroSized,
    Scalar(SolScalar),
    Slice { origin: SolGlobalSlot, length: usize },
    Indirect { origin: SolGlobalSlot, offset: SolOffset },
    Pointer { origin: SolGlobalSlot, offset: SolOffset },
}

/*
 * Builtins
 */

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolInstanceKind {
    Regular,
    Builtin(SolBuiltinFunc),

    /* drop operation */
    DropEmpty,
    DropGlued(Box<SolType>),

    /* instance that has a MIR body */
    VTableShim,
    ReifyShim,
    FnPtrShim(Box<SolType>),
    FnPtrAddrShim(Box<SolType>),
    CloneShim(Box<SolType>),
    ClosureOnceShim,

    /* instance that does not have a MIR body */
    Intrinsic,
    Virtual(usize),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolBuiltinFunc {
    /* operations */
    IntrinsicsAbort,
    IntrinsicsAssertFailed,
    IntrinsicsPanic,
    IntrinsicsPanicFmt,
    IntrinsicsPanicNounwindFmt,
    IntrinsicsRawEq,
    IntrinsicsPtrOffsetFromUnsigned,
    IntrinsicsCtpop,
    IntrinsicsColdPath,
    /* alloc */
    AllocGlobalAllocImpl,
    AllocRustRealloc,
    AllocRustDealloc,
    AllocHandleAllocError,
    AllocRawVecHandleError,
    LayoutIsSizeAlignValid,
    SpecToString,
    /* formatter */
    DebugFmt,
    /* solana */
    SolInvokeSigned,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolBuiltinType {
    /* error */
    StdIoError,
    /* format arguments */
    FmtArgument,
    FmtArguments,
    FmtArgumentType,
    FmtFormatter,
}

/* --- END OF SYNC --- */

/*
 * Implementations
 */

impl From<IntTy> for SolTypeInt {
    fn from(t: IntTy) -> Self {
        match t {
            IntTy::I8 => Self::I8,
            IntTy::I16 => Self::I16,
            IntTy::I32 => Self::I32,
            IntTy::I64 => Self::I64,
            IntTy::I128 => Self::I128,
            IntTy::Isize => Self::Isize,
        }
    }
}

impl From<UintTy> for SolTypeInt {
    fn from(t: UintTy) -> Self {
        match t {
            UintTy::U8 => Self::U8,
            UintTy::U16 => Self::U16,
            UintTy::U32 => Self::U32,
            UintTy::U64 => Self::U64,
            UintTy::U128 => Self::U128,
            UintTy::Usize => Self::Usize,
        }
    }
}

impl From<FloatTy> for SolTypeFloat {
    fn from(t: FloatTy) -> Self {
        match t {
            FloatTy::F16 => Self::F16,
            FloatTy::F32 => Self::F32,
            FloatTy::F64 => Self::F64,
            FloatTy::F128 => Self::F128,
        }
    }
}

impl SolBuiltinFunc {
    fn regex(&self) -> Regex {
        let pattern = match self {
            /* operations */
            Self::IntrinsicsAbort => r"std::intrinsics::abort",
            Self::IntrinsicsAssertFailed => r"assert_failed::<.*>",
            Self::IntrinsicsPanic => r"panic",
            Self::IntrinsicsPanicFmt => r"panic_fmt",
            Self::IntrinsicsPanicNounwindFmt => r"panic_nounwind_fmt",
            Self::IntrinsicsRawEq => r"raw_eq::<.*>",
            Self::IntrinsicsPtrOffsetFromUnsigned => r"ptr_offset_from_unsigned::<.*>",
            Self::IntrinsicsCtpop => r"ctpop::<.*>",
            Self::IntrinsicsColdPath => r"std::intrinsics::cold_path",
            /* alloc */
            Self::AllocGlobalAllocImpl => r"std::alloc::Global::alloc_impl",
            Self::AllocRustRealloc => r"alloc::alloc::__rust_realloc",
            Self::AllocRustDealloc => r"alloc::alloc::__rust_dealloc",
            Self::AllocHandleAllocError => r"handle_alloc_error",
            Self::AllocRawVecHandleError => r"alloc::raw_vec::handle_error",
            Self::LayoutIsSizeAlignValid => r"Layout::is_size_align_valid",
            Self::SpecToString => r"<.* as string::SpecToString>::spec_to_string",
            /* formatter */
            Self::DebugFmt => r"<.* as Debug>::fmt",
            /* solana */
            Self::SolInvokeSigned => r"sol_invoke_signed",
        };

        Regex::new(pattern).unwrap_or_else(|e| {
            bug!("[invariant] failed to compile regex for builtin function: {e}")
        })
    }

    fn all() -> Vec<Self> {
        vec![
            /* operations */
            Self::IntrinsicsAbort,
            Self::IntrinsicsAssertFailed,
            Self::IntrinsicsPanic,
            Self::IntrinsicsPanicFmt,
            Self::IntrinsicsPanicNounwindFmt,
            Self::IntrinsicsRawEq,
            Self::IntrinsicsPtrOffsetFromUnsigned,
            Self::IntrinsicsCtpop,
            Self::IntrinsicsColdPath,
            /* alloc */
            Self::AllocGlobalAllocImpl,
            Self::AllocRustRealloc,
            Self::AllocRustDealloc,
            Self::AllocHandleAllocError,
            Self::AllocRawVecHandleError,
            Self::LayoutIsSizeAlignValid,
            Self::SpecToString,
            /* formatter */
            Self::DebugFmt,
            /* solana */
            Self::SolInvokeSigned,
        ]
    }
}

impl SolBuiltinType {
    fn regex(&self) -> Regex {
        let pattern = match self {
            Self::StdIoError => r"std::io::Error",
            Self::FmtArgument => r"core::fmt::rt::Argument<.*>",
            Self::FmtArguments => r"Arguments<.*>",
            Self::FmtArgumentType => r"core::fmt::rt::ArgumentType<.*>",
            Self::FmtFormatter => r"Formatter<.*>",
        };

        Regex::new(pattern).unwrap_or_else(|e| {
            bug!("[invariant] failed to compile regex for builtin datatype: {e}")
        })
    }

    fn all() -> Vec<Self> {
        vec![
            Self::StdIoError,
            Self::FmtArgument,
            Self::FmtArguments,
            Self::FmtArgumentType,
            Self::FmtFormatter,
        ]
    }
}

impl Display for SolPathDescWithArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/*
 * Utilities
 */

/// A depth counter for indentation in logging
struct Depth {
    level: usize,
}

impl Depth {
    /// Create a new depth counter starting at level 0
    fn new() -> Self {
        Self { level: 0 }
    }

    /// Increment the depth counter to the next level
    fn push(&mut self) {
        self.level += 1;
    }

    /// Decrement the depth counter to the previous level
    fn pop(&mut self) {
        if self.level == 0 {
            bug!("[invariant] context stack underflows");
        }
        self.level -= 1;
    }
}

impl Display for Depth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", "  ".repeat(self.level))
    }
}
