use std::collections::BTreeMap;
use std::fmt::Display;

use rustc_abi::Size;
use rustc_ast::Mutability;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{LOCAL_CRATE, LocalDefId};
use rustc_hir::definitions::DefPathData;
use rustc_middle::bug;
use rustc_middle::mir::interpret::{AllocId, AllocRange, GlobalAlloc, Scalar};
use rustc_middle::mir::{
    AggregateKind, BinOp, BorrowKind, CastKind, Const as OpConst, ConstValue, MirPhase,
    NonDivergingIntrinsic, NullOp, Operand, Place, PlaceElem, RawPtrKind, RuntimePhase, Rvalue,
    Statement, StatementKind, Terminator, TerminatorKind, UnOp, UnwindAction,
};
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::fast_reject::SimplifiedType;
use rustc_middle::ty::{
    self, AdtDef, AdtKind, Const as TyConst, ConstKind as TyConstKind, EarlyBinder, FloatTy,
    GenericArg, GenericArgKind, GenericArgsRef, ImplSubject, Instance, IntTy, Ty, TyCtxt,
    TypingEnv, UintTy,
};
use rustc_span::DUMMY_SP;
use rustc_span::def_id::DefId;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::solana::builtin::BuiltinFunction;
use crate::solana::common::SolEnv;

/// A builder for creating a Solana context
pub(crate) struct SolContextBuilder<'tcx> {
    /// compiler context
    tcx: TyCtxt<'tcx>,

    /// solana environment
    sol: SolEnv,

    /// inherent impls for types
    inherent_impls: FxHashMap<DefId, LocalDefId>,

    /// incoherent impls for types
    incoherent_impls: FxHashMap<DefId, SolTagType>,

    /// depth of the current context
    depth: Depth,

    /// generics context
    generics: Vec<GenericArgsRef<'tcx>>,

    /// a cache of id to identifier mappings
    id_cache: FxHashMap<DefId, SolIdent>,

    /// a cache of alloc id to global slot mappings
    alloc_map: FxHashMap<AllocId, SolGlobalSlot>,

    /// collected definitions of datatypes
    ty_defs: BTreeMap<SolIdent, BTreeMap<Vec<SolGenericArg>, Option<SolTyDef>>>,

    /// collected definitions of functions
    fn_defs: BTreeMap<SolIdent, BTreeMap<Vec<SolGenericArg>, Option<SolFnDef>>>,

    /// collected definitions of global variables
    globals: BTreeMap<SolGlobalSlot, SolGlobalObject>,

    /// functions without MIRs available
    dep_fns: BTreeMap<SolIdent, BTreeMap<Vec<SolGenericArg>, SolInstDesc>>,
}

impl<'tcx> SolContextBuilder<'tcx> {
    /// Create a new context builder
    pub(crate) fn new(tcx: TyCtxt<'tcx>, sol: SolEnv) -> Self {
        // construct the inherent impl to type mapping
        let (all_impls, result) = tcx.crate_inherent_impls(());
        result.unwrap_or_else(|_| bug!("[invariant] failed to retrieve inherent impls"));

        let mut inherent_impls = FxHashMap::default();
        for (local_ty_id, impls) in all_impls.inherent_impls.iter() {
            for impl_id in impls {
                let existing = inherent_impls.insert(*impl_id, *local_ty_id);
                if existing.is_some() {
                    bug!("an inherent impl is mapped to more than one type");
                }
            }
        }

        // construct the incoherent impl to type mapping
        let mut incoherent_impls = FxHashMap::default();
        for (tag_ty, simplified_ty) in SolTagType::supported_simplified_types() {
            for impl_id in tcx.incoherent_impls(simplified_ty) {
                let existing = incoherent_impls.insert(*impl_id, tag_ty.clone());
                if existing.is_some() {
                    bug!("an incoherent impl is mapped to more than one type");
                }
            }
        }

        // construct the context builder
        Self {
            tcx,
            sol,
            inherent_impls,
            incoherent_impls,
            depth: Depth::new(),
            generics: Vec::new(),
            id_cache: FxHashMap::default(),
            alloc_map: FxHashMap::default(),
            ty_defs: BTreeMap::new(),
            fn_defs: BTreeMap::new(),
            globals: BTreeMap::new(),
            dep_fns: BTreeMap::new(),
        }
    }

    /// Create an identifier
    fn mk_ident(&mut self, def_id: DefId) -> SolIdent {
        // check cache first
        if let Some(ident) = self.id_cache.get(&def_id) {
            return ident.clone();
        }

        // now construct the identifier
        let def_path = self.tcx.def_path(def_id);
        let def_path_str = self.tcx.def_path_str(def_id);
        let krate = self.tcx.crate_name(def_path.krate).to_ident_string();

        // shortcut if we are at the crate root
        if def_id.is_crate_root() {
            return SolIdent::CrateRoot(krate);
        }

        // resolve parent
        let parent = self.mk_ident(self.tcx.parent(def_id));
        if parent.krate() != krate {
            bug!("[invariant] parent crate name does not match: {} vs {krate}", parent.krate());
        }
        let parent = Box::new(parent);

        // resolve the last path segment
        let segment = match def_path.data.last() {
            None => bug!("[invariant] no segment in def path: {def_path_str}"),
            Some(s) => s,
        };

        let ident = match &segment.data {
            DefPathData::CrateRoot => {
                bug!("[invariant] unexpected crate root segment in def path: {def_path_str}");
            }
            DefPathData::TypeNs(name) => {
                SolIdent::TypeNs { parent, name: SolTypeNs(name.to_ident_string()) }
            }
            DefPathData::ValueNs(name) => match self.tcx.def_kind(def_id) {
                DefKind::Fn | DefKind::AssocFn => {
                    SolIdent::FuncNs { parent, name: SolFuncNs(name.to_ident_string()) }
                }
                _ => {
                    bug!(
                        "[invariant] unexpected value name {} for def kind {}",
                        name.to_ident_string(),
                        self.tcx.def_kind_descr(self.tcx.def_kind(def_id), def_id)
                    );
                }
            },
            DefPathData::Impl => match self.tcx.trait_id_of_impl(def_id) {
                None => match self.inherent_impls.get(&def_id) {
                    None => match self.incoherent_impls.get(&def_id) {
                        None => {
                            let subject = self.tcx.instantiate_and_normalize_erasing_regions(
                                self.generics.last().unwrap_or_else(|| {
                                    bug!("[invariant] no generic arguments in context")
                                }),
                                TypingEnv::fully_monomorphized(),
                                self.tcx.impl_subject(def_id),
                            );
                            match subject {
                                ImplSubject::Trait(..) => {
                                    bug!(
                                        "[invariant] unexpected trait as impl subject for {def_path_str}"
                                    )
                                }
                                ImplSubject::Inherent(subject_ty) => SolIdent::TypeImpl {
                                    parent,
                                    ty: Box::new(self.mk_type(subject_ty)),
                                },
                            }
                        }
                        Some(type_tag) => {
                            SolIdent::OtherImpl { parent, type_tag: type_tag.clone() }
                        }
                    },
                    Some(local_ty_id) => {
                        let self_ident = self.mk_ident(local_ty_id.to_def_id());
                        SolIdent::SelfImpl { parent, self_ident: Box::new(self_ident) }
                    }
                },
                Some(trait_id) => {
                    SolIdent::TraitImpl { parent, trait_ident: Box::new(self.mk_ident(trait_id)) }
                }
            },
            DefPathData::ForeignMod => SolIdent::Extern { parent },

            // unsupported or unrelated
            DefPathData::MacroNs(name) => {
                bug!("[assumption] unexpected macro segment {name} in def path: {def_path_str}");
            }
            DefPathData::LifetimeNs(name) => {
                bug!("[assumption] unexpected lifetime segment {name} in def path: {def_path_str}");
            }
            DefPathData::Use => {
                bug!("[assumption] unexpected use segment in def path: {def_path_str}");
            }
            DefPathData::Closure => {
                bug!("[assumption] unexpected closure segment in def path: {def_path_str}");
            }
            DefPathData::Ctor => {
                bug!("[assumption] unexpected ctor segment in def path: {def_path_str}");
            }
            DefPathData::GlobalAsm => {
                bug!("[unsupported] global asm segment in def path: {def_path_str}");
            }
            DefPathData::AnonConst
            | DefPathData::OpaqueTy
            | DefPathData::OpaqueLifetime(..)
            | DefPathData::AnonAssocTy(..)
            | DefPathData::SyntheticCoroutineBody
            | DefPathData::NestedStatic => {
                bug!("[invariant] unexpected segment {segment:?} in def path: {def_path_str}");
            }
        };

        // insert into the cache
        self.id_cache.insert(def_id, ident.clone());

        // return ident
        ident
    }

    /// Create a type
    fn mk_type(&mut self, ty: Ty<'tcx>) -> SolType {
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
                let (ident, ty_args) = self.make_type_function(def_id, generics);
                SolType::Function(ident, ty_args)
            }
            ty::Closure(def_id, generics) => {
                let (ident, ty_args) = self.make_type_closure(def_id, generics);
                SolType::Closure(ident, ty_args)
            }
            ty::FnPtr(..) => {
                bug!("[unsupported] function pointer type: {ty}");
            }
            ty::Dynamic(..) => {
                bug!("[unsupported] dynamic type: {ty}");
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
        let ident = self.mk_ident(adt_def.did());
        let ty_args: Vec<_> = generics.iter().map(|ty_arg| self.mk_ty_arg(ty_arg)).collect();

        // if already defined or is being defined, return the key
        if self.ty_defs.get(&ident).map_or(false, |inner| inner.contains_key(&ty_args)) {
            return (ident, ty_args);
        }

        // mark start
        self.depth.push();
        info!("{}-> adt definition", self.depth);
        self.generics.push(generics);

        // first update the entry to mark that type definition in progress
        self.ty_defs.entry(ident.clone()).or_default().insert(ty_args.clone(), None);

        // now create the type definition
        let ty_def = match adt_def.adt_kind() {
            AdtKind::Struct => {
                if adt_def.variants().len() != 1 {
                    bug!(
                        "[invariant] struct {} has multiple variants",
                        self.tcx.def_path_str(adt_def.did())
                    );
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
                    bug!(
                        "[invariant] union {} has multiple variants",
                        self.tcx.def_path_str(adt_def.did())
                    );
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
        self.generics.pop();
        info!("{}<- adt definition", self.depth);
        self.depth.pop();

        // return the key to the lookup table
        (ident, ty_args)
    }

    fn make_type_function(
        &mut self,
        def_id: DefId,
        generics: GenericArgsRef<'tcx>,
    ) -> (SolIdent, Vec<SolGenericArg>) {
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
    ) -> (SolIdent, Vec<SolGenericArg>) {
        // resolve the closure instance
        let closure_ty_args = generics.as_closure();
        let instance =
            Instance::resolve_closure(self.tcx, def_id, generics, closure_ty_args.kind());

        // return the key to the lookup table
        self.make_instance(instance)
    }

    /// Create a generic argument
    fn mk_ty_arg(&mut self, ty_arg: GenericArg<'tcx>) -> SolGenericArg {
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
                        | PointerCoercion::Unsize => SolOpcodeCast::Reinterpret,
                        PointerCoercion::ReifyFnPointer
                        | PointerCoercion::UnsafeFnPointer
                        | PointerCoercion::ClosureFnPointer(_) => {
                            bug!("[unsupported] fn pointer coercion in cast op");
                        }
                    },
                    CastKind::PointerExposeProvenance | CastKind::PointerWithExposedProvenance => {
                        bug!("[assumption] unexpected cast kind: {cast_kind:?}")
                    }
                };
                SolExpr::Cast { opcode, place: self.mk_operand(operand), ty: self.mk_type(*ty) }
            }
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
                    AggregateKind::Closure(..) => {
                        bug!("[assumption] unexpected closure in aggregate conversion");
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
            Rvalue::ShallowInitBox(..) => {
                bug!("[unsupported] unexpected shallow initialization of box in rvalue conversion");
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
    ) -> (SolIdent, Vec<SolGenericArg>) {
        let def_id = instance.def_id();
        let def_desc = self.tcx.def_path_str(def_id);

        // locate the key of the definition
        let ident = self.mk_ident(def_id);
        let ty_args: Vec<_> = instance.args.iter().map(|ty_arg| self.mk_ty_arg(ty_arg)).collect();

        // if already defined or is being defined, return the key
        if self.fn_defs.get(&ident).map_or(false, |inner| inner.contains_key(&ty_args)) {
            return (ident, ty_args);
        }

        // check if this is a known function
        match BuiltinFunction::try_to_resolve(&ident) {
            None => (), /* continue construction */
            Some(_) => {
                info!("{}-- builtin function: {def_desc}", self.depth);
                return (ident, ty_args);
            }
        }

        // convert the instance to monomorphised MIR
        if !self.tcx.is_mir_available(def_id) {
            info!("{}-- external dependency: {def_desc}", self.depth);
            let deps_by_ident = self.dep_fns.entry(ident.clone()).or_default();
            if !deps_by_ident.contains_key(&ty_args) {
                warn!("external dependency: {def_desc}");
                let inst_desc = self.tcx.def_path_str_with_args(def_id, instance.args);
                deps_by_ident.insert(ty_args.clone(), SolInstDesc(inst_desc));
            }
            return (ident, ty_args);
        }

        // mark start
        self.depth.push();
        warn!("{}-> function {def_desc}", self.depth);
        self.generics.push(instance.args);

        // convert the instance to monomorphised MIR
        let instance_mir = self.tcx.instance_mir(instance.def).clone();
        if instance_mir.phase != MirPhase::Runtime(RuntimePhase::Optimized) {
            bug!("[assumption] converted instance is not runtime optimized: {def_desc}");
        }

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
        self.fn_defs.entry(ident.clone()).or_default().insert(ty_args.clone(), Some(fn_def));

        // mark end
        self.generics.pop();
        warn!("{}<- function {def_desc}", self.depth);
        self.depth.pop();

        // return the key to the lookup table
        (ident, ty_args)
    }

    /// Create a global variable definition
    pub(crate) fn make_global(&mut self, alloc_id: AllocId) -> SolGlobalSlot {
        // if already defined or is being defined, return the key
        if let Some(slot) = self.alloc_map.get(&alloc_id) {
            return slot.clone();
        }

        // now analyze the global variable
        let global = match self.tcx.global_alloc(alloc_id) {
            GlobalAlloc::Function { instance } => {
                let (ident, ty_args) = self.make_instance(instance);
                SolGlobalObject::Function(ident, ty_args)
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
            GlobalAlloc::Static(..) => bug!("[invariant] unexpected static global variable"),
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
        if !self.generics.is_empty() {
            bug!("[invariant] generics stack is not empty");
        }

        // unpack the fields
        let krate = self.tcx.crate_name(LOCAL_CRATE).to_ident_string();

        let mut ty_defs = vec![];
        for (ident, defs) in self.ty_defs.into_iter() {
            for (mono, def) in defs.into_iter() {
                match def {
                    None => bug!("[invariant] missing type definition"),
                    Some(val) => ty_defs.push((ident.clone(), mono, val)),
                }
            }
        }

        let mut fn_defs = vec![];
        for (ident, defs) in self.fn_defs.into_iter() {
            for (mono, def) in defs.into_iter() {
                match def {
                    None => bug!("[invariant] missing function definition"),
                    Some(val) => fn_defs.push((ident.clone(), mono, val)),
                }
            }
        }

        let mut globals = vec![];
        for (slot, global) in self.globals.into_iter() {
            globals.push((slot, global));
        }

        let mut dep_fns = vec![];
        for (ident, descs) in self.dep_fns.into_iter() {
            for (mono, desc) in descs.into_iter() {
                dep_fns.push((ident.clone(), mono, desc));
            }
        }

        (self.sol, SolContext { krate, ty_defs, fn_defs, globals, dep_fns })
    }
}

/*
 * Context
 */

/// A complete Solana context
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolContext {
    krate: String,
    ty_defs: Vec<(SolIdent, Vec<SolGenericArg>, SolTyDef)>,
    fn_defs: Vec<(SolIdent, Vec<SolGenericArg>, SolFnDef)>,
    globals: Vec<(SolGlobalSlot, SolGlobalObject)>,
    dep_fns: Vec<(SolIdent, Vec<SolGenericArg>, SolInstDesc)>,
}

/*
 * Naming
 */

/// An identifier in the Solana context
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolIdent {
    CrateRoot(String),
    TypeNs { parent: Box<SolIdent>, name: SolTypeNs },
    FuncNs { parent: Box<SolIdent>, name: SolFuncNs },
    SelfImpl { parent: Box<SolIdent>, self_ident: Box<SolIdent> },
    TraitImpl { parent: Box<SolIdent>, trait_ident: Box<SolIdent> },
    TypeImpl { parent: Box<SolIdent>, ty: Box<SolType> },
    OtherImpl { parent: Box<SolIdent>, type_tag: SolTagType },
    Extern { parent: Box<SolIdent> },
}

/// A description of an instance
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolInstDesc(String);

/// A typing namespace item
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolTypeNs(pub String);

/// A function namespace item
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolFuncNs(pub String);

/// A field name
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolFieldName(String);

/// A field index
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolFieldIndex(usize);

/// A variant name
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolVariantName(String);

/// A variant index
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolVariantIndex(usize);

/// A slot number for locals
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolLocalSlot(usize);

/// A basic block index
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolBlockId(usize);

/// A location id for global values
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolGlobalSlot(usize);

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
    Function(SolIdent, Vec<SolGenericArg>),
    Closure(SolIdent, Vec<SolGenericArg>),
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
    index: SolFieldIndex,
    name: SolFieldName,
    ty: SolType,
}

/// A field definition in an ADT
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolVariant {
    index: SolVariantIndex,
    name: SolVariantName,
    fields: Vec<SolField>,
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolTagType {
    Bool,
    Char,
    Int(SolTypeInt),
    Float(SolTypeFloat),
    Str,
    Array,
    Slice,
    ImmRef,
    MutRef,
    ImmPtr,
    MutPtr,
}

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

impl SolTagType {
    /// A mapping of supported simplified types
    fn supported_simplified_types() -> BTreeMap<Self, SimplifiedType> {
        let mut map = BTreeMap::new();

        map.insert(Self::Bool, SimplifiedType::Bool);
        map.insert(Self::Char, SimplifiedType::Char);
        map.insert(Self::Int(SolTypeInt::I8), SimplifiedType::Int(IntTy::I8));
        map.insert(Self::Int(SolTypeInt::U8), SimplifiedType::Uint(UintTy::U8));
        map.insert(Self::Int(SolTypeInt::I16), SimplifiedType::Int(IntTy::I16));
        map.insert(Self::Int(SolTypeInt::U16), SimplifiedType::Uint(UintTy::U16));
        map.insert(Self::Int(SolTypeInt::I32), SimplifiedType::Int(IntTy::I32));
        map.insert(Self::Int(SolTypeInt::U32), SimplifiedType::Uint(UintTy::U32));
        map.insert(Self::Int(SolTypeInt::I64), SimplifiedType::Int(IntTy::I64));
        map.insert(Self::Int(SolTypeInt::U64), SimplifiedType::Uint(UintTy::U64));
        map.insert(Self::Int(SolTypeInt::I128), SimplifiedType::Int(IntTy::I128));
        map.insert(Self::Int(SolTypeInt::U128), SimplifiedType::Uint(UintTy::U128));
        map.insert(Self::Float(SolTypeFloat::F16), SimplifiedType::Float(FloatTy::F16));
        map.insert(Self::Float(SolTypeFloat::F32), SimplifiedType::Float(FloatTy::F32));
        map.insert(Self::Float(SolTypeFloat::F64), SimplifiedType::Float(FloatTy::F64));
        map.insert(Self::Float(SolTypeFloat::F128), SimplifiedType::Float(FloatTy::F128));
        map.insert(Self::Str, SimplifiedType::Str);
        map.insert(Self::Array, SimplifiedType::Array);
        map.insert(Self::Slice, SimplifiedType::Slice);
        map.insert(Self::ImmRef, SimplifiedType::Ref(Mutability::Not));
        map.insert(Self::MutRef, SimplifiedType::Ref(Mutability::Mut));
        map.insert(Self::ImmPtr, SimplifiedType::Ptr(Mutability::Not));
        map.insert(Self::MutPtr, SimplifiedType::Ptr(Mutability::Mut));

        map
    }
}

/*
 * Value
 */

/// A place
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolPlace {
    local: SolLocalSlot,
    projection: Vec<SolProjection>,
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
    id: SolBlockId,
    statements: Vec<SolStatement>,
    terminator: SolTerminator,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolFnDef {
    args: Vec<SolType>,
    ret_ty: SolType,
    locals: Vec<SolType>,
    blocks: Vec<SolBasicBlock>,
}

/*
 * Global
 */

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolGlobalObject {
    ImmData { bytes: Vec<u8>, provenance: BTreeMap<SolOffset, SolGlobalSlot>, align: SolAlign },
    MutData { bytes: Vec<u8>, provenance: BTreeMap<SolOffset, SolGlobalSlot>, align: SolAlign },
    Function(SolIdent, Vec<SolGenericArg>),
}

/*
 * Shared
 */

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolOffset(usize);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolAlign(usize);

/// A scalar value
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolScalar {
    bits: usize,
    value: u128,
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

impl SolIdent {
    /// Find the root identifier
    pub(crate) fn krate(&self) -> &str {
        match self {
            Self::CrateRoot(name) => name,
            Self::TypeNs { parent, name: _ }
            | Self::FuncNs { parent, name: _ }
            | Self::SelfImpl { parent, self_ident: _ }
            | Self::TraitImpl { parent, trait_ident: _ }
            | Self::TypeImpl { parent, ty: _ }
            | Self::OtherImpl { parent, type_tag: _ }
            | Self::Extern { parent } => parent.krate(),
        }
    }

    /// Create a new identifier for the crate root
    pub(crate) fn from_root(name: &str) -> Self {
        Self::CrateRoot(name.to_string())
    }

    pub(crate) fn with_type_ns(self: &Self, name: &str) -> Self {
        Self::TypeNs { parent: Box::new(self.clone()), name: SolTypeNs(name.to_string()) }
    }

    pub(crate) fn with_func_ns(self: &Self, name: &str) -> Self {
        Self::FuncNs { parent: Box::new(self.clone()), name: SolFuncNs(name.to_string()) }
    }

    // FIXME: remove the mark
    #[allow(dead_code)]
    pub(crate) fn with_self_impl(self: &Self, self_ident: Self) -> Self {
        Self::SelfImpl { parent: Box::new(self.clone()), self_ident: Box::new(self_ident) }
    }

    pub(crate) fn with_trait_impl(self: &Self, trait_ident: Self) -> Self {
        Self::TraitImpl { parent: Box::new(self.clone()), trait_ident: Box::new(trait_ident) }
    }

    pub(crate) fn with_type_impl(self: &Self, ty: SolType) -> Self {
        Self::TypeImpl { parent: Box::new(self.clone()), ty: Box::new(ty) }
    }

    // FIXME: remove the mark
    #[allow(dead_code)]
    pub(crate) fn with_other_impl(self: &Self, type_tag: SolTagType) -> Self {
        Self::OtherImpl { parent: Box::new(self.clone()), type_tag }
    }

    // FIXME: remove the mark
    #[allow(dead_code)]
    pub(crate) fn with_extern(self: &Self) -> Self {
        Self::Extern { parent: Box::new(self.clone()) }
    }
}
