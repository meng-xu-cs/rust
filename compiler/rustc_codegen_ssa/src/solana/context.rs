use std::collections::BTreeMap;
use std::fmt::Display;

use regex::Regex;
use rustc_abi::{
    BackendRepr, FieldIdx, FieldsShape, HasDataLayout, Primitive, Scalar as ScalarAbi, Size,
    TagEncoding, VariantIdx, Variants,
};
use rustc_ast::Mutability;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::bug;
use rustc_middle::mir::interpret::{AllocRange, Allocation, CtfeProvenance, GlobalAlloc, Scalar};
use rustc_middle::mir::{
    AggregateKind, BinOp, BorrowKind, CastKind, Const as OpConst, ConstValue, MirPhase,
    NonDivergingIntrinsic, NullOp, Operand, Place, PlaceElem, RawPtrKind, RuntimePhase, Rvalue,
    Statement, StatementKind, Terminator, TerminatorKind, UnOp, UnwindAction,
};
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::{
    self, AdtDef, AdtKind, Const as TyConst, ConstKind as TyConstKind, EarlyBinder,
    ExistentialPredicate, FloatTy, GenericArg, GenericArgKind, GenericArgsRef, Instance,
    InstanceKind, IntTy, ScalarInt, TermKind, Ty, TyCtxt, TypingEnv, UintTy, ValTree, VariantDiscr,
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
    builtin_fns: BTreeMap<SolBuiltinFunc, Regex>,

    /// depth of the current context
    depth: Depth,

    /// a cache of id to identifier mappings
    id_cache: FxHashMap<DefId, (SolIdent, SolPathDesc)>,

    /// collected definitions of datatypes
    ty_defs: BTreeMap<SolIdent, BTreeMap<Vec<SolGenericArg>, Option<SolTyDef>>>,

    /// collected definitions of functions
    fn_defs: BTreeMap<
        SolIdent,
        BTreeMap<Vec<SolGenericArg>, BTreeMap<SolInstanceKind, Option<SolFnDef>>>,
    >,

    /// collected definitions of global variables
    globals: BTreeMap<SolIdent, (SolType, SolConst)>,

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
            builtin_fns: SolBuiltinFunc::all()
                .into_iter()
                .map(|item| {
                    let regex = item.regex();
                    (item, regex)
                })
                .collect(),
            depth: Depth::new(),
            id_cache: FxHashMap::default(),
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
                            trait_refs.push((trait_ident, trait_ty_args, None));
                        }
                        ExistentialPredicate::Projection(trait_proj) => {
                            let trait_ident = self.mk_ident(trait_proj.def_id);
                            let trait_ty_args = trait_proj
                                .args
                                .iter()
                                .map(|ty_arg| self.mk_ty_arg(ty_arg))
                                .collect();
                            let trait_term = match trait_proj.term.kind() {
                                TermKind::Ty(term_ty) => SolTyTerm::Type(self.mk_type(term_ty)),
                                TermKind::Const(term_const) => {
                                    SolTyTerm::Const(self.mk_ty_const(term_const))
                                }
                            };
                            trait_refs.push((trait_ident, trait_ty_args, Some(trait_term)));
                        }
                        ExistentialPredicate::AutoTrait(auto_trait_id) => {
                            let trait_ident = self.mk_ident(auto_trait_id);
                            trait_refs.push((trait_ident, vec![], None));
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
                let mut variants = vec![];
                let mut last_discr_value = 0;
                for (variant_idx, variant_def) in adt_def.variants().iter_enumerated() {
                    let variant_index = SolVariantIndex(variant_idx.index());
                    let variant_name = SolVariantName(variant_def.name.to_ident_string());
                    let variant_descr = match variant_def.discr {
                        VariantDiscr::Relative(pos) => {
                            SolVariantDiscr(last_discr_value + pos as u128)
                        }
                        VariantDiscr::Explicit(did) => {
                            let discr = adt_def.eval_explicit_discr(self.tcx, did).unwrap_or_else(|_| {
                                    bug!("[invariant] failed to evaluate discriminant for enum {def_desc}")
                                });
                            last_discr_value = discr.val;
                            SolVariantDiscr(discr.val)
                        }
                    };
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
                    variants.push(SolVariant {
                        index: variant_index,
                        name: variant_name,
                        discr: variant_descr,
                        fields,
                    });
                }
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

    /// Create a constant by parsing the value tree
    fn mk_typed_valtree(&mut self, ty: Ty<'tcx>, tree: ValTree<'tcx>) -> SolConst {
        match tree.try_to_scalar() {
            None => match tree.try_to_branch() {
                None => {
                    bug!("[invariant] non-scalar non-branch valtree {tree:?} with type {ty}");
                }
                Some(branches) => {
                    if branches.len() != 1 {
                        bug!("[unsupported] multi-branch valtree {branches:#?} with type {ty}");
                    }
                    self.mk_typed_valtree(ty, *branches.into_iter().next().unwrap())
                }
            },
            Some(Scalar::Int(scalar_int)) => self.mk_val_const_from_scalar_val(ty, scalar_int),
            Some(Scalar::Ptr(..)) => {
                bug!("[unsupported] pointer as valtree constant {tree:?} in type {ty}")
            }
        }
    }

    /// Create a constant known in the type system
    fn mk_ty_const(&mut self, ty_const: TyConst<'tcx>) -> SolTyConst {
        match ty_const.kind() {
            TyConstKind::Value(val) => {
                let const_ty = self.mk_type(val.ty);
                let normalized_ty =
                    self.tcx.normalize_erasing_regions(TypingEnv::fully_monomorphized(), val.ty);
                let const_val = if val.valtree.is_zst() {
                    self.mk_val_const_from_zst(normalized_ty)
                } else {
                    self.mk_typed_valtree(normalized_ty, val.valtree)
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

    /// Create a constant from a zero-sized type (ZST)
    fn mk_val_const_from_zst(&mut self, ty: Ty<'tcx>) -> SolConst {
        match ty.kind() {
            ty::Tuple(elems) if elems.is_empty() => SolConst::Tuple(vec![]),
            ty::Adt(def, generics) => match def.adt_kind() {
                AdtKind::Struct => {
                    let variant = def.variant(VariantIdx::from_usize(0));
                    let mut fields = vec![];
                    for (field_idx, field_def) in variant.fields.iter_enumerated() {
                        let field_val =
                            self.mk_val_const_from_zst(field_def.ty(self.tcx, generics));
                        fields.push((SolFieldIndex(field_idx.as_usize()), field_val));
                    }
                    SolConst::Struct(fields)
                }
                AdtKind::Enum if def.variants().len() == 1 => {
                    let (variant_idx, variant_def) =
                        def.variants().iter_enumerated().next().unwrap();

                    let mut fields = vec![];
                    for (field_idx, field_def) in variant_def.fields.iter_enumerated() {
                        let field_val =
                            self.mk_val_const_from_zst(field_def.ty(self.tcx, generics));
                        fields.push((SolFieldIndex(field_idx.as_usize()), field_val));
                    }
                    SolConst::Enum(SolVariantIndex(variant_idx.index()), fields)
                }
                _ => bug!("[invariant] the ADT definition is not a ZST struct: {ty}"),
            },
            ty::FnDef(def_id, generics) => {
                let (kind, ident, ty_args) = self.make_type_function(*def_id, generics);
                SolConst::FuncDef(kind, ident, ty_args)
            }
            ty::Closure(def_id, generics) => {
                let (kind, ident, ty_args) = self.make_type_closure(*def_id, generics);
                SolConst::Closure(kind, ident, ty_args)
            }
            _ => bug!("[invariant] making ZST const out of a non-ZST type {ty}"),
        }
    }
    /// Create a constant if the type is a ZST
    fn mk_val_const_when_zst(&mut self, ty: Ty<'tcx>) -> Option<SolConst> {
        match ty.kind() {
            ty::Tuple(elems) if elems.is_empty() => Some(SolConst::Tuple(vec![])),
            ty::Adt(def, _) => match def.adt_kind() {
                AdtKind::Struct if def.all_fields().count() == 0 => Some(SolConst::Struct(vec![])),
                _ => None,
            },
            _ => None,
        }
    }

    /// Create a value constant from a MIR scalar
    fn mk_val_const_from_scalar_val(&mut self, ty: Ty<'tcx>, val: ScalarInt) -> SolConst {
        match ty.kind() {
            // value types
            ty::Bool => SolConst::Bool(!val.is_null()),
            ty::Char => SolConst::Char(val.to_u8() as char),
            ty::Int(int_ty) => match int_ty {
                IntTy::I8 => SolConst::Int(SolConstInt::I8(val.to_i8())),
                IntTy::I16 => SolConst::Int(SolConstInt::I16(val.to_i16())),
                IntTy::I32 => SolConst::Int(SolConstInt::I32(val.to_i32())),
                IntTy::I64 => SolConst::Int(SolConstInt::I64(val.to_i64())),
                IntTy::I128 => SolConst::Int(SolConstInt::I128(val.to_i128())),
                IntTy::Isize => {
                    SolConst::Int(SolConstInt::Isize(val.to_target_isize(self.tcx) as isize))
                }
            },
            ty::Uint(uint_ty) => match uint_ty {
                UintTy::U8 => SolConst::Int(SolConstInt::U8(val.to_u8())),
                UintTy::U16 => SolConst::Int(SolConstInt::U16(val.to_u16())),
                UintTy::U32 => SolConst::Int(SolConstInt::U32(val.to_u32())),
                UintTy::U64 => SolConst::Int(SolConstInt::U64(val.to_u64())),
                UintTy::U128 => SolConst::Int(SolConstInt::U128(val.to_u128())),
                UintTy::Usize => {
                    SolConst::Int(SolConstInt::Usize(val.to_target_usize(self.tcx) as usize))
                }
            },
            ty::Float(float_ty) => match float_ty {
                FloatTy::F16 => SolConst::Float(SolConstFloat::F16(val.to_f16().to_string())),
                FloatTy::F32 => SolConst::Float(SolConstFloat::F32(val.to_f32().to_string())),
                FloatTy::F64 => SolConst::Float(SolConstFloat::F64(val.to_f64().to_string())),
                FloatTy::F128 => SolConst::Float(SolConstFloat::F128(val.to_f128().to_string())),
            },

            // single-field datatype
            ty::Adt(def, generics) => match def.adt_kind() {
                AdtKind::Enum => {
                    let tag_value = val.to_bits_unchecked();

                    // we need to drill down to the encoding
                    let layout = self
                        .tcx
                        .layout_of(TypingEnv::fully_monomorphized().as_query_input(ty))
                        .unwrap_or_else(|e| {
                            bug!("[invariant] unable to get layout of type {ty}: {e}")
                        })
                        .layout;

                    if !matches!(layout.backend_repr, BackendRepr::Scalar(_)) {
                        bug!("[invariant] {ty} is not represented as a scalar");
                    }

                    match &layout.variants {
                        Variants::Multiple { tag_encoding, .. } => match tag_encoding {
                            TagEncoding::Direct => {
                                // discriminant is stored directly as the scalar value
                                if def.all_fields().count() != 0 {
                                    bug!("[assumption] {ty} is not discriminant-only as scalar");
                                }

                                // handle enum discriminant for scalar constant
                                for (variant_idx, discr) in def.discriminants(self.tcx) {
                                    if discr.val == tag_value {
                                        return SolConst::Enum(
                                            SolVariantIndex(variant_idx.index()),
                                            vec![],
                                        );
                                    }
                                }
                                bug!("[invariant] no discriminant matches {tag_value} for {ty}");
                            }
                            TagEncoding::Niche {
                                untagged_variant,
                                niche_variants,
                                niche_start,
                            } => {
                                // NOTE: the discriminant and variant index of each variant coincide
                                let tag_index = tag_value
                                    .wrapping_sub(*niche_start)
                                    .wrapping_add(niche_variants.start().index() as u128);

                                let variant_idx = if tag_index >= def.variants().len() as u128 {
                                    *untagged_variant
                                } else {
                                    VariantIdx::from_usize(tag_index as usize)
                                };
                                let variant = def.variant(variant_idx);

                                // collect field values for the variant
                                let mut field_values = vec![];
                                if variant_idx == *untagged_variant {
                                    // for untagged variant, the scalar value is not yet consumed
                                    let mut consumed = false;
                                    for (field_idx, field_def) in variant.fields.iter_enumerated() {
                                        let field_ty = field_def.ty(self.tcx, generics);
                                        match (consumed, self.mk_val_const_when_zst(field_ty)) {
                                            (true, Some(zst_val)) => {
                                                field_values.push((
                                                    SolFieldIndex(field_idx.index()),
                                                    zst_val,
                                                ));
                                            }
                                            (true, None) => {
                                                bug!(
                                                    "[invariant] expect ZST field {} in {ty} after consuming the scalar",
                                                    field_def.name
                                                );
                                            }
                                            (false, None) => {
                                                let field_val = self
                                                    .mk_val_const_from_scalar_val(field_ty, val);
                                                field_values.push((
                                                    SolFieldIndex(field_idx.index()),
                                                    field_val,
                                                ));
                                                consumed = true;
                                            }
                                            (false, Some(zst_val)) => {
                                                field_values.push((
                                                    SolFieldIndex(field_idx.index()),
                                                    zst_val,
                                                ));
                                            }
                                        }
                                    }
                                } else {
                                    // for tagged variant, the scalar value is consumed as the tag
                                    for (field_idx, field_def) in variant.fields.iter_enumerated() {
                                        let field_ty = field_def.ty(self.tcx, generics);
                                        match self.mk_val_const_when_zst(field_ty) {
                                            None => bug!(
                                                "[invariant] expect ZST field {} in {ty} for tagged variant",
                                                field_def.name
                                            ),
                                            Some(zst_val) => field_values
                                                .push((SolFieldIndex(field_idx.index()), zst_val)),
                                        }
                                    }
                                };

                                // done with the enum constant
                                SolConst::Enum(SolVariantIndex(variant_idx.index()), field_values)
                            }
                        },
                        Variants::Empty | Variants::Single { .. } => {
                            bug!("[invariant] expect multiple variants for scalar enum type {ty}")
                        }
                    }
                }
                AdtKind::Union => bug!("[unsupported] union type {ty} for scalar constant"),
                AdtKind::Struct => {
                    let fields = &def.variant(VariantIdx::ZERO).fields;
                    if fields.len() != 1 {
                        bug!("[unsupported] {ty} is not single-field struct for scalar constant");
                    }
                    let field = fields.iter().next().unwrap();
                    let field_ty = field.ty(self.tcx, generics);
                    let field_val = self.mk_val_const_from_scalar_val(field_ty, val);
                    SolConst::Struct(vec![(SolFieldIndex(0), field_val)])
                }
            },

            // all others
            _ => {
                bug!("[invariant] unexpected type {ty} for value scalar constant {val}");
            }
        }
    }

    fn read_const_from_memory_and_layout(
        &mut self,
        memory: &Allocation,
        offset: Size,
        ty: Ty<'tcx>,
    ) -> (Size, SolConst) {
        // utility
        let read_primitive = |tcx: TyCtxt<'tcx>, start, size: Size, is_provenane: bool| {
            memory.read_scalar(&tcx, AllocRange { start, size }, is_provenane).unwrap_or_else(|e| {
                bug!("[invariant] failed to read a primitive in memory allocation: {e:?}")
            })
        };

        // get the layout of the type
        let layout = self
            .tcx
            .layout_of(TypingEnv::fully_monomorphized().as_query_input(ty))
            .unwrap_or_else(|e| bug!("[invariant] unable to get layout of type {ty}: {e}"))
            .layout;

        // case by type
        let parsed = match ty.kind() {
            // primitives
            ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Float(_) => {
                match layout.variants {
                    Variants::Single { index } if index.index() == 0 => {}
                    _ => bug!("[invariant] expect single 0-variant for value type {ty}"),
                }
                if !matches!(layout.fields, FieldsShape::Primitive) {
                    bug!("[invariant] expect a primitive field for value type {ty}");
                }

                let v = match read_primitive(self.tcx, offset, layout.size, false) {
                    Scalar::Int(scalar_int) => scalar_int,
                    Scalar::Ptr(..) => {
                        bug!("[invariant] unexpected pointer scalar for value type {ty}")
                    }
                };
                self.mk_val_const_from_scalar_val(ty, v)
            }

            // pointers
            ty::RawPtr(sub_ty, mutability) | ty::Ref(_, sub_ty, mutability) => {
                // sanity check on the variants
                match &layout.variants {
                    Variants::Single { index } if index.index() == 0 => {}
                    _ => bug!("[invariant] expect single 0-variant for pointer type {ty}"),
                }

                // read and parse the pointer
                let pointer_size = self.tcx.data_layout().pointer_size();
                let pointer_parsed = match read_primitive(self.tcx, offset, pointer_size, true) {
                    Scalar::Ptr(ptr, _) => {
                        let (prov, offset) = ptr.into_raw_parts();
                        let offset = SolOffset(offset.bytes_usize());
                        let parsed_prov = self.make_provenance(prov, *sub_ty, *mutability);
                        Some((parsed_prov, offset))
                    }
                    Scalar::Int(int) => {
                        if !ty.is_raw_ptr() {
                            bug!("[invariant] unexpected integer for non-ptr type {ty} in memory");
                        }
                        if int.is_null() {
                            // the normal case when faced with a null pointer
                            None
                        } else {
                            // a rare but possible case where the pointer is a constant integer
                            if layout.size != pointer_size {
                                bug!("[invariant] invalid layout size for addr-ptr in type {ty}");
                            }
                            match &layout.backend_repr {
                                BackendRepr::Scalar(ScalarAbi::Initialized {
                                    value: Primitive::Pointer(_),
                                    valid_range: _,
                                }) => (),
                                _ => {
                                    bug!("[invariant] invalid layout for addr-ptr in type {ty}");
                                }
                            }

                            // sanity check passed, short-circuit parsing from here
                            return (
                                layout.size,
                                SolConst::PtrAddress(int.to_target_usize(self.tcx) as usize),
                            );
                        }
                    }
                };

                // check whether the metadata is a ZST
                // NOTE: we only handle size-based metadata for now
                let need_metadata = !sub_ty.is_sized(self.tcx, TypingEnv::fully_monomorphized());

                // switch by whether we need metadata
                if need_metadata {
                    // sanity check on the shape of the fields
                    if pointer_size * 2 != layout.size {
                        bug!(
                            "[invariant] invalid layout size of ptr/ref type {ty}, expect metadata but got {}",
                            layout.size.bytes_usize()
                        );
                    }

                    // derive the metadata offset
                    let offset_metadata = match &layout.fields {
                        FieldsShape::Arbitrary { offsets, memory_index: _ } => {
                            if offsets.len() != 2 {
                                bug!("[invariant] expect two fields for ptr/ref type {ty}");
                            }
                            let mut offsets_iter = offsets.into_iter();
                            let off_pointer = *offsets_iter.next().unwrap();
                            let off_metadata = *offsets_iter.next().unwrap();
                            if off_pointer.bytes_usize() != 0 || off_metadata != pointer_size {
                                bug!("[invariant] unexpected offsets for ptr/ref type {ty}");
                            }
                            off_metadata
                        }
                        _ => bug!("[invariant] expect arbitrary layout for ptr/ref type {ty}"),
                    };

                    // read the metadata
                    let metadata = match read_primitive(
                        self.tcx,
                        offset + offset_metadata,
                        pointer_size,
                        false,
                    ) {
                        Scalar::Int(scalar_int) => scalar_int.to_target_usize(self.tcx) as usize,
                        Scalar::Ptr(..) => {
                            bug!("[invariant] unexpected pointer scalar for metadata for type {ty}")
                        }
                    };
                    let ptr_metadata = SolPtrMetadata(metadata);

                    // now create the pointer constant
                    match pointer_parsed {
                        None => {
                            if metadata != 0 {
                                bug!("[invariant] unexpected non-zero metadata for null pointers");
                            }
                            match mutability {
                                Mutability::Not => SolConst::PtrMetaSizedNullImm,
                                Mutability::Mut => SolConst::PtrMetaSizedNullMut,
                            }
                        }
                        Some((parsed_prov, offset)) => {
                            // construct the constant based on mutability and whether it is a raw pointer
                            match (parsed_prov, ty.is_raw_ptr()) {
                                (SolProvenance::Const(val), true) => {
                                    SolConst::PtrMetaSizedConst(ptr_metadata, val.into(), offset)
                                }
                                (SolProvenance::Const(val), false) => {
                                    SolConst::RefMetaSizedConst(ptr_metadata, val.into(), offset)
                                }
                                (SolProvenance::VarImm(ident), true) => {
                                    SolConst::PtrMetaSizedVarImm(ptr_metadata, ident, offset)
                                }
                                (SolProvenance::VarImm(ident), false) => {
                                    SolConst::RefMetaSizedVarImm(ptr_metadata, ident, offset)
                                }
                                (SolProvenance::VarMut(ident), true) => {
                                    SolConst::PtrMetaSizedVarMut(ptr_metadata, ident, offset)
                                }
                                (SolProvenance::VarMut(ident), false) => {
                                    SolConst::RefMetaSizedVarMut(ptr_metadata, ident, offset)
                                }
                            }
                        }
                    }
                } else {
                    // sanity check on the shape of the fields
                    if pointer_size != layout.size {
                        bug!(
                            "[invariant] invalid layout size of ptr/ref type {ty}, expect no metadata but got {}",
                            layout.size.bytes_usize()
                        );
                    }
                    if !matches!(layout.fields, FieldsShape::Primitive) {
                        bug!("[invariant] expect a primitive field for pointer type {ty}");
                    }

                    // now create the pointer constant
                    match pointer_parsed {
                        None => match mutability {
                            Mutability::Not => SolConst::PtrSizedNullImm,
                            Mutability::Mut => SolConst::PtrSizedNullMut,
                        },
                        Some((parsed_prov, offset)) => {
                            // construct the constant based on mutability and whether it is a raw pointer
                            match (parsed_prov, ty.is_raw_ptr()) {
                                (SolProvenance::Const(val), true) => {
                                    SolConst::PtrSizedConst(val.into(), offset)
                                }
                                (SolProvenance::Const(val), false) => {
                                    SolConst::RefSizedConst(val.into(), offset)
                                }
                                (SolProvenance::VarImm(ident), true) => {
                                    SolConst::PtrSizedVarImm(ident, offset)
                                }
                                (SolProvenance::VarImm(ident), false) => {
                                    SolConst::RefSizedVarImm(ident, offset)
                                }
                                (SolProvenance::VarMut(ident), true) => {
                                    SolConst::PtrSizedVarMut(ident, offset)
                                }
                                (SolProvenance::VarMut(ident), false) => {
                                    SolConst::RefSizedVarMut(ident, offset)
                                }
                            }
                        }
                    }
                }
            }

            // unsupported
            ty::FnPtr(..) => {
                // sanity check on the variants
                match &layout.variants {
                    Variants::Single { index } if index.index() == 0 => {}
                    _ => bug!("[invariant] expect single 0-variant for fn-ptr type {ty}"),
                }

                // read and parse the pointer
                let pointer_size = self.tcx.data_layout().pointer_size();
                if layout.size != pointer_size {
                    bug!("[invariant] unexpected layout size for function pointer type {ty}");
                }
                match read_primitive(self.tcx, offset, pointer_size, true) {
                    Scalar::Ptr(ptr, _) => {
                        let (prov, offset) = ptr.into_raw_parts();
                        if offset != Size::ZERO {
                            bug!("[invariant] unexpected non-zero offset for function pointer");
                        }

                        match self.tcx.global_alloc(prov.alloc_id()) {
                            GlobalAlloc::Function { instance } => {
                                let (kind, ident, ty_args) = self.make_instance(instance);
                                SolConst::FuncPtr(kind, ident, ty_args)
                            }
                            _ => {
                                bug!("[invariant] unexpected allocation for function pointer");
                            }
                        }
                    }
                    Scalar::Int(int) => {
                        if !int.is_null() {
                            bug!("[invariant] unexpected non-null integer for function pointer");
                        }
                        SolConst::FuncPtrNull
                    }
                }
            }

            // vector-alike
            ty::Array(elem_ty, _) => {
                match &layout.variants {
                    Variants::Single { index } if index.index() == 0 => {}
                    _ => bug!("[invariant] expect single 0-variant for array type {ty}"),
                }
                match &layout.fields {
                    FieldsShape::Array { stride, count } => {
                        let mut elements = vec![];
                        for i in 0..*count {
                            let elem_offset = offset + *stride * i;
                            let (_, elem) = self.read_const_from_memory_and_layout(
                                memory,
                                elem_offset,
                                *elem_ty,
                            );
                            elements.push(elem);
                        }
                        SolConst::Array(elements)
                    }
                    _ => bug!("[invariant] expect array field layout for array type {ty}"),
                }
            }
            ty::Slice(elem_ty) => {
                match &layout.variants {
                    Variants::Single { index } if index.index() == 0 => {}
                    _ => bug!("[invariant] expect single 0-variant for slice type {ty}"),
                }
                match &layout.fields {
                    FieldsShape::Array { stride, count } => {
                        let mut elements = vec![];
                        for i in 0..*count {
                            let elem_offset = offset + *stride * i;
                            let (_, elem) = self.read_const_from_memory_and_layout(
                                memory,
                                elem_offset,
                                *elem_ty,
                            );
                            elements.push(elem);
                        }
                        SolConst::Slice(elements)
                    }
                    _ => bug!("[invariant] expect an array field layout for slice type {ty}"),
                }
            }
            ty::Str => {
                match &layout.variants {
                    Variants::Single { index } if index.index() == 0 => {}
                    _ => bug!("[invariant] expect single 0-variant for str type"),
                }
                match &layout.fields {
                    FieldsShape::Array { stride, count } => {
                        let range = AllocRange { start: Size::ZERO, size: *stride * *count };
                        let num_prov = memory.provenance().get_range(&self.tcx, range).count();
                        if num_prov != 0 {
                            bug!("[invariant] string memory contains provenance");
                        }

                        let bytes = memory
                            .get_bytes_strip_provenance(&self.tcx, range)
                            .unwrap_or_else(|e| {
                                bug!("[invariant] failed to read bytes for string memory: {e:?}");
                            });
                        SolConst::String(String::from_utf8(bytes.to_vec()).unwrap_or_else(|e| {
                            bug!("[invariant] non utf-8 string memory: {e}");
                        }))
                    }
                    _ => bug!("[invariant] expect an array field for str type"),
                }
            }

            // packed
            ty::Tuple(elem_tys) => {
                match &layout.variants {
                    Variants::Single { index } if index.index() == 0 => {}
                    _ => bug!("[invariant] expect a single 0-variant for tuple type {ty}"),
                }
                match &layout.fields {
                    FieldsShape::Arbitrary { offsets, memory_index: _ } => {
                        if offsets.len() != elem_tys.len() {
                            bug!("[invariant] field count mismatch for tuple type {ty}");
                        }

                        let mut elements = vec![];
                        for (i, elem_ty) in elem_tys.iter().enumerate() {
                            let field_idx = FieldIdx::from_usize(i);
                            let field_offset = *offsets.get(field_idx).unwrap_or_else(|| {
                                bug!("[invariant] no offset for field {i} in tuple type {ty}");
                            });
                            let (_, elem) = self.read_const_from_memory_and_layout(
                                memory,
                                offset + field_offset,
                                elem_ty,
                            );
                            elements.push(elem);
                        }
                        SolConst::Tuple(elements)
                    }
                    _ => bug!("[invariant] expect an arbitrary field for tuple type {ty}"),
                }
            }

            ty::Adt(def, ty_args) => match def.adt_kind() {
                AdtKind::Struct => {
                    let field_details = def
                        .non_enum_variant()
                        .fields
                        .iter_enumerated()
                        .map(|(field_idx, field_def)| {
                            (field_idx, field_def.name, field_def.ty(self.tcx, ty_args))
                        })
                        .collect::<Vec<_>>();

                    match &layout.variants {
                        Variants::Single { index } if index.index() == 0 => {}
                        _ => bug!("[invariant] expect a single 0-variant for array type {ty}"),
                    }
                    match &layout.fields {
                        FieldsShape::Arbitrary { offsets, memory_index: _ } => {
                            if offsets.len() != field_details.len() {
                                bug!("[invariant] field count mismatch for struct type {ty}");
                            }

                            let mut elements = vec![];
                            for (field_idx, field_name, field_ty) in field_details {
                                let field_offset = *offsets.get(field_idx).unwrap_or_else(|| {
                                    bug!(
                                        "[invariant] no offset for field {field_name} in struct type {ty}",
                                    );
                                });
                                let (_, elem) = self.read_const_from_memory_and_layout(
                                    memory,
                                    offset + field_offset,
                                    field_ty,
                                );
                                elements.push((SolFieldIndex(field_idx.index()), elem));
                            }
                            SolConst::Struct(elements)
                        }
                        _ => bug!("[invariant] expect an arbitrary field for struct type {ty}"),
                    }
                }
                AdtKind::Union => {
                    match &layout.variants {
                        Variants::Single { index } if index.index() == 0 => {}
                        _ => bug!("[invariant] expect a single 0-variant for union type {ty}"),
                    }
                    match &layout.fields {
                        FieldsShape::Union(..) => bug!("[unsupported] union constant in memory"),
                        _ => bug!("[invariant] expect a union field for union type {ty}"),
                    }
                }
                AdtKind::Enum => {
                    let (variant_idx, variant_layout) = match &layout.variants {
                        Variants::Multiple { tag, tag_encoding, tag_field, variants } => {
                            // get the tag type
                            let tag_type = match tag {
                                ScalarAbi::Initialized { value, valid_range: _ } => value,
                                ScalarAbi::Union { .. } => {
                                    bug!("[invariant] unexpected tag specification for type {ty}");
                                }
                            };

                            // get the offset for the tag field
                            let tag_offset =  match &layout.fields {
                                FieldsShape::Arbitrary { offsets, memory_index: _ } => {
                                    *offsets.get(*tag_field).unwrap_or_else(|| {
                                        bug!("[invariant] no offset for tag field in constant for type {ty}");
                                    })
                                }
                                _ => bug!("[invariant] tagged variants does not have field specification"),
                            };

                            // parse the tag value
                            let tag_size = tag_type.size(&self.tcx);
                            let tag_value = match read_primitive(
                                self.tcx,
                                offset + tag_offset,
                                tag_size,
                                matches!(tag_type, Primitive::Pointer(_)),
                            ) {
                                Scalar::Int(val) => val.to_uint(tag_size),
                                Scalar::Ptr(..) => bug!("[unsupported] non-null pointer as tag"),
                            };

                            // derive variant index
                            let variant_idx = match tag_encoding {
                                TagEncoding::Direct => {
                                    // look for a variant with the same discriminant value
                                    let mut last_discr_value = 0;
                                    let mut variant_idx_found = None;
                                    for (variant_idx, variant) in def.variants().iter_enumerated() {
                                        let discr_value = match variant.discr {
                                            VariantDiscr::Relative(pos) => {
                                                last_discr_value + pos as u128
                                            }
                                            VariantDiscr::Explicit(did) => {
                                                let discr = def.eval_explicit_discr(self.tcx, did).unwrap_or_else(|_| {
                                                    bug!("[invariant] failed to evaluate discriminant for enum {ty}");
                                                });
                                                last_discr_value = discr.val;
                                                discr.val
                                            }
                                        };
                                        if discr_value == tag_value {
                                            variant_idx_found = Some(variant_idx);
                                            break;
                                        }
                                    }
                                    variant_idx_found.unwrap_or_else(|| {
                                        bug!("[invariant] invalid discriminant {tag_value} for enum type {ty}");
                                    })
                                }
                                TagEncoding::Niche {
                                    untagged_variant,
                                    niche_variants,
                                    niche_start,
                                } => {
                                    // NOTE: the discriminant and variant index of each variant coincide
                                    let tag_index = tag_value
                                        .wrapping_sub(*niche_start)
                                        .wrapping_add(niche_variants.start().index() as u128);
                                    if tag_index >= def.variants().len() as u128 {
                                        *untagged_variant
                                    } else {
                                        VariantIdx::from_usize(tag_index as usize)
                                    }
                                }
                            };

                            // get variant layout
                            let variant_layout = variants.get(variant_idx).unwrap_or_else(|| {
                                bug!(
                                    "[invariant] invalid variant index {} for type {ty}",
                                    variant_idx.index()
                                )
                            });

                            // return both the index and the layout
                            (variant_idx, variant_layout)
                        }
                        Variants::Empty | Variants::Single { .. } => {
                            bug!("[invariant] expect multiple variants for enum type {ty}")
                        }
                    };

                    // get the variant definition from typing
                    let variant_def = def.variant(variant_idx);

                    // sanity check on the variant layout
                    if !matches!(variant_layout.variants, Variants::Single { index } if index == variant_idx)
                    {
                        bug!(
                            "[invariant] expect single indexed({})-variant for enum variant of type {ty}",
                            variant_idx.index()
                        );
                    }

                    // get variant fields from layout
                    let field_offsets = match &variant_layout.fields {
                        FieldsShape::Arbitrary { offsets, memory_index: _ } => {
                            if offsets.len() != variant_def.fields.len() {
                                bug!(
                                    "[invariant] field count mismatch for variant {} in type {ty}",
                                    variant_def.name
                                );
                            }
                            offsets
                        }
                        _ => bug!(
                            "[invariant] expect an arbitrary field for enum variant of type {ty}"
                        ),
                    };

                    // construct values for the variant fields
                    let mut elements = vec![];
                    for (field_idx, field_def) in variant_def.fields.iter_enumerated() {
                        let field_offset = *field_offsets.get(field_idx).unwrap_or_else(|| {
                            bug!(
                                "[invariant] no offset for field {} in enum type {ty} variant {}",
                                field_def.name,
                                variant_def.name
                            );
                        });
                        let (_, elem) = self.read_const_from_memory_and_layout(
                            memory,
                            offset + field_offset,
                            field_def.ty(self.tcx, ty_args),
                        );
                        elements.push((SolFieldIndex(field_idx.index()), elem));
                    }
                    SolConst::Enum(SolVariantIndex(variant_idx.index()), elements)
                }
            },

            // unexpected
            ty::Never
            | ty::Pat(..)
            | ty::FnDef(..)
            | ty::Closure(..)
            | ty::Dynamic(..)
            | ty::Coroutine(..)
            | ty::CoroutineClosure(..)
            | ty::CoroutineWitness(..)
            | ty::Foreign(..)
            | ty::Alias(..)
            | ty::Param(..)
            | ty::Bound(..)
            | ty::UnsafeBinder(..)
            | ty::Infer(..)
            | ty::Placeholder(..)
            | ty::Error(..) => {
                bug!("[invariant] unexpected type in memory allocation: {ty}");
            }
        };

        // return both the size and the parsed constant
        (layout.size, parsed)
    }

    /// Create a constant known in the value system together with its type
    fn mk_val_const_with_ty(&mut self, val_const: ConstValue, ty: Ty<'tcx>) -> SolConst {
        match val_const {
            ConstValue::ZeroSized => self.mk_val_const_from_zst(ty),
            ConstValue::Scalar(scalar) => match ty.kind() {
                // value types
                ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::Adt(..) => {
                    let val = match scalar {
                        Scalar::Int(scalar_int) => scalar_int,
                        Scalar::Ptr(..) => {
                            bug!("[invariant] unexpected pointer scalar for value type {ty}");
                        }
                    };
                    self.mk_val_const_from_scalar_val(ty, val)
                }

                // pointers or references
                ty::Ref(_, sub_ty, mutability) | ty::RawPtr(sub_ty, mutability) => {
                    // we only know how to convert a scalar to a sized reference/pointer for now
                    if !sub_ty.is_sized(self.tcx, TypingEnv::fully_monomorphized()) {
                        bug!("[unsupported] unsized pointee type {sub_ty} as const scalar");
                    }

                    // get the pointer (or possibly null if RawPtr type)
                    let ptr = match scalar {
                        Scalar::Ptr(scalar_ptr, _) => scalar_ptr,
                        Scalar::Int(scalar_int) => {
                            if !ty.is_raw_ptr() {
                                bug!("[invariant] unexpected int scalar for non-ptr type {ty}");
                            }
                            if !scalar_int.is_null() {
                                bug!("[invariant] unexpected non-null for pointer type {ty}");
                            }

                            // shortcut and early return for null pointers
                            return match mutability {
                                Mutability::Not => SolConst::PtrSizedNullImm,
                                Mutability::Mut => SolConst::PtrSizedNullMut,
                            };
                        }
                    };

                    let (prov, offset) = ptr.into_raw_parts();
                    let offset = SolOffset(offset.bytes_usize());

                    // try to probe the memory behind the pointer
                    let parsed_prov = self.make_provenance(prov, *sub_ty, *mutability);

                    // construct the constant based on mutability and whether it is a raw pointer
                    match (parsed_prov, ty.is_raw_ptr()) {
                        (SolProvenance::Const(val), true) => {
                            SolConst::PtrSizedConst(val.into(), offset)
                        }
                        (SolProvenance::Const(val), false) => {
                            SolConst::RefSizedConst(val.into(), offset)
                        }
                        (SolProvenance::VarImm(ident), true) => {
                            SolConst::PtrSizedVarImm(ident, offset)
                        }
                        (SolProvenance::VarImm(ident), false) => {
                            SolConst::RefSizedVarImm(ident, offset)
                        }
                        (SolProvenance::VarMut(ident), true) => {
                            SolConst::PtrSizedVarMut(ident, offset)
                        }
                        (SolProvenance::VarMut(ident), false) => {
                            SolConst::RefSizedVarMut(ident, offset)
                        }
                    }
                }

                // function pointers
                ty::FnPtr(..) => {
                    bug!("[unsupported] function pointer for scalar constant");
                }

                // all others
                _ => {
                    bug!("[assumption] unexpected type {ty} for scalar {scalar}");
                }
            },
            ConstValue::Slice { alloc_id, meta } => {
                let memory = match self.tcx.global_alloc(alloc_id) {
                    GlobalAlloc::Memory(a) => a.inner(),
                    _ => bug!("[invariant] slice const should be of the memory allocation type"),
                };
                if !matches!(memory.mutability, Mutability::Not) {
                    bug!("[invariant] slice const refers to a mutable memory allocation");
                }

                let slice_ty = match ty.kind() {
                    ty::Ref(_, sub_ty, Mutability::Not) => sub_ty,
                    _ => bug!("[invariant] unexpected type for slice const: {ty}"),
                };
                match slice_ty.kind() {
                    ty::Slice(elem_ty) => {
                        let mut elements = vec![];
                        let mut offset = Size::ZERO;
                        for _ in 0..meta {
                            let (elem_size, elem) =
                                self.read_const_from_memory_and_layout(memory, offset, *elem_ty);
                            elements.push(elem);
                            offset += elem_size;
                        }
                        SolConst::RefSlice(elements)
                    }
                    ty::Str => {
                        let range = AllocRange { start: Size::ZERO, size: Size::from_bytes(meta) };
                        let num_prov = memory.provenance().get_range(&self.tcx, range).count();
                        if num_prov != 0 {
                            bug!("[invariant] string constant contains provenance");
                        }

                        let bytes = memory
                            .get_bytes_strip_provenance(&self.tcx, range)
                            .unwrap_or_else(|e| {
                                bug!("[invariant] failed to read bytes for string constant: {e:?}");
                            });
                        SolConst::RefString(String::from_utf8(bytes.to_vec()).unwrap_or_else(|e| {
                            bug!("[invariant] non utf-8 string constant: {e}");
                        }))
                    }
                    _ => bug!("[invariant] unexpected type for slice const: {ty}"),
                }
            }
            ConstValue::Indirect { alloc_id, offset } => {
                let memory = match self.tcx.global_alloc(alloc_id) {
                    GlobalAlloc::Memory(a) => a.inner(),
                    _ => bug!("[invariant] indirect const should be of the memory allocation type"),
                };
                if !matches!(memory.mutability, Mutability::Not) {
                    bug!("[invariant] indirect const refers to a mutable memory allocation");
                }
                let (_, parsed) = self.read_const_from_memory_and_layout(memory, offset, ty);
                parsed
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
                            bug!("[invariant] type constant mismatches its value type: {op_const}");
                        }
                        const_val
                    }
                };
                SolOpConst::Type(const_ty, const_val)
            }
            OpConst::Val(val_const, ty) => {
                let const_ty = self.mk_type(ty);

                let normalized_ty =
                    self.tcx.normalize_erasing_regions(TypingEnv::fully_monomorphized(), ty);
                let const_val = self.mk_val_const_with_ty(val_const, normalized_ty);

                SolOpConst::Value(const_ty, const_val)
            }
            OpConst::Unevaluated(unevaluated, ty) => {
                let val_const = self
                    .tcx
                    .const_eval_resolve(TypingEnv::fully_monomorphized(), unevaluated, DUMMY_SP)
                    .unwrap_or_else(|e| {
                        bug!("[invariant] unable to resolve unevaluated constant {op_const}: {e:?}")
                    });

                let const_ty = self.mk_type(ty);

                let normalized_ty =
                    self.tcx.normalize_erasing_regions(TypingEnv::fully_monomorphized(), ty);
                let const_val = self.mk_val_const_with_ty(val_const, normalized_ty);

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
            PlaceElem::Downcast(sym_opt, variant) => {
                let symbol = match sym_opt {
                    None => bug!("[invariant] missing variant symbol for downcast projection"),
                    Some(sym) => SolVariantName(sym.to_ident_string()),
                };
                let variant_index = SolVariantIndex(variant.index());
                SolProjection::Downcast { symbol, variant: variant_index }
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
                    CastKind::PtrToPtr => SolOpcodeCast::PtrToPtr,
                    CastKind::FnPtrToPtr => SolOpcodeCast::FnPtrToPtr,
                    CastKind::Transmute => SolOpcodeCast::Transmute,
                    CastKind::PointerCoercion(coercion_type, _) => match coercion_type {
                        PointerCoercion::UnsafeFnPointer => SolOpcodeCast::SafeToUnsafe,
                        PointerCoercion::Unsize => SolOpcodeCast::Unsize,
                        PointerCoercion::ReifyFnPointer | PointerCoercion::ClosureFnPointer(_) => {
                            SolOpcodeCast::ReifyFnPtr
                        }
                        PointerCoercion::ArrayToPointer | PointerCoercion::MutToConstPointer => {
                            bug!("[invariant] unexpected pointer coercion in cast: {cast_kind:?}");
                        }
                    },
                    CastKind::PointerExposeProvenance => SolOpcodeCast::PtrToAddr,
                    CastKind::PointerWithExposedProvenance => SolOpcodeCast::AddrToPtr,
                };
                SolExpr::Cast { opcode, place: self.mk_operand(operand), ty: self.mk_type(*ty) }
            }
            Rvalue::ShallowInitBox(operand, ty) => SolExpr::Cast {
                opcode: SolOpcodeCast::Boxify,
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
            Rvalue::ThreadLocalRef(def_id) => {
                let ident = self.mk_ident(*def_id);
                // FIXME: the TLS support is currently incomplete
                SolExpr::ThreadLocalRef(ident)
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
                for (builtin, regex) in self.builtin_fns.iter() {
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
                    warn!("{}-- external dependency: {def_desc}", self.depth);
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

    fn make_provenance(
        &mut self,
        prov: CtfeProvenance,
        ty: Ty<'tcx>,
        mutability: Mutability,
    ) -> SolProvenance {
        match self.tcx.global_alloc(prov.alloc_id()) {
            GlobalAlloc::Memory(allocated) => {
                if !matches!(mutability, Mutability::Not) {
                    bug!("[assumption] mutable memory allocation in provenance");
                }

                let (_, const_val) =
                    self.read_const_from_memory_and_layout(allocated.inner(), Size::ZERO, ty);
                SolProvenance::Const(const_val)
            }
            GlobalAlloc::Static(def_id) => {
                let ident = self.mk_ident(def_id);

                // check if we have an initializer already
                if !self.globals.contains_key(&ident) {
                    let initializer =
                        self.tcx.eval_static_initializer(def_id).unwrap_or_else(|_| {
                            bug!(
                                "[invariant] unable to evaluate static initializer for {}",
                                self.tcx.def_path_str(def_id)
                            )
                        });

                    let init_ty = self.mk_type(ty);
                    let (_, init_val) =
                        self.read_const_from_memory_and_layout(initializer.inner(), Size::ZERO, ty);
                    self.globals.insert(ident.clone(), (init_ty, init_val));
                }

                // split by mutability
                match mutability {
                    Mutability::Not => SolProvenance::VarImm(ident),
                    Mutability::Mut => SolProvenance::VarMut(ident),
                }
            }
            GlobalAlloc::Function { .. } => bug!("[invariant] unexpected function in provenance"),
            GlobalAlloc::VTable(..) => bug!("[unsupported] vtable in provenance"),
            GlobalAlloc::TypeId { .. } => bug!("[unsupported] type id in provenance"),
        }
    }

    /// Build the context
    pub(crate) fn build(self) -> (SolEnv, SolContext) {
        // check we are balanced on stack
        if self.depth.level != 0 {
            bug!("[invariant] depth stack is not balanced");
        }

        // unpack the fields
        let krate = SolCrateName(self.tcx.crate_name(LOCAL_CRATE).to_ident_string());

        let mut id_desc = vec![];

        // we don't care about the ordering of the values
        #[allow(rustc::potential_query_instability)]
        for (ident, desc) in self.id_cache.into_values() {
            id_desc.push((ident, desc));
        }

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
        for (ident, (ty, val)) in self.globals.into_iter() {
            globals.push((ident, ty, val));
        }

        let mut dep_fns = vec![];
        for (ident, l1) in self.dep_fns.into_iter() {
            for (mono, l2) in l1.into_iter() {
                for (kind, desc) in l2.into_iter() {
                    dep_fns.push((kind, ident.clone(), mono.clone(), desc));
                }
            }
        }

        (self.sol, SolContext { krate, id_desc, ty_defs, fn_defs, globals, dep_fns })
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
    pub(crate) id_desc: Vec<(SolIdent, SolPathDesc)>,
    pub(crate) ty_defs: Vec<(SolIdent, Vec<SolGenericArg>, SolTyDef)>,
    pub(crate) fn_defs: Vec<(SolInstanceKind, SolIdent, Vec<SolGenericArg>, SolFnDef)>,
    pub(crate) globals: Vec<(SolIdent, SolType, SolConst)>,
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

/// A variant discrimanant
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolVariantDiscr(pub(crate) u128);

/// A slot number for locals
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolLocalSlot(pub(crate) usize);

/// A basic block index
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolBlockId(pub(crate) usize);

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
    Dynamic(Vec<(SolIdent, Vec<SolGenericArg>, Option<SolTyTerm>)>),
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
    pub(crate) discr: SolVariantDiscr,
    pub(crate) fields: Vec<SolField>,
}

/// A constant known in the type system
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolTyConst {
    Simple { ty: SolType, val: SolConst },
}

/// A term known in the type system
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolTyTerm {
    Type(SolType),
    Const(SolTyConst),
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
    Downcast { symbol: SolVariantName, variant: SolVariantIndex },
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
    ThreadLocalRef(SolIdent),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolOpcodeCast {
    IntToInt,
    FloatToFloat,
    IntToFloat,
    FloatToInt,
    PtrToPtr,
    FnPtrToPtr,
    Transmute,
    ReifyFnPtr,
    SafeToUnsafe,
    Unsize,
    Boxify,
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
pub(crate) struct SolOffset(pub(crate) usize);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolPtrMetadata(pub(crate) usize);

/// A provenance origin
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolProvenance {
    Const(SolConst),
    VarImm(SolIdent),
    VarMut(SolIdent),
}

/*
 * Constant
 */

/// An integer constant
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolConstInt {
    I8(i8),
    U8(u8),
    I16(i16),
    U16(u16),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    I128(i128),
    U128(u128),
    Isize(isize),
    Usize(usize),
}

/// A floating-point constant
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolConstFloat {
    F16(String),
    F32(String),
    F64(String),
    F128(String),
}

/// A constant
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolConst {
    /* data types */
    Bool(bool),
    Char(char),
    Int(SolConstInt),
    Float(SolConstFloat),
    Array(Vec<SolConst>),
    Tuple(Vec<SolConst>),
    Struct(Vec<(SolFieldIndex, SolConst)>),
    Union(SolFieldIndex, Box<SolConst>),
    Enum(SolVariantIndex, Vec<(SolFieldIndex, SolConst)>),

    /* proxy data */
    Slice(Vec<SolConst>),
    String(String),

    /* code types */
    FuncDef(SolInstanceKind, SolIdent, Vec<SolGenericArg>),
    Closure(SolInstanceKind, SolIdent, Vec<SolGenericArg>),

    /* references and pointers */
    RefSlice(Vec<SolConst>),
    RefString(String),

    RefSizedConst(Box<SolConst>, SolOffset),
    RefSizedVarImm(SolIdent, SolOffset),
    RefSizedVarMut(SolIdent, SolOffset),
    RefMetaSizedConst(SolPtrMetadata, Box<SolConst>, SolOffset),
    RefMetaSizedVarImm(SolPtrMetadata, SolIdent, SolOffset),
    RefMetaSizedVarMut(SolPtrMetadata, SolIdent, SolOffset),

    /* pointers */
    PtrSizedConst(Box<SolConst>, SolOffset),
    PtrSizedVarImm(SolIdent, SolOffset),
    PtrSizedVarMut(SolIdent, SolOffset),
    PtrMetaSizedConst(SolPtrMetadata, Box<SolConst>, SolOffset),
    PtrMetaSizedVarImm(SolPtrMetadata, SolIdent, SolOffset),
    PtrMetaSizedVarMut(SolPtrMetadata, SolIdent, SolOffset),

    PtrSizedNullImm,
    PtrSizedNullMut,
    PtrMetaSizedNullImm,
    PtrMetaSizedNullMut,

    /* special */
    PtrAddress(usize),

    /* function pointers */
    FuncPtrNull,
    FuncPtr(SolInstanceKind, SolIdent, Vec<SolGenericArg>),
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
    /* panics */
    IntrinsicsAssertFailedInner,
    IntrinsicsPanic,
    IntrinsicsPanicNounwind,
    IntrinsicsPanicFmt,
    IntrinsicsPanicNounwindFmt,
    IntrinsicsResultUnwrapFailed,
    /* alloc */
    AllocGlobalAllocImpl,
    AllocRustAlloc,
    AllocRustAllocZeroed,
    AllocRustRealloc,
    AllocRustDealloc,
    AllocExchangeMalloc,
    AllocHandleAllocError,
    AllocRawVecHandleError,
    LayoutIsSizeAlignValid,
    /* formatter */
    StdFmtWrite,
    DebugFmt,
    SpecToString,
    /* solana */
    SolInvokeSigned,
}

/*
 * Platform-specifics
 */

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolAnchorInstruction {
    pub(crate) function: SolIdent,
    pub(crate) ty_state: SolIdent,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolSplTestEntrypoint {
    pub(crate) function: SolIdent,
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
            /* panics */
            Self::IntrinsicsAssertFailedInner => r"panicking::assert_failed_inner",
            Self::IntrinsicsPanic => r"panic",
            Self::IntrinsicsPanicNounwind => r"panic_nounwind",
            Self::IntrinsicsPanicFmt => r"panic_fmt",
            Self::IntrinsicsPanicNounwindFmt => r"panic_nounwind_fmt",
            Self::IntrinsicsResultUnwrapFailed => r"result::unwrap_failed",
            /* alloc */
            Self::AllocGlobalAllocImpl => r"std::alloc::Global::alloc_impl",
            Self::AllocRustAlloc => r"alloc::alloc::__rust_alloc",
            Self::AllocRustAllocZeroed => r"alloc::alloc::__rust_alloc_zeroed",
            Self::AllocRustRealloc => r"alloc::alloc::__rust_realloc",
            Self::AllocRustDealloc => r"alloc::alloc::__rust_dealloc",
            Self::AllocExchangeMalloc => r"alloc::alloc::exchange_malloc",
            Self::AllocHandleAllocError => r"handle_alloc_error",
            Self::AllocRawVecHandleError => r"alloc::raw_vec::handle_error",
            Self::LayoutIsSizeAlignValid => r"Layout::is_size_align_valid",
            /* formatter */
            Self::StdFmtWrite => r"std::fmt::write",
            Self::DebugFmt => r"<.* as Debug>::fmt",
            Self::SpecToString => r"<.* as string::SpecToString>::spec_to_string",
            /* solana */
            Self::SolInvokeSigned => r"sol_invoke_signed",
        };

        Regex::new(pattern).unwrap_or_else(|e| {
            bug!("[invariant] failed to compile regex for builtin function: {e}")
        })
    }

    fn all() -> Vec<Self> {
        vec![
            /* panics */
            Self::IntrinsicsAssertFailedInner,
            Self::IntrinsicsPanic,
            Self::IntrinsicsPanicNounwind,
            Self::IntrinsicsPanicFmt,
            Self::IntrinsicsPanicNounwindFmt,
            Self::IntrinsicsResultUnwrapFailed,
            /* alloc */
            Self::AllocGlobalAllocImpl,
            Self::AllocRustAlloc,
            Self::AllocRustAllocZeroed,
            Self::AllocRustRealloc,
            Self::AllocRustDealloc,
            Self::AllocExchangeMalloc,
            Self::AllocHandleAllocError,
            Self::AllocRawVecHandleError,
            Self::LayoutIsSizeAlignValid,
            /* formatter */
            Self::StdFmtWrite,
            Self::DebugFmt,
            Self::SpecToString,
            /* solana */
            Self::SolInvokeSigned,
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
