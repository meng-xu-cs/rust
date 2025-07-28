use std::collections::BTreeMap;
use std::fmt::Display;

use rustc_ast::Mutability;
use rustc_middle::bug;
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::ty::{
    self, AdtDef, AdtKind, Const as TyConst, ConstKind as TyConstKind, FloatTy, GenericArg,
    GenericArgKind, GenericArgsRef, Instance, IntTy, Ty, TyCtxt, TypingEnv, UintTy,
};
use rustc_span::def_id::DefId;
use serde::{Deserialize, Serialize};
use tracing::info;

/// A builder for creating a Solana context
pub(crate) struct SolContextBuilder<'tcx> {
    /// compiler context
    tcx: TyCtxt<'tcx>,

    /// depth of the current context
    depth: Depth,

    /// collected definitions of datatypes
    ty_defs: BTreeMap<SolIdent, BTreeMap<Vec<SolGenericArg>, Option<SolTyDef>>>,

    /// collected definitions of functions
    fn_defs: BTreeMap<SolIdent, BTreeMap<Vec<SolGenericArg>, Option<()>>>,
}

impl<'tcx> SolContextBuilder<'tcx> {
    /// Create a new context builder
    pub(crate) fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self { tcx, depth: Depth::new(), ty_defs: BTreeMap::new(), fn_defs: BTreeMap::new() }
    }

    /// Create an identifier
    fn mk_ident(&mut self, def_id: DefId) -> SolIdent {
        let def_path = self.tcx.def_path(def_id);
        let krate = self.tcx.crate_name(def_path.krate).to_string();
        let paths = def_path.data.iter().map(|segment| segment.as_sym(false).to_string()).collect();
        SolIdent { krate, paths }
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
            ty::Pat(..) => {
                bug!("[unsupported] pattern type: {ty}");
            }
            ty::Dynamic(..) => {
                bug!("[unsupported] dynamic (dyn trait) type: {ty}");
            }
            ty::Coroutine(..) | ty::CoroutineClosure(..) | ty::CoroutineWitness(..) => {
                bug!("[unsupported] coroutine-related type: {ty}");
            }
            ty::FnPtr(..) => {
                bug!("[assumption] unexpected function pointer type: {ty}");
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
            ty::Never | ty::Infer(..) | ty::Placeholder(..) | ty::Error(..) => {
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
                        let field_name = SolFieldName(field_def.name.to_string());
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
                        let field_name = SolFieldName(field_def.name.to_string());
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
                        let variant_name = SolVariantName(variant_def.name.to_string());
                        let fields = variant_def
                            .fields
                            .iter_enumerated()
                            .map(|(field_idx, field_def)| {
                                let field_index = SolFieldIndex(field_idx.index());
                                let field_name = SolFieldName(field_def.name.to_string());
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
    ) -> (SolIdent, Vec<SolGenericArg>) {
        let ident = self.mk_ident(def_id);
        let ty_args: Vec<_> = generics.iter().map(|ty_arg| self.mk_ty_arg(ty_arg)).collect();

        // if already defined or is being defined, return the key
        if self.fn_defs.get(&ident).map_or(false, |inner| inner.contains_key(&ty_args)) {
            return (ident, ty_args);
        }

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

        // mark start
        self.depth.push();
        info!("{}-> function definition", self.depth);

        // first update the entry to mark that function definition in progress
        self.fn_defs.entry(ident.clone()).or_default().insert(ty_args.clone(), None);

        // TODO

        // update the function definition
        self.fn_defs.entry(ident.clone()).or_default().insert(ty_args.clone(), Some(fn_def));

        // mark end
        info!("{}<- function definition", self.depth);
        self.depth.pop();

        // return the key to the lookup table
        (ident, ty_args)
    }

    fn make_type_closure(
        &mut self,
        def_id: DefId,
        generics: GenericArgsRef<'tcx>,
    ) -> (SolIdent, Vec<SolGenericArg>) {
        let ident = self.mk_ident(def_id);
        let ty_args: Vec<_> = generics.iter().map(|ty_arg| self.mk_ty_arg(ty_arg)).collect();

        // if already defined or is being defined, return the key
        if self.fn_defs.get(&ident).map_or(false, |inner| inner.contains_key(&ty_args)) {
            return (ident, ty_args);
        }

        // resolve the closure instance
        let closure_ty_args = generics.as_closure();
        let instance =
            Instance::resolve_closure(self.tcx, def_id, generics, closure_ty_args.kind());

        // mark start
        self.depth.push();
        info!("{}-> closure definition", self.depth);

        // first update the entry to mark that closure definition in progress
        self.fn_defs.entry(ident.clone()).or_default().insert(ty_args.clone(), None);

        // TODO

        // update the closure definition
        self.fn_defs.entry(ident.clone()).or_default().insert(ty_args.clone(), Some(fn_def));

        // mark end
        info!("{}<- closure definition", self.depth);
        self.depth.pop();

        // return the key to the lookup table
        (ident, ty_args)
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

    /// Create a generic argument
    fn mk_ty_arg(&mut self, ty_arg: GenericArg<'tcx>) -> SolGenericArg {
        match ty_arg.kind() {
            GenericArgKind::Type(ty) => SolGenericArg::Type(self.mk_type(ty)),
            GenericArgKind::Const(ty_const) => SolGenericArg::Const(self.mk_ty_const(ty_const)),
            GenericArgKind::Lifetime(_) => SolGenericArg::Lifetime,
        }
    }
}

/*
 * Naming
 */

/// An identifier in the Solana context
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolIdent {
    krate: String,
    paths: Vec<String>,
}

/// A field name
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolFieldName(String);

/// A field index
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolFieldIndex(usize);

/// A variant name
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolVariantName(String);

/// A variant index
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolVariantIndex(usize);

/*
 * Typing
 */

/// Primitive integer types
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
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
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolTypeFloat {
    F16,
    F32,
    F64,
    F128,
}

/// All supported types in the Solana context
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolType {
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
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolTyDef {
    Struct { fields: Vec<SolField> },
    Union { fields: Vec<SolField> },
    Enum { variants: Vec<SolVariant> },
}

/// A field definition in an ADT
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolField {
    index: SolFieldIndex,
    name: SolFieldName,
    ty: SolType,
}

/// A field definition in an ADT
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolVariant {
    index: SolVariantIndex,
    name: SolVariantName,
    fields: Vec<SolField>,
}

/// A constant known in the type system
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolTyConst {
    Simple { ty: SolType, val: SolConst },
}

/// A generic argument
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolGenericArg {
    Type(SolType),
    Const(SolTyConst),
    Lifetime,
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

/*
 * Shared
 */

/// A scalar value
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolScalar {
    bits: usize,
    value: u128,
}

/// A constant, used by both the type system and the value system
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolConst {
    ZeroSized,
    Scalar(SolScalar),
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
