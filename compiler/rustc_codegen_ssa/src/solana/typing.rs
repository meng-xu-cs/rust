use rustc_middle::bug;
use rustc_middle::ty::{self, Ty, TyCtxt};
use serde::{Deserialize, Serialize};

use crate::solana::ident::Ident;

#[derive(Serialize, Deserialize)]
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

impl From<ty::IntTy> for SolTypeInt {
    fn from(t: ty::IntTy) -> Self {
        match t {
            ty::IntTy::I8 => Self::I8,
            ty::IntTy::I16 => Self::I16,
            ty::IntTy::I32 => Self::I32,
            ty::IntTy::I64 => Self::I64,
            ty::IntTy::I128 => Self::I128,
            ty::IntTy::Isize => Self::Isize,
        }
    }
}

impl From<ty::UintTy> for SolTypeInt {
    fn from(t: ty::UintTy) -> Self {
        match t {
            ty::UintTy::U8 => Self::U8,
            ty::UintTy::U16 => Self::U16,
            ty::UintTy::U32 => Self::U32,
            ty::UintTy::U64 => Self::U64,
            ty::UintTy::U128 => Self::U128,
            ty::UintTy::Usize => Self::Usize,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) enum SolTypeFloat {
    F16,
    F32,
    F64,
    F128,
}

impl From<ty::FloatTy> for SolTypeFloat {
    fn from(t: ty::FloatTy) -> Self {
        match t {
            ty::FloatTy::F16 => Self::F16,
            ty::FloatTy::F32 => Self::F32,
            ty::FloatTy::F64 => Self::F64,
            ty::FloatTy::F128 => Self::F128,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) enum SolType {
    Bool,
    Char,
    Int(SolTypeInt),
    Float(SolTypeFloat),
    Str,
    Array(Box<SolType>, usize),
    Tuple(Vec<SolType>),
    Struct {
        name: Ident,
        mono: Vec<SolGenericArg>,
        fields: Vec<(String, SolType)>,
    },
    Union {
        name: Ident,
        mono: Vec<SolGenericArg>,
        fields: Vec<(String, SolType)>,
    },
    Enum {
        name: Ident,
        mono: Vec<SolGenericArg>,
        variants: Vec<(String, Vec<(String, SolType)>)>,
    },
    Slice(Box<SolType>),
    ImmRef(Box<SolType>),
    MutRef(Box<SolType>),
    ImmPtr(Box<SolType>),
    MutPtr(Box<SolType>),
    Function {
        name: Ident,
        mono: Vec<SolGenericArg>,
        /* FIXME: entire function definition here */
    },
    Closure {
        name: Ident,
        mono: Vec<SolGenericArg>,
        /* FIXME: entire closure definition here */
    },
}

#[derive(Serialize, Deserialize)]
pub(crate) enum SolGenericArg {
    Type(SolType),
    Const(usize),
    Lifetime,
}

impl SolType {
    pub(crate) fn convert<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Self {
        match *ty.kind() {
            ty::Bool => Self::Bool,
            ty::Char => Self::Char,
            ty::Int(t) => Self::Int(t.into()),
            ty::Uint(t) => Self::Int(t.into()),
            ty::Float(t) => Self::Float(t.into()),
            ty::Str => Self::Str,
            ty::Array(sub_ty, size_const) => Self::Array(
                Box::new(Self::convert(tcx, sub_ty)),
                size_const.try_to_target_usize(tcx).unwrap_or_else(|| {
                    bug!("[assumption] unexpected non-constant array size {size_const}");
                }) as usize,
            ),
            ty::Tuple(tys) => Self::Tuple(tys.iter().map(|t| Self::convert(tcx, t)).collect()),
            ty::Adt(adt_def, adt_args) => {
                let name = Ident::new(tcx, adt_def.did());
                let mono = adt_args.iter().map(|arg| SolGenericArg::convert(tcx, arg)).collect();
                match adt_def.adt_kind() {
                    ty::AdtKind::Struct => Self::Struct {
                        name,
                        mono,
                        fields: adt_def
                            .all_fields()
                            .map(|field| {
                                (
                                    field.name.to_string(),
                                    Self::convert(tcx, field.ty(tcx, adt_args)),
                                )
                            })
                            .collect(),
                    },
                    ty::AdtKind::Union => Self::Union {
                        name,
                        mono,
                        fields: adt_def
                            .all_fields()
                            .map(|field| {
                                (
                                    field.name.to_string(),
                                    Self::convert(tcx, field.ty(tcx, adt_args)),
                                )
                            })
                            .collect(),
                    },
                    ty::AdtKind::Enum => Self::Enum {
                        name,
                        mono,
                        variants: adt_def
                            .variants()
                            .iter()
                            .map(|variant| {
                                (
                                    variant.name.to_string(),
                                    variant
                                        .fields
                                        .iter()
                                        .map(|field| {
                                            (
                                                field.name.to_string(),
                                                Self::convert(tcx, field.ty(tcx, adt_args)),
                                            )
                                        })
                                        .collect(),
                                )
                            })
                            .collect(),
                    },
                }
            }
            ty::Slice(sub_ty) => Self::Slice(Box::new(Self::convert(tcx, sub_ty))),
            ty::Ref(_, sub_ty, mutability) => match mutability {
                ty::Mutability::Mut => Self::MutPtr(Box::new(Self::convert(tcx, sub_ty))),
                ty::Mutability::Not => Self::ImmPtr(Box::new(Self::convert(tcx, sub_ty))),
            },
            ty::RawPtr(sub_ty, mutability) => match mutability {
                ty::Mutability::Mut => Self::MutPtr(Box::new(Self::convert(tcx, sub_ty))),
                ty::Mutability::Not => Self::ImmPtr(Box::new(Self::convert(tcx, sub_ty))),
            },
            ty::FnDef(def_id, fn_ty_args) => {
                let mono = fn_ty_args.iter().map(|arg| SolGenericArg::convert(tcx, arg)).collect();
                Self::Function { name: Ident::new(tcx, def_id), mono }
            }
            ty::Closure(def_id, closure_ty_args) => {
                let mono =
                    closure_ty_args.iter().map(|arg| SolGenericArg::convert(tcx, arg)).collect();
                Self::Closure { name: Ident::new(tcx, def_id), mono }
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
                bug!("[assumption] unexpected alias type {ty}");
            }
            ty::Param(..) | ty::Bound(..) | ty::UnsafeBinder(..) => {
                bug!("[invariant] unexpected generic type: {ty}");
            }
            ty::Never | ty::Infer(..) | ty::Placeholder(..) | ty::Error(..) => {
                bug!("[invariant] unexpected type used in type analysis: {ty}");
            }
        }
    }
}

impl SolGenericArg {
    pub(crate) fn convert<'tcx>(tcx: TyCtxt<'tcx>, arg: ty::GenericArg<'tcx>) -> Self {
        match arg.kind() {
            ty::GenericArgKind::Type(ty) => Self::Type(SolType::convert(tcx, ty)),
            ty::GenericArgKind::Const(c) => {
                let size = c.try_to_target_usize(tcx).unwrap_or_else(|| {
                    bug!("[assumption] unexpected non-integer generic argument {c}");
                });
                Self::Const(size as usize)
            }
            ty::GenericArgKind::Lifetime(_) => Self::Lifetime,
        }
    }
}
