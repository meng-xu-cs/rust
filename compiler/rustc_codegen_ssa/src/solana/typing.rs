use rustc_middle::bug;
use rustc_middle::ty::{self, Ty, TyCtxt};
use serde::{Deserialize, Serialize};
use tracing::warn;

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

impl<'a> From<&'a ty::IntTy> for SolTypeInt {
    fn from(t: &'a ty::IntTy) -> Self {
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

impl<'a> From<&'a ty::UintTy> for SolTypeInt {
    fn from(t: &'a ty::UintTy) -> Self {
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

impl<'a> From<&'a ty::FloatTy> for SolTypeFloat {
    fn from(t: &'a ty::FloatTy) -> Self {
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
        name: String,
        /* FIXME: instantiation */
        fields: Vec<(String, SolType)>,
    },
    Union {
        name: String,
        /* FIXME: instantiation */
        fields: Vec<(String, SolType)>,
    },
    Enum {
        name: String,
        /* FIXME: instantiation */
        variants: Vec<(String, Vec<(String, SolType)>)>,
    },
}

impl SolType {
    pub(crate) fn convert<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Self {
        match ty.kind() {
            ty::Bool => Self::Bool,
            ty::Char => Self::Char,
            ty::Int(t) => Self::Int(t.into()),
            ty::Uint(t) => Self::Int(t.into()),
            ty::Float(t) => Self::Float(t.into()),
            ty::Str => Self::Str,
            ty::Array(sub_ty, size_const) => Self::Array(
                Box::new(Self::convert(tcx, *sub_ty)),
                size_const.try_to_target_usize(tcx).unwrap_or_else(|| {
                    bug!("[assumption] unexpected non-constant array size {size_const}");
                }) as usize,
            ),
            ty::Tuple(tys) => Self::Tuple(tys.iter().map(|t| Self::convert(tcx, t)).collect()),
            ty::Adt(adt_def, adt_args) => match adt_def.adt_kind() {
                ty::AdtKind::Struct => Self::Struct {
                    name: tcx.def_path_str(adt_def.did()),
                    fields: adt_def
                        .all_fields()
                        .map(|field| {
                            (field.name.to_string(), Self::convert(tcx, field.ty(tcx, adt_args)))
                        })
                        .collect(),
                    // FIXME: instantiations
                },
                ty::AdtKind::Union => Self::Union {
                    name: tcx.def_path_str(adt_def.did()),
                    fields: adt_def
                        .all_fields()
                        .map(|field| {
                            (field.name.to_string(), Self::convert(tcx, field.ty(tcx, adt_args)))
                        })
                        .collect(),
                    // FIXME: instantiations
                },
                ty::AdtKind::Enum => Self::Enum {
                    name: tcx.def_path_str(adt_def.did()),
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
            },
            ty::Coroutine(..) | ty::CoroutineClosure(..) | ty::CoroutineWitness(..) => {
                bug!("[unsupported] coroutine-related type: {ty}");
            }
            ty::Foreign(..) => {
                bug!("[assumption] unexpected foreign type {ty}");
            }
            ty::Alias(..) => {
                bug!("[assumption] unexpected alias type {ty}");
            }
            ty::Param(..) | ty::Bound(..) | ty::UnsafeBinder(..) => {
                bug!("[assumption] unexpected generic type: {ty}");
            }
            ty::Never | ty::Infer(..) | ty::Placeholder(..) | ty::Error(..) => {
                bug!("[invariant] unexpected type used in type analysis: {ty}");
            }
            _ => {
                warn!("Unsupported type for SolType: {ty}");
                Self::Bool
            }
        }
    }
}
