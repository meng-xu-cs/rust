use rustc_middle::bug;
use rustc_middle::ty::{self, Instance, Ty, TyCtxt, TypingEnv};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::solana::common::Depth;
use crate::solana::function::SolFunc;
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
    Struct { name: Ident, mono: Vec<SolGenericArg>, fields: Vec<(String, SolType)> },
    Union { name: Ident, mono: Vec<SolGenericArg>, fields: Vec<(String, SolType)> },
    Enum { name: Ident, mono: Vec<SolGenericArg>, variants: Vec<(String, Vec<(String, SolType)>)> },
    Slice(Box<SolType>),
    ImmRef(Box<SolType>),
    MutRef(Box<SolType>),
    ImmPtr(Box<SolType>),
    MutPtr(Box<SolType>),
    Function { name: Ident, mono: Vec<SolGenericArg>, func: Box<SolFunc> },
    Closure { name: Ident, mono: Vec<SolGenericArg>, func: Box<SolFunc> },
}

#[derive(Serialize, Deserialize)]
pub(crate) enum SolGenericArg {
    Type(SolType),
    Const(usize),
    Lifetime,
}

impl SolType {
    pub(crate) fn convert<'tcx>(tcx: TyCtxt<'tcx>, depth: Depth, ty: Ty<'tcx>) -> Self {
        // mark start
        info!("{depth}-> type {ty}");

        // force normalize the type due to lazy normalization (this is needed for at least associated types)
        let normalized_ty = tcx.normalize_erasing_regions(TypingEnv::fully_monomorphized(), ty);
        info!("{depth}-- normalized to {normalized_ty}");

        // switch by type kind
        let converted = match *normalized_ty.kind() {
            ty::Bool => Self::Bool,
            ty::Char => Self::Char,
            ty::Int(t) => Self::Int(t.into()),
            ty::Uint(t) => Self::Int(t.into()),
            ty::Float(t) => Self::Float(t.into()),
            ty::Str => Self::Str,
            ty::Array(sub_ty, size_const) => {
                info!("{depth}-- array element type");
                Self::Array(
                    Box::new(Self::convert(tcx, depth.next(), sub_ty)),
                    size_const.try_to_target_usize(tcx).unwrap_or_else(|| {
                        bug!("[assumption] unexpected non-constant array size {size_const}");
                    }) as usize,
                )
            }
            ty::Tuple(tys) => {
                let mut elem_tys = vec![];
                for (i, t) in tys.iter().enumerate() {
                    info!("{depth}-- tuple element {i}: {t}");
                    elem_tys.push(Self::convert(tcx, depth.next(), t));
                }
                Self::Tuple(elem_tys)
            }
            ty::Adt(adt_def, adt_args) => {
                let name = Ident::new(tcx, adt_def.did());
                let mut mono = vec![];
                for (i, arg) in adt_args.iter().enumerate() {
                    info!("{depth}-- adt type argument {i}: {arg}");
                    mono.push(SolGenericArg::convert(tcx, depth.next(), arg));
                }
                match adt_def.adt_kind() {
                    ty::AdtKind::Struct => {
                        let mut fields = vec![];
                        for (i, field) in adt_def.all_fields().enumerate() {
                            info!("{depth}-- struct field {i}: {}", field.name);
                            fields.push((
                                field.name.to_string(),
                                Self::convert(tcx, depth.next(), field.ty(tcx, adt_args)),
                            ));
                        }
                        Self::Struct { name, mono, fields }
                    }
                    ty::AdtKind::Union => {
                        let mut fields = vec![];
                        for (i, field) in adt_def.all_fields().enumerate() {
                            info!("{depth}-- union field {i}: {}", field.name);
                            fields.push((
                                field.name.to_string(),
                                Self::convert(tcx, depth.next(), field.ty(tcx, adt_args)),
                            ));
                        }
                        Self::Union { name, mono, fields }
                    }
                    ty::AdtKind::Enum => {
                        let mut variants = vec![];
                        for (i, variant) in adt_def.variants().iter().enumerate() {
                            info!("{depth}-- enum variant {i}: {}", variant.name);
                            let mut fields = vec![];
                            for (j, field) in variant.fields.iter().enumerate() {
                                info!("{depth}-- variant field {j}: {}", field.name);
                                fields.push((
                                    field.name.to_string(),
                                    Self::convert(tcx, depth.next(), field.ty(tcx, adt_args)),
                                ));
                            }
                            variants.push((variant.name.to_string(), fields));
                        }
                        Self::Enum { name, mono, variants }
                    }
                }
            }
            ty::Slice(sub_ty) => {
                info!("{depth}-- slice element type");
                Self::Slice(Box::new(Self::convert(tcx, depth.next(), sub_ty)))
            }
            ty::Ref(_, sub_ty, mutability) => {
                info!("{depth}-- ref inner type");
                match mutability {
                    ty::Mutability::Mut => {
                        Self::MutRef(Box::new(Self::convert(tcx, depth.next(), sub_ty)))
                    }
                    ty::Mutability::Not => {
                        Self::ImmRef(Box::new(Self::convert(tcx, depth.next(), sub_ty)))
                    }
                }
            }
            ty::RawPtr(sub_ty, mutability) => {
                info!("{depth}-- ptr inner type");
                match mutability {
                    ty::Mutability::Mut => {
                        Self::MutPtr(Box::new(Self::convert(tcx, depth.next(), sub_ty)))
                    }
                    ty::Mutability::Not => {
                        Self::ImmPtr(Box::new(Self::convert(tcx, depth.next(), sub_ty)))
                    }
                }
            }
            ty::FnDef(def_id, fn_ty_args) => {
                let name = Ident::new(tcx, def_id);

                // convert the function type arguments
                let mut mono = vec![];
                for (i, arg) in fn_ty_args.iter().enumerate() {
                    info!("{depth}-- fn type argument {i}: {arg}");
                    mono.push(SolGenericArg::convert(tcx, depth.next(), arg));
                }

                // resolve the function instance
                let instance = match Instance::try_resolve(
                    tcx,
                    TypingEnv::fully_monomorphized(),
                    def_id,
                    fn_ty_args,
                ) {
                    Ok(Some(instance)) => instance,
                    Ok(None) => {
                        bug!("[assumption] unresolved function type: {ty}");
                    }
                    Err(_) => {
                        bug!("[invariant] failed to resolve function type {ty}");
                    }
                };

                // convert the function instance
                info!("{depth}-- fn type instance: {}", tcx.def_path_str(def_id));
                let func = SolFunc::convert(tcx, depth.next(), instance);

                // pack the information
                Self::Function { name, mono, func: func.into() }
            }
            ty::Closure(def_id, recorded_ty_args) => {
                let name = Ident::new(tcx, def_id);

                // convert the closure type arguments
                let mut mono = vec![];
                for (i, arg) in recorded_ty_args.iter().enumerate() {
                    info!("{depth}-- closure type argument {i}: {arg}");
                    mono.push(SolGenericArg::convert(tcx, depth.next(), arg));
                }

                // resolve the closure instance
                let closure_ty_args = recorded_ty_args.as_closure();
                let instance = Instance::resolve_closure(
                    tcx,
                    def_id,
                    recorded_ty_args,
                    closure_ty_args.kind(),
                );

                // convert the closure instance
                info!("{depth}-- closure type instance: {}", tcx.def_path_str(def_id));
                let func = SolFunc::convert(tcx, depth.next(), instance);

                // pack the information
                Self::Closure { name, mono, func: func.into() }
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
        info!("{depth}<- type {ty}");

        // done
        converted
    }
}

impl SolGenericArg {
    pub(crate) fn convert<'tcx>(
        tcx: TyCtxt<'tcx>,
        depth: Depth,
        arg: ty::GenericArg<'tcx>,
    ) -> Self {
        match arg.kind() {
            ty::GenericArgKind::Type(ty) => Self::Type(SolType::convert(tcx, depth, ty)),
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
