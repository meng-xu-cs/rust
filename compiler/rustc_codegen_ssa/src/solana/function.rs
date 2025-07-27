use rustc_middle::bug;
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::{
    AggregateKind, BinOp, BorrowKind, CastKind, Const, ConstValue, MirPhase, NullOp, Operand,
    Place, ProjectionElem, RawPtrKind, RuntimePhase, Rvalue, StatementKind, UnOp,
};
use rustc_middle::ty::{self, Instance, Ty, TyCtxt};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::solana::common::Depth;
use crate::solana::ident::Ident;
use crate::solana::typing::SolType;

#[derive(Serialize, Deserialize)]
pub(crate) struct SolFunc {
    name: Ident,
    args: Vec<SolType>,
    ret_ty: SolType,
    locals: Vec<SolType>,
    blocks: Vec<SolBasicBlock>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct SolBasicBlock {
    index: usize,
    stmts: Vec<SolStatement>,
}

#[derive(Serialize, Deserialize)]
pub(crate) enum SolStatement {
    Deinit(SolPlace),
    StorageLive(usize),
    StorageDead(usize),
    PlaceMention(SolPlace),
    Assign { lhs: SolPlace, rhs: SolExpression },
    SetDiscriminant { place: SolPlace, variant: usize },
}

#[derive(Serialize, Deserialize)]
pub(crate) enum SolProjection {
    Deref,
    Field { field: usize, ty: SolType },
    Index { element: usize },
    ConstantIndex { offset: usize, min_length: usize, from_end: bool },
    Subslice { from: usize, to: usize, from_end: bool },
    Downcast { variant: usize },
    Subtype(SolType),
}

#[derive(Serialize, Deserialize)]
pub(crate) struct SolPlace {
    local: usize,
    projection: Vec<SolProjection>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct SolScalar {
    bits: usize,
    value: u128,
}

#[derive(Serialize, Deserialize)]
pub(crate) enum SolConst {
    ZeroSized,
    Scalar(SolScalar),
}

#[derive(Serialize, Deserialize)]
pub(crate) struct SolTypedConst {
    ty: SolType,
    const_: SolConst,
}

#[derive(Serialize, Deserialize)]
pub(crate) enum SolOperand {
    Copy(SolPlace),
    Move(SolPlace),
    Constant(SolTypedConst),
}

#[derive(Serialize, Deserialize)]
pub(crate) enum SolCastKind {
    IntToInt,
    FloatToFloat,
    IntToFloat,
    FloatToInt,
}

#[derive(Serialize, Deserialize)]
pub(crate) enum SolExpression {
    Use(SolOperand),
    Repeat(SolOperand, usize),
    BorrowImm(SolPlace),
    BorrowMut(SolPlace),
    PointerImm(SolPlace),
    PointerMut(SolPlace),
    Length(SolPlace),
    Cast { kind: SolCastKind, place: SolOperand, ty: SolType },
    OpNullary { opcode: SolOpcodeNullary, ty: SolType },
    OpUnary { opcode: SolOpcodeUnary, operand: SolOperand },
    OpBinary { opcode: SolOpcodeBinary, v1: SolOperand, v2: SolOperand },
    Discriminant(SolPlace),
    Aggregate { opcode: SolOpcodeAggregate, values: Vec<(usize, SolOperand)> },
    Load(SolPlace),
}

#[derive(Serialize, Deserialize)]
pub(crate) enum SolOpcodeNullary {
    SizeOf,
    AlignOf,
    OffsetOf(Vec<(usize, usize)>),
}

#[derive(Serialize, Deserialize)]
pub(crate) enum SolOpcodeUnary {
    SizeOf,
    Not,
    Neg,
}

#[derive(Serialize, Deserialize)]
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

#[derive(Serialize, Deserialize)]
pub(crate) enum SolOpcodeAggregate {
    Tuple,
    Array(SolType /* element type */),
    Struct { ty: SolType },
    Union { ty: SolType, field: usize },
    Enum { ty: SolType, variant: usize },
}

impl SolFunc {
    pub(crate) fn convert<'tcx>(tcx: TyCtxt<'tcx>, depth: Depth, instance: Instance<'tcx>) -> Self {
        let def_id = instance.def_id();
        let def_desc = tcx.def_path_str(def_id);

        // mark start
        info!("{depth}-> function {def_desc}");

        // convert the instance to monomorphised MIR
        let instance_mir = tcx.instance_mir(instance.def).clone();
        if instance_mir.phase != MirPhase::Runtime(RuntimePhase::Optimized) {
            bug!("[assumption] converted instance is not runtime optimized: {def_desc}");
        }

        // NOTE: lasy normalization seems to be in effect here, at least associated types
        // are not easily resolved after this transformation.
        let body = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            instance_mir.typing_env(tcx),
            ty::EarlyBinder::bind(instance_mir),
        );

        // convert function signatures
        let name = Ident::new(tcx, def_id);

        let mut args = vec![];
        for arg_idx in body.args_iter() {
            info!("{depth}-- argument {}", arg_idx.index());

            let decl = &body.local_decls[arg_idx];
            let norm_ty =
                tcx.normalize_erasing_regions(ty::TypingEnv::fully_monomorphized(), decl.ty);

            args.push(SolType::convert(tcx, depth.next(), norm_ty));
        }

        info!("{depth}-- return type");
        let norm_ret_ty =
            tcx.normalize_erasing_regions(ty::TypingEnv::fully_monomorphized(), body.return_ty());
        let ret_ty = SolType::convert(tcx, depth.next(), norm_ret_ty);

        // convert function body
        let mut locals = vec![];
        for local_idx in body.vars_and_temps_iter() {
            info!("{depth}-- local {}", local_idx.index());

            let decl = &body.local_decls[local_idx];
            let norm_ty =
                tcx.normalize_erasing_regions(ty::TypingEnv::fully_monomorphized(), decl.ty);

            locals.push(SolType::convert(tcx, depth.next(), norm_ty));
        }

        let mut blocks = vec![];
        for (block_id, block_data) in body.basic_blocks.iter_enumerated() {
            info!("{depth}-- basic block {}", block_id.index());

            // sanity check
            if block_data.is_cleanup {
                bug!("[assumption] unexpected cleanup block found in function {def_desc}");
            }

            // process statements
            let stmt_depth = depth.next();
            let mut statements = vec![];
            for stmt in block_data.statements.iter() {
                info!("{stmt_depth}-- statement {}", stmt.kind.name());
                let converted = match &stmt.kind {
                    // assign
                    StatementKind::Assign(assignment) => {
                        let (place, value) = assignment.as_ref();
                        SolStatement::Assign {
                            lhs: SolPlace::convert(tcx, stmt_depth.next(), *place),
                            rhs: SolExpression::convert(tcx, stmt_depth.next(), value),
                        }
                    }
                    StatementKind::SetDiscriminant { place, variant_index } => {
                        SolStatement::SetDiscriminant {
                            place: SolPlace::convert(tcx, stmt_depth.next(), **place),
                            variant: variant_index.index(),
                        }
                    }
                    // storage
                    StatementKind::Deinit(place) => {
                        SolStatement::Deinit(SolPlace::convert(tcx, stmt_depth.next(), **place))
                    }
                    StatementKind::StorageLive(local_idx) => {
                        SolStatement::StorageLive(local_idx.index())
                    }
                    StatementKind::StorageDead(local_idx) => {
                        SolStatement::StorageDead(local_idx.index())
                    }
                    StatementKind::PlaceMention(place) => {
                        SolStatement::Deinit(SolPlace::convert(tcx, stmt_depth.next(), **place))
                    }
                    // no-op
                    StatementKind::Nop
                    | StatementKind::ConstEvalCounter
                    | StatementKind::FakeRead(..)
                    | StatementKind::AscribeUserType(..)
                    | StatementKind::BackwardIncompatibleDropHint { .. } => continue,
                    // should not appear
                    StatementKind::Retag(..) => {
                        bug!("[invariant] unexpected retag statement in function {def_desc}");
                    }
                    StatementKind::Coverage(..) => {
                        bug!("[invariant] unexpected coverage statement in function {def_desc}");
                    }
                    StatementKind::Intrinsic(..) => {
                        bug!("[invariant] unexpected intrinsic statement in function {def_desc}");
                    }
                };
                statements.push(converted);
            }

            // FIXME: process terminator

            // pack information
            blocks.push(SolBasicBlock { index: block_id.index(), stmts: statements });
        }

        // mark end
        info!("{depth}<- function {def_desc}");

        // park the information
        Self { name, args, ret_ty, locals, blocks }
    }
}

impl SolPlace {
    pub(crate) fn convert<'tcx>(tcx: TyCtxt<'tcx>, depth: Depth, place: Place<'tcx>) -> Self {
        let local = place.local.index();
        let projection = place
            .projection
            .iter()
            .map(|proj| match proj {
                ProjectionElem::Deref => SolProjection::Deref,
                ProjectionElem::Field(index, ty) => SolProjection::Field {
                    field: index.index(),
                    ty: SolType::convert(tcx, depth.next(), ty),
                },
                ProjectionElem::Index(index) => SolProjection::Index { element: index.index() },
                ProjectionElem::ConstantIndex { offset, min_length, from_end } => {
                    SolProjection::ConstantIndex {
                        offset: offset as usize,
                        min_length: min_length as usize,
                        from_end,
                    }
                }
                ProjectionElem::Subslice { from, to, from_end } => {
                    SolProjection::Subslice { from: from as usize, to: to as usize, from_end }
                }
                ProjectionElem::Downcast(_, variant) => {
                    SolProjection::Downcast { variant: variant.index() }
                }
                ProjectionElem::Subtype(ty) => {
                    SolProjection::Subtype(SolType::convert(tcx, depth.next(), ty))
                }
                ProjectionElem::OpaqueCast(ty) => {
                    bug!("[invariant] unexpected opaque cast in place conversion: {ty}");
                }
                ProjectionElem::UnwrapUnsafeBinder(ty) => {
                    bug!("[invariant] unexpected unwrap unsafe binder in place conversion: {ty}");
                }
            })
            .collect();

        Self { local, projection }
    }
}

impl SolTypedConst {
    pub(crate) fn convert<'tcx>(tcx: TyCtxt<'tcx>, depth: Depth, const_: Const<'tcx>) -> Self {
        match const_ {
            Const::Val(value, ty) => {
                let converted_ty = SolType::convert(tcx, depth, ty);
                let converted_const = match value {
                    ConstValue::Scalar(Scalar::Int(scalar)) => SolConst::Scalar(SolScalar {
                        bits: scalar.size().bits_usize(),
                        value: scalar.to_bits_unchecked(),
                    }),
                    ConstValue::Scalar(Scalar::Ptr(..)) => {
                        bug!("[assumption] unexpected pointer scalar: {const_}");
                    }
                    ConstValue::ZeroSized => SolConst::ZeroSized,
                    ConstValue::Slice { .. } | ConstValue::Indirect { .. } => {
                        bug!("[assumption] unexpected constant kind: {const_}");
                    }
                };
                Self { ty: converted_ty, const_: converted_const }
            }
            Const::Unevaluated(..) => {
                bug!("[assumption] unexpected unevaluated constant in scalar conversion: {const_}");
            }
            Const::Ty(..) => {
                bug!("[assumption] unexpected type constant in scalar conversion: {const_}");
            }
        }
    }
}

impl SolOperand {
    pub(crate) fn convert<'tcx>(tcx: TyCtxt<'tcx>, depth: Depth, operand: &Operand<'tcx>) -> Self {
        match operand {
            Operand::Copy(place) => Self::Copy(SolPlace::convert(tcx, depth, *place)),
            Operand::Move(place) => Self::Move(SolPlace::convert(tcx, depth, *place)),
            Operand::Constant(cval) => {
                Self::Constant(SolTypedConst::convert(tcx, depth, cval.const_))
            }
        }
    }
}

impl SolExpression {
    pub(crate) fn convert<'tcx>(tcx: TyCtxt<'tcx>, depth: Depth, rvalue: &Rvalue<'tcx>) -> Self {
        match rvalue {
            Rvalue::Use(operand) => Self::Use(SolOperand::convert(tcx, depth, operand)),
            Rvalue::Repeat(value, count) => Self::Repeat(
                SolOperand::convert(tcx, depth, value),
                count.try_to_target_usize(tcx).unwrap_or_else(|| {
                    bug!("[assumption] unexpected non-constant array size {count}");
                }) as usize,
            ),
            Rvalue::Ref(_, borrow_kind, place) => {
                let converted_place = SolPlace::convert(tcx, depth, *place);
                match borrow_kind {
                    BorrowKind::Shared => Self::BorrowImm(converted_place),
                    BorrowKind::Mut { .. } => Self::BorrowMut(converted_place),
                    BorrowKind::Fake(..) => bug!("[invariant] fake borrow not expected"),
                }
            }
            Rvalue::RawPtr(raw_ptr_kind, place) => {
                let converted_place = SolPlace::convert(tcx, depth, *place);
                match raw_ptr_kind {
                    RawPtrKind::Const => Self::PointerImm(converted_place),
                    RawPtrKind::Mut => Self::PointerMut(converted_place),
                    RawPtrKind::FakeForPtrMetadata => bug!("[invariant] fake raw ptr not expected"),
                }
            }
            Rvalue::Len(place) => Self::Length(SolPlace::convert(tcx, depth, *place)),
            Rvalue::Cast(cast_kind, operand, ty) => {
                let kind = match cast_kind {
                    CastKind::IntToInt => SolCastKind::IntToInt,
                    CastKind::FloatToFloat => SolCastKind::FloatToFloat,
                    CastKind::IntToFloat => SolCastKind::IntToFloat,
                    CastKind::FloatToInt => SolCastKind::FloatToInt,
                    _ => bug!("[assumption] unexpected cast kind: {cast_kind:?}"),
                };
                Self::Cast {
                    kind,
                    place: SolOperand::convert(tcx, depth, operand),
                    ty: SolType::convert(tcx, depth, *ty),
                }
            }
            Rvalue::NullaryOp(op, ty) => {
                let opcode = match op {
                    NullOp::SizeOf => SolOpcodeNullary::SizeOf,
                    NullOp::AlignOf => SolOpcodeNullary::AlignOf,
                    NullOp::OffsetOf(indices) => SolOpcodeNullary::OffsetOf(
                        indices
                            .iter()
                            .map(|(variant_idx, field_idx)| {
                                (variant_idx.index(), field_idx.index())
                            })
                            .collect(),
                    ),
                    _ => bug!("[invariant] unexpected nullary opcode: {op:?}"),
                };
                Self::OpNullary { opcode, ty: SolType::convert(tcx, depth, *ty) }
            }
            Rvalue::UnaryOp(op, operand) => {
                let opcode = match op {
                    UnOp::Not => SolOpcodeUnary::Not,
                    UnOp::Neg => SolOpcodeUnary::Neg,
                    _ => bug!("[invariant] unexpected unary opcode: {op:?}"),
                };
                Self::OpUnary { opcode, operand: SolOperand::convert(tcx, depth, operand) }
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
                Self::OpBinary {
                    opcode,
                    v1: SolOperand::convert(tcx, depth, v1),
                    v2: SolOperand::convert(tcx, depth, v2),
                }
            }
            Rvalue::Discriminant(place) => {
                Self::Discriminant(SolPlace::convert(tcx, depth, *place))
            }
            Rvalue::Aggregate(kind, operands) => {
                let opcode = match kind.as_ref() {
                    AggregateKind::Tuple => SolOpcodeAggregate::Tuple,
                    AggregateKind::Array(ty) => {
                        SolOpcodeAggregate::Array(SolType::convert(tcx, depth, *ty))
                    }
                    AggregateKind::Adt(def_id, variant_idx, ty_args, _, union_field_idx) => {
                        let adt_def = tcx.adt_def(def_id);
                        let converted_adt_ty =
                            SolType::convert(tcx, depth.next(), Ty::new_adt(tcx, adt_def, ty_args));

                        match adt_def.adt_kind() {
                            ty::AdtKind::Struct => {
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
                                SolOpcodeAggregate::Struct { ty: converted_adt_ty }
                            }
                            ty::AdtKind::Union => {
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
                                        ty: converted_adt_ty,
                                        field: field_idx.index(),
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
                                    ty: converted_adt_ty,
                                    variant: variant_idx.index(),
                                }
                            }
                        }
                    }
                    AggregateKind::Closure(..) => {
                        bug!("[assumption] unexpected closure in aggregate conversion");
                    }
                    AggregateKind::Coroutine(..) | AggregateKind::CoroutineClosure(..) => {
                        bug!("[invariant] unexpected coroutine in aggregate conversion");
                    }
                    AggregateKind::RawPtr(..) => {
                        bug!("[assumption] unexpected raw ptr in aggregate conversion");
                    }
                };

                let mut values = vec![];
                for (field_idx, operand) in operands.iter_enumerated() {
                    let value = SolOperand::convert(tcx, depth.next(), operand);
                    values.push((field_idx.index(), value));
                }
                Self::Aggregate { opcode, values }
            }
            Rvalue::CopyForDeref(place) => Self::Load(SolPlace::convert(tcx, depth, *place)),

            _ => {
                bug!("unhandled");
            }
        }
    }
}
