use rustc_middle::bug;
use rustc_middle::mir::{MirPhase, Place, ProjectionElem, RuntimePhase, StatementKind};
use rustc_middle::ty::{self, Instance, TyCtxt};
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
    Assign { lhs: SolPlace /* FIXME: RHS */ },
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
                        let (place, _value) = assignment.as_ref();
                        // FIXME: rvalue
                        SolStatement::Assign {
                            lhs: SolPlace::convert(tcx, stmt_depth.next(), *place),
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
