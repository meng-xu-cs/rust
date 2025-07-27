use rustc_middle::bug;
use rustc_middle::mir::{MirPhase, RuntimePhase};
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
            let norm_arg_ty =
                tcx.normalize_erasing_regions(ty::TypingEnv::fully_monomorphized(), decl.ty);

            args.push(SolType::convert(tcx, depth.next(), norm_arg_ty));
        }

        info!("{depth}-- return type");
        let norm_ret_ty =
            tcx.normalize_erasing_regions(ty::TypingEnv::fully_monomorphized(), body.return_ty());
        let ret_ty = SolType::convert(tcx, depth.next(), norm_ret_ty);

        // mark end
        info!("{depth}<- function {def_desc}");

        // done
        Self { name, args, ret_ty }
    }
}
