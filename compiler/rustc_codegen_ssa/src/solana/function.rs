use rustc_middle::bug;
use rustc_middle::mir::{MirPhase, RuntimePhase};
use rustc_middle::ty::{self, Instance, TyCtxt};
use serde::{Deserialize, Serialize};

use crate::solana::ident::Ident;
use crate::solana::typing::SolType;

#[derive(Serialize, Deserialize)]
pub(crate) struct SolFunc {
    name: Ident,
    args: Vec<SolType>,
    ret_ty: SolType,
}

impl SolFunc {
    pub(crate) fn convert<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> Self {
        let def_id = instance.def_id();
        let def_desc = tcx.def_path_str(def_id);

        // convert the instance to monomorphised MIR
        let instance_mir = tcx.instance_mir(instance.def).clone();
        if instance_mir.phase != MirPhase::Runtime(RuntimePhase::Optimized) {
            bug!("[assumption] converted instance is not runtime optimized: {def_desc}");
        }

        let body = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            instance_mir.typing_env(tcx),
            ty::EarlyBinder::bind(instance_mir),
        );

        // construct the function signature
        Self {
            name: Ident::new(tcx, def_id),
            args: body
                .args_iter()
                .map(|arg_idx| {
                    let decl = &body.local_decls[arg_idx];
                    SolType::convert(tcx, decl.ty)
                })
                .collect(),
            ret_ty: SolType::convert(tcx, body.return_ty()),
        }
    }
}
