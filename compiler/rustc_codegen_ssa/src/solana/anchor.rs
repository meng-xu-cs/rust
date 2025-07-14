use rustc_hir::def::DefKind;
use rustc_middle::bug;
use rustc_middle::mir::mono::MonoItem;
use rustc_middle::ty::{self, InstanceKind, TyCtxt};
use tracing::{debug, info};

fn collect_instructions(tcx: TyCtxt<'_>) {
    let partitions = tcx.collect_and_partition_mono_items(());
    for unit in partitions.codegen_units {
        for (item, _) in unit.items_in_deterministic_order(tcx) {
            let instance = match item {
                MonoItem::Fn(instance) => {
                    // an instruction must be locally defined
                    if !instance.def_id().is_local() {
                        continue;
                    }
                    instance
                }
                MonoItem::Static(_) => continue,
                MonoItem::GlobalAsm(_) => {
                    bug!("[assumption] unexpected assembly item in codegen unit")
                }
            };

            // an instruction must be a `fn` item
            if !matches!(instance.def, InstanceKind::Item(_)) {
                continue;
            }

            let def_id = instance.def_id();
            if !matches!(tcx.def_kind(def_id), DefKind::Fn) {
                continue;
            }

            // check parameter types
            let sig_decl = tcx.fn_sig(def_id);
            let sig_inst = sig_decl.instantiate(tcx, instance.args);
            let sig_norm = tcx.instantiate_bound_regions_with_erased(sig_inst);

            let ty_param0 = match sig_norm.inputs().first() {
                None => continue, // skip if no parameters
                Some(ty) => ty,
            };

            let ty_state = match ty_param0.kind() {
                ty::Adt(adt_def, adt_ty_args) => {
                    // the first parameter of an instruction must be `anchor_lang::context::Context`
                    if tcx.def_path_str(adt_def.did()) != "anchor_lang::context::Context" {
                        continue;
                    }

                    // extract the type that represents the state
                    if adt_ty_args.len() != 1 {
                        bug!(
                            "expect one and only one type argument for anchor_lang::context::Context"
                        );
                    }
                    adt_ty_args.first().unwrap().expect_ty()
                }
                _ => continue, // skip for all non-adt types
            };

            // extract the state type definition
            match ty_state.kind() {
                ty::Adt(adt_def, _adt_ty_args) => {
                    info!(
                        "- found instruction {} taking state {}",
                        tcx.def_path_str(instance.def_id()),
                        tcx.def_path_str(adt_def.did())
                    );
                }
                _ => bug!("expect the state type to be an adt, found: {ty_state}"),
            }
        }
    }
}

/// Processes an anchor-based compilation
pub(crate) fn process_compilation(tcx: TyCtxt<'_>) {
    collect_instructions(tcx);

    // FIXME: probing, to be removed later
    let partitions = tcx.collect_and_partition_mono_items(());
    for unit in partitions.codegen_units {
        debug!("  -> codegen unit {}", unit.name());
        for (k, _v) in unit.items_in_deterministic_order(tcx) {
            match k {
                MonoItem::Fn(instance) => {
                    let id = instance.def_id();
                    debug!("    - instance[{}]: {}", id.is_local(), tcx.def_path_str(id));
                }
                MonoItem::Static(id) => {
                    debug!("    - static[{}]: {}", id.is_local(), tcx.def_path_str(id));
                }
                MonoItem::GlobalAsm(_) => {
                    bug!("[assumption] unexpected assembly item in codegen unit")
                }
            }
        }
        debug!("  <- codegen unit {}", unit.name());
    }
}
