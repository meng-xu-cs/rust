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

            // an instruction must be an `fn` item
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

            let ty = match sig_inst.inputs().iter().next() {
                None => continue, // skip if no parameters
                Some(bounded) => bounded.skip_binder(),
            };

            match ty.kind() {
                ty::Adt(adt_def, adt_ty_args) => {
                    if tcx.def_path_str(adt_def.did()) == "anchor_lang::context::Context" {
                        info!("- found instruction: {}", tcx.def_path_str(instance.def_id()));
                    }
                    if !adt_ty_args.len() == 1 {
                        bug!(
                            "expect one and only one type argument for anchor_lang::context::Context"
                        );
                    }
                }
                _ => continue, // skip for all non-adt types
            };
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
