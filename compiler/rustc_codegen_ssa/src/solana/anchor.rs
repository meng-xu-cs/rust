use rustc_middle::bug;
use rustc_middle::mir::mono::MonoItem;
use rustc_middle::ty::TyCtxt;
use tracing::info;

/// Processes an anchor-based compilation
pub(crate) fn process_compilation(tcx: TyCtxt<'_>) {
    let partitions = tcx.collect_and_partition_mono_items(());
    for unit in partitions.codegen_units {
        info!("  -> codegen unit {}", unit.name());
        for (k, _v) in unit.items_in_deterministic_order(tcx) {
            match k {
                MonoItem::Fn(instance) => {
                    let id = instance.def_id();
                    info!("    - instance[{}]: {}", id.is_local(), tcx.def_path_str(id));
                }
                MonoItem::Static(id) => {
                    info!("    - static[{}]: {}", id.is_local(), tcx.def_path_str(id));
                }
                MonoItem::GlobalAsm(_) => {
                    bug!("[assumption] unexpected assembly item in codegen unit")
                }
            }
        }
        info!("  <- codegen unit {}", unit.name());
    }
}
