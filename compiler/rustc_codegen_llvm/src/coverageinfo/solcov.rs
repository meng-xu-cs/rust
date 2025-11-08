use rustc_abi::Size;
use rustc_codegen_ssa::traits::{BuilderMethods, ConstCodegenMethods};
use rustc_middle::bug;
use rustc_middle::ty::Instance;

use crate::builder::Builder;
use crate::llvm;

/// Coverage kind for PC tracking
const COV_KIND_PC: u16 = 1;

fn output_u32<'a, 'll, 'tcx>(
    builder: &mut Builder<'a, 'll, 'tcx>,
    slot: &'ll llvm::Value,
    value: u32,
) {
    let align = builder.tcx.data_layout.i32_align.abi;
    builder.store(builder.const_u32(value), slot, align);
    builder.call_intrinsic("llvm.fake.use", &[], &[slot]);
}

fn output_u64<'a, 'll, 'tcx>(
    builder: &mut Builder<'a, 'll, 'tcx>,
    slot: &'ll llvm::Value,
    value: u64,
) {
    output_u32(builder, slot, (value >> 32) as u32);
    output_u32(builder, slot, value as u32);
}

/// Process Solana coverage information by emitting special markers
pub(crate) fn process_solcov<'a, 'll, 'tcx>(
    builder: &mut Builder<'a, 'll, 'tcx>,
    instance: Instance<'tcx>,
    kind: u16,
    value: u32,
) {
    // create the bitmap slots if not already created
    if !builder.solcov_cx().mcdc_condition_bitmap_map.borrow().contains_key(&instance) {
        let align = builder.tcx.data_layout.i32_align.abi;
        let size = Size::from_bytes(4);
        let slot = builder.alloca(size, align);
        llvm::set_value_name(slot, format!("solcov.slot").as_bytes());
        output_u32(builder, slot, 0);
        builder.solcov_cx().mcdc_condition_bitmap_map.borrow_mut().insert(instance, vec![slot]);
    }

    // output in sequence
    let slot = builder.solcov_cx().mcdc_condition_bitmap_map.borrow().get(&instance).unwrap()[0];
    output_u32(builder, slot, 0x63_6c_6f_73); // "solc" encoded backwards
    output_u32(builder, slot, 0x00_00_76_6f | (kind as u32) << 16); // "ov"|kind encoded backwards

    match kind {
        COV_KIND_PC => {
            let def_path_hash = builder.tcx.def_path_hash(instance.def_id());
            output_u64(builder, slot, def_path_hash.stable_crate_id().as_u64());
            output_u64(builder, slot, def_path_hash.local_hash().as_u64());
            output_u32(builder, slot, value);
        }
        _ => bug!("[invariant] unexpected solana coverage kind {kind}"),
    }
}
