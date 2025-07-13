//! This pass encapsulate all the transformations related to Solana's Anchor framework.

use std::fs::File;

use rustc_hir::def::DefKind;
use rustc_middle::mir::Body;
use rustc_middle::mir::graphviz::write_mir_fn_graphviz;
use rustc_middle::mir::pretty::{PrettyPrintMirOptions, write_mir_fn};
use rustc_middle::ty::{InstanceKind, TyCtxt};
use rustc_middle::{bug, ty};
use tracing::{info, warn};

use crate::solana::common::SolanaContext;

pub(crate) fn phase_bootstrap<'tcx>(tcx: TyCtxt<'tcx>, sol: SolanaContext, body: &mut Body<'tcx>) {
    info!(
        "phase: {} on {} with source {}",
        sol.phase,
        sol.build_system,
        sol.src_file_name.display()
    );

    // debug marking
    let def_id = body.source.instance.def_id();
    let def_desc = tcx.def_path_str(def_id);
    info!("processing {def_desc}");

    // skip source files that are not under the local source prefix
    if !sol.src_path_full.starts_with(&sol.source_dir) {
        info!("- skipped, not a local source file");
        return;
    }

    // skip non-local items (e.g., generics from external crates)
    if !def_id.is_local() {
        info!("- skipped, not a local definition");
        return;
    }

    // an instruction must be a `fn` item
    match body.source.instance {
        InstanceKind::Item(did) if tcx.def_kind(did) == DefKind::Fn => (),
        _ => {
            info!("- skipped, not a fn item");
            return;
        }
    };

    // check parameter types
    let ty_state = match body.args_iter().next() {
        None => {
            info!("- skipped, no parameters");
            return;
        }
        Some(arg) => {
            let decl = &body.local_decls[arg];
            match decl.ty.kind() {
                ty::Adt(adt_def, adt_ty_args) => {
                    // the first parameter of an instruction must be `anchor_lang::context::Context`
                    if tcx.def_path_str(adt_def.did()) != "anchor_lang::context::Context" {
                        info!("- skipped, first parameter is not Context");
                        return;
                    }

                    // extract the type that represents the state
                    // the definiton is `struct Context<'a, 'b, 'c, 'info, T: Bumps>`
                    if adt_ty_args.len() != 5 {
                        bug!(
                            "[assumption] anchor_lang::context::Context should take 5 type arguments, found: {}",
                            adt_ty_args.len()
                        );
                    }

                    // unpack the type arguments
                    let mut iter = adt_ty_args.iter();
                    iter.next().unwrap().expect_region();
                    iter.next().unwrap().expect_region();
                    iter.next().unwrap().expect_region();
                    iter.next().unwrap().expect_region();
                    let last_ty_arg = iter.next().unwrap().expect_ty();
                    if !iter.next().is_none() {
                        bug!(
                            "[invariant] anchor_lang::context::Context takes more than 5 type arguments"
                        );
                    }
                    last_ty_arg
                }
                _ => {
                    info!("- skipped, first parameter is not an adt");
                    return;
                }
            }
        }
    };

    // need subsequent processing
    warn!("- found instruction {def_desc} with argument {ty_state}");

    // prepare for output directory
    let instance_outdir = sol.instance_output_dir();

    // dump the MIR to a file
    let mut file_mir = File::create(instance_outdir.join("body.mir"))
        .unwrap_or_else(|e| bug!("[invariant] failed to create MIR file: {e}"));
    write_mir_fn(
        tcx,
        body,
        &mut |_, _| Ok(()),
        &mut file_mir,
        PrettyPrintMirOptions::from_cli(tcx),
    )
    .unwrap_or_else(|e| bug!("[invariant] failed to write MIR to file: {e}"));

    // dump the CFG to a file
    let mut file_dot = File::create(instance_outdir.join("body.dot"))
        .unwrap_or_else(|e| bug!("[invariant] failed to create Dot file: {e}"));
    write_mir_fn_graphviz(tcx, body, false, &mut file_dot)
        .unwrap_or_else(|e| bug!("[invariant] failed to write Dot to file: {e}"));

    // extract the state type definition
    match ty_state.kind() {
        ty::Adt(_adt_def, _adt_ty_args) => {}
        _ => bug!("[assumption] expect the state type to be an adt, found: {ty_state}"),
    }
}
