//@ edition: 2021

#![feature(rustc_private)]
#![deny(warnings)]

extern crate rustc_codegen_ssa;
extern crate rustc_data_structures;
extern crate rustc_metadata;
extern crate rustc_middle;
extern crate rustc_session;

use std::any::Any;
use std::io::Write;
use std::sync::OnceLock;

use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_codegen_ssa::{CompiledModules, CrateInfo, NlaiProducerIdentity};
use rustc_data_structures::fx::FxIndexMap;
use rustc_metadata::EncodedMetadata;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_session::config::OutputFilenames;

struct HandoffBackend {
    identity: OnceLock<NlaiProducerIdentity>,
}

impl CodegenBackend for HandoffBackend {
    fn name(&self) -> &'static str {
        "nlai-handoff-backend"
    }

    fn target_cpu(&self, _sess: &Session) -> String {
        "generic".to_owned()
    }

    fn init(&self, sess: &Session) {
        let identity = sess
            .nlai_producer_identity::<NlaiProducerIdentity>()
            .expect("dynamic backend did not receive the driver-captured NLAI identity")
            .clone();
        self.identity.set(identity).expect("dynamic backend initialized twice");
    }

    fn codegen_crate(&self, _tcx: TyCtxt<'_>) -> Box<dyn Any> {
        let identity = self.identity.get().expect("dynamic backend was not initialized");
        let launcher = std::env::current_exe().expect("discover current rustc path in backend");
        let launcher_len = std::fs::metadata(&launcher)
            .expect("read current rustc path metadata in backend")
            .len();
        let marker = std::env::var_os("NLAI_HANDOFF_MARKER")
            .expect("NLAI_HANDOFF_MARKER must name the backend observation file");
        let mut marker = std::fs::File::create(marker).expect("create backend observation file");
        writeln!(marker, "commit={}", identity.commit()).unwrap();
        writeln!(marker, "source={}", identity.source_state_fingerprint()).unwrap();
        writeln!(marker, "executable={}", identity.executable_fingerprint()).unwrap();
        writeln!(marker, "current_path_len={launcher_len}").unwrap();

        Box::new(CompiledModules { modules: vec![], allocator_module: None })
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        _sess: &Session,
        _outputs: &OutputFilenames,
        _crate_info: &CrateInfo,
    ) -> (CompiledModules, FxIndexMap<WorkProductId, WorkProduct>) {
        let modules = ongoing_codegen
            .downcast::<CompiledModules>()
            .expect("ongoing codegen must be CompiledModules");
        (*modules, FxIndexMap::default())
    }

    fn link(
        &self,
        _sess: &Session,
        _compiled_modules: CompiledModules,
        _crate_info: CrateInfo,
        _metadata: EncodedMetadata,
        _outputs: &OutputFilenames,
    ) {
    }
}

/// Entry point for the separately loaded test backend.
#[no_mangle]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    Box::new(HandoffBackend { identity: OnceLock::new() })
}
