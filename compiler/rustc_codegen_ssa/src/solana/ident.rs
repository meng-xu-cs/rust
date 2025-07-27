use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub(crate) struct Ident {
    pub krate: String,
    pub paths: Vec<String>,
}

impl Ident {
    pub(crate) fn new(tcx: TyCtxt<'_>, def_id: DefId) -> Self {
        let def_path = tcx.def_path(def_id);
        let krate = tcx.crate_name(def_path.krate).to_string();
        let paths = def_path.data.iter().map(|segment| segment.as_sym(false).to_string()).collect();
        Self { krate, paths }
    }
}
