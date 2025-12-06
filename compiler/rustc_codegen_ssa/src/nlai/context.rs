use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_middle::ty::TyCtxt;
use serde::{Deserialize, Serialize};

/// A builder for creating a nlai module
pub(crate) struct Builder<'tcx> {
    /// compiler context
    tcx: TyCtxt<'tcx>,

    /// a cache of id to identifier mappings
    id_cache: FxHashMap<DefId, (SolIdent, SolPathDesc)>,
}

impl<'tcx> Builder<'tcx> {
    /// Create a new builder
    pub(crate) fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self { tcx, id_cache: FxHashMap::default() }
    }

    /// Create an identifier
    fn mk_ident(&mut self, def_id: DefId) -> SolIdent {
        // check cache first
        if let Some((ident, _)) = self.id_cache.get(&def_id) {
            return ident.clone();
        }

        // now construct the identifier
        let def_path_hash = self.tcx.def_path_hash(def_id);
        let ident = SolIdent {
            krate: SolHash64(def_path_hash.stable_crate_id().as_u64()),
            local: SolHash64(def_path_hash.local_hash().as_u64()),
        };

        // insert into the cache
        let desc = SolPathDesc(self.tcx.def_path_debug_str(def_id));
        self.id_cache.insert(def_id, (ident.clone(), desc));

        // return ident
        ident
    }

    /// Build the module
    pub(crate) fn build(mut self) -> SolModule {
        // collect crate-level information
        let crate_name = SolCrateName(self.tcx.crate_name(LOCAL_CRATE).to_ident_string());
        let crate_ident = self.mk_ident(LOCAL_CRATE.as_def_id());

        // collect crate-level comments
        let mut crate_comments = Vec::new();
        for attr in self.tcx.hir_krate_attrs() {
            if let Some((comment, _)) = attr.doc_str_and_comment_kind() {
                crate_comments.push(comment.to_string());
            }
        }

        // unpack the builder
        let Self { tcx: _, id_cache } = self;

        // collect the id to description mappings
        let mut id_desc = vec![];

        // we don't care about the ordering of the values
        #[allow(rustc::potential_query_instability)]
        for (ident, desc) in id_cache.into_values() {
            id_desc.push((ident, desc));
        }

        // construct the module
        SolModule { crate_name, crate_ident, crate_comments, id_desc }
    }
}

/* --- BEGIN OF SYNC --- */

/*
 * Module
 */

/// A complete nlai module
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolModule {
    pub(crate) crate_name: SolCrateName,
    pub(crate) crate_ident: SolIdent,
    pub(crate) crate_comments: Vec<String>,
    pub(crate) id_desc: Vec<(SolIdent, SolPathDesc)>,
}

/*
 * Naming
 */

/// An identifier in the Solana context
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolIdent {
    pub(crate) krate: SolHash64,
    pub(crate) local: SolHash64,
}

/// A 64-bit hash
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolHash64(pub(crate) u64);

/// A description of a definition path
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolPathDesc(pub(crate) String);

/// A crate name
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolCrateName(pub(crate) String);

/* --- END OF SYNC --- */
