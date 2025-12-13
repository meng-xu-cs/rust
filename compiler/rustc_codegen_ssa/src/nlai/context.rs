use rustc_ast::AttrStyle;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::attrs::AttributeKind;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_hir::{Attribute, CRATE_HIR_ID, HirId, ItemKind, Mod};
use rustc_middle::bug;
use rustc_middle::ty::TyCtxt;
use rustc_span::Ident;
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

    /// Collect all information about a module
    fn mk_module(&mut self, hir_id: HirId, ident: Ident, module: &'tcx Mod<'tcx>) -> SolModule {
        // must have a def_id
        let def_id = hir_id.expect_owner().to_def_id();

        // collect comments
        let mut doc_comments = Vec::new();
        for attr in self.tcx.hir_attrs(hir_id) {
            match attr {
                Attribute::Parsed(AttributeKind::DocComment {
                    style,
                    kind: _,
                    span: _,
                    comment,
                }) => match style {
                    AttrStyle::Outer => {
                        doc_comments.push(SolDocComment::Outer(comment.to_string()))
                    }
                    AttrStyle::Inner => {
                        doc_comments.push(SolDocComment::Inner(comment.to_string()))
                    }
                },
                _ => continue,
            }
        }

        // collect items defined in the module
        let mut item_modules = vec![];

        // iterate over all items in the module
        for item_id in module.item_ids {
            let item = self.tcx.hir_item(*item_id);
            let item_hir_id = item_id.hir_id();
            match item.kind {
                // dependencies
                ItemKind::ExternCrate(..) | ItemKind::Use(..) => {
                    // we don't dump information about dependencies or naming aliases
                    // as they have been already encoded in the identifier we dump.
                    continue;
                }

                // macro
                ItemKind::Macro(..) => {
                    // we don't dump information about macros under the assumption that
                    // they are expanded away during compilation, however, this also means
                    // that we will lose information that is only present in macros, e.g.,
                    // their comments, or some higher-level abstraction.
                    continue;
                }

                // globals
                ItemKind::Static(..) => {}
                ItemKind::Const(..) => {}

                // functions
                ItemKind::Fn { .. } => {}

                // datatypes
                ItemKind::Struct(..) => {}
                ItemKind::Enum(..) => {}
                ItemKind::Union(..) => {}
                ItemKind::TyAlias(..) => {}

                // traits
                ItemKind::Trait(..) => {}
                ItemKind::TraitAlias(..) => {}

                // nested modules
                ItemKind::Mod(submod_ident, submod_content) => {
                    let submod_parsed = self.mk_module(item_hir_id, submod_ident, submod_content);
                    item_modules.push(submod_parsed);
                }

                // impl blocks
                ItemKind::Impl(..) => {}

                // foreign interfaces
                ItemKind::ForeignMod { .. } => {}

                // unsupported items
                ItemKind::GlobalAsm { .. } => {
                    bug!("[unsupported] global assembly");
                }
            }
        }

        // construct the module
        SolModule {
            name: SolSymbol(ident.name.to_ident_string()),
            ident: self.mk_ident(def_id),
            doc_comments,
            item_modules,
        }
    }

    /// Build the crate
    pub(crate) fn build(mut self) -> SolCrate {
        // collect crate-level information
        let ident = self.mk_ident(LOCAL_CRATE.as_def_id());

        // recursively build the modules starting from the root module
        let module = self.mk_module(
            CRATE_HIR_ID,
            Ident::with_dummy_span(self.tcx.crate_name(LOCAL_CRATE)),
            self.tcx.hir_root_module(),
        );

        // unpack the builder
        let Self { tcx: _, id_cache } = self;

        // collect the id to description mappings
        let mut id_desc = vec![];

        // we don't care about the ordering of the values
        #[allow(rustc::potential_query_instability)]
        for (ident, desc) in id_cache.into_values() {
            id_desc.push((ident, desc));
        }

        // construct the crate
        SolCrate { ident, module, id_desc }
    }
}

/* --- BEGIN OF SYNC --- */

/*
 * Crate
 */

/// A complete crate
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolCrate {
    pub(crate) ident: SolIdent,
    pub(crate) module: SolModule,
    pub(crate) id_desc: Vec<(SolIdent, SolPathDesc)>,
}

/*
 * Module
 */

/// A module
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolModule {
    pub(crate) name: SolSymbol,
    pub(crate) ident: SolIdent,
    pub(crate) doc_comments: Vec<SolDocComment>,
    pub(crate) item_modules: Vec<SolModule>,
}

/*
* Comment
 */

/// A documentation comment
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolDocComment {
    Outer(String),
    Inner(String),
}

/*
 * Naming
 */

/// An identifier in the crate
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolIdent {
    pub(crate) krate: SolHash64,
    pub(crate) local: SolHash64,
}

/// A 64-bit hash
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolHash64(pub(crate) u64);

/// A name
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolSymbol(pub(crate) String);

/// A description of a definition path
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolPathDesc(pub(crate) String);

/* --- END OF SYNC --- */
