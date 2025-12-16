use std::path::PathBuf;

use rustc_ast::AttrStyle;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::attrs::AttributeKind;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_hir::{
    Attribute, CRATE_HIR_ID, GenericParam, GenericParamKind, Generics, HirId, ItemKind,
    LifetimeParamKind, MissingLifetimeKind, Mod, ParamName,
};
use rustc_middle::bug;
use rustc_middle::ty::TyCtxt;
use rustc_span::{DUMMY_SP, Ident, RemapPathScopeComponents, Span, StableSourceFileId, Symbol};
use serde::{Deserialize, Serialize};

/// A builder for creating a nlai module
pub(crate) struct Builder<'tcx> {
    /// compiler context
    tcx: TyCtxt<'tcx>,

    /// source directory
    src_dir: PathBuf,

    /// source cache
    src_cache: FxHashSet<StableSourceFileId>,

    /// a cache of id to identifier mappings
    id_cache: FxHashMap<DefId, (SolIdent, SolPathDesc)>,
}

impl<'tcx> Builder<'tcx> {
    /// Create a new builder
    pub(crate) fn new(tcx: TyCtxt<'tcx>, src_dir: PathBuf) -> Self {
        Self { tcx, src_dir, src_cache: FxHashSet::default(), id_cache: FxHashMap::default() }
    }

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

    fn mk_symbol(&self, ident: Ident) -> SolSymbol {
        SolSymbol(ident.name.to_ident_string())
    }

    fn mk_span(&mut self, span: Span) -> SolSpan {
        let source_map = self.tcx.sess.source_map();

        // skip invisible spans
        if !span.is_visible(source_map) {
            return SolSpan {
                file_id: SolHash128(0),
                start_line: 0,
                start_column: 0,
                end_line: 0,
                end_column: 0,
            };
        }

        // ensure the span is within a single file
        let start_loc = source_map.lookup_char_pos(span.lo());
        let end_loc = source_map.lookup_char_pos(span.hi());

        let file_id = start_loc.file.stable_id;
        if end_loc.file.stable_id != file_id {
            bug!(
                "[invariant] span crosses multiple files: {} to {}",
                start_loc.file.name.display(RemapPathScopeComponents::OBJECT),
                end_loc.file.name.display(RemapPathScopeComponents::OBJECT)
            );
        }

        // dump the source content if not yet cached
        if !self.src_cache.contains(&file_id) {
            let src_code = start_loc.file.src.as_ref().map_or_else(
                || bug!("[invariant] visible span has no source code"),
                |src| src.clone(),
            );

            let src_path = self.src_dir.join(format!("{:x}", file_id.0.as_u128()));
            if src_path.exists() {
                bug!("[invariant] source file {} already exists", src_path.display());
            }

            std::fs::write(&src_path, src_code.as_str()).unwrap_or_else(|e| {
                bug!("[invariant] failed to write source code to file {}: {e}", src_path.display())
            });
            self.src_cache.insert(file_id.clone());
        }

        // construct the span
        SolSpan {
            file_id: SolHash128(file_id.0.as_u128()),
            start_line: start_loc.line as u32,
            start_column: start_loc.col.0 as u32,
            end_line: end_loc.line as u32,
            end_column: end_loc.col.0 as u32,
        }
    }

    fn mk_base(&mut self, hir_id: HirId, span: Span) -> SolBase {
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

        // construct the base
        SolBase { span: self.mk_span(span), ident: self.mk_ident(def_id), doc_comments }
    }

    fn mk_generic_param<'hir>(&mut self, param: GenericParam<'hir>) -> SolGenericParam {
        // sanity check
        if param.hir_id.expect_owner().def_id != param.def_id {
            bug!("[invariant] generic param hir_id and def_id do not match");
        }
        if matches!(param.name, ParamName::Error(_)) {
            bug!("[invariant] generic param has error name");
        }

        // construct the base
        let base = self.mk_base(param.hir_id, param.span);

        // switch case by kind
        let kind = match param.kind {
            GenericParamKind::Lifetime { kind } => match kind {
                LifetimeParamKind::Elided(reason) => {
                    match param.name {
                        ParamName::Plain(_) => {
                            bug!("[invariant] elided generic lifetime param has user-defined name");
                        }
                        ParamName::Fresh => {}
                        ParamName::Error(_) => unreachable!(),
                    }
                    match reason {
                        MissingLifetimeKind::Underscore => {
                            SolGenericParamKind::Lifetime { name: SolLifetimeName::ElidedExplicit }
                        }
                        MissingLifetimeKind::Ampersand => {
                            SolGenericParamKind::Lifetime { name: SolLifetimeName::ElidedImplicit }
                        }
                        MissingLifetimeKind::Comma => {
                            bug!("[unsupported] MissingLifetimeKind::Comma");
                        }
                        MissingLifetimeKind::Brackets => {
                            bug!("[unsupported] MissingLifetimeKind::Brackets");
                        }
                    }
                }
                LifetimeParamKind::Explicit => match param.name {
                    ParamName::Plain(ident) => SolGenericParamKind::Lifetime {
                        name: SolLifetimeName::Named(self.mk_symbol(ident)),
                    },
                    ParamName::Fresh => {
                        bug!("[invariant] generic lifetime param has fresh name");
                    }
                    ParamName::Error(_) => unreachable!(),
                },

                LifetimeParamKind::Error => {
                    bug!("[invariant] generic lifetime param has error kind");
                }
            },
            GenericParamKind::Type { default: _, synthetic } => {
                if synthetic {
                    bug!("[unsupported] generic type param is synthetic");
                }
                match param.name {
                    ParamName::Plain(ident) => {
                        SolGenericParamKind::Type { name: self.mk_symbol(ident) }
                    }
                    ParamName::Fresh => {
                        bug!("[invariant] generic type param has fresh name");
                    }
                    ParamName::Error(_) => unreachable!(),
                }
            }
            GenericParamKind::Const { ty: _, default: _ } => match param.name {
                ParamName::Plain(ident) => {
                    SolGenericParamKind::Const { name: self.mk_symbol(ident) }
                }
                ParamName::Fresh => {
                    bug!("[invariant] generic const param has fresh name");
                }
                ParamName::Error(_) => unreachable!(),
            },
        };

        // construct the generic param
        SolGenericParam { base, kind }
    }

    fn mk_generics<'hir>(&mut self, generics: &Generics<'hir>) -> SolGenerics {
        let mut params = vec![];
        for param in generics.params {
            params.push(self.mk_generic_param(*param));
        }
        SolGenerics { params }
    }

    fn mk_struct<'hir>(&mut self, name: Symbol, generics: &Generics<'hir>) -> SolStruct {
        let generics = self.mk_generics(generics);

        // construct the struct
        SolStruct { name: SolSymbol(name.to_ident_string()), generics }
    }

    fn mk_module(&mut self, name: Symbol, module: &'tcx Mod<'tcx>) -> SolModule {
        // collect items defined in the module
        let mut items = vec![];

        // iterate over all items in the module
        for item_id in module.item_ids {
            let item = self.tcx.hir_item(*item_id);
            let item_kind = match item.kind {
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

                // datatypes
                ItemKind::Struct(ident, generics, _) => {
                    SolItemKind::Struct(self.mk_struct(ident.name, generics))
                }
                ItemKind::Enum(..) => todo!(),
                ItemKind::Union(..) => todo!(),
                ItemKind::TyAlias(..) => todo!(),

                // traits
                ItemKind::Trait(..) => todo!(),
                ItemKind::TraitAlias(..) => todo!(),

                // functions
                ItemKind::Fn { .. } => todo!(),

                // impl blocks
                ItemKind::Impl(..) => todo!(),

                // foreign interfaces
                ItemKind::ForeignMod { .. } => todo!(),

                // globals
                ItemKind::Static(..) => todo!(),
                ItemKind::Const(..) => todo!(),

                // nested modules
                ItemKind::Mod(submod_ident, submod_content) => {
                    SolItemKind::Module(self.mk_module(submod_ident.name, submod_content))
                }

                // unsupported items
                ItemKind::GlobalAsm { .. } => bug!("[unsupported] global assembly"),
            };

            // parse item base and make the final item
            let item_base = self.mk_base(item_id.hir_id(), item.span);
            items.push(SolItem { base: item_base, kind: item_kind });
        }

        // construct the module
        SolModule {
            name: SolSymbol(name.to_ident_string()),
            scope: self.mk_span(module.spans.inner_span),
            items,
        }
    }

    /// Build the crate
    pub(crate) fn build(mut self) -> SolCrate {
        // construct the base
        let base = self.mk_base(CRATE_HIR_ID, DUMMY_SP);

        // recursively build the modules starting from the root module
        let module = self.mk_module(self.tcx.crate_name(LOCAL_CRATE), self.tcx.hir_root_module());

        // unpack the builder
        let Self { tcx: _, src_dir: _, src_cache: _, id_cache } = self;

        // collect the id to description mappings
        let mut id_desc = vec![];

        // we don't care about the ordering of the values
        #[allow(rustc::potential_query_instability)]
        for (ident, desc) in id_cache.into_values() {
            id_desc.push((ident, desc));
        }

        // construct the crate
        SolCrate { base, module, id_desc }
    }
}

/* --- BEGIN OF SYNC --- */

/*
* Common
 */

/// The base information associated with any HIR element
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolBase {
    pub(crate) span: SolSpan,
    pub(crate) ident: SolIdent,
    pub(crate) doc_comments: Vec<SolDocComment>,
}

/*
 * Crate
 */

/// A complete crate
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolCrate {
    pub(crate) base: SolBase,
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
    pub(crate) scope: SolSpan,
    pub(crate) items: Vec<SolItem>,
}

/*
 * Item
 */

/// Details associated with an item
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolItemKind {
    Module(SolModule),
    Struct(SolStruct),
}

/// An item recognizable in the crate
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolItem {
    pub(crate) base: SolBase,
    pub(crate) kind: SolItemKind,
}

/*
* Generics
*/

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolLifetimeName {
    ElidedExplicit,
    ElidedImplicit,
    Named(SolSymbol),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolGenericParamKind {
    Lifetime { name: SolLifetimeName },
    Type { name: SolSymbol },
    Const { name: SolSymbol },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolGenericParam {
    pub(crate) base: SolBase,
    pub(crate) kind: SolGenericParamKind,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolGenerics {
    pub(crate) params: Vec<SolGenericParam>,
}

/*
* Type
*/

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolStruct {
    pub(crate) name: SolSymbol,
    pub(crate) generics: SolGenerics,
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

/// A 128-bit hash
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolHash128(pub(crate) u128);

/// A name
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolSymbol(pub(crate) String);

/// A description of a definition path
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolPathDesc(pub(crate) String);

/// A span description
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolSpan {
    pub(crate) file_id: SolHash128,
    pub(crate) start_line: u32,
    pub(crate) start_column: u32,
    pub(crate) end_line: u32,
    pub(crate) end_column: u32,
}

/* --- END OF SYNC --- */
