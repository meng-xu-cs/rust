use std::fmt::Debug;
use std::path::PathBuf;

use rustc_abi::ExternAbi;
use rustc_ast::AttrStyle;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::attrs::AttributeKind;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_hir::{Attribute, CRATE_HIR_ID, HirId, Item, ItemKind, Mod, Safety};
use rustc_middle::bug;
use rustc_middle::thir::{BodyTy, ExprId, Thir};
use rustc_middle::ty::{FnSig, Ty, TyCtxt};
use rustc_span::{DUMMY_SP, RemapPathScopeComponents, Span, StableSourceFileId, Symbol};
use serde::de::DeserializeOwned;
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

    fn mk_doc_comments(&self, hir_id: HirId) -> Vec<SolDocComment> {
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
        doc_comments
    }

    #[allow(dead_code)]
    fn mk_ast<T: SolIR>(&mut self, span: Span, data: T) -> SolAST<T> {
        SolAST { span: self.mk_span(span), data }
    }

    fn mk_mir<T: SolIR>(&mut self, hir_id: HirId, span: Span, data: T) -> SolMIR<T> {
        SolMIR {
            ident: self.mk_ident(hir_id.expect_owner().to_def_id()),
            span: self.mk_span(span),
            doc_comments: self.mk_doc_comments(hir_id),
            data,
        }
    }

    fn mk_module(&mut self, name: Symbol, module: Mod<'tcx>) -> SolModule {
        let Mod { spans, item_ids } = module;

        // collect items defined in the module
        let mut items = vec![];

        // iterate over all items in the module
        for item_id in item_ids {
            let Item { owner_id, kind, span, vis_span: _, has_delayed_lints: _, eii: _ } =
                *self.tcx.hir_item(*item_id);

            let item_mir = match kind {
                // dependencies
                ItemKind::ExternCrate(..) | ItemKind::Use(..) | ItemKind::ForeignMod { .. } => {
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
                ItemKind::Struct(..)
                | ItemKind::Enum(..)
                | ItemKind::Union(..)
                | ItemKind::TyAlias(..) => {
                    // we don't dump datatype definitions, instead, we will dump the MIR types
                    // which are referred to in THIR
                    continue;
                }

                // traits
                ItemKind::Trait(..) | ItemKind::TraitAlias(..) => {
                    // we don't dump trait definitions, instead we will dump the THIR function
                    // bodies with traits resolved
                    continue;
                }

                // functions and function-alikes
                ItemKind::Fn { .. } | ItemKind::Impl(..) | ItemKind::GlobalAsm { .. } => {
                    // we don't dump function-alikes at HIR level, we will dump their THIR
                    continue;
                }

                // globals
                ItemKind::Static(..) | ItemKind::Const(..) => {
                    // we don't dump global constant definitions at HIR level, we will dump their THIR
                    continue;
                }

                // nested modules
                ItemKind::Mod(mod_ident, mod_content) => {
                    let module_data = self.mk_module(mod_ident.name, *mod_content);
                    self.mk_mir(
                        HirId::make_owner(owner_id.def_id),
                        span,
                        SolItem::Module(module_data),
                    )
                }
            };

            // make the final item
            items.push(item_mir);
        }

        // construct the module
        SolModule {
            name: SolSymbol(name.to_ident_string()),
            scope: self.mk_span(spans.inner_span),
            items,
        }
    }

    pub(crate) fn mk_abi(
        &mut self,
        abi: ExternAbi,
        c_variadic: bool,
        safety: Safety,
    ) -> SolExternAbi {
        match abi {
            ExternAbi::Rust => {
                if c_variadic {
                    bug!("[invariant] Rust ABI should not be variadic");
                }
                match safety {
                    Safety::Safe => SolExternAbi::Rust { safety: true },
                    Safety::Unsafe => SolExternAbi::Rust { safety: false },
                }
            }
            ExternAbi::C { unwind: _ } => SolExternAbi::C { variadic: c_variadic },
            ExternAbi::System { unwind: _ } => {
                if c_variadic {
                    bug!("[invariant] System ABI should not be variadic");
                }
                SolExternAbi::System
            }
            _ => bug!("[unsupported] ABI {:?}", abi),
        }
    }

    pub(crate) fn mk_type(&mut self, _ty: Ty<'tcx>) -> SolType {
        todo!()
    }

    pub(crate) fn mk_exec(&mut self, thir: Thir<'tcx>, _expr: ExprId) -> SolExec {
        // switch-case on the body type
        match thir.body_type {
            BodyTy::Fn(sig) => {
                let FnSig { abi, c_variadic, safety, inputs_and_output: _ } = sig;

                // parse function signature
                let parsed_abi = self.mk_abi(abi, c_variadic, safety);
                let ret_ty = self.mk_type(sig.output());
                let mut params = vec![];
                for input_ty in sig.inputs() {
                    params.push(self.mk_type(*input_ty));
                }

                // parse parameters
                if thir.params.len() != params.len() {
                    bug!(
                        "[invariant] parameter count mismatch: THIR has {} but signature has {}",
                        thir.params.len(),
                        params.len()
                    );
                }

                // pack the information
                SolExec::Function(SolFnDef { abi: parsed_abi, ret_ty, params })
            }
            BodyTy::Const(ty) => {
                // sanity checks
                if !thir.params.is_empty() {
                    bug!("[invariant] constant body should not have parameters");
                }

                // parse type
                let const_ty = self.mk_type(ty);

                // pack the information
                SolExec::Constant(SolConst { ty: const_ty })
            }
            BodyTy::GlobalAsm(..) => bug!("[unsupported] global assembly"),
        }
    }

    /// Build the crate
    pub(crate) fn build(mut self) -> SolCrate {
        // recursively build the modules starting from the root module
        let module_data =
            self.mk_module(self.tcx.crate_name(LOCAL_CRATE), *self.tcx.hir_root_module());
        let module_mir = self.mk_mir(CRATE_HIR_ID, DUMMY_SP, module_data);

        // process all body owners in this crate
        let mut executables = vec![];
        for owner_id in self.tcx.hir_body_owners() {
            let (thir_body, thir_expr) = self.tcx.thir_body(owner_id).unwrap_or_else(|_| {
                panic!(
                    "[invariant] failed to retrieve THIR body for {}",
                    self.tcx.def_path_debug_str(owner_id.to_def_id())
                )
            });

            // build the executable body
            let exec = self.mk_exec(thir_body.steal(), thir_expr);
            let hir_id = HirId::make_owner(owner_id);
            executables.push(self.mk_mir(hir_id, self.tcx.hir_span_with_body(hir_id), exec));
        }

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
        SolCrate { root: module_mir, id_desc }
    }
}

/* --- BEGIN OF SYNC --- */

/// A trait alias for all sorts IR elements
pub(crate) trait SolIR =
    Debug + Clone + PartialEq + Eq + PartialOrd + Ord + Serialize + DeserializeOwned;

/*
* Common
 */

/// Anything that has a span but does not have an hir_id
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(bound = "T: Serialize + DeserializeOwned")]
pub(crate) struct SolAST<T: SolIR> {
    pub(crate) span: SolSpan,
    pub(crate) data: T,
}

/// The base information associated with anything that has an hir_id (and span)
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(bound = "T: Serialize + DeserializeOwned")]
pub(crate) struct SolMIR<T: SolIR> {
    pub(crate) ident: SolIdent,
    pub(crate) span: SolSpan,
    pub(crate) doc_comments: Vec<SolDocComment>,
    pub(crate) data: T,
}

/*
 * Crate
 */

/// A complete crate
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolCrate {
    pub(crate) root: SolMIR<SolModule>,
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
    pub(crate) items: Vec<SolMIR<SolItem>>,
}

/*
 * Item
 */

/// Details associated with an item
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolItem {
    Module(SolModule),
}

/*
 * Exec (THIR)
 */

/// The main body of a THIR (e.g., a function or a constant)
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolExec {
    Function(SolFnDef),
    Constant(SolConst),
}

/// THIR of a function
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolFnDef {
    pub(crate) abi: SolExternAbi,
    pub(crate) ret_ty: SolType,
    pub(crate) params: Vec<SolType>,
}

/// THIR of a constant
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolConst {
    pub(crate) ty: SolType,
}

/// External ABI of a function
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolExternAbi {
    C { variadic: bool },
    Rust { safety: bool },
    System,
}

/*
 * Typing
 */

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolType {
    // baseline
    Never,
    // primitive types
    Bool,
    Char,
    I8,
    I16,
    I32,
    I64,
    I128,
    Isize,
    U8,
    U16,
    U32,
    U64,
    U128,
    Usize,
    F16,
    F32,
    F64,
    F128,
    Str,
    // compound types
    Tuple(Vec<SolType>),
    Slice(Box<SolType>),
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
