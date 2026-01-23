use std::collections::{BTreeMap, VecDeque};
use std::fmt::Debug;
use std::path::PathBuf;

use rustc_abi::{ExternAbi, FieldIdx, VariantIdx};
use rustc_ast::{
    AttrStyle, BindingMode, ByRef, FloatTy, IntTy, LitFloatType, LitIntType, LitKind, Mutability,
    UintTy,
};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::attrs::AttributeKind;
use rustc_hir::def_id::{CRATE_DEF_ID, DefId, LOCAL_CRATE, LocalDefId};
use rustc_hir::{
    Attribute, CRATE_HIR_ID, HirId, Item, ItemKind, MatchSource, Mod, OwnerId, Safety,
};
use rustc_middle::mir::{AssignOp, BinOp, BorrowKind, UnOp};
use rustc_middle::thir::{
    AdtExpr, AdtExprBase, Arm, Block, BlockId, BlockSafety, BodyTy, ClosureExpr,
    DerefPatBorrowMode, Expr, ExprId, ExprKind, FieldExpr, FieldPat, FruInfo, LocalVarId,
    LogicalOp, Pat, PatKind, Stmt, StmtId, StmtKind, Thir,
};
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::{
    AdtDef, AdtKind, AliasTyKind, Clause, ClauseKind, Const, ConstKind, FnHeader, FnSig,
    GenericArg, GenericArgKind, GenericArgsRef, GenericParamDef, GenericParamDefKind, List,
    OutlivesPredicate, ParamConst, ParamTy, Pattern, PatternKind, PredicatePolarity,
    ProjectionPredicate, ScalarInt, Term, TermKind, TraitDef, TraitPredicate, Ty, TyCtxt,
    TypingEnv, UpvarArgs, ValTreeKind, Value, VariantDiscr,
};
use rustc_middle::{bug, ty};
use rustc_span::{DUMMY_SP, RemapPathScopeComponents, Span, StableSourceFileId, Symbol};
use rustc_trait_selection::infer::TyCtxtInferExt;
use rustc_trait_selection::traits;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tracing::info;

/// A builder for the overall item-level NLAI representation
struct BaseBuilder<'tcx> {
    /// compiler context
    tcx: TyCtxt<'tcx>,

    /// source directory
    src_dir: PathBuf,

    /// source cache
    src_cache: FxHashSet<StableSourceFileId>,

    /// a cache of id to identifier mappings
    id_cache: FxHashMap<DefId, (SolIdent, SolPathDesc)>,
}

/// A builder for each executable THIR body
struct ExecBuilder<'tcx> {
    /// compiler context
    tcx: TyCtxt<'tcx>,

    /// typing env, for normalization and region erasure
    typing_env: TypingEnv<'tcx>,

    /// base builder
    base: BaseBuilder<'tcx>,

    /// owner id
    owner_id: OwnerId,

    /// generics declarations
    generics: Vec<SolGenericParam>,

    /// collected definitions of datatypes
    adt_defs: BTreeMap<SolIdent, BTreeMap<Vec<SolGenericArg>, Option<SolAdtDef>>>,

    /// collected definitions of traits
    trait_defs: BTreeMap<SolIdent, BTreeMap<Vec<SolGenericArg>, Option<SolTraitDef>>>,

    /// log stack
    log_stack: LogStack,
}

impl<'tcx> BaseBuilder<'tcx> {
    /// Create a new builder
    pub(crate) fn new(tcx: TyCtxt<'tcx>, src_dir: PathBuf) -> Self {
        Self { tcx, src_dir, src_cache: FxHashSet::default(), id_cache: FxHashMap::default() }
    }

    /// Record an identifier (with caching)
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

    /// Record a span
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

    /// Record the doc comments associated with a hir_id
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

    /// Record a MIR node in the THIR body with its metadata
    fn mk_mir<T: SolIR>(
        &mut self,
        def_id: LocalDefId,
        hir_id: HirId,
        span: Span,
        data: T,
    ) -> SolMIR<T> {
        SolMIR {
            ident: self.mk_ident(def_id.to_def_id()),
            span: self.mk_span(span),
            doc_comments: self.mk_doc_comments(hir_id),
            data,
        }
    }

    /// Record a module
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
                        owner_id.def_id,
                        item_id.hir_id(),
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
            name: SolModuleName(name.to_ident_string()),
            scope: self.mk_span(spans.inner_span),
            items,
        }
    }
}

impl<'tcx> ExecBuilder<'tcx> {
    /// Create a new builder
    pub(crate) fn new(
        base: BaseBuilder<'tcx>,
        owner_id: OwnerId,
        generics: Vec<SolGenericParam>,
    ) -> Self {
        let tcx = base.tcx;
        Self {
            tcx,
            typing_env: TypingEnv::post_analysis(tcx, owner_id).with_post_analysis_normalized(tcx),
            base,
            owner_id,
            generics,
            adt_defs: BTreeMap::new(),
            trait_defs: BTreeMap::new(),
            log_stack: LogStack::new(),
        }
    }

    /// Get a generic parameter by index with cross-checking on name and kind
    fn get_param(&self, index: u32, name: Symbol, kind: SolGenericKind) -> SolIdent {
        // validate the type parameter against the generics declarations
        let param_def = self.generics.get(index as usize).unwrap_or_else(|| {
            bug!("[invariant] generic parameter {name} index out of bounds: {index}",)
        });
        if param_def.kind != kind {
            bug!("[invariant] generic parameter {name} at index {index} is not a {kind:?}")
        }
        if param_def.name.0 != name.to_ident_string() {
            bug!(
                "[invariant] generic parameter name mismatch at index {index}: {} vs {name}",
                param_def.name.0,
            )
        }

        // done with validation, return the ident
        param_def.ident.clone()
    }

    /// Record an identifier (with caching)
    #[inline]
    fn mk_ident(&mut self, def_id: DefId) -> SolIdent {
        self.base.mk_ident(def_id)
    }

    /// Record a span
    #[inline]
    fn mk_span(&mut self, span: Span) -> SolSpan {
        self.base.mk_span(span)
    }

    /// Record the doc comments associated with a hir_id
    fn mk_doc_comments(&self, hir_id: HirId) -> Vec<SolDocComment> {
        self.base.mk_doc_comments(hir_id)
    }

    /// Record a HIR node in the THIR body with its metadata
    fn mk_hir<T: SolIR>(&mut self, hir_id: HirId, data: T) -> SolHIR<T> {
        if hir_id.is_owner() {
            bug!("[invariant] expected non-owner HirId, found owner: {:?}", hir_id);
        }
        SolHIR { doc_comments: self.mk_doc_comments(hir_id), data }
    }

    /// Record a function ABI
    pub(crate) fn mk_abi(
        &mut self,
        abi: ExternAbi,
        c_variadic: bool,
        safety: Safety,
    ) -> SolExternAbi {
        match abi {
            ExternAbi::Rust | ExternAbi::RustCall => {
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
            _ => bug!("[unsupported] ABI"),
        }
    }

    /// Record a generic argument
    pub(crate) fn mk_generic_arg(&mut self, ty_arg: GenericArg<'tcx>) -> SolGenericArg {
        match ty_arg.kind() {
            GenericArgKind::Type(ty) => SolGenericArg::Type(self.mk_type(ty)),
            GenericArgKind::Const(val) => SolGenericArg::Const(self.mk_const(val)),
            GenericArgKind::Lifetime(region) => {
                if !(region.is_erased() || region.is_static()) {
                    bug!("[invariant] regions should be erased or static in THIR: {region}");
                }
                SolGenericArg::Lifetime
            }
        }
    }

    /// Record a projection term in the type system
    pub(crate) fn mk_projection_term(&mut self, term: Term<'tcx>) -> SolProjTerm {
        match term.kind() {
            TermKind::Ty(ty) => SolProjTerm::Type(self.mk_type(ty)),
            TermKind::Const(cval) => SolProjTerm::Const(self.mk_const(cval)),
        }
    }

    /// Record a pattern type
    pub(crate) fn mk_ty_pat(&mut self, pat: Pattern<'tcx>) -> SolTyPat {
        match *pat {
            PatternKind::NotNull => SolTyPat::NotNull,
            PatternKind::Range { start, end } => {
                SolTyPat::Range(self.mk_const(start), self.mk_const(end))
            }
            PatternKind::Or(patterns) => {
                SolTyPat::Or(patterns.iter().map(|sub_pat| self.mk_ty_pat(sub_pat)).collect())
            }
        }
    }

    /// Record the definition of an ADT
    pub(crate) fn mk_adt(
        &mut self,
        adt_def: AdtDef<'tcx>,
        ty_args: GenericArgsRef<'tcx>,
    ) -> (SolIdent, Vec<SolGenericArg>) {
        let def_id = adt_def.did();

        // locate the key of the definition
        let ident = self.mk_ident(def_id);
        let generic_args = ty_args.iter().map(|arg| self.mk_generic_arg(arg)).collect();

        // if already defined or is being defined, return the key
        if self.adt_defs.get(&ident).map_or(false, |inner| inner.contains_key(&generic_args)) {
            return (ident, generic_args);
        }

        // mark start
        let def_desc = util_debug_symbol(self.tcx, def_id, ty_args);
        self.log_stack.push("ADT", def_desc.clone());

        // first update the entry to mark that type definition in progress
        self.adt_defs.entry(ident.clone()).or_default().insert(generic_args.clone(), None);

        // now create the type definition
        let parsed_def = match adt_def.adt_kind() {
            AdtKind::Struct => {
                if adt_def.variants().len() != 1 {
                    bug!("[invariant] struct {def_desc} has multiple variants");
                }
                let fields = adt_def
                    .non_enum_variant()
                    .fields
                    .iter_enumerated()
                    .map(|(field_idx, field_def)| SolField {
                        index: SolFieldIndex(field_idx.index()),
                        name: SolFieldName(field_def.name.to_ident_string()),
                        ty: self.mk_type(field_def.ty(self.tcx, ty_args)),
                        default: field_def.value.map(|did| self.mk_ident(did)),
                    })
                    .collect();
                SolAdtDef::Struct { fields }
            }
            AdtKind::Union => {
                if adt_def.variants().len() != 1 {
                    bug!("[invariant] union {def_desc} has multiple variants");
                }
                let fields = adt_def
                    .non_enum_variant()
                    .fields
                    .iter_enumerated()
                    .map(|(field_idx, field_def)| SolField {
                        index: SolFieldIndex(field_idx.index()),
                        name: SolFieldName(field_def.name.to_ident_string()),
                        ty: self.mk_type(field_def.ty(self.tcx, ty_args)),
                        default: field_def.value.map(|did| self.mk_ident(did)),
                    })
                    .collect();
                SolAdtDef::Union { fields }
            }
            AdtKind::Enum => {
                let mut variants = vec![];
                let mut last_discr_value = 0;
                for (variant_idx, variant_def) in adt_def.variants().iter_enumerated() {
                    let variant_index = SolVariantIndex(variant_idx.index());
                    let variant_name = SolVariantName(variant_def.name.to_ident_string());
                    let variant_descr = match variant_def.discr {
                        VariantDiscr::Relative(pos) => {
                            SolVariantDiscr(last_discr_value + pos as u128)
                        }
                        VariantDiscr::Explicit(did) => {
                            let discr = adt_def.eval_explicit_discr(self.tcx, did).unwrap_or_else(|_| {
                                    bug!("[invariant] failed to evaluate discriminant for enum {def_desc}")
                                });
                            if !matches!(discr.ty.kind(), ty::Int(_) | ty::Uint(_)) {
                                bug!("[invariant] non-integral discriminant for enum {def_desc}");
                            }
                            last_discr_value = discr.val;
                            SolVariantDiscr(discr.val)
                        }
                    };
                    let fields = variant_def
                        .fields
                        .iter_enumerated()
                        .map(|(field_idx, field_def)| SolField {
                            index: SolFieldIndex(field_idx.index()),
                            name: SolFieldName(field_def.name.to_ident_string()),
                            ty: self.mk_type(field_def.ty(self.tcx, ty_args)),
                            default: field_def.value.map(|did| self.mk_ident(did)),
                        })
                        .collect();
                    variants.push(SolVariant {
                        index: variant_index,
                        name: variant_name,
                        discr: variant_descr,
                        fields,
                    });
                }
                SolAdtDef::Enum { variants }
            }
        };

        // update the type definition
        self.adt_defs
            .entry(ident.clone())
            .or_default()
            .insert(generic_args.clone(), Some(parsed_def));

        // mark end
        self.log_stack.pop();

        // return the result
        (ident, generic_args)
    }

    /// Record the definition of a trait
    pub(crate) fn mk_trait(
        &mut self,
        trait_def: &TraitDef,
        ty_args: GenericArgsRef<'tcx>,
    ) -> (SolIdent, Vec<SolGenericArg>) {
        let def_id = trait_def.def_id;

        // locate the key of the definition
        let ident = self.mk_ident(def_id);
        // NOTE: skip the first generic arg which is Self type
        let generic_args = ty_args.iter().skip(1).map(|arg| self.mk_generic_arg(arg)).collect();

        // if already defined or is being defined, return the key
        if self.trait_defs.get(&ident).map_or(false, |inner| inner.contains_key(&generic_args)) {
            return (ident, generic_args);
        }

        // mark start
        let def_desc = util_debug_symbol(self.tcx, def_id, ty_args);
        self.log_stack.push("Trait", def_desc.clone());

        // first update the entry to mark that trait definition in progress
        self.trait_defs.entry(ident.clone()).or_default().insert(generic_args.clone(), None);

        // collect explicit predicates of the trait
        let mut parsed_clauses = vec![];
        for (clause, _) in
            self.tcx.explicit_predicates_of(def_id).instantiate(self.tcx, ty_args).into_iter()
        {
            parsed_clauses.push(self.mk_clause(clause));
        }

        // FIXME: check `TyCtxt::trait_explicit_predicates_and_bounds`,
        // we might want to collect constraints for associated type bounds as well

        // update the trait definition
        self.trait_defs
            .entry(ident.clone())
            .or_default()
            .insert(generic_args.clone(), Some(SolTraitDef { clauses: parsed_clauses }));

        // mark end
        self.log_stack.pop();

        // return the result
        (ident, generic_args)
    }

    /// Record a clause in a predicate
    pub(crate) fn mk_clause(&mut self, clause: Clause<'tcx>) -> SolClause {
        self.log_stack.push("Clause", format!("{clause}"));

        let parsed = match self.tcx.instantiate_bound_regions_with_erased(clause.kind()) {
            ClauseKind::Trait(TraitPredicate { trait_ref, polarity }) => {
                let (trait_ident, trait_ty_args) =
                    self.mk_trait(self.tcx.trait_def(trait_ref.def_id), trait_ref.args);
                match polarity {
                    PredicatePolarity::Positive => SolClause::TraitImpl(trait_ident, trait_ty_args),
                    PredicatePolarity::Negative => {
                        SolClause::TraitNotImpl(trait_ident, trait_ty_args)
                    }
                }
            }

            // less important clauses
            ClauseKind::WellFormed(term) => SolClause::WellFormed(self.mk_projection_term(term)),
            ClauseKind::Projection(ProjectionPredicate { projection_term, term }) => {
                SolClause::Projection(
                    self.mk_projection_term(projection_term.to_term(self.tcx)),
                    self.mk_projection_term(term),
                )
            }
            ClauseKind::TypeOutlives(OutlivesPredicate(ty, region)) => {
                if !(region.is_erased() || region.is_static()) {
                    bug!("[invariant] regions should be erased or static in THIR: {region}");
                }
                SolClause::TypeOutlives(self.mk_type(ty))
            }
            ClauseKind::RegionOutlives(OutlivesPredicate(r1, r2)) => {
                if !(r1.is_erased() || r1.is_static()) {
                    bug!("[invariant] regions should be erased or static in THIR: {r1}");
                }
                if !(r2.is_erased() || r2.is_static()) {
                    bug!("[invariant] regions should be erased or static in THIR: {r2}");
                }
                SolClause::RegionOutlives
            }
            ClauseKind::ConstArgHasType(cval, ty) => {
                SolClause::ConstHasType(self.mk_const(cval), self.mk_type(ty))
            }
            ClauseKind::ConstEvaluatable(cval) => SolClause::ConstEvaluatable(self.mk_const(cval)),

            // unsupported
            ClauseKind::HostEffect(..) => {
                bug!("[unsupported] clause");
            }

            // unexpected
            ClauseKind::UnstableFeature(..) => {
                bug!("[invariant] unexpected unstable feature clause: {clause}");
            }
        };

        self.log_stack.pop();
        parsed
    }

    /// Record a type in MIR/THIR context
    pub(crate) fn mk_type(&mut self, ty: Ty<'tcx>) -> SolType {
        self.log_stack.push("Type", format!("{ty}"));

        let parsed = match ty.kind() {
            // baseline
            ty::Never => SolType::Never,

            // primitive types
            ty::Bool => SolType::Bool,
            ty::Char => SolType::Char,
            ty::Int(int_ty) => match int_ty {
                ty::IntTy::I8 => SolType::I8,
                ty::IntTy::I16 => SolType::I16,
                ty::IntTy::I32 => SolType::I32,
                ty::IntTy::I64 => SolType::I64,
                ty::IntTy::I128 => SolType::I128,
                ty::IntTy::Isize => SolType::Isize,
            },
            ty::Uint(uint_ty) => match uint_ty {
                ty::UintTy::U8 => SolType::U8,
                ty::UintTy::U16 => SolType::U16,
                ty::UintTy::U32 => SolType::U32,
                ty::UintTy::U64 => SolType::U64,
                ty::UintTy::U128 => SolType::U128,
                ty::UintTy::Usize => SolType::Usize,
            },
            ty::Float(float_ty) => match float_ty {
                ty::FloatTy::F16 => SolType::F16,
                ty::FloatTy::F32 => SolType::F32,
                ty::FloatTy::F64 => SolType::F64,
                ty::FloatTy::F128 => SolType::F128,
            },
            ty::Str => SolType::Str,

            // pattern types
            ty::Pat(base_ty, pattern) => {
                SolType::Pat(Box::new(self.mk_type(*base_ty)), Box::new(self.mk_ty_pat(*pattern)))
            }

            // dependencies
            ty::Foreign(def_id) => SolType::Foreign(self.mk_ident(*def_id)),
            ty::Adt(adt_def, ty_args) => {
                let (ident, generic_args) = self.mk_adt(*adt_def, ty_args);
                SolType::Adt(ident, generic_args)
            }

            // reference types
            ty::RawPtr(pointee_ty, mutability) => {
                let sub_ty = self.mk_type(*pointee_ty);
                match mutability {
                    Mutability::Not => SolType::ImmPtr(Box::new(sub_ty)),
                    Mutability::Mut => SolType::MutPtr(Box::new(sub_ty)),
                }
            }
            ty::Ref(region, referent_ty, mutability) => {
                if !(region.is_erased() || region.is_static()) {
                    bug!("[invariant] regions should be erased or static in THIR: {region}");
                }
                let sub_ty = self.mk_type(*referent_ty);
                match mutability {
                    Mutability::Not => SolType::ImmRef(Box::new(sub_ty)),
                    Mutability::Mut => SolType::MutRef(Box::new(sub_ty)),
                }
            }

            // type parameter
            ty::Param(ParamTy { index, name }) => {
                SolType::Param(self.get_param(*index, *name, SolGenericKind::Type))
            }

            // compound types
            ty::Tuple(sub_tys) => {
                SolType::Tuple(sub_tys.iter().map(|sub_ty| self.mk_type(sub_ty)).collect())
            }
            ty::Slice(elem_ty) => SolType::Slice(Box::new(self.mk_type(*elem_ty))),
            ty::Array(elem_ty, size) => {
                SolType::Array(Box::new(self.mk_type(*elem_ty)), Box::new(self.mk_const(*size)))
            }

            // function pointer
            ty::FnDef(def_id, ty_args) => {
                let ident = self.mk_ident(*def_id);
                let generic_args = ty_args.iter().map(|arg| self.mk_generic_arg(arg)).collect();
                SolType::Function(ident, generic_args)
            }
            ty::Closure(def_id, ty_args) => {
                let ident = self.mk_ident(*def_id);
                let generic_args = ty_args.iter().map(|arg| self.mk_generic_arg(arg)).collect();
                SolType::Closure(ident, generic_args)
            }
            ty::FnPtr(sig_binder, FnHeader { c_variadic, abi, safety }) => {
                let sig = self.tcx.instantiate_bound_regions_with_erased(*sig_binder);
                let abi = self.mk_abi(*abi, *c_variadic, *safety);

                let ret_ty = self.mk_type(sig.output());
                let params: Vec<_> =
                    sig.inputs().iter().map(|input_ty| self.mk_type(*input_ty)).collect();
                SolType::FnPtr(abi, params, Box::new(ret_ty))
            }

            // dynamic
            ty::Dynamic(predicates, region) => {
                if !(region.is_erased() || region.is_static()) {
                    bug!("[invariant] regions should be erased or static in THIR: {region}");
                }
                let clauses = predicates
                    .iter()
                    .map(|pred|
                        // NOTE: we use the dynamic type itself as the self type
                        // so that we can instantiate the clauses properly
                        self.mk_clause(pred.with_self_ty(self.tcx, ty)))
                    .collect();
                SolType::Dynamic(clauses)
            }

            // alias
            ty::Alias(AliasTyKind::Projection, alias_ty) => {
                match self.tcx.try_normalize_erasing_regions(self.typing_env, ty) {
                    Ok(norm_ty) if norm_ty != ty => SolType::Alias(Box::new(self.mk_type(norm_ty))),
                    _ => {
                        // It is possible that normalization does not change the type, for example,
                        // when the projection comes from a type parameter (which implements a trait),
                        // e.g., <P as Foo>::T, where P is the type parameter of the THIR definition.
                        //
                        // Another possiblility is that we treat the `dyn Trait` it itself as Self,
                        // and the projection is from the trait object, e.g., `<dyn Foo as Foo>::T`.
                        // We can't further normalize the `dyn Foo` here.
                        let (trait_ref, own_ty_args) = alias_ty.trait_ref_and_own_args(self.tcx);
                        let (trait_ident, trait_ty_args) =
                            self.mk_trait(self.tcx.trait_def(trait_ref.def_id), trait_ref.args);
                        let item_ty_args =
                            own_ty_args.iter().map(|arg| self.mk_generic_arg(*arg)).collect();

                        // construct the associated type
                        SolType::Assoc {
                            trait_ident,
                            trait_ty_args,
                            item_ident: self.mk_ident(alias_ty.def_id),
                            item_ty_args,
                        }
                    }
                }
            }
            ty::Alias(AliasTyKind::Inherent, _) => {
                let norm_ty = self.tcx.normalize_erasing_regions(self.typing_env, ty);
                assert_ne!(norm_ty, ty, "[invariant] inherent alias type should be normalized");
                SolType::Alias(Box::new(self.mk_type(norm_ty)))
            }
            ty::Alias(AliasTyKind::Opaque, alias_ty) => {
                let actual_ty = self.mk_type(
                    self.tcx.type_of(alias_ty.def_id).instantiate(self.tcx, alias_ty.args),
                );
                SolType::Opaque(Box::new(actual_ty))
            }
            ty::Alias(AliasTyKind::Free, _) => {
                let norm_ty = self.tcx.normalize_erasing_regions(self.typing_env, ty);
                assert_ne!(norm_ty, ty, "[invariant] free alias type should be normalized");
                SolType::Opaque(Box::new(self.mk_type(norm_ty)))
            }

            // unsupported
            ty::Coroutine(..) | ty::CoroutineClosure(..) | ty::CoroutineWitness(..) => {
                bug!("[unsupported] coroutine type")
            }
            ty::UnsafeBinder(..) => {
                bug!("[unsupported] unsafe binder")
            }

            // unexpected
            ty::Infer(..) | ty::Bound(..) | ty::Placeholder(..) | ty::Error(..) => {
                bug!("[invariant] unexpected type: {ty}")
            }
        };

        self.log_stack.pop();
        parsed
    }

    /// Record a pattern in the THIR context
    pub(crate) fn mk_pat(&mut self, pat: &Pat<'tcx>) -> SolPattern {
        let Pat { ty, span, extra: _, kind } = pat;

        // parse the basics
        let pat_ty = self.mk_type(*ty);
        let pat_span = self.mk_span(*span);

        // MAYFIX: right now we don't care about the extra info, but maybe we want to record them?

        // parse the pattern kind
        let pat_rule = match kind {
            PatKind::Missing => SolPatRule::Missing,
            PatKind::Wild => SolPatRule::Wild,
            PatKind::Never => SolPatRule::Never,

            PatKind::Binding {
                name,
                mode: BindingMode(by_ref, mutability),
                var,
                ty,
                subpattern,
                is_primary: _,
                is_shorthand: _,
            } => {
                let binding_mode = match (by_ref, mutability) {
                    (ByRef::No, Mutability::Not) => SolBindMode::ImmByValue,
                    (ByRef::No, Mutability::Mut) => SolBindMode::MutByValue,
                    (ByRef::Yes(_, Mutability::Not), Mutability::Not) => SolBindMode::ImmByImmRef,
                    (ByRef::Yes(_, Mutability::Not), Mutability::Mut) => SolBindMode::ImmByMutRef,
                    (ByRef::Yes(_, Mutability::Mut), Mutability::Not) => SolBindMode::MutByImmRef,
                    (ByRef::Yes(_, Mutability::Mut), Mutability::Mut) => SolBindMode::MutByMutRef,
                };

                let var_name = SolLocalVarName(name.to_ident_string());
                let var_index = SolLocalVarIndex(var.0.local_id.index());
                let var_type = self.mk_type(*ty);
                let var_subpat = subpattern.as_ref().map(|sub_pat| Box::new(self.mk_pat(sub_pat)));
                SolPatRule::Bind {
                    name: var_name,
                    var_id: var_index,
                    ty: var_type,
                    mode: binding_mode,
                    subpat: var_subpat,
                }
            }
            PatKind::Variant { adt_def, args, variant_index, subpatterns } => {
                let (adt_ident, adt_ty_args) = self.mk_adt(*adt_def, args);
                let variant = SolVariantIndex(variant_index.index());
                let sub_pats = subpatterns
                    .iter()
                    .map(|FieldPat { field, pattern }| {
                        (SolFieldIndex(field.index()), self.mk_pat(pattern))
                    })
                    .collect();
                SolPatRule::Variant { adt_ident, adt_ty_args, variant, fields: sub_pats }
            }
            PatKind::Leaf { subpatterns } => {
                let sub_pats = subpatterns
                    .iter()
                    .map(|FieldPat { field, pattern }| {
                        (SolFieldIndex(field.index()), self.mk_pat(pattern))
                    })
                    .collect();
                SolPatRule::Leaf { fields: sub_pats }
            }
            PatKind::Slice { box prefix, slice, box suffix } => {
                let prefix_pats = prefix.iter().map(|pat| self.mk_pat(pat)).collect();
                let slice_pat = slice.as_ref().map(|pat| Box::new(self.mk_pat(pat)));
                let suffix_pats = suffix.iter().map(|pat| self.mk_pat(pat)).collect();
                SolPatRule::Slice { prefix: prefix_pats, slice: slice_pat, suffix: suffix_pats }
            }
            PatKind::Array { prefix, slice, suffix } => {
                let prefix_pats = prefix.iter().map(|pat| self.mk_pat(pat)).collect();
                let slice_pat = slice.as_ref().map(|pat| Box::new(self.mk_pat(pat)));
                let suffix_pats = suffix.iter().map(|pat| self.mk_pat(pat)).collect();
                SolPatRule::Array { prefix: prefix_pats, slice: slice_pat, suffix: suffix_pats }
            }

            PatKind::Deref { subpattern } => SolPatRule::Deref(Box::new(self.mk_pat(subpattern))),
            PatKind::DerefPattern { subpattern, borrow } => {
                let sub_pat = Box::new(self.mk_pat(subpattern));
                match borrow {
                    DerefPatBorrowMode::Box => SolPatRule::DerefBox(sub_pat),
                    DerefPatBorrowMode::Borrow(Mutability::Not) => SolPatRule::DerefImm(sub_pat),
                    DerefPatBorrowMode::Borrow(Mutability::Mut) => SolPatRule::DerefMut(sub_pat),
                }
            }

            PatKind::Constant { value } => SolPatRule::Constant(self.mk_value(*value)),
            PatKind::Or { box pats } => {
                SolPatRule::Or(pats.iter().map(|sub_pat| self.mk_pat(sub_pat)).collect())
            }

            // unsupported
            PatKind::Range(..) => bug!("[unsupported] range pattern"),

            // unreachable
            PatKind::Error(..) => bug!("[invariant] unreachable pattern {kind:?}"),
        };

        // construct the final pattern
        SolPattern { ty: pat_ty, rule: pat_rule, span: pat_span }
    }

    /// Record a constant in MIR/THIR context
    pub(crate) fn mk_const(&mut self, cval: Const<'tcx>) -> SolConst {
        self.log_stack.push("Const", format!("{cval}"));

        let parsed = match cval.kind() {
            ConstKind::Param(ParamConst { index, name }) => {
                SolConst::Param(self.get_param(index, name, SolGenericKind::Const))
            }
            ConstKind::Value(val) => SolConst::Value(self.mk_value(val)),

            // evaluate unevaluated constants
            ConstKind::Unevaluated(_) => {
                // evaluate it with an inference context
                let evaluated = traits::try_evaluate_const(
                    &self.tcx.infer_ctxt().build(self.typing_env.typing_mode),
                    cval,
                    self.typing_env.param_env,
                )
                .unwrap_or_else(|_| bug!("[unsupported] unevaluatable const"));

                if matches!(evaluated.kind(), ConstKind::Unevaluated(_)) {
                    bug!("[invariant] unable to fully evaluate const: {cval} -> {evaluated}");
                }
                self.mk_const(evaluated)
            }

            // unsupported
            ConstKind::Expr(..) => {
                bug!("[unsupported] const expr")
            }

            // unexpected
            ConstKind::Infer(..)
            | ConstKind::Bound(..)
            | ConstKind::Placeholder(..)
            | ConstKind::Error(..) => bug!("[invariant] unexpected const: {cval}"),
        };

        self.log_stack.pop();
        parsed
    }

    /// Record a (constant) value in MIR/THIR context
    pub(crate) fn mk_value(&mut self, val: Value<'tcx>) -> SolValue {
        let Value { ty, valtree } = val;
        match *valtree {
            ValTreeKind::Leaf(leaf) => self.mk_value_from_scalar(val.ty, *leaf),
            ValTreeKind::Branch(box []) => self.mk_value_from_zst(ty),
            ValTreeKind::Branch(box branches) => {
                let consts: Vec<_> = branches.iter().map(|cval| self.mk_const(*cval)).collect();
                self.mk_value_from_branch_consts(ty, consts)
            }
        }
    }

    /// Record a (constant) value based on a scalar integer
    pub(crate) fn mk_value_from_scalar(&mut self, ty: Ty<'tcx>, scalar: ScalarInt) -> SolValue {
        match ty.kind() {
            // primitives
            ty::Bool => SolValue::Bool(scalar.try_to_bool().unwrap()),
            ty::Char => SolValue::Char(scalar.try_into().unwrap()),
            ty::Int(IntTy::I8) => SolValue::I8(scalar.to_i8()),
            ty::Int(IntTy::I16) => SolValue::I16(scalar.to_i16()),
            ty::Int(IntTy::I32) => SolValue::I32(scalar.to_i32()),
            ty::Int(IntTy::I64) => SolValue::I64(scalar.to_i64()),
            ty::Int(IntTy::I128) => SolValue::I128(scalar.to_i128()),
            ty::Int(IntTy::Isize) => SolValue::Isize(scalar.to_target_isize(self.tcx) as isize),
            ty::Uint(UintTy::U8) => SolValue::U8(scalar.to_u8()),
            ty::Uint(UintTy::U16) => SolValue::U16(scalar.to_u16()),
            ty::Uint(UintTy::U32) => SolValue::U32(scalar.to_u32()),
            ty::Uint(UintTy::U64) => SolValue::U64(scalar.to_u64()),
            ty::Uint(UintTy::U128) => SolValue::U128(scalar.to_u128()),
            ty::Uint(UintTy::Usize) => SolValue::Usize(scalar.to_target_usize(self.tcx) as usize),
            ty::Float(FloatTy::F16) => SolValue::F16(scalar.to_f16().to_string()),
            ty::Float(FloatTy::F32) => SolValue::F32(scalar.to_f32().to_string()),
            ty::Float(FloatTy::F64) => SolValue::F64(scalar.to_f64().to_string()),
            ty::Float(FloatTy::F128) => SolValue::F128(scalar.to_f128().to_string()),
            ty::RawPtr(inner_ty, mutability) => {
                if !scalar.is_null() {
                    bug!("[unsupported] non-null scalar pointer");
                }

                // as far as we concern, only null pointers are meaningful constant values
                let pointee_ty = self.mk_type(*inner_ty);
                match mutability {
                    Mutability::Not => SolValue::ImmPtrNull(pointee_ty),
                    Mutability::Mut => SolValue::MutPtrNull(pointee_ty),
                }
            }

            // reference
            ty::Ref(_, inner_ty, mutability) => {
                let inner_val = self.mk_value_from_scalar(*inner_ty, scalar);
                match mutability {
                    Mutability::Not => SolValue::ImmRef(Box::new(inner_val)),
                    Mutability::Mut => SolValue::MutRef(Box::new(inner_val)),
                }
            }

            // unexpected
            _ => bug!("[invariant] unhandled scalar value {scalar} for type {ty}"),
        }
    }

    /// Record a (constant) value from a HIR literal
    pub(crate) fn mk_value_from_lit_and_ty(&mut self, lit: LitKind, ty: Ty<'tcx>) -> SolValue {
        match (lit, ty.kind()) {
            // basic
            (LitKind::Bool(v), ty::Bool) => SolValue::Bool(v),
            (LitKind::Char(v), ty::Char) => SolValue::Char(v),
            (LitKind::Byte(v), ty::Uint(UintTy::U8)) => SolValue::U8(v),

            // int
            (LitKind::Int(v, LitIntType::Signed(lit_ty)), ty::Int(int_ty)) => {
                match (lit_ty, int_ty) {
                    (IntTy::I8, IntTy::I8) => SolValue::I8(v.0 as i8),
                    (IntTy::I16, IntTy::I16) => SolValue::I16(v.0 as i16),
                    (IntTy::I32, IntTy::I32) => SolValue::I32(v.0 as i32),
                    (IntTy::I64, IntTy::I64) => SolValue::I64(v.0 as i64),
                    (IntTy::I128, IntTy::I128) => SolValue::I128(v.0 as i128),
                    (IntTy::Isize, IntTy::Isize) => SolValue::Isize(v.0 as isize),
                    _ => bug!("[invariant] int literal type mismatch: {lit_ty:?} vs {int_ty:?}"),
                }
            }
            (LitKind::Int(v, LitIntType::Unsigned(lit_ty)), ty::Uint(int_ty)) => {
                match (lit_ty, int_ty) {
                    (UintTy::U8, UintTy::U8) => SolValue::U8(v.0 as u8),
                    (UintTy::U16, UintTy::U16) => SolValue::U16(v.0 as u16),
                    (UintTy::U32, UintTy::U32) => SolValue::U32(v.0 as u32),
                    (UintTy::U64, UintTy::U64) => SolValue::U64(v.0 as u64),
                    (UintTy::U128, UintTy::U128) => SolValue::U128(v.0 as u128),
                    (UintTy::Usize, UintTy::Usize) => SolValue::Usize(v.0 as usize),
                    _ => bug!("[invariant] uint literal type mismatch: {lit_ty:?} vs {int_ty:?}"),
                }
            }
            (LitKind::Int(v, LitIntType::Unsuffixed), ty::Int(int_ty)) => {
                // unsuffixed integer literal, try to fit into the target type
                match int_ty {
                    IntTy::I8 => SolValue::I8(v.0 as i8),
                    IntTy::I16 => SolValue::I16(v.0 as i16),
                    IntTy::I32 => SolValue::I32(v.0 as i32),
                    IntTy::I64 => SolValue::I64(v.0 as i64),
                    IntTy::I128 => SolValue::I128(v.0 as i128),
                    IntTy::Isize => SolValue::Isize(v.0 as isize),
                }
            }
            (LitKind::Int(v, LitIntType::Unsuffixed), ty::Uint(int_ty)) => {
                // unsuffixed integer literal, try to fit into the target type
                match int_ty {
                    UintTy::U8 => SolValue::U8(v.0 as u8),
                    UintTy::U16 => SolValue::U16(v.0 as u16),
                    UintTy::U32 => SolValue::U32(v.0 as u32),
                    UintTy::U64 => SolValue::U64(v.0 as u64),
                    UintTy::U128 => SolValue::U128(v.0 as u128),
                    UintTy::Usize => SolValue::Usize(v.0 as usize),
                }
            }

            // float
            (LitKind::Float(v, LitFloatType::Suffixed(lit_ty)), ty::Float(float_ty)) => {
                match (lit_ty, float_ty) {
                    (FloatTy::F16, FloatTy::F16) => SolValue::F16(v.to_string()),
                    (FloatTy::F32, FloatTy::F32) => SolValue::F32(v.to_string()),
                    (FloatTy::F64, FloatTy::F64) => SolValue::F64(v.to_string()),
                    (FloatTy::F128, FloatTy::F128) => SolValue::F128(v.to_string()),
                    _ => {
                        bug!("[invariant] float literal type mismatch: {lit_ty:?} vs {float_ty:?}")
                    }
                }
            }
            (LitKind::Float(v, LitFloatType::Unsuffixed), ty::Float(float_ty)) => {
                // unsuffixed float literal, try to fit into the target type
                match float_ty {
                    FloatTy::F16 => SolValue::F16(v.to_string()),
                    FloatTy::F32 => SolValue::F32(v.to_string()),
                    FloatTy::F64 => SolValue::F64(v.to_string()),
                    FloatTy::F128 => SolValue::F128(v.to_string()),
                }
            }

            // strings
            (LitKind::Str(v, _), ty::Ref(_, inner_ty, Mutability::Not))
                if matches!(inner_ty.kind(), ty::Str) =>
            {
                SolValue::ImmRef(Box::new(SolValue::Str(v.to_ident_string())))
            }
            (LitKind::ByteStr(v, _), ty::Ref(_, inner_ty, Mutability::Not)) => {
                let inner_val = match inner_ty.kind() {
                    ty::Slice(elem_ty) if matches!(elem_ty.kind(), ty::Uint(UintTy::U8)) => {
                        SolValue::Slice(
                            SolType::U8,
                            v.as_byte_str()
                                .iter()
                                .map(|b| SolConst::Value(SolValue::U8(*b)))
                                .collect(),
                        )
                    }
                    ty::Array(elem_ty, _) if matches!(elem_ty.kind(), ty::Uint(UintTy::U8)) => {
                        SolValue::Array(
                            SolType::U8,
                            v.as_byte_str()
                                .iter()
                                .map(|b| SolConst::Value(SolValue::U8(*b)))
                                .collect(),
                        )
                    }
                    _ => bug!("[invariant] literal and type mismatch: {lit} vs {ty}"),
                };
                SolValue::ImmRef(Box::new(inner_val))
            }

            // unexpected
            (LitKind::Err(..), _) => bug!("[invariant] unexpected literal {lit}"),

            // unsupported
            (LitKind::CStr(..), _) => bug!("[unsupported] CStr literal"),
            (_, ty::Pat(..)) => bug!("[unsupported] literal for pattern type"),

            // all other cases are considered type mismatches
            _ => bug!("[invariant] literal and type mismatch: {lit} vs {ty}"),
        }
    }

    /// Record a (constant) value for a ZST
    pub(crate) fn mk_value_from_zst(&mut self, ty: Ty<'tcx>) -> SolValue {
        if let Some(zst_val) = self.mk_value_when_zst(ty) {
            return zst_val;
        }

        // special-case for empty string and empty slice
        match ty.kind() {
            ty::Str => SolValue::Str(String::new()),
            ty::Slice(elem_ty) => SolValue::Slice(self.mk_type(*elem_ty), vec![]),

            // reference
            ty::Ref(_, inner_ty, Mutability::Not) => {
                let inner_val = match inner_ty.kind() {
                    ty::Str => SolValue::Str(String::new()),
                    ty::Slice(elem_ty) => SolValue::Slice(self.mk_type(*elem_ty), vec![]),
                    _ => bug!("[invariant] failed to create a value for ZST type {ty}"),
                };
                SolValue::ImmRef(Box::new(inner_val))
            }

            // unexpected
            _ => bug!("[invariant] failed to create a value for ZST type {ty}"),
        }
    }

    /// Try to record a (constant) value for a ZST or None of the type is not zero-sized
    fn mk_value_when_zst(&mut self, ty: Ty<'tcx>) -> Option<SolValue> {
        let zst_val = match ty.kind() {
            ty::Tuple(elems) => {
                // a tuple is a ZST if all its elements are ZSTs (including the empty tuple)
                let mut elem_vals = vec![];
                for elem_ty in elems.iter() {
                    let elem_val = self.mk_value_when_zst(elem_ty)?;
                    elem_vals.push(SolConst::Value(elem_val));
                }
                SolValue::Tuple(elem_vals)
            }
            ty::Array(elem_ty, length) => {
                // an array is a ZST if its element type is a ZST or its length is zero
                let size = length.try_to_target_usize(self.tcx)?;
                if size == 0 {
                    SolValue::Array(self.mk_type(*elem_ty), vec![])
                } else {
                    let elem_val = self.mk_value_when_zst(*elem_ty)?;
                    SolValue::Array(
                        self.mk_type(*elem_ty),
                        vec![SolConst::Value(elem_val); size as usize],
                    )
                }
            }
            ty::Adt(def, ty_args) => {
                let (adt_ident, adt_ty_args) = self.mk_adt(*def, ty_args);
                match def.adt_kind() {
                    AdtKind::Struct => {
                        // a struct is a ZST if all its fields are ZSTs
                        let variant = def.non_enum_variant();
                        let mut fields = vec![];
                        for (field_idx, field_def) in variant.fields.iter_enumerated() {
                            let field_val =
                                self.mk_value_when_zst(field_def.ty(self.tcx, ty_args))?;
                            fields.push((
                                SolFieldIndex(field_idx.index()),
                                SolConst::Value(field_val),
                            ));
                        }
                        SolValue::Struct(adt_ident, adt_ty_args, fields)
                    }
                    AdtKind::Union => {
                        // a union is a ZST if it has only one feasible field and the field is a ZST
                        let field_idx = get_uniquely_feasible_field(self.tcx, *def, ty_args)?;
                        let field_def = def.non_enum_variant().fields.get(field_idx).unwrap();
                        let field_val = self.mk_value_when_zst(field_def.ty(self.tcx, ty_args))?;
                        SolValue::Union(
                            adt_ident,
                            adt_ty_args,
                            SolFieldIndex(field_idx.index()),
                            Box::new(SolConst::Value(field_val)),
                        )
                    }
                    AdtKind::Enum => {
                        // an enum is a ZST if it has only one feasible variant and all fields in that variant are ZSTs
                        let variant_idx = get_uniquely_feasible_variant(self.tcx, *def, ty_args)?;

                        // parse the fields in that feasible variant
                        let variant_def = def.variant(variant_idx);
                        let mut fields = vec![];
                        for (field_idx, field_def) in variant_def.fields.iter_enumerated() {
                            let field_val =
                                self.mk_value_when_zst(field_def.ty(self.tcx, ty_args))?;
                            fields.push((
                                SolFieldIndex(field_idx.index()),
                                SolConst::Value(field_val),
                            ));
                        }
                        SolValue::Enum(
                            adt_ident,
                            adt_ty_args,
                            SolVariantIndex(variant_idx.index()),
                            fields,
                        )
                    }
                }
            }

            ty::FnDef(def_id, ty_args) => {
                let ident = self.mk_ident(*def_id);
                let generic_args = ty_args.iter().map(|arg| self.mk_generic_arg(arg)).collect();
                SolValue::FuncDef(ident, generic_args)
            }
            ty::Closure(def_id, ty_args) => {
                let ident = self.mk_ident(*def_id);
                let generic_args = ty_args.iter().map(|arg| self.mk_generic_arg(arg)).collect();
                SolValue::Closure(ident, generic_args)
            }
            ty::Ref(_, inner_ty, Mutability::Not) => {
                SolValue::ImmRef(Box::new(self.mk_value_when_zst(*inner_ty)?))
            }

            _ => {
                // double check that we should have captured all ZST cases above
                if self
                    .tcx
                    .layout_of(self.typing_env.as_query_input(ty))
                    .unwrap_or_else(|_| bug!("[invariant] unable to query layout of {ty}"))
                    .is_zst()
                {
                    bug!("[invariant] unhandled ZST type: {ty}")
                }
                return None;
            }
        };
        Some(zst_val)
    }

    /// Record a (constant) value based on vector of branches
    pub(crate) fn mk_value_from_branch_consts(
        &mut self,
        ty: Ty<'tcx>,
        consts: Vec<SolConst>,
    ) -> SolValue {
        match ty.kind() {
            // FIXME: we should check that the values and types are consistent
            ty::Str => SolValue::Str(util_values_to_string(&consts)),
            ty::Slice(elem_ty) => SolValue::Slice(self.mk_type(*elem_ty), consts),

            ty::Tuple(_) => SolValue::Tuple(consts),
            ty::Array(elem_ty, _) => SolValue::Array(self.mk_type(*elem_ty), consts),
            ty::Adt(def, ty_args) => {
                let (adt_ident, adt_ty_args) = self.mk_adt(*def, ty_args);
                match def.adt_kind() {
                    AdtKind::Struct => {
                        let variant = def.non_enum_variant();
                        assert_eq!(
                            consts.len(),
                            variant.fields.len(),
                            "[invariant] struct value field count mismatch"
                        );
                        SolValue::Struct(
                            adt_ident,
                            adt_ty_args,
                            variant
                                .fields
                                .iter_enumerated()
                                .zip(consts)
                                .map(|((field_idx, _), field_const)| {
                                    (SolFieldIndex(field_idx.index()), field_const)
                                })
                                .collect(),
                        )
                    }
                    AdtKind::Enum => bug!("[unsupported] enum const value"),
                    AdtKind::Union => bug!("[unsupported] union const value"),
                }
            }

            // reference types
            ty::Ref(_, inner_ty, Mutability::Not) => {
                let inner_val = self.mk_value_from_branch_consts(*inner_ty, consts);
                SolValue::ImmRef(Box::new(inner_val))
            }

            // we should have covered all branch cases above
            _ => bug!("[invariant] unhandled type for valtree branch: {ty}"),
        }
    }

    /// Record a executable, i.e., a THIR body (for a function or a constant/static)
    pub(crate) fn mk_exec(&mut self, thir: &Thir<'tcx>, expr: ExprId) -> SolExec {
        // switch-case on the body type
        match thir.body_type {
            BodyTy::Fn(sig) => {
                let FnSig { abi, c_variadic, safety, inputs_and_output: _ } = sig;

                // ignore unsupported case
                if c_variadic {
                    bug!("[unsupported] variadic function");
                }

                // parse function signature
                let parsed_abi = self.mk_abi(abi, c_variadic, safety);
                let ret_ty = self.mk_type(sig.output());
                let params: Vec<_> =
                    sig.inputs().iter().map(|input_ty| self.mk_type(*input_ty)).collect();

                // parse parameters
                let mut param_iter = thir.params.iter_enumerated();
                if thir.params.len() == params.len() + 1 {
                    // special case for closure: first parameter must be closure-related
                    let (index0, param0) = param_iter.next().unwrap();
                    assert_eq!(index0.index(), 0, "[invariant] expect parameter index 0");

                    let (closure_id, closure_ty_args) = match param0.ty.kind() {
                        ty::Ref(_, inner_ty, _) => match inner_ty.kind() {
                            ty::Closure(c_id, c_ty_args) => (*c_id, *c_ty_args),
                            _ => bug!("[invariant] expect &closure as param 0: got {}", param0.ty),
                        },
                        ty::Closure(c_id, c_ty_args) => (*c_id, *c_ty_args),
                        _ => bug!("[invariant] expect &closure as param 0: got {}", param0.ty),
                    };

                    // sanity check with previous recordings
                    assert_eq!(
                        self.owner_id.to_def_id(),
                        closure_id,
                        "[invariant] closure id mismatch",
                    );

                    /*
                     * NOTE: the actual closure type arguments have three synthetic generics:
                     * <closure_kind>: I16,
                     * <closure_signature>: fn(..) -> ..
                     * <upvars>: (..) (a.k.a, a tuple)
                     * We skip these three synthetic type arguments in the generics declarations
                     */
                    assert_eq!(
                        self.generics.len() + 3,
                        closure_ty_args.len(),
                        "[invariant] closure generics count mismatch",
                    );

                    // sanity check that the closure type arguments matches the generic parameters
                    for (ty_arg, ty_param) in closure_ty_args.iter().zip(self.generics.clone()) {
                        let parsed_ty_arg = self.mk_generic_arg(ty_arg);
                        match (parsed_ty_arg, ty_param.kind) {
                            (SolGenericArg::Type(ty), SolGenericKind::Type) => {
                                assert_eq!(
                                    ty,
                                    SolType::Param(ty_param.ident),
                                    "[invariant] closure generic type argument mismatch"
                                );
                            }
                            (SolGenericArg::Const(cval), SolGenericKind::Const) => {
                                assert_eq!(
                                    cval,
                                    SolConst::Param(ty_param.ident),
                                    "[invariant] closure generic const argument mismatch"
                                );
                            }
                            (SolGenericArg::Lifetime, SolGenericKind::Lifetime) => {
                                // nothing to check for lifetime
                            }
                            _ => bug!("[invariant] closure generic argument/parameter mismatch"),
                        }
                    }
                } else {
                    assert_eq!(
                        thir.params.len(),
                        params.len(),
                        "[invariant] parameter count mismatch",
                    );
                }

                // declared parameters should match type declarations
                for (param_ty, (_param_id, param_decl)) in params.iter().zip(param_iter) {
                    let declared_param_ty = self.mk_type(param_decl.ty);
                    assert_eq!(param_ty, &declared_param_ty, "[invariant] parameter type mismatch");

                    // FIXME: record parameter patterns as well
                }

                // parse the body expression
                let body = self.mk_expr(thir, expr);

                // pack the information
                SolExec::Function(SolFnDef { abi: parsed_abi, ret_ty, params, body })
            }
            BodyTy::Const(ty) => {
                // sanity checks
                if !thir.params.is_empty() {
                    bug!("[invariant] constant body should not have parameters");
                }

                // parse type
                let const_ty = self.mk_type(ty);

                // parse the body expression
                let body = self.mk_expr(thir, expr);

                // pack the information
                SolExec::Constant(SolCEval { ty: const_ty, body })
            }
            BodyTy::GlobalAsm(..) => bug!("[unsupported] global assembly"),
        }
    }

    /// Record a THIR expression
    pub(crate) fn mk_expr(&mut self, thir: &Thir<'tcx>, expr_id: ExprId) -> SolExpr {
        let Expr { kind, ty, temp_scope_id: _, span } = &thir.exprs[expr_id];
        self.log_stack.push("Expr", format!("{kind:?}"));

        // record type and span
        let expr_ty = self.mk_type(*ty);
        let expr_span = self.mk_span(*span);

        // switch-case on expression kind
        let expr_op = match kind {
            // markers
            ExprKind::Scope { region_scope: _, hir_id, value } => {
                let inner_expr = self.mk_expr(thir, *value);
                SolOp::Scope(self.mk_hir(*hir_id, inner_expr))
            }
            ExprKind::PlaceTypeAscription { source, user_ty: _, user_ty_span: _ }
            | ExprKind::ValueTypeAscription { source, user_ty: _, user_ty_span: _ } => {
                // MAYFIX: maybe record user type annotation as well?
                SolOp::TypeAscribe(self.mk_expr(thir, *source))
            }

            // use
            ExprKind::Use { source } => SolOp::Use(self.mk_expr(thir, *source)),
            ExprKind::VarRef { id: LocalVarId(var_id) } => {
                // FIXME: whlie we can't enforce the following check because closure may refer to upvar,
                // we need to find a better way to ensure the correctness of local var references.
                // assert_eq!(var_id.owner, self.owner_id, "[invariant] local var owner mismatch");
                SolOp::VarRef(SolLocalVarIndex(var_id.local_id.index()))
            }
            ExprKind::UpvarRef { closure_def_id, var_hir_id: LocalVarId(var_id) } => {
                // FIXME: whlie we can't enforce the following check because closure may refer to upvar,
                // we need to find a better way to ensure the correctness of upvar references.
                // assert_eq!(var_id.owner, self.owner_id, "[invariant] local var owner mismatch");
                assert_eq!(
                    closure_def_id,
                    &self.owner_id.to_def_id(),
                    "[invariant] closure def_id mismatch"
                );
                SolOp::UpVarRef(
                    self.mk_ident(*closure_def_id),
                    SolLocalVarIndex(var_id.local_id.index()),
                )
            }
            ExprKind::ConstParam { param: ParamConst { index, name }, def_id } => {
                let param_ident = self.get_param(*index, *name, SolGenericKind::Const);
                let const_ident = self.mk_ident(*def_id);
                assert_eq!(param_ident, const_ident, "[invariant] const param ident mismatch",);
                SolOp::ConstParam(param_ident)
            }
            ExprKind::NamedConst { def_id, args, user_ty: _ } => {
                // MAYFIX: maybe record user type annotation as well?
                let const_ident = self.mk_ident(*def_id);
                let generic_args = args.iter().map(|arg| self.mk_generic_arg(arg)).collect();
                SolOp::ConstValue(const_ident, generic_args)
            }
            ExprKind::StaticRef { .. } => bug!("[unsupported] static ref"),

            // intrinsics
            ExprKind::Box { value } => SolOp::Box(self.mk_expr(thir, *value)),
            ExprKind::Deref { arg } => SolOp::Deref(self.mk_expr(thir, *arg)),

            // operators
            ExprKind::Binary { op, lhs, rhs } => {
                let lhs_expr = self.mk_expr(thir, *lhs);
                let rhs_expr = self.mk_expr(thir, *rhs);
                match op {
                    BinOp::Add => SolOp::Add(lhs_expr, rhs_expr),
                    BinOp::AddUnchecked => SolOp::AddUnchecked(lhs_expr, rhs_expr),
                    BinOp::AddWithOverflow => SolOp::AddWithOverflow(lhs_expr, rhs_expr),
                    BinOp::Sub => SolOp::Sub(lhs_expr, rhs_expr),
                    BinOp::SubUnchecked => SolOp::SubUnchecked(lhs_expr, rhs_expr),
                    BinOp::SubWithOverflow => SolOp::SubWithOverflow(lhs_expr, rhs_expr),
                    BinOp::Mul => SolOp::Mul(lhs_expr, rhs_expr),
                    BinOp::MulUnchecked => SolOp::MulUnchecked(lhs_expr, rhs_expr),
                    BinOp::MulWithOverflow => SolOp::MulWithOverflow(lhs_expr, rhs_expr),
                    BinOp::Div => SolOp::Div(lhs_expr, rhs_expr),
                    BinOp::Rem => SolOp::Rem(lhs_expr, rhs_expr),
                    BinOp::BitXor => SolOp::BitXor(lhs_expr, rhs_expr),
                    BinOp::BitAnd => SolOp::BitAnd(lhs_expr, rhs_expr),
                    BinOp::BitOr => SolOp::BitOr(lhs_expr, rhs_expr),
                    BinOp::Shl => SolOp::Shl(lhs_expr, rhs_expr),
                    BinOp::ShlUnchecked => SolOp::ShlUnchecked(lhs_expr, rhs_expr),
                    BinOp::Shr => SolOp::Shr(lhs_expr, rhs_expr),
                    BinOp::ShrUnchecked => SolOp::ShrUnchecked(lhs_expr, rhs_expr),
                    BinOp::Eq => SolOp::Eq(lhs_expr, rhs_expr),
                    BinOp::Ne => SolOp::Ne(lhs_expr, rhs_expr),
                    BinOp::Lt => SolOp::Lt(lhs_expr, rhs_expr),
                    BinOp::Le => SolOp::Le(lhs_expr, rhs_expr),
                    BinOp::Gt => SolOp::Gt(lhs_expr, rhs_expr),
                    BinOp::Ge => SolOp::Ge(lhs_expr, rhs_expr),
                    BinOp::Cmp => SolOp::Cmp(lhs_expr, rhs_expr),
                    BinOp::Offset => bug!("[unsupported] offset binary op in THIR"),
                }
            }
            ExprKind::LogicalOp { op, lhs, rhs } => {
                let lhs_expr = self.mk_expr(thir, *lhs);
                let rhs_expr = self.mk_expr(thir, *rhs);
                match op {
                    LogicalOp::And => SolOp::LogicalAnd(lhs_expr, rhs_expr),
                    LogicalOp::Or => SolOp::LogicalOr(lhs_expr, rhs_expr),
                }
            }
            ExprKind::Unary { op, arg } => {
                let operand = self.mk_expr(thir, *arg);
                match op {
                    UnOp::Not => SolOp::Not(operand),
                    UnOp::Neg => SolOp::Neg(operand),
                    UnOp::PtrMetadata => bug!("[invariant] pointer metadata unary in THIR"),
                }
            }

            // assignment
            ExprKind::Assign { lhs, rhs } => {
                SolOp::Assign { lhs: self.mk_expr(thir, *lhs), rhs: self.mk_expr(thir, *rhs) }
            }
            ExprKind::AssignOp { op, lhs, rhs } => {
                let lhs_expr = self.mk_expr(thir, *lhs);
                let rhs_expr = self.mk_expr(thir, *rhs);
                match op {
                    AssignOp::AddAssign => SolOp::AddAssign { lhs: lhs_expr, rhs: rhs_expr },
                    AssignOp::SubAssign => SolOp::SubAssign { lhs: lhs_expr, rhs: rhs_expr },
                    AssignOp::MulAssign => SolOp::MulAssign { lhs: lhs_expr, rhs: rhs_expr },
                    AssignOp::DivAssign => SolOp::DivAssign { lhs: lhs_expr, rhs: rhs_expr },
                    AssignOp::RemAssign => SolOp::RemAssign { lhs: lhs_expr, rhs: rhs_expr },
                    AssignOp::BitXorAssign => SolOp::BitXorAssign { lhs: lhs_expr, rhs: rhs_expr },
                    AssignOp::BitAndAssign => SolOp::BitAndAssign { lhs: lhs_expr, rhs: rhs_expr },
                    AssignOp::BitOrAssign => SolOp::BitOrAssign { lhs: lhs_expr, rhs: rhs_expr },
                    AssignOp::ShlAssign => SolOp::ShlAssign { lhs: lhs_expr, rhs: rhs_expr },
                    AssignOp::ShrAssign => SolOp::ShrAssign { lhs: lhs_expr, rhs: rhs_expr },
                }
            }

            // casts
            ExprKind::Cast { source } => SolOp::Cast(self.mk_expr(thir, *source)),
            ExprKind::PointerCoercion { cast, source, is_from_as_cast: _ } => {
                let operand = self.mk_expr(thir, *source);
                match cast {
                    PointerCoercion::ReifyFnPointer(safety) => {
                        SolOp::CastReifyFnPtr(operand, matches!(safety, Safety::Safe))
                    }
                    PointerCoercion::UnsafeFnPointer => SolOp::CastUnsafeFnPtr(operand),
                    PointerCoercion::ClosureFnPointer(safety) => {
                        SolOp::CastClosureFnPtr(operand, matches!(safety, Safety::Safe))
                    }
                    PointerCoercion::MutToConstPointer => SolOp::CastMutToConstPtr(operand),
                    PointerCoercion::ArrayToPointer => SolOp::CastUnsizeArrayPtr(operand),
                    PointerCoercion::Unsize => SolOp::CastUnsizeArrayRef(operand),
                }
            }
            ExprKind::NeverToAny { source } => SolOp::NeverToAny(self.mk_expr(thir, *source)),

            // packing
            ExprKind::Repeat { value, count } => {
                SolOp::Repeat(self.mk_expr(thir, *value), self.mk_const(*count))
            }
            ExprKind::Array { fields } => {
                SolOp::Array(fields.iter().map(|e| self.mk_expr(thir, *e)).collect())
            }
            ExprKind::Tuple { fields } => {
                SolOp::Tuple(fields.iter().map(|e| self.mk_expr(thir, *e)).collect())
            }
            ExprKind::Adt(box AdtExpr {
                adt_def,
                variant_index,
                args,
                user_ty: _,
                fields,
                base: base_expr,
            }) => {
                // parse the definition
                let (adt_ident, adt_args) = self.mk_adt(*adt_def, *args);

                // parse the fields
                let mut parsed_fields = vec![];
                for FieldExpr { name: field_idx, expr: field_expr } in fields.iter() {
                    parsed_fields
                        .push((SolFieldIndex(field_idx.index()), self.mk_expr(thir, *field_expr)));
                }

                // parse the base
                let adt_base = match base_expr {
                    AdtExprBase::None => SolAdtBase::None,
                    AdtExprBase::Base(FruInfo { base, box field_types }) => SolAdtBase::Overlay(
                        self.mk_expr(thir, *base),
                        field_types.iter().map(|field_ty| self.mk_type(*field_ty)).collect(),
                    ),
                    AdtExprBase::DefaultFields(box default_field_tys) => SolAdtBase::Default(
                        default_field_tys.iter().map(|field_ty| self.mk_type(*field_ty)).collect(),
                    ),
                };

                // MAYFIX: maybe record user type annotation as well?

                // pack the ADT expression
                SolOp::Adt {
                    adt_ident,
                    adt_args,
                    variant: SolVariantIndex(variant_index.index()),
                    fields: parsed_fields,
                    base: adt_base,
                }
            }

            // access
            ExprKind::Field { lhs, variant_index, name } => SolOp::Field {
                base: self.mk_expr(thir, *lhs),
                variant: SolVariantIndex(variant_index.index()),
                field: SolFieldIndex(name.index()),
            },
            ExprKind::Index { lhs, index } => {
                SolOp::Index { base: self.mk_expr(thir, *lhs), index: self.mk_expr(thir, *index) }
            }

            // control-folow
            ExprKind::If { if_then_scope: _, cond, then, else_opt } => SolOp::If {
                cond: self.mk_expr(thir, *cond),
                then: self.mk_expr(thir, *then),
                else_opt: else_opt.map(|e| self.mk_expr(thir, e)),
            },
            ExprKind::Loop { body } => SolOp::Loop(self.mk_expr(thir, *body)),
            ExprKind::Break { label: _, value } => {
                // FIXME: handle scope (must be done)
                SolOp::Break(value.map(|e| self.mk_expr(thir, e)))
            }
            ExprKind::Continue { label: _ } => {
                // FIXME: handle scope (must be done)
                SolOp::Continue
            }
            ExprKind::Return { value } => SolOp::Return(value.map(|e| self.mk_expr(thir, e))),

            // borrow
            ExprKind::Borrow { borrow_kind, arg } => {
                let borrowed = self.mk_expr(thir, *arg);
                match borrow_kind {
                    BorrowKind::Shared => SolOp::ImmBorrow(borrowed),
                    BorrowKind::Mut { kind: _ } => SolOp::MutBorrow(borrowed),
                    BorrowKind::Fake(..) => bug!("[unsupported] fake borrow in THIR"),
                }
            }
            ExprKind::RawBorrow { mutability, arg } => {
                let borrowed = self.mk_expr(thir, *arg);
                match mutability {
                    Mutability::Not => SolOp::ImmRawPtr(borrowed),
                    Mutability::Mut => SolOp::MutRawPtr(borrowed),
                }
            }

            // function call
            ExprKind::Call { ty, fun, args, from_hir_call: _, fn_span: _ } => SolOp::Call {
                target: self.mk_type(*ty),
                callee: self.mk_expr(thir, *fun),
                args: args.iter().map(|arg| self.mk_expr(thir, *arg)).collect(),
            },
            ExprKind::ConstBlock { did, args } => SolOp::ConstBlock(
                self.mk_ident(*did),
                args.iter().map(|arg| self.mk_generic_arg(arg)).collect(),
            ),

            // pattern
            ExprKind::Let { expr, pat } => SolOp::Let(self.mk_expr(thir, *expr), self.mk_pat(pat)),
            ExprKind::Match { scrutinee, arms, match_source } => {
                if matches!(match_source, MatchSource::AwaitDesugar) {
                    bug!("[unsupported] coroutine await match");
                }

                // parse scrutinee
                let scrutinee_expr = self.mk_expr(thir, *scrutinee);

                // parse arms
                let mut parsed_arms = vec![];
                for arm in arms.iter() {
                    let Arm { pattern, guard, body, hir_id, scope: _, span } = &thir.arms[*arm];
                    let arm_pattern = self.mk_pat(pattern);
                    let arm_guard = guard.map(|e| self.mk_expr(thir, e));
                    let arm_body = self.mk_expr(thir, *body);
                    let arm_span = self.mk_span(*span);
                    let arm_hir = self.mk_hir(
                        *hir_id,
                        SolMatchArm {
                            pat: arm_pattern,
                            guard: arm_guard,
                            body: arm_body,
                            span: arm_span,
                        },
                    );
                    parsed_arms.push(arm_hir);
                }

                // pack the match expression
                SolOp::Match(scrutinee_expr, parsed_arms)
            }
            ExprKind::Block { block } => SolOp::Block(self.mk_block(thir, *block)),

            // literals
            ExprKind::Literal { lit, neg } => SolOp::BaseLiteral(
                *neg,
                self.mk_value_from_lit_and_ty(lit.node, *ty),
                self.mk_span(lit.span),
            ),
            ExprKind::NonHirLiteral { lit, user_ty: _ } => {
                // MAYFIX: record user type annotation as well?
                SolOp::ScalarLiteral(self.mk_value_from_scalar(*ty, *lit))
            }
            ExprKind::ZstLiteral { user_ty: _ } => {
                // MAYFIX: record user type annotation as well?
                SolOp::ZstLiteral(self.mk_value_from_zst(*ty))
            }

            // closure
            ExprKind::Closure(box ClosureExpr {
                closure_id,
                args,
                box upvars,
                movability: _,
                fake_reads: _,
            }) => {
                let closure_ident = self.mk_ident(closure_id.to_def_id());
                let closure_ty_args = match args {
                    UpvarArgs::Closure(ty_args) => {
                        ty_args.iter().map(|arg| self.mk_generic_arg(arg)).collect()
                    }
                    UpvarArgs::Coroutine(..) | UpvarArgs::CoroutineClosure(..) => {
                        bug!("[unsupported] coroutine closure");
                    }
                };
                let closure_upvars = upvars.iter().map(|v| self.mk_expr(thir, *v)).collect();
                // MAYFIX: maybe we should do something about the movability and fake reads as well?
                SolOp::Closure(closure_ident, closure_ty_args, closure_upvars)
            }

            // unsupported
            ExprKind::ByUse { .. } => bug!("[unsupported] by-use"),
            ExprKind::LoopMatch { .. } => bug!("[unsupported] loop-match"),
            ExprKind::ConstContinue { .. } => bug!("[unsupported] const-continue"),
            ExprKind::InlineAsm { .. } => bug!("[unsupported] inline assembly"),
            ExprKind::Yield { .. } => bug!("[unsupported] yield"),
            ExprKind::ThreadLocalRef(..) => bug!("[unsupported] thread-local reference"),
            ExprKind::Become { .. } => bug!("[unsupported] become"),

            ExprKind::WrapUnsafeBinder { .. } => bug!("[unsupported] wrap unsafe binder"),
            ExprKind::PlaceUnwrapUnsafeBinder { .. } => {
                bug!("[unsupported] place unwrap unsafe binder")
            }
            ExprKind::ValueUnwrapUnsafeBinder { .. } => {
                bug!("[unsupported] value unwrap unsafe binder")
            }
        };

        // pop the log stack
        self.log_stack.pop();

        // pack the expression
        SolExpr { ty: expr_ty, span: expr_span, op: Box::new(expr_op) }
    }

    /// Record a THIR statement
    pub(crate) fn mk_stmt(&mut self, thir: &Thir<'tcx>, stmt_id: StmtId) -> SolStmt {
        let Stmt { kind } = &thir.stmts[stmt_id];
        match kind {
            StmtKind::Expr { scope: _, expr } => SolStmt::Expr(self.mk_expr(thir, *expr)),
            StmtKind::Let {
                remainder_scope: _,
                init_scope: _,
                pattern,
                initializer,
                else_block,
                hir_id,
                span,
            } => {
                let let_pattern = self.mk_pat(pattern);
                let let_init = match (initializer, else_block) {
                    (None, None) => None,
                    (Some(init), None) => Some((self.mk_expr(thir, *init), None)),
                    (Some(init), Some(else_blk)) => {
                        Some((self.mk_expr(thir, *init), Some(self.mk_block(thir, *else_blk))))
                    }
                    (None, Some(_)) => {
                        bug!("[invariant] let statement with else block must have an initializer");
                    }
                };
                let let_span = self.mk_span(*span);

                // pack the let statement
                SolStmt::Bind(self.mk_hir(
                    *hir_id,
                    SolLetBinding { pat: let_pattern, init: let_init, span: let_span },
                ))
            }
        }
    }

    /// Record a THIR block
    pub(crate) fn mk_block(&mut self, thir: &Thir<'tcx>, block_id: BlockId) -> SolBlock {
        let Block { targeted_by_break: _, region_scope: _, span, stmts, expr, safety_mode } =
            &thir.blocks[block_id];

        let block_safety = match safety_mode {
            BlockSafety::Safe => true,
            BlockSafety::BuiltinUnsafe | BlockSafety::ExplicitUnsafe(..) => false,
        };
        let block_span = self.mk_span(*span);
        let block_stmts = stmts.iter().map(|stmt| self.mk_stmt(thir, *stmt)).collect();
        let block_expr = expr.map(|e| self.mk_expr(thir, e));

        SolBlock { safety: block_safety, stmts: block_stmts, expr: block_expr, span: block_span }
    }
}

/// Build the crate
pub(crate) fn build<'tcx>(tcx: TyCtxt<'tcx>, src_dir: PathBuf) -> SolCrate {
    let mut base_builder = BaseBuilder::new(tcx, src_dir);

    // recursively build the modules starting from the root module
    let module_data = base_builder.mk_module(tcx.crate_name(LOCAL_CRATE), *tcx.hir_root_module());
    let module_mir = base_builder.mk_mir(CRATE_DEF_ID, CRATE_HIR_ID, DUMMY_SP, module_data);

    // process all body owners in this crate
    let mut bundles = vec![];

    for owner_id in tcx.hir_body_owners() {
        let def_id = owner_id.to_def_id();
        let def_desc = util_debug_symbol(tcx, def_id, &List::empty());

        // check if the owner is a closure
        let is_closure = tcx.is_closure_like(def_id);

        // skip coroutine-related owners
        if is_closure
            && matches!(
                tcx.type_of(def_id).instantiate_identity().kind(),
                ty::Coroutine(..) | ty::CoroutineClosure(..)
            )
        {
            continue;
        }

        // retrieve the THIR body
        let (thir_body, thir_expr) = tcx
            .thir_body(owner_id)
            .unwrap_or_else(|_| panic!("[invariant] failed to retrieve THIR body for {def_desc}"));

        // collect the generics of the body
        let mut bundle_generics = vec![];
        let generics = tcx.generics_of(def_id);
        for i in 0..generics.count() {
            let GenericParamDef {
                name: param_symbol,
                def_id: param_def_id,
                index: param_index,
                pure_wrt_drop: _,
                kind: param_def_kind,
            } = generics.param_at(i, tcx);
            assert_eq!(
                *param_index as usize, i,
                "[invariant] generic parameter {param_symbol} index mismatch in {def_desc}"
            );

            // construct the generic parameter
            let param_ident = base_builder.mk_ident(*param_def_id);
            let param_name = SolParamName(param_symbol.to_ident_string());
            let param_kind = match param_def_kind {
                GenericParamDefKind::Lifetime => SolGenericKind::Lifetime,
                GenericParamDefKind::Type { has_default: _, synthetic: _ } => {
                    // skip injected type parameters in closures
                    if is_closure
                        && matches!(
                            param_symbol.as_str(),
                            "<closure_kind>" | "<closure_signature>" | "<upvars>"
                        )
                    {
                        continue;
                    }
                    SolGenericKind::Type
                }
                GenericParamDefKind::Const { has_default: _ } => SolGenericKind::Const,
            };
            bundle_generics.push(SolGenericParam {
                ident: param_ident,
                name: param_name,
                kind: param_kind,
            });

            // FIXME: collect the explict predicates of each parameter
            // e.g., tcx.explicit_predicates_of(*param_def_id).instantiate_identity(tcx);
        }

        // create an exec builder
        let mut exec_builder =
            ExecBuilder::new(base_builder, OwnerId { def_id: owner_id }, bundle_generics);

        // mark start
        exec_builder.log_stack.push("THIR", def_desc);

        // build the executable body
        let thir_lock = thir_body.borrow();
        let exec = exec_builder.mk_exec(&*thir_lock, thir_expr);

        // mark end
        exec_builder.log_stack.pop();

        // unpack the exec builder
        let ExecBuilder {
            tcx: _,
            typing_env: _,
            mut base,
            owner_id: _,
            generics,
            adt_defs,
            trait_defs,
            log_stack,
        } = exec_builder;
        assert!(log_stack.is_empty(), "[invariant] log stack is not empty");

        // collect the datatype definitions
        let mut flat_adt_defs = vec![];
        for (ident, l1) in adt_defs.into_iter() {
            for (mono, def) in l1.into_iter() {
                match def {
                    None => bug!("[invariant] missing type definition"),
                    Some(val) => flat_adt_defs.push((ident.clone(), mono, val)),
                }
            }
        }

        // collect the trait definitions
        let mut flat_trait_defs = vec![];
        for (ident, l1) in trait_defs.into_iter() {
            for (mono, def) in l1.into_iter() {
                match def {
                    None => bug!("[invariant] missing trait definition"),
                    Some(val) => flat_trait_defs.push((ident.clone(), mono, val)),
                }
            }
        }

        // complete the bundle construction
        let hir_id = tcx.local_def_id_to_hir_id(owner_id);
        let exec_full = base.mk_mir(owner_id, hir_id, tcx.hir_span_with_body(hir_id), exec);
        bundles.push(SolBundle {
            generics,
            adt_defs: flat_adt_defs,
            trait_defs: flat_trait_defs,
            executable: exec_full,
        });

        // re-assign the base builder for next iteration
        base_builder = base;
    }

    // unpack the builder
    let BaseBuilder { tcx: _, src_dir: _, src_cache: _, id_cache } = base_builder;

    // collect the id to description mappings
    let mut id_desc = vec![];

    // we sort the values alphabetically by description first, then by id, but after collection
    #[allow(rustc::potential_query_instability)]
    for (ident, desc) in id_cache.into_values() {
        id_desc.push((ident, desc));
    }
    id_desc.sort_by(|(ident_a, desc_a), (ident_b, desc_b)| {
        desc_a.cmp(desc_b).then_with(|| ident_a.cmp(ident_b))
    });

    // construct the crate
    SolCrate { root: module_mir, bundles, id_desc }
}

/* --- BEGIN OF SYNC --- */

/// A trait alias for all sorts IR elements
pub(crate) trait SolIR =
    Debug + Clone + PartialEq + Eq + PartialOrd + Ord + Serialize + DeserializeOwned;

/*
* Common
 */

/// The base information associated with anything that has an hir_id, span, but no def_id
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(bound = "T: Serialize + DeserializeOwned")]
pub(crate) struct SolHIR<T: SolIR> {
    pub(crate) doc_comments: Vec<SolDocComment>,
    pub(crate) data: T,
}

/// The base information associated with anything that has a def_id, span, and maybe hir_id
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
    pub(crate) bundles: Vec<SolBundle>,
    pub(crate) id_desc: Vec<(SolIdent, SolPathDesc)>,
}

/// A complete execution context
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolBundle {
    pub(crate) generics: Vec<SolGenericParam>,
    pub(crate) adt_defs: Vec<(SolIdent, Vec<SolGenericArg>, SolAdtDef)>,
    pub(crate) trait_defs: Vec<(SolIdent, Vec<SolGenericArg>, SolTraitDef)>,
    pub(crate) executable: SolMIR<SolExec>,
}

/*
 * Module
 */

/// A module
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolModule {
    pub(crate) name: SolModuleName,
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
    Constant(SolCEval),
}

/// THIR of a function
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolFnDef {
    pub(crate) abi: SolExternAbi,
    pub(crate) ret_ty: SolType,
    pub(crate) params: Vec<SolType>,
    pub(crate) body: SolExpr,
}

/// THIR of a constant evaluation
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolCEval {
    pub(crate) ty: SolType,
    pub(crate) body: SolExpr,
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
    // pattern types
    Pat(Box<SolType>, Box<SolTyPat>),
    // dependencies
    Foreign(SolIdent),
    Adt(SolIdent, Vec<SolGenericArg>),
    // reference types
    ImmPtr(Box<SolType>),
    MutPtr(Box<SolType>),
    ImmRef(Box<SolType>),
    MutRef(Box<SolType>),
    // type parameter
    Param(SolIdent),
    // compound types
    Tuple(Vec<SolType>),
    Slice(Box<SolType>),
    Array(Box<SolType>, Box<SolConst>),
    // function pointer
    Function(SolIdent, Vec<SolGenericArg>),
    Closure(SolIdent, Vec<SolGenericArg>),
    FnPtr(SolExternAbi, Vec<SolType>, Box<SolType>),
    // dynamic types
    Dynamic(Vec<SolClause>),
    Assoc {
        trait_ident: SolIdent,
        trait_ty_args: Vec<SolGenericArg>,
        item_ident: SolIdent,
        item_ty_args: Vec<SolGenericArg>,
    },
    // alias types
    Alias(Box<SolType>),
    Opaque(Box<SolType>),
}

/// A pattern type
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolTyPat {
    NotNull,
    Range(SolConst, SolConst),
    Or(Vec<SolTyPat>),
}

/// A projection term
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolProjTerm {
    Type(SolType),
    Const(SolConst),
}

/// Differentiate kinds of generics
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolGenericKind {
    Lifetime,
    Const,
    Type,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolGenericParam {
    pub(crate) ident: SolIdent,
    pub(crate) name: SolParamName,
    pub(crate) kind: SolGenericKind,
}

/// A generic argument
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolGenericArg {
    Type(SolType),
    Const(SolConst),
    Lifetime,
}

/// User-defined type, i.e., an algebraic data type (ADT)
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolAdtDef {
    Struct { fields: Vec<SolField> },
    Union { fields: Vec<SolField> },
    Enum { variants: Vec<SolVariant> },
}

/// A field definition in an ADT
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolField {
    pub(crate) index: SolFieldIndex,
    pub(crate) name: SolFieldName,
    pub(crate) ty: SolType,
    pub(crate) default: Option<SolIdent>,
}

/// A field definition in an ADT
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolVariant {
    pub(crate) index: SolVariantIndex,
    pub(crate) name: SolVariantName,
    pub(crate) discr: SolVariantDiscr,
    pub(crate) fields: Vec<SolField>,
}

/// Trait definition
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolTraitDef {
    pub(crate) clauses: Vec<SolClause>,
}

/// A clause in a predicate
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolClause {
    TraitImpl(SolIdent, Vec<SolGenericArg>),
    TraitNotImpl(SolIdent, Vec<SolGenericArg>),
    WellFormed(SolProjTerm),
    Projection(SolProjTerm, SolProjTerm),
    TypeOutlives(SolType),
    RegionOutlives,
    ConstHasType(SolConst, SolType),
    ConstEvaluatable(SolConst),
}

/*
 * Constant
 */

/// A compile-time constant
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolConst {
    Param(SolIdent),
    Value(SolValue),
}

/// A constant with concrete value and type
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolValue {
    // primitives
    Bool(bool),
    Char(char),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    I128(i128),
    Isize(isize),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    U128(u128),
    Usize(usize),
    F16(String),
    F32(String),
    F64(String),
    F128(String),
    // string
    Str(String),
    // composite
    Tuple(Vec<SolConst>),
    Slice(SolType, Vec<SolConst>),
    Array(SolType, Vec<SolConst>),
    Struct(SolIdent, Vec<SolGenericArg>, Vec<(SolFieldIndex, SolConst)>),
    Union(SolIdent, Vec<SolGenericArg>, SolFieldIndex, Box<SolConst>),
    Enum(SolIdent, Vec<SolGenericArg>, SolVariantIndex, Vec<(SolFieldIndex, SolConst)>),
    // reference
    ImmRef(Box<SolValue>),
    MutRef(Box<SolValue>),
    ImmPtrNull(SolType),
    MutPtrNull(SolType),
    // function pointers
    FuncDef(SolIdent, Vec<SolGenericArg>),
    Closure(SolIdent, Vec<SolGenericArg>),
}

/*
 * Expression
 */

/// An expression in THIR
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolExpr {
    pub(crate) ty: SolType,
    pub(crate) op: Box<SolOp>,
    pub(crate) span: SolSpan,
}

/// Details of the operation in an expression
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolOp {
    // markers
    Scope(SolHIR<SolExpr>),
    TypeAscribe(SolExpr),
    // use
    Use(SolExpr),
    VarRef(SolLocalVarIndex),
    UpVarRef(SolIdent, SolLocalVarIndex),
    ConstParam(SolIdent),
    ConstValue(SolIdent, Vec<SolGenericArg>),
    // intrisics
    Box(SolExpr),
    Deref(SolExpr),
    // operators
    Not(SolExpr),
    Neg(SolExpr),
    Add(SolExpr, SolExpr),
    AddUnchecked(SolExpr, SolExpr),
    AddWithOverflow(SolExpr, SolExpr),
    Sub(SolExpr, SolExpr),
    SubUnchecked(SolExpr, SolExpr),
    SubWithOverflow(SolExpr, SolExpr),
    Mul(SolExpr, SolExpr),
    MulUnchecked(SolExpr, SolExpr),
    MulWithOverflow(SolExpr, SolExpr),
    Div(SolExpr, SolExpr),
    Rem(SolExpr, SolExpr),
    BitXor(SolExpr, SolExpr),
    BitAnd(SolExpr, SolExpr),
    BitOr(SolExpr, SolExpr),
    Shl(SolExpr, SolExpr),
    ShlUnchecked(SolExpr, SolExpr),
    Shr(SolExpr, SolExpr),
    ShrUnchecked(SolExpr, SolExpr),
    Eq(SolExpr, SolExpr),
    Ne(SolExpr, SolExpr),
    Lt(SolExpr, SolExpr),
    Le(SolExpr, SolExpr),
    Gt(SolExpr, SolExpr),
    Ge(SolExpr, SolExpr),
    Cmp(SolExpr, SolExpr),
    LogicalAnd(SolExpr, SolExpr),
    LogicalOr(SolExpr, SolExpr),
    // assignments
    Assign {
        lhs: SolExpr,
        rhs: SolExpr,
    },
    AddAssign {
        lhs: SolExpr,
        rhs: SolExpr,
    },
    SubAssign {
        lhs: SolExpr,
        rhs: SolExpr,
    },
    MulAssign {
        lhs: SolExpr,
        rhs: SolExpr,
    },
    DivAssign {
        lhs: SolExpr,
        rhs: SolExpr,
    },
    RemAssign {
        lhs: SolExpr,
        rhs: SolExpr,
    },
    BitXorAssign {
        lhs: SolExpr,
        rhs: SolExpr,
    },
    BitAndAssign {
        lhs: SolExpr,
        rhs: SolExpr,
    },
    BitOrAssign {
        lhs: SolExpr,
        rhs: SolExpr,
    },
    ShlAssign {
        lhs: SolExpr,
        rhs: SolExpr,
    },
    ShrAssign {
        lhs: SolExpr,
        rhs: SolExpr,
    },
    // casts
    Cast(SolExpr),
    CastReifyFnPtr(SolExpr, bool),
    CastUnsafeFnPtr(SolExpr),
    CastClosureFnPtr(SolExpr, bool),
    CastMutToConstPtr(SolExpr),
    CastUnsizeArrayPtr(SolExpr),
    CastUnsizeArrayRef(SolExpr),
    NeverToAny(SolExpr),
    // access
    Field {
        base: SolExpr,
        variant: SolVariantIndex,
        field: SolFieldIndex,
    },
    Index {
        base: SolExpr,
        index: SolExpr,
    },
    // packing
    Repeat(SolExpr, SolConst),
    Array(Vec<SolExpr>),
    Tuple(Vec<SolExpr>),
    Adt {
        adt_ident: SolIdent,
        adt_args: Vec<SolGenericArg>,
        variant: SolVariantIndex,
        fields: Vec<(SolFieldIndex, SolExpr)>,
        base: SolAdtBase,
    },
    // borrow
    ImmBorrow(SolExpr),
    MutBorrow(SolExpr),
    ImmRawPtr(SolExpr),
    MutRawPtr(SolExpr),
    // control-flow
    If {
        cond: SolExpr,
        then: SolExpr,
        else_opt: Option<SolExpr>,
    },
    Loop(SolExpr),
    Break(Option<SolExpr>),
    Continue,
    Return(Option<SolExpr>),
    // function calls
    Call {
        target: SolType,
        callee: SolExpr,
        args: Vec<SolExpr>,
    },
    ConstBlock(SolIdent, Vec<SolGenericArg>),
    // pattern
    Let(SolExpr, SolPattern),
    Match(SolExpr, Vec<SolHIR<SolMatchArm>>),
    // compound
    Block(SolBlock),
    // literals
    BaseLiteral(bool, SolValue, SolSpan),
    ScalarLiteral(SolValue),
    ZstLiteral(SolValue),
    // closure
    Closure(SolIdent, Vec<SolGenericArg>, Vec<SolExpr>),
}

/// A pattern matcher in THIR
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolPattern {
    pub(crate) ty: SolType,
    pub(crate) rule: SolPatRule,
    pub(crate) span: SolSpan,
}

/// A pattern matching rule in THIR
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolPatRule {
    Missing,
    Wild,
    Never,
    Bind {
        name: SolLocalVarName,
        var_id: SolLocalVarIndex,
        mode: SolBindMode,
        ty: SolType,
        subpat: Option<Box<SolPattern>>,
    },
    Variant {
        adt_ident: SolIdent,
        adt_ty_args: Vec<SolGenericArg>,
        variant: SolVariantIndex,
        fields: Vec<(SolFieldIndex, SolPattern)>,
    },
    Leaf {
        fields: Vec<(SolFieldIndex, SolPattern)>,
    },
    Slice {
        prefix: Vec<SolPattern>,
        slice: Option<Box<SolPattern>>,
        suffix: Vec<SolPattern>,
    },
    Array {
        prefix: Vec<SolPattern>,
        slice: Option<Box<SolPattern>>,
        suffix: Vec<SolPattern>,
    },
    Deref(Box<SolPattern>),
    DerefBox(Box<SolPattern>),
    DerefImm(Box<SolPattern>),
    DerefMut(Box<SolPattern>),
    Constant(SolValue),
    Or(Vec<SolPattern>),
}

/// Binding mode
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolBindMode {
    ImmByValue,
    MutByValue,
    ImmByImmRef,
    ImmByMutRef,
    MutByImmRef,
    MutByMutRef,
}

/// A match arm in THIR
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolMatchArm {
    pub(crate) pat: SolPattern,
    pub(crate) guard: Option<SolExpr>,
    pub(crate) body: SolExpr,
    pub(crate) span: SolSpan,
}

/// A block of expressiosn in THIR
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolBlock {
    pub(crate) safety: bool,
    pub(crate) stmts: Vec<SolStmt>,
    pub(crate) expr: Option<SolExpr>,
    pub(crate) span: SolSpan,
}

/// A statement in THIR
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolStmt {
    Expr(SolExpr),
    Bind(SolHIR<SolLetBinding>),
}

/// Let binding in a statement in THIR
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolLetBinding {
    pat: SolPattern,
    init: Option<(SolExpr, Option<SolBlock>)>,
    span: SolSpan,
}

/// Base
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolAdtBase {
    None,
    Overlay(SolExpr, Vec<SolType>),
    Default(Vec<SolType>),
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

/// A parameter name
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolModuleName(pub(crate) String);

/// A parameter name
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolParamName(pub(crate) String);

/// A name to a local variable
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolLocalVarName(pub(crate) String);

/// A index to a local variable
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolLocalVarIndex(pub(crate) usize);

/// A field name
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolFieldName(pub(crate) String);

/// A field index
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolFieldIndex(pub(crate) usize);

/// A variant name
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolVariantName(pub(crate) String);

/// A variant index
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolVariantIndex(pub(crate) usize);

/// A variant discrimanant
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolVariantDiscr(pub(crate) u128);

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

/*
 * Utilities
 */

/// A stack for context logging
struct LogStack {
    stack: VecDeque<(&'static str, String)>,
}

impl LogStack {
    /// Create a new log stack
    fn new() -> Self {
        Self { stack: VecDeque::new() }
    }

    /// Increment the depth context
    fn push(&mut self, tag: &'static str, msg: String) {
        let indent = "  ".repeat(self.stack.len());
        info!("{indent}-> |{tag}| {msg}");
        self.stack.push_back((tag, msg.to_string()));
    }

    /// Decrement the depth context
    fn pop(&mut self) {
        let (tag, msg) = self.stack.pop_back().expect("[invariant] context stack underflows");
        let indent = "  ".repeat(self.stack.len());
        info!("{indent}<- |{tag}| {msg}");
    }

    /// Check if the stack is empty
    fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }
}

/// Create a fully qualified path string with crate name
#[inline]
fn util_debug_symbol<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    ty_args: GenericArgsRef<'tcx>,
) -> String {
    let path_str = tcx.def_path_str_with_args(def_id, ty_args);
    if def_id.is_local() {
        let krate = tcx.crate_name(LOCAL_CRATE).to_ident_string();
        format!("{krate}::{path_str}")
    } else {
        path_str
    }
}

/// helper for unpacking strings
#[inline]
fn util_values_to_string(bytes: &[SolConst]) -> String {
    String::from_utf8(
        bytes
            .iter()
            .map(|v| match v {
                SolConst::Value(SolValue::U8(b)) => *b,
                _ => bug!("[invariant] expect u8 value for string bytes"),
            })
            .collect(),
    )
    .unwrap_or_else(|_| bug!("[invariant] invalid utf-8 string"))
}

/// Check whether a type has any feasible value
#[inline]
fn has_feasible_value<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    match ty.kind() {
        // infeasible
        ty::Never => false,

        // feasible
        ty::Bool
        | ty::Char
        | ty::Int(_)
        | ty::Uint(_)
        | ty::Float(_)
        | ty::Str
        | ty::FnDef(..)
        | ty::Closure(..)
        | ty::FnPtr(..) => true,

        // conditional
        ty::Tuple(elems) => elems.iter().all(|e| has_feasible_value(tcx, e)),
        ty::Adt(adt_def, generics) => match adt_def.adt_kind() {
            AdtKind::Struct => adt_def
                .non_enum_variant()
                .fields
                .iter()
                .all(|f| has_feasible_value(tcx, f.ty(tcx, generics))),
            AdtKind::Union => adt_def
                .non_enum_variant()
                .fields
                .iter()
                .any(|f| has_feasible_value(tcx, f.ty(tcx, generics))),
            AdtKind::Enum => adt_def.variants().iter().any(|variant| {
                variant.fields.iter().all(|f| has_feasible_value(tcx, f.ty(tcx, generics)))
            }),
        },

        // inner-type
        ty::Array(sub, _) | ty::Slice(sub) | ty::Ref(_, sub, _) | ty::RawPtr(sub, _) => {
            has_feasible_value(tcx, *sub)
        }

        // unsupported
        ty::Dynamic(..) => bug!("[unsupported] feasibility query for dynamic type"),

        // should not appear
        _ => bug!("[invariant] unexpected type in feasibility query: {ty}"),
    }
}

/// Get a uniquely feasible field for a union
#[inline]
fn get_uniquely_feasible_field<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: AdtDef<'tcx>,
    generics: GenericArgsRef<'tcx>,
) -> Option<FieldIdx> {
    let mut feasible_field = None;
    for (field_idx, field_def) in def.non_enum_variant().fields.iter_enumerated() {
        if has_feasible_value(tcx, field_def.ty(tcx, generics)) {
            if feasible_field.is_some() {
                // found more than one feasible fields
                return None;
            }
            feasible_field = Some(field_idx);
        }
    }
    // return the uniquely feasible field, if any
    feasible_field
}

/// Get a uniquely feasible variant for an enum
#[inline]
fn get_uniquely_feasible_variant<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: AdtDef<'tcx>,
    generics: GenericArgsRef<'tcx>,
) -> Option<VariantIdx> {
    let mut feasible_variant = None;
    for (variant_idx, variant_def) in def.variants().iter_enumerated() {
        if variant_def.fields.iter().all(|f| has_feasible_value(tcx, f.ty(tcx, generics))) {
            if feasible_variant.is_some() {
                // found more than one feasible variants
                return None;
            }
            feasible_variant = Some(variant_idx);
        }
    }
    // return the uniquely feasible variant, if any
    feasible_variant
}
