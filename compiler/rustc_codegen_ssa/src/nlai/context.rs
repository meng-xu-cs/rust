// ignore-tidy-filelength

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;
use std::path::PathBuf;

use rustc_abi::{
    ExternAbi, FieldIdx, FieldsShape, Primitive, Scalar as ScalarAbi, Size, TagEncoding,
    VariantIdx, Variants,
};
use rustc_ast::{
    AttrStyle, BindingMode, ByRef, FloatTy, IntTy, LitFloatType, LitIntType, LitKind, Mutability,
    UintTy,
};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::attrs::AttributeKind;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{CRATE_DEF_ID, DefId, LOCAL_CRATE, LocalDefId};
use rustc_hir::{
    Attribute, CRATE_HIR_ID, HirId, Item, ItemKind, MatchSource, Mod, OwnerId, RangeEnd, Safety,
};
use rustc_middle::hir::place::PlaceBase as HirPlaceBase;
use rustc_middle::middle::region::{Scope, ScopeData};
use rustc_middle::mir::interpret::{AllocRange, Allocation, GlobalAlloc, Scalar};
use rustc_middle::mir::{AssignOp, BinOp, BorrowKind, UnOp};
use rustc_middle::thir::{
    AdtExpr, AdtExprBase, Arm, Block, BlockId, BlockSafety, BodyTy, ClosureExpr, Expr, ExprId,
    ExprKind, FieldExpr, FieldPat, FruInfo, LocalVarId, LogicalOp, Param, Pat, PatKind, PatRange,
    PatRangeBoundary, Stmt, StmtId, StmtKind, Thir,
};
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::print::{
    with_no_trimmed_paths, with_no_visible_paths, with_resolve_crate_name,
};
use rustc_middle::ty::util::Discr;
use rustc_middle::ty::{
    AdtDef, AdtKind, AliasTyKind, BoundTy, BoundTyKind, BoundVar, Clause, ClauseKind, Const,
    ConstKind, FieldDef, GenericArg, GenericArgKind, GenericArgsRef, GenericParamDef,
    GenericParamDefKind, Instance, InstanceKind, List, OutlivesPredicate, ParamConst, ParamTy,
    Pattern, PatternKind, PredicatePolarity, ProjectionPredicate, ScalarInt, Term, TermKind,
    TraitDef, TraitPredicate, Ty, TyCtxt, TypeFoldable, TypingEnv, UniverseIndex, Unnormalized,
    UpvarArgs, ValTreeKind, Value, Visibility,
};
use rustc_middle::{bug, ty};
use rustc_span::{DUMMY_SP, RemapPathScopeComponents, Span, StableSourceFileId, Symbol};
use rustc_trait_selection::infer::TyCtxtInferExt;
use rustc_trait_selection::traits::{self, EvaluateConstErr};
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
    id_cache: FxHashMap<DefId, (SolIdent, SolPathDesc, SolSpan)>,
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

    /// constant param types
    cp_types: BTreeMap<SolIdent, SolType>,

    /// collected definitions of datatypes
    adt_defs: BTreeMap<SolIdent, BTreeMap<Vec<SolGenericArg>, Option<SolAdtDef>>>,

    /// collected definitions of traits
    trait_defs: BTreeMap<SolIdent, BTreeMap<Vec<SolGenericArg>, Option<SolTraitDef>>>,

    /// dynamic types and their associated clauses
    dyn_types: BTreeMap<SolDynTypeIndex, (Ty<'tcx>, DynTypeStatus)>,

    /// static initializers
    static_inits: BTreeMap<SolIdent, Option<SolValue>>,

    /// upvar declarations
    upvar_decls: BTreeMap<SolLocalVarIndex, (SolLocalVarName, Option<SolType>)>,

    /// local variables
    var_scope: BTreeMap<SolLocalVarIndex, SolLocalVarName>,

    /// expr id and adjusted stmt id set
    inst_ids: BTreeSet<SolInstIndex>,

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
        if let Some((ident, _, _)) = self.id_cache.get(&def_id) {
            return ident.clone();
        }

        // now construct the identifier
        let def_path_hash = self.tcx.def_path_hash(def_id);
        let ident = SolIdent {
            krate: SolHash64(def_path_hash.stable_crate_id().as_u64()),
            local: SolHash64(def_path_hash.local_hash().as_u64()),
        };

        // insert into the cache
        let desc = SolPathDesc(util_ident_string(self.tcx, def_id));
        let span = self.mk_span(self.tcx.def_span(def_id));
        self.id_cache.insert(def_id, (ident.clone(), desc, span));

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
            let external_src = start_loc.file.external_src.borrow();
            let src_code = start_loc.file.src.as_ref().map_or_else(
                || {
                    external_src
                        .get_source()
                        .unwrap_or_else(|| bug!("[invariant] visible span has no source code"))
                },
                |src| src.as_str(),
            );

            let src_path = self.src_dir.join(format!("{:x}", file_id.0.as_u128()));
            if src_path.exists() {
                bug!("[invariant] source file {} already exists", src_path.display());
            }
            std::fs::write(&src_path, src_code).unwrap_or_else(|e| {
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
                ItemKind::Trait { .. } | ItemKind::TraitAlias(..) => {
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
        let typing_env = tcx.typing_env_normalized_for_post_analysis(owner_id);
        Self {
            tcx,
            typing_env,
            base,
            owner_id,
            generics,
            cp_types: BTreeMap::new(),
            adt_defs: BTreeMap::new(),
            trait_defs: BTreeMap::new(),
            dyn_types: BTreeMap::new(),
            static_inits: BTreeMap::new(),
            upvar_decls: BTreeMap::new(),
            var_scope: BTreeMap::new(),
            inst_ids: BTreeSet::new(),
            log_stack: LogStack::new(),
        }
    }

    /// Utility to normalize a type foldable value by erasing regions and applying type normalization
    #[inline]
    fn tcx_try_normalize<T>(&self, value: Unnormalized<'tcx, T>) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>> + Copy,
    {
        self.tcx
            .try_normalize_erasing_regions(self.typing_env, value)
            .unwrap_or_else(|_| self.tcx.erase_and_anonymize_regions(value.skip_norm_wip()))
    }

    /// Utility to get the type of a field after instantiating it with the generic arguments.
    #[inline]
    fn tcx_field_ty(&self, field: &FieldDef, ty_args: GenericArgsRef<'tcx>) -> Ty<'tcx> {
        self.tcx_try_normalize(field.ty(self.tcx, ty_args))
    }

    /// Get a generic parameter by index with cross-checking on name and kind
    fn get_param(&self, index: u32, name: Symbol, kind: SolGenericKind) -> SolIdent {
        // validate the type parameter against the generics declarations
        let param_def = self.generics.get(index as usize).unwrap_or_else(|| {
            bug!("[invariant] generic parameter {name} index out of bounds: {index}",)
        });

        // FIXME: ideally the right lookup is to filter the generics by index and extract the unique one.
        // The current workaround is fragile but will fail-early so we can detect and fix if it happens.
        if param_def.index.0 != index as usize {
            bug!(
                "[invariant] generic parameter {name} index mismatch: {} vs {index}",
                param_def.index.0,
            )
        }
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
    #[inline]
    fn mk_doc_comments(&self, hir_id: HirId) -> Vec<SolDocComment> {
        self.base.mk_doc_comments(hir_id)
    }

    /// Record a HIR node in the THIR body with its metadata
    #[inline]
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

    /// Record a vector of generic arguments
    #[inline]
    pub(crate) fn mk_generic_args(&mut self, ty_args: GenericArgsRef<'tcx>) -> Vec<SolGenericArg> {
        ty_args.iter().map(|arg| self.mk_generic_arg(arg)).collect()
    }

    /// Record the details of an ADT field
    pub(crate) fn mk_adt_field(
        &mut self,
        ty_args: GenericArgsRef<'tcx>,
        field_idx: FieldIdx,
        field_def: &FieldDef,
    ) -> SolField {
        SolField {
            index: SolFieldIndex(field_idx.index()),
            name: SolFieldName(field_def.name.to_ident_string()),
            ty: self.mk_type(self.tcx_field_ty(field_def, ty_args)),
            default: field_def.value.map(|did| self.mk_ident(did)),
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
        let generic_args = self.mk_generic_args(ty_args);

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
                    .map(|(field_idx, field_def)| self.mk_adt_field(ty_args, field_idx, field_def))
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
                    .map(|(field_idx, field_def)| self.mk_adt_field(ty_args, field_idx, field_def))
                    .collect();
                SolAdtDef::Union { fields }
            }
            AdtKind::Enum => {
                let mut variants = vec![];
                for (variant_idx, variant_def) in adt_def.variants().iter_enumerated() {
                    let variant_index = SolVariantIndex(variant_idx.index());
                    let variant_name = SolVariantName(variant_def.name.to_ident_string());
                    let discr = adt_def.discriminant_for_variant(self.tcx, variant_idx);
                    if !matches!(discr.ty.kind(), ty::Int(_) | ty::Uint(_)) {
                        bug!("[invariant] non-integral discriminant for enum {def_desc}");
                    }
                    let variant_descr = SolVariantDiscr(discr.val);
                    let fields = variant_def
                        .fields
                        .iter_enumerated()
                        .map(|(field_idx, field_def)| {
                            self.mk_adt_field(ty_args, field_idx, field_def)
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
        let generic_args = self.mk_generic_args(ty_args);

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
            parsed_clauses.push(self.mk_clause(self.tcx_try_normalize(clause)));
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
            ClauseKind::HostEffect(..) => bug!("[unsupported] host effect clause"),

            // unexpected
            ClauseKind::UnstableFeature(..) => {
                bug!("[invariant] unexpected unstable feature clause: {clause}")
            }
        };

        self.log_stack.pop();
        parsed
    }

    /// Record a projection term in the type system
    pub(crate) fn mk_projection_term(&mut self, term: Term<'tcx>) -> SolProjTerm {
        match term.kind() {
            TermKind::Ty(ty) => SolProjTerm::Type(self.mk_type(ty)),
            TermKind::Const(cval) => SolProjTerm::Const(self.mk_const(cval)),
        }
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
                match adt_def.adt_kind() {
                    AdtKind::Struct => SolType::Struct(ident, generic_args),
                    AdtKind::Union => SolType::Union(ident, generic_args),
                    AdtKind::Enum => SolType::Enum(ident, generic_args),
                }
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
            ty::FnDef(_, _) => match self.try_resolve_instance(ty) {
                ResolvedFunction::FuncDef(ident, generics) => SolType::FuncDef(ident, generics),
                ResolvedFunction::FuncSym(ident, generics) => SolType::FuncSym(ident, generics),
                ResolvedFunction::AdtCtor(ident, generics) => SolType::AdtCtor(ident, generics),
                ResolvedFunction::Virtual(ident, generics) => SolType::Virtual(ident, generics),
                ResolvedFunction::Libfunc(ident, generics) => SolType::Libfunc(ident, generics),
                ResolvedFunction::External(ident, generics) => SolType::External(ident, generics),
                ResolvedFunction::Intrinsic(ident, generics) => SolType::Intrinsic(ident, generics),
            },
            ty::Closure(def_id, ty_args) => {
                let ident = self.mk_ident(*def_id);
                let generic_args = self.mk_generic_args(*ty_args);
                SolType::Closure(ident, generic_args)
            }
            ty::FnPtr(sig_binder, header) => {
                let sig = self.tcx.instantiate_bound_regions_with_erased(*sig_binder);
                let abi = self.mk_abi(header.abi(), header.c_variadic(), header.safety());
                let ret_ty = self.mk_type(sig.output());
                let params = sig.inputs().iter().map(|input_ty| self.mk_type(*input_ty)).collect();
                SolType::FnPtr(SolFnSig { abi, param_tys: params, output_ty: Box::new(ret_ty) })
            }

            // dynamic
            ty::Dynamic(predicates, region) => {
                if !(region.is_erased() || region.is_static()) {
                    bug!("[invariant] regions should be erased or static in THIR: {region}");
                }

                // re-used the index of a previously processed dynamic type
                match self.dyn_types.iter().find(|(_, (t, s))| {
                    *t == ty
                        && match s {
                            DynTypeStatus::Processed(_) => true,
                            DynTypeStatus::Resolving(_) => bug!("[unsupported] recursive dyn type"),
                        }
                }) {
                    Some((dyn_idx, _)) => SolType::Dynamic(dyn_idx.clone()),
                    None => {
                        // create a placeholder Self type for the trait object
                        let var_idx = self.dyn_types.len();
                        let universe = self
                            .tcx
                            .infer_ctxt()
                            .build(self.typing_env.typing_mode())
                            .create_next_universe();
                        let self_ty = Ty::new_placeholder(
                            self.tcx,
                            ty::PlaceholderType::new(
                                universe,
                                BoundTy {
                                    var: BoundVar::from_usize(var_idx),
                                    kind: BoundTyKind::Anon,
                                },
                            ),
                        );

                        // mark the beginning of clause collection
                        let dyn_idx = SolDynTypeIndex(var_idx);
                        self.dyn_types
                            .insert(dyn_idx.clone(), (ty, DynTypeStatus::Resolving(universe)));

                        // resolve the predicates with Self type
                        let clauses = predicates
                            .iter()
                            .map(|pred| self.mk_clause(pred.with_self_ty(self.tcx, self_ty)))
                            .collect();

                        // update the collected clauses
                        let (_, status) = self.dyn_types.get_mut(&dyn_idx).unwrap();
                        *status = DynTypeStatus::Processed(clauses);

                        // the type will be returned as the index only
                        SolType::Dynamic(dyn_idx)
                    }
                }
            }
            ty::Placeholder(holder) => {
                // NOTE: under normal THIR, there should be no placeholder types at all. Hence, the only reason
                // why we can see a placeholder type here is when we are creating a Self type for a trait object
                // which is used in resolving the dynamic type's predicates (see above handling of ty::Dynamic).
                assert!(matches!(holder.bound.kind, BoundTyKind::Anon));
                let dyn_idx = SolDynTypeIndex(holder.bound.var.index());
                assert!(
                    self.dyn_types
                        .get(&dyn_idx)
                        .map_or(false, |(_, status)| matches!(status, DynTypeStatus::Resolving(ui) if *ui == holder.universe)),
                );
                SolType::Dynamic(dyn_idx)
            }

            // alias
            ty::Alias(alias_ty) => match alias_ty.kind {
                AliasTyKind::Projection { .. } => {
                    // It is possible that normalization does not change the type, for example,
                    // when the projection comes from a type parameter (which implements a trait),
                    // e.g., <P as Foo>::T, where P is the type parameter of the THIR definition.
                    let (trait_ref, own_ty_args) = alias_ty.trait_ref_and_own_args(self.tcx);
                    let (trait_ident, trait_ty_args) =
                        self.mk_trait(self.tcx.trait_def(trait_ref.def_id), trait_ref.args);
                    let item_ty_args =
                        own_ty_args.iter().map(|arg| self.mk_generic_arg(*arg)).collect();

                    // construct the associated type
                    SolType::Assoc {
                        trait_ident,
                        trait_ty_args,
                        item_ident: self.mk_ident(alias_ty.kind.def_id()),
                        item_ty_args,
                    }
                }
                AliasTyKind::Opaque { def_id } => {
                    let hidden_ty = self.tcx.type_of(def_id).instantiate(self.tcx, alias_ty.args);
                    let norm_ty = self.tcx_try_normalize(hidden_ty);
                    assert_ne!(norm_ty, ty, "[invariant] opaque alias type should be revealed");
                    // MAYFIX: keep track of opaque alias types separately, i.e., SolType::Opaque(..)?
                    self.mk_type(norm_ty)
                }
                AliasTyKind::Inherent { .. } => {
                    bug!("[invariant] inherent alias type should be normalized")
                }
                AliasTyKind::Free { .. } => {
                    bug!("[invariant] free alias type should be normalized")
                }
            },

            // unsupported
            ty::Coroutine(..) | ty::CoroutineClosure(..) | ty::CoroutineWitness(..) => {
                bug!("[unsupported] coroutine type")
            }
            ty::UnsafeBinder(..) => {
                bug!("[unsupported] unsafe binder")
            }

            // unexpected
            ty::Infer(..) | ty::Bound(..) | ty::Error(..) => {
                bug!("[invariant] unexpected type: {ty}")
            }
        };

        self.log_stack.pop();
        parsed
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

    /// Record a local variable identifier
    pub(crate) fn mk_local_var_id(&mut self, LocalVarId(var_id): LocalVarId) -> SolLocalVarIndex {
        SolLocalVarIndex(var_id.owner.def_id.local_def_index.index(), var_id.local_id.index())
    }

    /// Record a pattern in the THIR context
    pub(crate) fn mk_pat(&mut self, pat: &Pat<'tcx>) -> SolPattern {
        let Pat { ty, span, extra: _, kind } = pat;

        // mark the beginning
        self.log_stack.push("Pattern", format!("{kind:?}"));

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
                is_primary,
                is_shorthand: _,
            } => {
                let binding_mode = match (by_ref, mutability) {
                    (ByRef::No, Mutability::Not) => SolBindMode::ImmByValue,
                    (ByRef::No, Mutability::Mut) => SolBindMode::MutByValue,
                    (ByRef::Yes(_, Mutability::Not), Mutability::Not) => SolBindMode::ImmByImmRef,
                    (ByRef::Yes(_, Mutability::Not), Mutability::Mut) => SolBindMode::MutByImmRef,
                    (ByRef::Yes(_, Mutability::Mut), Mutability::Not) => SolBindMode::ImmByMutRef,
                    (ByRef::Yes(_, Mutability::Mut), Mutability::Mut) => SolBindMode::MutByMutRef,
                };

                let var_name = SolLocalVarName(name.to_ident_string());
                let var_index = self.mk_local_var_id(*var);
                let var_type = self.mk_type(*ty);
                let var_subpat = subpattern.as_ref().map(|sub_pat| Box::new(self.mk_pat(sub_pat)));

                // record the binding if primary, check consistency otherwise
                let var_hir_id = var.0;
                if *is_primary {
                    let existing = self.var_scope.insert(var_index.clone(), var_name.clone());
                    assert!(
                        existing.is_none(),
                        "[invariant] local variable {name} (with id: {var_hir_id}) is already defined"
                    );
                } else {
                    let existing = self.var_scope.get(&var_index).unwrap_or_else(|| panic!(
                        "[invariant] non-primary local variable {name} (with id: {var_hir_id}) is not defined yet"
                    ));
                    assert_eq!(
                        &var_name, existing,
                        "[invariant] non-primary local variable (with id: {var_hir_id}) name mismatch: {} vs {}",
                        existing.0, var_name.0
                    );
                }

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
            PatKind::Slice { prefix, slice, suffix } => {
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

            PatKind::Deref { pin: _, subpattern } => {
                SolPatRule::Deref(Box::new(self.mk_pat(subpattern)))
            }

            PatKind::Range(pat_range) => {
                let PatRange { lo, hi, end, ty } = pat_range.as_ref();
                let bound_lo = match lo {
                    PatRangeBoundary::PosInfinity => bug!("[invariant] +inf in lo range"),
                    PatRangeBoundary::NegInfinity => None,
                    PatRangeBoundary::Finite(valtree) => {
                        Some(self.mk_value_from_scalar(*ty, valtree.to_leaf()))
                    }
                };
                let bound_hi = match hi {
                    PatRangeBoundary::NegInfinity => bug!("[invariant] -inf in hi range"),
                    PatRangeBoundary::PosInfinity => None,
                    PatRangeBoundary::Finite(valtree) => {
                        Some(self.mk_value_from_scalar(*ty, valtree.to_leaf()))
                    }
                };
                let end_inclusive = match end {
                    RangeEnd::Included => true,
                    RangeEnd::Excluded => false,
                };
                SolPatRule::Range { lo: bound_lo, hi: bound_hi, end_inclusive }
            }
            PatKind::Constant { value } => SolPatRule::Constant(self.mk_value(*value)),
            PatKind::Or { pats } => {
                SolPatRule::Or(pats.iter().map(|sub_pat| self.mk_pat(sub_pat)).collect())
            }

            // unsupported
            PatKind::Guard { .. } => bug!("[unsupported] guard pattern"),
            PatKind::DerefPattern { .. } => bug!("[unsupported] deref pattern"),

            // unreachable
            PatKind::Error(..) => bug!("[invariant] unreachable pattern {kind:?}"),
        };

        // end of parsing
        self.log_stack.pop();

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
            ConstKind::Unevaluated(uneval) => {
                // evaluate it with an inference context
                match traits::try_evaluate_const(
                    &self.tcx.infer_ctxt().build(self.typing_env.typing_mode()),
                    cval,
                    self.typing_env.param_env,
                ) {
                    Ok(evaluated) => {
                        if matches!(evaluated.kind(), ConstKind::Unevaluated(_)) {
                            bug!("[invariant] fail to evaluate const: {cval} -> {evaluated}");
                        }
                        self.mk_const(evaluated)
                    }
                    Err(EvaluateConstErr::HasGenericsOrInfers) => {
                        // we hit an associated item that can't be evaluated now
                        let const_ty = self.tcx_try_normalize(
                            self.tcx.type_of(uneval.def).instantiate(self.tcx, uneval.args),
                        );
                        SolConst::Unevaluated(
                            self.mk_type(const_ty),
                            self.mk_ident(uneval.def),
                            self.mk_generic_args(uneval.args),
                        )
                    }
                    Err(_) => bug!("[invariant] failed to evaluate const {cval}"),
                }
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
            ValTreeKind::Branch(branches) => {
                if branches.is_empty() {
                    self.mk_value_from_zst(ty)
                } else {
                    let consts: Vec<_> = branches.iter().map(|cval| self.mk_const(cval)).collect();
                    self.mk_value_from_branch_consts(ty, consts)
                }
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
                // NOTE: we allow non-zero pointer values as meaningful (and nullptr) constant values
                let ptrval = scalar.to_target_usize(self.tcx) as usize;
                let pointee_ty = self.mk_type(*inner_ty);
                match mutability {
                    Mutability::Not => SolValue::ImmPtrNull(pointee_ty, ptrval),
                    Mutability::Mut => SolValue::MutPtrNull(pointee_ty, ptrval),
                }
            }

            // pattern type: delegate to base type
            ty::Pat(base_ty, _) => self.mk_value_from_scalar(*base_ty, scalar),

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
                SolValue::ImmRef(Box::new(SolValue::Str(v.as_str().to_string())))
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
                            let field_ty = self.tcx_field_ty(field_def, ty_args);
                            let field_val = self.mk_value_when_zst(field_ty)?;
                            fields.push((
                                SolFieldIndex(field_idx.index()),
                                SolConst::Value(field_val),
                            ));
                        }
                        SolValue::Struct(adt_ident, adt_ty_args, fields)
                    }
                    AdtKind::Union => {
                        // a union is a ZST if it has only one feasible field and the field is a ZST
                        let field_idx = self.get_uniquely_feasible_field(*def, ty_args)?;
                        let field_def = def.non_enum_variant().fields.get(field_idx).unwrap();
                        let field_ty = self.tcx_field_ty(field_def, ty_args);
                        let field_val = self.mk_value_when_zst(field_ty)?;
                        SolValue::Union(
                            adt_ident,
                            adt_ty_args,
                            SolFieldIndex(field_idx.index()),
                            Box::new(SolConst::Value(field_val)),
                        )
                    }
                    AdtKind::Enum => {
                        // an enum is a ZST if it has only one feasible variant and all fields in that variant are ZSTs
                        let variant_idx = self.get_uniquely_feasible_variant(*def, ty_args)?;

                        // parse the fields in that feasible variant
                        let variant_def = def.variant(variant_idx);
                        let mut fields = vec![];
                        for (field_idx, field_def) in variant_def.fields.iter_enumerated() {
                            let field_ty = self.tcx_field_ty(field_def, ty_args);
                            let field_val = self.mk_value_when_zst(field_ty)?;
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

            ty::FnDef(..) => match self.try_resolve_instance(ty) {
                ResolvedFunction::FuncDef(ident, generics) => SolValue::FuncDef(ident, generics),
                ResolvedFunction::FuncSym(ident, generics) => SolValue::FuncSym(ident, generics),
                ResolvedFunction::AdtCtor(ident, generics) => SolValue::AdtCtor(ident, generics),
                ResolvedFunction::Virtual(ident, generics) => SolValue::Virtual(ident, generics),
                ResolvedFunction::Libfunc(ident, generics) => SolValue::Libfunc(ident, generics),
                ResolvedFunction::External(ident, generics) => SolValue::External(ident, generics),
                ResolvedFunction::Intrinsic(ident, generics) => {
                    SolValue::Intrinsic(ident, generics)
                }
            },
            ty::Closure(def_id, ty_args) => {
                let ident = self.mk_ident(*def_id);
                let generic_args = self.mk_generic_args(ty_args);
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
            ty::Str => SolValue::Str(util_values_to_string(&consts)),

            ty::Slice(elem_ty) => {
                let const_ty = self.mk_type(*elem_ty);
                consts.iter().for_each(|c| {
                    assert_eq!(
                        self.type_of_const(c),
                        const_ty,
                        "[invariant] slice element type mismatch"
                    )
                });
                SolValue::Slice(const_ty, consts)
            }

            ty::Tuple(elem_tys) => {
                assert_eq!(
                    consts.len(),
                    elem_tys.len(),
                    "[invariant] tuple element count mismatch"
                );
                consts.iter().zip(elem_tys.iter()).for_each(|(c, ty)| {
                    let const_ty = self.type_of_const(c);
                    let expected_ty = self.mk_type(ty);
                    assert_eq!(const_ty, expected_ty, "[invariant] tuple element type mismatch")
                });
                SolValue::Tuple(consts)
            }

            ty::Array(elem_ty, len) => {
                let const_ty = self.mk_type(*elem_ty);
                consts.iter().for_each(|c| {
                    assert_eq!(
                        self.type_of_const(c),
                        const_ty,
                        "[invariant] array element type mismatch"
                    )
                });
                let size = len
                    .try_to_target_usize(self.tcx)
                    .unwrap_or_else(|| bug!("[invariant] unable to determine array size"))
                    as usize;
                assert_eq!(consts.len(), size, "[invariant] array element count mismatch");
                SolValue::Array(const_ty, consts)
            }

            ty::Adt(def, ty_args) => {
                let (adt_ident, adt_ty_args) = self.mk_adt(*def, ty_args);
                match def.adt_kind() {
                    AdtKind::Struct => {
                        let variant = def.non_enum_variant();
                        assert_eq!(
                            consts.len(),
                            variant.fields.len(),
                            "[invariant] struct field count mismatch"
                        );
                        SolValue::Struct(
                            adt_ident,
                            adt_ty_args,
                            variant
                                .fields
                                .iter_enumerated()
                                .zip(consts)
                                .map(|((field_idx, field_def), field_const)| {
                                    let field_ty = self.tcx_field_ty(field_def, ty_args);
                                    assert_eq!(
                                        self.type_of_const(&field_const),
                                        self.mk_type(field_ty),
                                        "[invariant] struct field type mismatch"
                                    );
                                    (SolFieldIndex(field_idx.index()), field_const)
                                })
                                .collect(),
                        )
                    }
                    AdtKind::Union => bug!("[unsupported] union const value"),
                    AdtKind::Enum => {
                        // the first const is the variant index
                        if consts.is_empty() {
                            bug!("[invariant] enum const value with no variant");
                        }

                        let variant_index = match &consts[0] {
                            SolConst::Value(SolValue::U32(idx)) => VariantIdx::from_u32(*idx),
                            _ => bug!("[invariant] expect enum variant index as u32 const"),
                        };

                        let variant_def = def.variant(variant_index);
                        assert_eq!(
                            consts.len() - 1,
                            variant_def.fields.len(),
                            "[invariant] enum field count mismatch"
                        );
                        SolValue::Enum(
                            adt_ident,
                            adt_ty_args,
                            SolVariantIndex(variant_index.index()),
                            variant_def
                                .fields
                                .iter_enumerated()
                                .zip(&consts[1..])
                                .map(|((field_idx, field_def), field_const)| {
                                    let field_ty = self.tcx_field_ty(field_def, ty_args);
                                    assert_eq!(
                                        self.type_of_const(field_const),
                                        self.mk_type(field_ty),
                                        "[invariant] enum field type mismatch"
                                    );
                                    (SolFieldIndex(field_idx.index()), field_const.clone())
                                })
                                .collect(),
                        )
                    }
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

    /// Record a static initializer
    pub(crate) fn mk_static_init(&mut self, def_id: DefId, ty: Ty<'tcx>) -> SolIdent {
        let ident = self.mk_ident(def_id);

        // if already defined or is being defined, return the key
        if self.static_inits.contains_key(&ident) {
            return ident;
        }

        // mark start
        let def_desc = util_debug_symbol(self.tcx, def_id, &List::empty());
        self.log_stack.push("Static", format!("{def_desc}"));

        // first update the entry to mark that static initialization in progress
        self.static_inits.insert(ident.clone(), None);

        let static_kind = self.tcx.def_kind(def_id);

        // Externally defined statics are unsupported. Nested backing statics do
        // not have MIR available, but they are still const-evaluable allocations.
        if !self.tcx.is_mir_available(def_id)
            && !matches!(static_kind, DefKind::Static { nested: true, .. })
        {
            bug!("[unsupported] static without MIR");
        }

        // evaluate the static initializer
        let alloc = self
            .tcx
            .eval_static_initializer(def_id)
            .unwrap_or_else(|_| bug!("[invariant] unable to evaluate static initializer"));
        let memory = alloc.inner();

        // process the constant value from memory
        let (_, value) = self.read_const_from_memory_and_layout(memory, Size::ZERO, ty, None);

        // update the static initializer entry
        self.static_inits.insert(ident.clone(), Some(value));

        // mark end
        self.log_stack.pop();

        // return the identifier
        ident
    }

    /// Helper function to read a constant from memory with optional wide-pointer metadata
    fn read_const_from_memory_and_layout(
        &mut self,
        memory: &Allocation,
        offset: Size,
        ty: Ty<'tcx>,
        mut metadata: Option<usize>,
    ) -> (Size, SolValue) {
        // utility
        let read_primitive = |tcx: TyCtxt<'tcx>, start, size: Size, is_provenance: bool| {
            memory.read_scalar(&tcx, AllocRange { start, size }, is_provenance).unwrap_or_else(
                |e| bug!("[invariant] failed to read a primitive in memory allocation: {e:?}"),
            )
        };

        // special handlings before layout-based reading
        match ty.kind() {
            ty::Alias(..) => bug!("[invariant] alias type should be normalized for const reading"),
            ty::Pat(base_ty, _) => {
                return self.read_const_from_memory_and_layout(memory, offset, *base_ty, metadata);
            }
            _ => (), // all other types, continue to layout-based reading
        };

        // get the layout of the type
        let layout = self
            .tcx
            .layout_of(self.typing_env.as_query_input(ty))
            .unwrap_or_else(|_| bug!("[invariant] unable to query layout of type {ty}"))
            .layout;

        // case by type
        let value = match ty.kind() {
            // primitives
            ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Float(_) => {
                match layout.variants {
                    Variants::Single { index } if index.index() == 0 => {}
                    _ => bug!("[invariant] expect a 0-variant for primitive type: {ty}"),
                }
                if !matches!(layout.fields, FieldsShape::Primitive) {
                    bug!("[invariant] expect a primitive field for type: {ty}");
                }

                let scalar = read_primitive(self.tcx, offset, layout.size, false)
                    .to_scalar_int()
                    .unwrap_or_else(|_| bug!("[invariant] expect a scalar int for type {ty}"));
                self.mk_value_from_scalar(ty, scalar)
            }

            // pointers
            ty::RawPtr(sub_ty, _) | ty::Ref(_, sub_ty, _) => {
                // sanity check on the variants
                match &layout.variants {
                    Variants::Single { index } if index.index() == 0 => {}
                    _ => bug!("[invariant] expect a 0-variant for ptr type {ty}"),
                }

                // need to resolve the dynamic type (if relevant)
                let pointer_size = self.tcx.data_layout.pointer_size();
                let is_dyn_ty = matches!(sub_ty.kind(), ty::Dynamic(..));

                let actual_sub_ty = if is_dyn_ty {
                    // the dynamic type must carries a metadata
                    if layout.size != pointer_size * 2 {
                        bug!("[invariant] expect metadata in the layout for {ty}");
                    }

                    // derive the metadata offset
                    let offset_metadata = match &layout.fields {
                        FieldsShape::Arbitrary { offsets, in_memory_order: _ } => {
                            // we know the metadata should be the second field, but still verify it
                            if offsets.len() != 2 {
                                bug!("[invariant] expect two fields for {ty}");
                            }
                            let mut offsets_iter = offsets.into_iter();
                            let off_pointer = *offsets_iter.next().unwrap();
                            let off_metadata = *offsets_iter.next().unwrap();
                            if off_pointer.bytes_usize() != 0 || off_metadata != pointer_size {
                                bug!("[invariant] unexpected offsets for {ty}");
                            }
                            off_metadata
                        }
                        _ => bug!("[invariant] expect arbitrary layout for {ty}"),
                    };

                    // read the metadata
                    let alloc_id = match read_primitive(
                        self.tcx,
                        offset + offset_metadata,
                        pointer_size,
                        true,
                    ) {
                        Scalar::Int(_) => {
                            bug!("[invariant] expect vtable for {sub_ty} in {ty}");
                        }
                        Scalar::Ptr(scalar_ptr, _) => {
                            let (prov, offset) = scalar_ptr.into_raw_parts();
                            if offset != Size::ZERO {
                                bug!("[invariant] expect zero offset for vtable pointer");
                            }
                            prov.alloc_id()
                        }
                    };

                    // get the actual underlying type behind the vtable
                    let actual_ty = match self.tcx.global_alloc(alloc_id) {
                        GlobalAlloc::VTable(vty, _) => vty,
                        _ => bug!("[invariant] {ty} has non-vtable metadata"),
                    };

                    // ensure that the actual type is sized (so later we don't parse the metadata as size)
                    if !actual_ty.is_sized(self.tcx, self.typing_env) {
                        bug!("[assumption] actual type {actual_ty} behind vtable should be sized");
                    }

                    // return the actual type
                    actual_ty
                } else {
                    *sub_ty
                };

                // preserve size metadata for slice/str and other unsized pointees
                //
                // NOTE: we only handle size-based metadata for now. In addition, we know that
                //       the `actual_sub_ty` must be sized if it comes from a dyn type (checked above),
                //       so if we need metadata here, the metadata must hold a size value, not a vtable.
                let need_metadata = !actual_sub_ty.is_sized(self.tcx, self.typing_env);
                let pointee_metadata = if need_metadata {
                    // sanity check on the shape of the fields
                    if pointer_size * 2 != layout.size {
                        bug!(
                            "[invariant] invalid layout size for {ty},
                            expect metadata but got {}",
                            layout.size.bytes_usize()
                        );
                    }

                    // derive the metadata offset
                    let offset_metadata = match &layout.fields {
                        FieldsShape::Arbitrary { offsets, in_memory_order: _ } => {
                            if offsets.len() != 2 {
                                bug!("[invariant] expect two fields for {ty}");
                            }
                            let mut offsets_iter = offsets.into_iter();
                            let off_pointer = *offsets_iter.next().unwrap();
                            let off_metadata = *offsets_iter.next().unwrap();
                            if off_pointer.bytes_usize() != 0 || off_metadata != pointer_size {
                                bug!("[invariant] unexpected offsets for {ty}");
                            }
                            off_metadata
                        }
                        _ => bug!("[invariant] expect arbitrary layout for {ty}"),
                    };

                    // read the metadata
                    let size_metadata = match read_primitive(
                        self.tcx,
                        offset + offset_metadata,
                        pointer_size,
                        false,
                    ) {
                        Scalar::Int(scalar_int) => scalar_int.to_target_usize(self.tcx) as usize,
                        Scalar::Ptr(..) => {
                            bug!("[invariant] unexpected scalar ptr metadata for {ty}")
                        }
                    };
                    Some(size_metadata)
                } else {
                    // sanity check on the shape of the fields
                    if !is_dyn_ty {
                        if pointer_size != layout.size {
                            bug!(
                                "[invariant] invalid layout size for {ty},
                                expect no more metadata behind dyn reference but got {}",
                                layout.size.bytes_usize()
                            );
                        }
                        if !matches!(layout.fields, FieldsShape::Primitive) {
                            bug!("[invariant] expect a primitive field for {ty}");
                        }
                    }
                    None
                };

                // read and parse the pointer
                let const_val = match read_primitive(self.tcx, offset, pointer_size, true) {
                    Scalar::Ptr(ptr, _) => {
                        let (prov, offset) = ptr.into_raw_parts();

                        match self.tcx.global_alloc(prov.alloc_id()) {
                            GlobalAlloc::Memory(allocated) => {
                                let inner_memory = allocated.inner();
                                assert!(
                                    matches!(inner_memory.mutability, Mutability::Not),
                                    "[invariant] provenance to mutable memory: {ty}"
                                );
                                let (_, pointee_val) = self.read_const_from_memory_and_layout(
                                    inner_memory,
                                    offset,
                                    actual_sub_ty,
                                    pointee_metadata,
                                );

                                // now construct the value
                                match ty.kind() {
                                    ty::Ref(_, _, Mutability::Not) => {
                                        SolValue::ImmRef(Box::new(pointee_val))
                                    }
                                    ty::RawPtr(_, Mutability::Not) => {
                                        SolValue::ImmPtr(Box::new(pointee_val))
                                    }
                                    ty::Ref(_, _, Mutability::Mut)
                                    | ty::RawPtr(_, Mutability::Mut) => {
                                        bug!("[invariant] mut ptr/ref to imm memory: {ty}")
                                    }
                                    _ => bug!("[invariant] unreachable type: {ty}"),
                                }
                            }
                            GlobalAlloc::Static(def_id) => {
                                let static_ty = match self.tcx.def_kind(def_id) {
                                    // named statics should be decoded with their declared type. `actual_sub_ty`
                                    // may be an unsized view reached through a fat pointer to the static.
                                    DefKind::Static { nested: false, .. } => self
                                        .tcx_try_normalize(
                                            self.tcx.type_of(def_id).instantiate_identity(),
                                        ),
                                    // nested statics are anonymous backing allocations; their HIR owner is
                                    // synthetic, so `type_of` is not available. Reconstruct the sized backing
                                    // array type from pointer metadata when the pointer view is unsized.
                                    DefKind::Static { nested: true, .. } => {
                                        match (actual_sub_ty.kind(), pointee_metadata) {
                                            (ty::Slice(elem_ty), Some(len)) => {
                                                Ty::new_array(self.tcx, *elem_ty, len as u64)
                                            }
                                            (ty::Str, Some(len)) => Ty::new_array(
                                                self.tcx,
                                                self.tcx.types.u8,
                                                len as u64,
                                            ),
                                            _ => actual_sub_ty,
                                        }
                                    }
                                    kind => {
                                        bug!("[invariant] expected static allocation, got {kind:?}")
                                    }
                                };
                                let static_ident = self.mk_static_init(def_id, static_ty);
                                let pointee_ty = self.mk_type(actual_sub_ty);

                                // now construct the value
                                match ty.kind() {
                                    ty::RawPtr(_, Mutability::Not) => SolValue::ImmPtrStatic(
                                        pointee_ty,
                                        static_ident,
                                        offset.bytes_usize(),
                                    ),
                                    ty::RawPtr(_, Mutability::Mut) => SolValue::MutPtrStatic(
                                        pointee_ty,
                                        static_ident,
                                        offset.bytes_usize(),
                                    ),
                                    ty::Ref(_, _, Mutability::Not) => SolValue::ImmRefStatic(
                                        pointee_ty,
                                        static_ident,
                                        offset.bytes_usize(),
                                    ),
                                    ty::Ref(_, _, Mutability::Mut) => SolValue::MutRefStatic(
                                        pointee_ty,
                                        static_ident,
                                        offset.bytes_usize(),
                                    ),
                                    _ => bug!("[invariant] unreachable type: {ty}"),
                                }
                            }

                            // unsupported
                            GlobalAlloc::TypeId { .. } => bug!("[unsupported] type id provenance"),
                            GlobalAlloc::Function { .. } => {
                                bug!("[unsupported] function provenance casted as pointer")
                            }

                            // unexpected
                            GlobalAlloc::VTable(..) => {
                                bug!("[invariant] unexpected provenance of type: {ty}")
                            }
                        }
                    }
                    Scalar::Int(int) => {
                        // NOTE: we allow non-zero pointer values as meaningful (and nullptr) constant values
                        let ptrval = int.to_target_usize(self.tcx) as usize;
                        let pointee_ty = self.mk_type(actual_sub_ty);

                        // now construct the value
                        match ty.kind() {
                            ty::RawPtr(_, Mutability::Not) => {
                                SolValue::ImmPtrNull(pointee_ty, ptrval)
                            }
                            ty::RawPtr(_, Mutability::Mut) => {
                                SolValue::MutPtrNull(pointee_ty, ptrval)
                            }
                            ty::Ref(_, _, Mutability::Not) => {
                                SolValue::ImmRefNull(pointee_ty, ptrval)
                            }
                            ty::Ref(_, _, Mutability::Mut) => {
                                SolValue::MutRefNull(pointee_ty, ptrval)
                            }
                            _ => bug!("[invariant] unreachable type: {ty}"),
                        }
                    }
                };

                // return the constructed constant value
                const_val
            }

            // function pointer
            ty::FnPtr(..) => {
                // sanity check on the variants
                match &layout.variants {
                    Variants::Single { index } if index.index() == 0 => {}
                    _ => bug!("[invariant] expect a 0-variant for fn-ptr type {ty}"),
                }

                // construct the type
                let fn_sig = match self.mk_type(ty) {
                    SolType::FnPtr(fn_sig) => fn_sig,
                    _ => bug!("[invariant] expect a function pointer type for {ty}"),
                };

                // read and parse the pointer
                let pointer_size = self.tcx.data_layout.pointer_size();
                if layout.size != pointer_size {
                    bug!("[invariant] unexpected layout size for fn-ptr {ty}");
                }
                match read_primitive(self.tcx, offset, pointer_size, true) {
                    Scalar::Ptr(ptr, _) => {
                        let (prov, offset) = ptr.into_raw_parts();
                        if offset != Size::ZERO {
                            bug!("[invariant] unexpected non-zero offset for function pointer");
                        }

                        match self.tcx.global_alloc(prov.alloc_id()) {
                            GlobalAlloc::Function { instance } => {
                                let ident = self.mk_ident(instance.def_id());
                                let ty_args = self.mk_generic_args(instance.args);
                                SolValue::FnPtr(fn_sig, ident, ty_args)
                            }
                            _ => bug!("[invariant] unexpected allocation for function pointer"),
                        }
                    }
                    Scalar::Int(int) => {
                        if !int.is_null() {
                            bug!("[invariant] unexpected non-null integer for function pointer");
                        }
                        SolValue::FnPtrNull(fn_sig)
                    }
                }
            }

            // vector-alike
            ty::Array(elem_ty, _) => {
                match &layout.variants {
                    Variants::Single { index } if index.index() == 0 => {}
                    _ => bug!("[invariant] expect a 0-variant for array type {ty}"),
                }
                match &layout.fields {
                    FieldsShape::Array { stride, count } => {
                        let mut elements = vec![];
                        for i in 0..*count {
                            let elem_offset = offset + *stride * i;
                            let (_, elem) = self.read_const_from_memory_and_layout(
                                memory,
                                elem_offset,
                                *elem_ty,
                                None,
                            );
                            elements.push(SolConst::Value(elem));
                        }
                        SolValue::Array(self.mk_type(*elem_ty), elements)
                    }
                    _ => bug!("[invariant] expect array field layout for array {ty}"),
                }
            }
            ty::Slice(elem_ty) => {
                match &layout.variants {
                    Variants::Single { index } if index.index() == 0 => {}
                    _ => bug!("[invariant] expect a 0-variant for slice type {ty}"),
                }
                match &layout.fields {
                    FieldsShape::Array { stride, count } => {
                        // unsized slice layouts use count 0, the fat-ptr carries the real length.
                        // if layout ever provides a concrete count too, they must agree.
                        let count = match metadata {
                            Some(len) => {
                                let len = len as u64;
                                if *count != 0 && *count != len {
                                    bug!("[invariant] conflicting slice lengths for {ty}");
                                }
                                len
                            }
                            None => *count,
                        };
                        let mut elements = vec![];
                        for i in 0..count {
                            let elem_offset = offset + *stride * i;
                            let (_, elem) = self.read_const_from_memory_and_layout(
                                memory,
                                elem_offset,
                                *elem_ty,
                                None,
                            );
                            elements.push(SolConst::Value(elem));
                        }
                        SolValue::Slice(self.mk_type(*elem_ty), elements)
                    }
                    _ => bug!("[invariant] expect an array field layout for slice {ty}"),
                }
            }
            ty::Str => {
                match &layout.variants {
                    Variants::Single { index } if index.index() == 0 => {}
                    _ => bug!("[invariant] expect a 0-variant for str type"),
                }
                match &layout.fields {
                    FieldsShape::Array { stride, count } => {
                        // unsized str layouts use count 0, the fat-ptr carries the real byte length.
                        // if layout ever provides a concrete count too, it must agree.
                        let count = match metadata {
                            Some(len) => {
                                let len = len as u64;
                                if *count != 0 && *count != len {
                                    bug!("[invariant] conflicting str lengths");
                                }
                                len
                            }
                            None => *count,
                        };
                        let range = AllocRange { start: offset, size: *stride * count };
                        let num_prov = memory.provenance().get_range(range, &self.tcx).count();
                        if num_prov != 0 {
                            bug!("[invariant] string memory contains provenance");
                        }

                        let bytes = memory
                            .get_bytes_strip_provenance(&self.tcx, range)
                            .unwrap_or_else(|e| {
                                bug!("[invariant] failed to read bytes for string memory: {e:?}");
                            });
                        SolValue::Str(String::from_utf8(bytes.to_vec()).unwrap_or_else(|_| {
                            bug!("[unsupported] non utf-8 string");
                        }))
                    }
                    _ => bug!("[invariant] expect an array field for str type"),
                }
            }

            // packed
            ty::Tuple(elem_tys) => {
                match &layout.variants {
                    Variants::Single { index } if index.index() == 0 => {}
                    _ => bug!("[invariant] expect a 0-variant for tuple type {ty}"),
                }
                match &layout.fields {
                    FieldsShape::Arbitrary { offsets, in_memory_order: _ } => {
                        if offsets.len() != elem_tys.len() {
                            bug!("[invariant] field count mismatch for tuple type {ty}");
                        }

                        let mut elements = vec![];
                        for (i, elem_ty) in elem_tys.iter().enumerate() {
                            let elem_idx = FieldIdx::from_usize(i);
                            let elem_offset = *offsets.get(elem_idx).unwrap_or_else(|| {
                                bug!("[invariant] no offset for field {i} in {ty}");
                            });
                            let elem_metadata = if elem_ty.is_sized(self.tcx, self.typing_env) {
                                None
                            } else {
                                let size = metadata.take();
                                assert!(
                                    size.is_some(),
                                    "[invariant] expect metadata for unsized tuple element {i} in {ty}"
                                );
                                size
                            };
                            let (_, elem) = self.read_const_from_memory_and_layout(
                                memory,
                                offset + elem_offset,
                                elem_ty,
                                elem_metadata,
                            );
                            elements.push(SolConst::Value(elem));
                        }
                        SolValue::Tuple(elements)
                    }
                    _ => bug!("[invariant] expect arbitrary fields for tuple {ty}"),
                }
            }

            ty::Adt(def, ty_args) => {
                let (adt_ident, adt_ty_args) = self.mk_adt(*def, ty_args);
                match def.adt_kind() {
                    AdtKind::Struct => {
                        let field_details = def
                            .non_enum_variant()
                            .fields
                            .iter_enumerated()
                            .map(|(field_idx, field_def)| {
                                (field_idx, field_def.name, self.tcx_field_ty(field_def, ty_args))
                            })
                            .collect::<Vec<_>>();

                        match &layout.variants {
                            Variants::Single { index } if index.index() == 0 => {}
                            _ => bug!("[invariant] expect a 0-variant for struct type {ty}"),
                        }
                        match &layout.fields {
                            FieldsShape::Arbitrary { offsets, in_memory_order: _ } => {
                                if offsets.len() != field_details.len() {
                                    bug!("[invariant] field count mismatch for struct {ty}");
                                }

                                let mut elements = vec![];
                                for (field_idx, field_name, field_ty) in field_details {
                                    let field_offset = *offsets.get(field_idx).unwrap_or_else(|| {
                                        bug!("[invariant] no offset for field {field_name} in struct {ty}");
                                    });

                                    let field_metadata = if field_ty
                                        .is_sized(self.tcx, self.typing_env)
                                    {
                                        None
                                    } else {
                                        let size = metadata.take();
                                        assert!(
                                            size.is_some(),
                                            "[invariant] expect metadata for unsized field {field_name} in {ty}"
                                        );
                                        size
                                    };
                                    let (_, elem) = self.read_const_from_memory_and_layout(
                                        memory,
                                        offset + field_offset,
                                        field_ty,
                                        field_metadata,
                                    );
                                    elements.push((
                                        SolFieldIndex(field_idx.index()),
                                        SolConst::Value(elem),
                                    ));
                                }
                                SolValue::Struct(adt_ident, adt_ty_args, elements)
                            }
                            _ => bug!("[invariant] expect arbitrary fields for struct {ty}"),
                        }
                    }
                    AdtKind::Union => {
                        match &layout.variants {
                            Variants::Single { index } if index.index() == 0 => {}
                            _ => bug!("[invariant] expect a 0-variant for union {ty}"),
                        }
                        if !matches!(layout.fields, FieldsShape::Union(..)) {
                            bug!("[invariant] expect union fields for {ty}");
                        }

                        // expect only two fields in the union: one that matches the layout size and another that does not
                        let field_details = def
                            .non_enum_variant()
                            .fields
                            .iter_enumerated()
                            .map(|(field_idx, field_def)| {
                                (field_idx, self.tcx_field_ty(field_def, ty_args))
                            })
                            .collect::<Vec<_>>();

                        if field_details.len() != 2 {
                            bug!(
                                "[invariant] expect two variants in union {ty} only, found {}",
                                field_details.len()
                            );
                        }
                        let (f1_idx, f1_ty) = field_details[0];
                        let (f2_idx, f2_ty) = field_details[1];

                        let f1_layout_size = self
                            .tcx
                            .layout_of(self.typing_env.as_query_input(f1_ty))
                            .unwrap_or_else(|_| {
                                bug!("[invariant] unable to query layout of type {f1_ty}")
                            })
                            .layout
                            .size;
                        let f2_layout_size = self
                            .tcx
                            .layout_of(self.typing_env.as_query_input(f2_ty))
                            .unwrap_or_else(|_| {
                                bug!("[invariant] unable to query layout of type {f2_ty}")
                            })
                            .layout
                            .size;

                        let (field_match_idx, field_match_ty, field_value_idx, field_value_ty) =
                            match (f1_layout_size == layout.size, f2_layout_size == layout.size) {
                                (true, true) => {
                                    bug!("[invariant] found two matches in union {ty}");
                                }
                                (false, false) => {
                                    bug!("[invariant] found no matches in union {ty}");
                                }
                                (true, false) => (f1_idx, f1_ty, f2_idx, f2_ty),
                                (false, true) => (f2_idx, f2_ty, f1_idx, f1_ty),
                            };

                        // check whether the entire range is initialized
                        let range = AllocRange { start: offset, size: layout.size };
                        let initialized = memory.init_mask().is_range_initialized(range).is_ok();

                        if initialized {
                            // parse as the matched variant
                            let (_, val) = self.read_const_from_memory_and_layout(
                                memory,
                                offset,
                                field_match_ty,
                                None,
                            );
                            SolValue::Union(
                                adt_ident,
                                adt_ty_args,
                                SolFieldIndex(field_match_idx.as_usize()),
                                Box::new(SolConst::Value(val)),
                            )
                        } else {
                            // parse as the alternative variant
                            let (_, val) = self.read_const_from_memory_and_layout(
                                memory,
                                offset,
                                field_value_ty,
                                None,
                            );
                            SolValue::Union(
                                adt_ident,
                                adt_ty_args,
                                SolFieldIndex(field_value_idx.as_usize()),
                                Box::new(SolConst::Value(val)),
                            )
                        }
                    }
                    AdtKind::Enum => {
                        let (variant_idx, field_offsets) = match layout.variants() {
                            Variants::Single { index } => {
                                // the enum is known to be this single variant, maybe the other variants are all infeasible

                                // get variant field offsets from layout
                                let field_offsets = match layout.fields() {
                                    FieldsShape::Arbitrary { offsets, in_memory_order: _ } => {
                                        offsets
                                    }
                                    _ => bug!(
                                        "[invariant] expect an arbitrary field for enum variant in {ty}"
                                    ),
                                };

                                (*index, field_offsets)
                            }
                            Variants::Multiple { tag, tag_encoding, tag_field, variants } => {
                                // get the tag type
                                let tag_type = match tag {
                                    ScalarAbi::Initialized { value, valid_range: _ } => value,
                                    ScalarAbi::Union { .. } => {
                                        bug!("[invariant] unexpected tag spec for {ty}");
                                    }
                                };

                                // get the offset for the tag field
                                let tag_offset = match &layout.fields {
                                    FieldsShape::Arbitrary { offsets, in_memory_order: _ } => {
                                        *offsets.get(*tag_field).unwrap_or_else(|| {
                                            bug!("[invariant] no offset for tag field for {ty}");
                                        })
                                    }
                                    _ => bug!(
                                        "[invariant] tagged variants do not have field spec in {ty}"
                                    ),
                                };

                                // parse the tag value
                                let tag_size = tag_type.size(&self.tcx);
                                let tag_scalar = read_primitive(
                                    self.tcx,
                                    offset + tag_offset,
                                    tag_size,
                                    matches!(tag_type, Primitive::Pointer(_)),
                                );

                                // derive variant index
                                let variant_idx = match tag_encoding {
                                    TagEncoding::Direct => {
                                        // direct tag encoding, the scalar value is the discriminant
                                        let tag_value = tag_scalar.assert_scalar_int();
                                        get_variant_by_discriminant(self.tcx, *def, tag_value)
                                    }
                                    TagEncoding::Niche {
                                        untagged_variant,
                                        niche_variants,
                                        niche_start,
                                    } => {
                                        // recover the variant index from the niche tag
                                        match tag_scalar {
                                            Scalar::Int(int) => {
                                                // NOTE: the discriminant and variant index of each variant coincide
                                                let tag_index = int
                                                    .to_bits_unchecked()
                                                    .wrapping_sub(*niche_start)
                                                    .wrapping_add(
                                                        niche_variants.start.index() as u128
                                                    );

                                                if tag_index >= def.variants().len() as u128 {
                                                    *untagged_variant
                                                } else {
                                                    VariantIdx::from_usize(tag_index as usize)
                                                }
                                            }
                                            Scalar::Ptr(..) => {
                                                // if we get a pointer as a tag, it must be on the untagged variant
                                                *untagged_variant
                                            }
                                        }
                                    }
                                };

                                // get variant layout
                                let variant_layout =
                                    variants.get(variant_idx).unwrap_or_else(|| {
                                        bug!(
                                            "[invariant] invalid variant index {} for {ty}",
                                            variant_idx.index()
                                        )
                                    });

                                // return both the index and the layout
                                (variant_idx, &variant_layout.field_offsets)
                            }
                            Variants::Empty => {
                                bug!("[invariant] unexpected empty variants for {ty}")
                            }
                        };

                        // get the variant definition from typing
                        let variant_def = def.variant(variant_idx);

                        // sanity check on the field offsets derivation
                        if field_offsets.len() != variant_def.fields.len() {
                            bug!(
                                "[invariant] field count mismatch for variant {} in {ty}",
                                variant_def.name
                            );
                        }

                        // construct values for the variant fields
                        let mut elements = vec![];
                        for (field_idx, field_def) in variant_def.fields.iter_enumerated() {
                            let field_offset = *field_offsets.get(field_idx).unwrap_or_else(|| {
                                bug!(
                                    "[invariant] no offset for field {} in {ty} variant {}",
                                    field_def.name,
                                    variant_def.name
                                );
                            });
                            let (_, elem) = self.read_const_from_memory_and_layout(
                                memory,
                                offset + field_offset,
                                self.tcx_field_ty(field_def, ty_args),
                                None,
                            );
                            elements
                                .push((SolFieldIndex(field_idx.index()), SolConst::Value(elem)));
                        }
                        SolValue::Enum(
                            adt_ident,
                            adt_ty_args,
                            SolVariantIndex(variant_idx.index()),
                            elements,
                        )
                    }
                }
            }

            // closure
            ty::Closure(def_id, ty_args) => {
                SolValue::Closure(self.mk_ident(*def_id), self.mk_generic_args(ty_args))
            }

            // unexpected
            _ => bug!(
                "[invariant] unhandled type for reading constant from memory: {ty}\n[{}]",
                memory
                    .get_bytes_unchecked(AllocRange { start: offset, size: memory.size() - offset })
                    .iter()
                    .map(|b| format!("{b}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
        };

        // return the size consumed and the value
        (layout.size, value)
    }

    /// Record a executable, i.e., a THIR body (for a function or a constant/static)
    pub(crate) fn mk_exec(&mut self, thir: &Thir<'tcx>, expr: ExprId) -> SolExec {
        // switch-case on the body type
        match thir.body_type {
            BodyTy::Fn(sig) => {
                // ignore unsupported case
                if sig.c_variadic() {
                    bug!("[unsupported] variadic function");
                }

                // parse function signature
                let parsed_abi = self.mk_abi(sig.abi(), sig.c_variadic(), sig.safety());
                let ret_ty = sig.output();
                let params = sig.inputs();

                // parse parameters
                let mut visibility = None;
                let mut param_iter = thir.params.iter_enumerated();
                if thir.params.len() == params.len() + 1 {
                    // make sure this is a closure
                    assert_eq!(
                        self.tcx.def_kind(self.owner_id.to_def_id()),
                        DefKind::Closure,
                        "[invariant] expect closure"
                    );

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

                    // closure should have only one possible abi
                    assert_eq!(
                        parsed_abi,
                        SolExternAbi::Rust { safety: true },
                        "[invariant] closure abi mismatch",
                    );

                    // get the upvar captures and types
                    let closure_local_id = closure_id.as_local().unwrap_or_else(|| {
                        bug!("[invariant] closure definition should be local: {closure_id:?}")
                    });
                    let upvar_captures = self.tcx.closure_captures(closure_local_id);

                    // NOTE: `upvar_tys` are the types of the captured *places* (with disjoint closure
                    // captures, possibly field/by-ref views of a root variable), so they are at a
                    // different granularity than the root-keyed `SolClosure::upvars` and are used here
                    // only as a count cross-check.
                    let upvar_tys = UpvarArgs::Closure(closure_ty_args).upvar_tys();
                    if upvar_captures.len() != upvar_tys.len() {
                        bug!(
                            "[invariant] closure upvar captures/types count mismatch: {} vs {}",
                            upvar_captures.len(),
                            upvar_tys.len(),
                        );
                    }

                    let upvar_mention = self.tcx.upvars_mentioned(closure_id);

                    // collect the root variables that may appear as `ExprKind::UpvarRef` in the body.
                    //
                    // NOTE: We should NOT seed the upvar type from the capture metadata: a capture's
                    // `base_ty` is normalized independently of THIR, so for an HRTB GAT projection such as
                    // `<() as Container>::Item<'a>` it is resolved to the concrete type (`()`) while the
                    // THIR expression type keeps it symbolic. Seeding it would disagree with the type
                    // recorded from `UpvarRef` (tripping the equality check in `mk_expr`) and with how the
                    // same type is represented elsewhere in the IR. The type is therefore taken from the
                    // body's `UpvarRef`, keeping a single, consistent source of truth.
                    //
                    // NOTE: there may be multiple captured places for the same root variable, but
                    // `SolClosure::upvars` is keyed by the root HIR id that `ExprKind::UpvarRef` uses;
                    // duplicate roots collapse to one declaration, and equal keys carry equal `hir_id`s.
                    let mut declared_roots = BTreeMap::new();
                    for captured_place in upvar_captures.iter() {
                        let hir_id = match captured_place.place.base {
                            HirPlaceBase::Upvar(upvar_id) => upvar_id.var_path.hir_id,
                            ref base => {
                                bug!("[invariant] expected captured upvar, found {base:?}");
                            }
                        };

                        // consistency check
                        if let Some(upvars) = upvar_mention {
                            if !upvars.contains_key(&hir_id) {
                                bug!(
                                    "[invariant] captured upvar {} is not mentioned",
                                    self.tcx.hir_name(hir_id)
                                );
                            }
                        }

                        let var_id = self.mk_local_var_id(LocalVarId(hir_id));
                        declared_roots.insert(var_id, hir_id);

                        // NOTE: we intentionally do not check for duplicated root variables, as
                        // the same root variable can be captured multiple times (e.g., by nested
                        // closures or by capturing multiple fields of the same struct), and they
                        // will all share the same declaration in `SolClosure::upvars`.
                        //
                        // In addition, if two keys are the same, their values, `hir_id`, must
                        // also be the same, hence we don't check for value-equality.
                    }

                    // THIR classifies `ExprKind::UpvarRef` by lexical enclosing-body membership, not by
                    // the disjoint-capture set, so a mentioned variable can be referenced in the body
                    // without appearing in `closure_captures` (e.g. via infallible non-binding patterns).
                    // Declare those roots too so body refs find a declaration. A declared root that is
                    // never referenced gets no type and is dropped when packing the closure (see below).
                    if let Some(upvars) = upvar_mention {
                        for &hir_id in upvars.keys() {
                            let var_id = self.mk_local_var_id(LocalVarId(hir_id));
                            declared_roots.insert(var_id, hir_id);
                            // see above note for reasons not to check for duplicates here
                        }
                    }

                    // add to upvar declarations, the type is filled later from the body's `UpvarRef`
                    for (var_id, hir_id) in declared_roots {
                        let var_name = SolLocalVarName(self.tcx.hir_name(hir_id).to_ident_string());
                        if let Some((existing, _)) =
                            self.upvar_decls.insert(var_id, (var_name, None))
                        {
                            bug!("[invariant] duplicate upvar declaration for {}", existing.0);
                        }
                    }
                } else {
                    // make sure this is a function or an associated function
                    assert!(
                        matches!(
                            self.tcx.def_kind(self.owner_id.to_def_id()),
                            DefKind::Fn | DefKind::AssocFn
                        ),
                        "[invariant] expect function or associated function"
                    );

                    // check that argument count matches
                    assert_eq!(
                        thir.params.len(),
                        params.len(),
                        "[invariant] parameter count mismatch",
                    );

                    // get the visibility of the function
                    let vis = match self.tcx.visibility(self.owner_id.to_def_id()) {
                        Visibility::Public => SolVisibility::Public,
                        Visibility::Restricted(id) => SolVisibility::Restricted(self.mk_ident(id)),
                    };
                    visibility = Some(vis);
                }

                // declared parameters should match type declarations
                let mut param_decls = vec![];
                for (
                    (idx, param_ty),
                    (param_id, Param { pat, ty, ty_span: _, self_kind: _, hir_id: _ }),
                ) in params.iter().enumerate().zip(param_iter)
                {
                    assert_eq!(
                        idx + if visibility.is_none() { 1 } else { 0 },
                        param_id.index(),
                        "[invariant] parameter id not sequential"
                    );
                    assert_eq!(param_ty, ty, "[invariant] parameter type mismatch");

                    let decl_ty = self.mk_type(*ty);
                    let decl_pat = pat.as_ref().map(|p| self.mk_pat(p));
                    param_decls.push((decl_ty, decl_pat));
                }

                // record return type
                let ret_ty_decl = self.mk_type(ret_ty);

                // parse the body expression
                let body = self.mk_expr(thir, expr);

                // pack the information
                match visibility {
                    None => {
                        // unpack the upvars for closures
                        let upvars = self
                            .upvar_decls
                            .iter()
                            .map(|(var_id, (name, ty_opt))| {
                                let var_ty = ty_opt.clone().unwrap_or_else(|| {
                                    bug!("[invariant] upvar type not recorded for {}", name.0);
                                });
                                (var_id.clone(), var_ty, name.clone())
                            })
                            .collect();
                        SolExec::Closure(SolClosure {
                            ret_ty: ret_ty_decl,
                            params: param_decls,
                            upvars,
                            body,
                        })
                    }
                    Some(vis) => {
                        // function should not have upvars
                        assert!(
                            self.upvar_decls.is_empty(),
                            "[invariant] function should not have upvars"
                        );
                        SolExec::Function(SolFnDef {
                            abi: parsed_abi,
                            vis,
                            ret_ty: ret_ty_decl,
                            params: param_decls,
                            body,
                        })
                    }
                }
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
                SolExec::ConstEval(SolCEval { ty: const_ty, body })
            }
            BodyTy::GlobalAsm(..) => bug!("[unsupported] global assembly"),
        }
    }

    /// Record a THIR expression
    pub(crate) fn mk_expr(&mut self, thir: &Thir<'tcx>, expr_id: ExprId) -> SolExpr {
        let Expr { kind, ty, temp_scope_id, span } = &thir.exprs[expr_id];
        self.log_stack.push("Expr", format!("{kind:?}"));

        // record the id
        let id = SolInstIndex(expr_id.index());
        assert!(self.inst_ids.insert(id.clone()), "[invariant] duplicate instruction id: {}", id.0);

        // record type, span, and scope
        let expr_ty = self.mk_type(*ty);
        let expr_span = self.mk_span(*span);
        let expr_scope = self.mk_scope(Scope { local_id: *temp_scope_id, data: ScopeData::Node });

        // switch-case on expression kind
        let expr_op = match kind {
            // markers
            ExprKind::Scope { region_scope, hir_id, value } => {
                let inner_expr = self.mk_expr(thir, *value);
                SolOp::Scope(self.mk_hir(*hir_id, inner_expr), self.mk_scope(*region_scope))
            }
            ExprKind::PlaceTypeAscription { source, user_ty: _, user_ty_span: _ }
            | ExprKind::ValueTypeAscription { source, user_ty: _, user_ty_span: _ } => {
                // MAYFIX: maybe record user type annotation as well?
                SolOp::TypeAscribe(self.mk_expr(thir, *source))
            }

            // use
            ExprKind::Use { source } => SolOp::Use(self.mk_expr(thir, *source)),
            ExprKind::VarRef { id } => {
                let var_id = self.mk_local_var_id(*id);
                assert!(
                    self.var_scope.contains_key(&var_id),
                    "[invariant] reference to undeclared local variable {id:?}"
                );
                SolOp::LocalVar(var_id)
            }
            ExprKind::UpvarRef { closure_def_id, var_hir_id } => {
                assert_eq!(
                    closure_def_id,
                    &self.owner_id.to_def_id(),
                    "[invariant] closure def_id mismatch"
                );
                assert_ne!(
                    var_hir_id.0.owner, self.owner_id,
                    "[invariant] upvar cannot belong to local closure"
                );

                // ensure that the current variable scope does not contain the upvar
                let var_id = self.mk_local_var_id(*var_hir_id);
                assert!(
                    !self.var_scope.contains_key(&var_id),
                    "[invariant] upvar refers to a declared local variable {var_hir_id:?}"
                );

                // update the var declaration type if not yet recorded
                let (_, upvar_ty) = self
                    .upvar_decls
                    .get_mut(&var_id)
                    .unwrap_or_else(|| bug!("[invariant] upvar not declared"));
                if let Some(existing_ty) = upvar_ty {
                    assert_eq!(
                        *existing_ty, expr_ty,
                        "[invariant] upvar type mismatch for {var_hir_id:?}"
                    );
                } else {
                    *upvar_ty = Some(expr_ty.clone());
                }

                SolOp::UpVar(var_id)
            }

            // constant
            ExprKind::ConstParam { param: ParamConst { index, name }, def_id } => {
                let param_ident = self.get_param(*index, *name, SolGenericKind::Const);
                let const_ident = self.mk_ident(*def_id);
                assert_eq!(param_ident, const_ident, "[invariant] const param ident mismatch",);
                SolOp::ConstParam(param_ident)
            }
            ExprKind::NamedConst { def_id, args, user_ty: _ } => {
                // MAYFIX: maybe record user type annotation as well?
                SolOp::NamedConst(self.mk_ident(*def_id), self.mk_generic_args(args))
            }
            ExprKind::ConstBlock { did, args } => {
                SolOp::ConstBlock(self.mk_ident(*did), self.mk_generic_args(args))
            }
            ExprKind::StaticRef { alloc_id, ty: ref_ty, def_id } => {
                // sanity check
                match self.tcx.global_alloc(*alloc_id) {
                    GlobalAlloc::Static(static_def_id) if static_def_id == *def_id => {}
                    _ => bug!("[invariant] invalid static ref in THIR"),
                }

                // ensure type matches
                assert_eq!(ty, ref_ty, "[invariant] static ref type mismatch");
                let mem_ty = match ref_ty.kind() {
                    ty::Ref(_, inner_ty, _) | ty::RawPtr(inner_ty, _) => inner_ty,
                    _ => bug!("[invariant] static ref must be a ref or a ptr type"),
                };
                let static_ident = self.mk_static_init(*def_id, *mem_ty);
                SolOp::StaticRef(static_ident)
            }

            // intrinsics
            ExprKind::Deref { arg } => SolOp::Deref(self.mk_expr(thir, *arg)),

            // operators
            ExprKind::Binary { op, lhs, rhs } => {
                let lhs_expr = self.mk_expr(thir, *lhs);
                let rhs_expr = self.mk_expr(thir, *rhs);
                match op {
                    BinOp::Add => SolOp::Add(lhs_expr, rhs_expr),
                    BinOp::Sub => SolOp::Sub(lhs_expr, rhs_expr),
                    BinOp::Mul => SolOp::Mul(lhs_expr, rhs_expr),
                    BinOp::Div => SolOp::Div(lhs_expr, rhs_expr),
                    BinOp::Rem => SolOp::Rem(lhs_expr, rhs_expr),
                    BinOp::BitXor => SolOp::BitXor(lhs_expr, rhs_expr),
                    BinOp::BitAnd => SolOp::BitAnd(lhs_expr, rhs_expr),
                    BinOp::BitOr => SolOp::BitOr(lhs_expr, rhs_expr),
                    BinOp::Shl => SolOp::Shl(lhs_expr, rhs_expr),
                    BinOp::Shr => SolOp::Shr(lhs_expr, rhs_expr),
                    BinOp::Eq => SolOp::Eq(lhs_expr, rhs_expr),
                    BinOp::Ne => SolOp::Ne(lhs_expr, rhs_expr),
                    BinOp::Lt => SolOp::Lt(lhs_expr, rhs_expr),
                    BinOp::Le => SolOp::Le(lhs_expr, rhs_expr),
                    BinOp::Gt => SolOp::Gt(lhs_expr, rhs_expr),
                    BinOp::Ge => SolOp::Ge(lhs_expr, rhs_expr),
                    BinOp::Cmp => SolOp::Cmp(lhs_expr, rhs_expr),
                    // unexpected
                    BinOp::AddUnchecked
                    | BinOp::AddWithOverflow
                    | BinOp::SubUnchecked
                    | BinOp::SubWithOverflow
                    | BinOp::MulUnchecked
                    | BinOp::MulWithOverflow
                    | BinOp::ShlUnchecked
                    | BinOp::ShrUnchecked => {
                        bug!("[invariant] unexpected overflow-related operation")
                    }
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
            ExprKind::Adt(AdtExpr {
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
                    AdtExprBase::Base(FruInfo { base, field_types }) => SolAdtBase::Overlay(
                        self.mk_expr(thir, *base),
                        field_types.iter().map(|field_ty| self.mk_type(*field_ty)).collect(),
                    ),
                    AdtExprBase::DefaultFields(default_field_tys) => SolAdtBase::Default(
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
            ExprKind::If { if_then_scope, cond, then, else_opt } => SolOp::If {
                cond: self.mk_expr(thir, *cond),
                then: self.mk_expr(thir, *then),
                else_opt: else_opt.map(|e| self.mk_expr(thir, e)),
                scope: self.mk_scope(*if_then_scope),
            },
            ExprKind::Loop { body } => SolOp::Loop(self.mk_expr(thir, *body)),
            ExprKind::Break { label, value } => {
                SolOp::Break(self.mk_scope(*label), value.map(|e| self.mk_expr(thir, e)))
            }
            ExprKind::Continue { label } => SolOp::Continue(self.mk_scope(*label)),
            ExprKind::Return { value } => SolOp::Return(value.map(|e| self.mk_expr(thir, e))),

            // borrow
            ExprKind::Borrow { borrow_kind, arg } => {
                let borrowed = self.mk_expr(thir, *arg);
                match borrow_kind {
                    BorrowKind::Shared => SolOp::ImmBorrow(borrowed),
                    BorrowKind::Mut { kind: _ } => SolOp::MutBorrow(borrowed),
                    BorrowKind::Fake(..) => bug!("[invariant] unexpected fake borrow in THIR"),
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
                    let Arm { pattern, guard, body, hir_id, scope, span } = &thir.arms[*arm];
                    let arm_scope = self.mk_scope(*scope);
                    let arm_pattern = self.mk_pat(pattern);
                    let arm_guard = guard.map(|e| self.mk_expr(thir, e));
                    let arm_body = self.mk_expr(thir, *body);
                    let arm_span = self.mk_span(*span);
                    let arm_hir = self.mk_hir(
                        *hir_id,
                        SolMatchArm {
                            scope: arm_scope,
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
                SolOp::ConstValue(self.mk_value_from_scalar(*ty, *lit))
            }
            ExprKind::ZstLiteral { user_ty: _ } => {
                // MAYFIX: record user type annotation as well?
                SolOp::ConstValue(self.mk_value_from_zst(*ty))
            }

            // closure
            ExprKind::Closure(ClosureExpr {
                closure_id,
                args,
                upvars,
                movability: _,
                fake_reads: _,
            }) => {
                let closure_ident = self.mk_ident(closure_id.to_def_id());
                let closure_ty_args = match args {
                    UpvarArgs::Closure(ty_args) => self.mk_generic_args(ty_args),
                    UpvarArgs::Coroutine(..) | UpvarArgs::CoroutineClosure(..) => {
                        bug!("[unsupported] coroutine closure");
                    }
                };
                let closure_upvars = upvars.iter().map(|v| self.mk_expr(thir, *v)).collect();
                // MAYFIX: maybe we should do something about the movability and fake reads as well?
                SolOp::Closure(closure_ident, closure_ty_args, closure_upvars)
            }

            // unsupported
            ExprKind::Reborrow { .. } => bug!("[unsupported] reborrow"),
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
        SolExpr { id, ty: expr_ty, op: Box::new(expr_op), span: expr_span, scope: expr_scope }
    }

    /// Record a THIR statement
    pub(crate) fn mk_stmt(&mut self, thir: &Thir<'tcx>, stmt_id: StmtId) -> SolStmt {
        let Stmt { kind } = &thir.stmts[stmt_id];
        self.log_stack.push("Stmt", format!("{kind:?}"));

        // record the id
        let id = SolInstIndex(stmt_id.index() + thir.exprs.len());
        assert!(self.inst_ids.insert(id.clone()), "[invariant] duplicate instruction id: {}", id.0);

        let stmt = match kind {
            StmtKind::Expr { scope, expr } => {
                SolStmt::Expr(id, self.mk_scope(*scope), self.mk_expr(thir, *expr))
            }
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
                SolStmt::Bind(
                    id,
                    self.mk_hir(
                        *hir_id,
                        SolLetBinding { pat: let_pattern, init: let_init, span: let_span },
                    ),
                )
            }
        };

        // pop the log stack
        self.log_stack.pop();

        // return the statement
        stmt
    }

    /// Record a THIR block
    pub(crate) fn mk_block(&mut self, thir: &Thir<'tcx>, block_id: BlockId) -> SolBlock {
        let Block { targeted_by_break: _, region_scope, span, stmts, expr, safety_mode } =
            &thir.blocks[block_id];

        let block_scope = self.mk_scope(*region_scope);
        let block_safety = match safety_mode {
            BlockSafety::Safe => true,
            BlockSafety::BuiltinUnsafe | BlockSafety::ExplicitUnsafe(..) => false,
        };
        let block_span = self.mk_span(*span);
        let block_stmts = stmts.iter().map(|stmt| self.mk_stmt(thir, *stmt)).collect();
        let block_expr = expr.map(|e| self.mk_expr(thir, e));

        SolBlock {
            scope: block_scope,
            safety: block_safety,
            stmts: block_stmts,
            expr: block_expr,
            span: block_span,
        }
    }

    /// Record a scope
    pub(crate) fn mk_scope(&mut self, scope: Scope) -> SolScope {
        SolScope { index: SolScopeIndex(scope.local_id.index()) }
    }

    /// Derive the type of a value
    pub(crate) fn type_of_value(&self, val: &SolValue) -> SolType {
        match val {
            SolValue::Bool(_) => SolType::Bool,
            SolValue::Char(_) => SolType::Char,
            SolValue::I8(_) => SolType::I8,
            SolValue::I16(_) => SolType::I16,
            SolValue::I32(_) => SolType::I32,
            SolValue::I64(_) => SolType::I64,
            SolValue::I128(_) => SolType::I128,
            SolValue::Isize(_) => SolType::Isize,
            SolValue::U8(_) => SolType::U8,
            SolValue::U16(_) => SolType::U16,
            SolValue::U32(_) => SolType::U32,
            SolValue::U64(_) => SolType::U64,
            SolValue::U128(_) => SolType::U128,
            SolValue::Usize(_) => SolType::Usize,
            SolValue::F16(_) => SolType::F16,
            SolValue::F32(_) => SolType::F32,
            SolValue::F64(_) => SolType::F64,
            SolValue::F128(_) => SolType::F128,
            SolValue::Str(_) => SolType::Str,

            SolValue::Tuple(elems) => {
                SolType::Tuple(elems.iter().map(|elem| self.type_of_const(elem)).collect())
            }
            SolValue::Slice(elem_ty, _) => SolType::Slice(Box::new(elem_ty.clone())),
            SolValue::Array(elem_ty, elems) => SolType::Array(
                Box::new(elem_ty.clone()),
                Box::new(SolConst::Value(SolValue::Usize(elems.len()))),
            ),
            SolValue::Struct(adt_ident, adt_args, ..) => {
                SolType::Struct(adt_ident.clone(), adt_args.clone())
            }
            SolValue::Union(adt_ident, adt_args, ..) => {
                SolType::Union(adt_ident.clone(), adt_args.clone())
            }
            SolValue::Enum(adt_ident, adt_args, ..) => {
                SolType::Enum(adt_ident.clone(), adt_args.clone())
            }

            SolValue::ImmRef(inner_val) => SolType::ImmRef(Box::new(self.type_of_value(inner_val))),
            SolValue::MutRef(inner_val) => SolType::MutRef(Box::new(self.type_of_value(inner_val))),
            SolValue::ImmRefStatic(inner_ty, _, _) => SolType::ImmRef(Box::new(inner_ty.clone())),
            SolValue::MutRefStatic(inner_ty, _, _) => SolType::MutRef(Box::new(inner_ty.clone())),
            SolValue::ImmRefNull(inner_ty, _) => SolType::ImmRef(Box::new(inner_ty.clone())),
            SolValue::MutRefNull(inner_ty, _) => SolType::MutRef(Box::new(inner_ty.clone())),

            SolValue::ImmPtr(inner_val) => SolType::ImmPtr(Box::new(self.type_of_value(inner_val))),
            SolValue::MutPtr(inner_val) => SolType::MutPtr(Box::new(self.type_of_value(inner_val))),
            SolValue::ImmPtrStatic(inner_ty, _, _) => SolType::ImmPtr(Box::new(inner_ty.clone())),
            SolValue::MutPtrStatic(inner_ty, _, _) => SolType::MutPtr(Box::new(inner_ty.clone())),
            SolValue::ImmPtrNull(inner_ty, _) => SolType::ImmPtr(Box::new(inner_ty.clone())),
            SolValue::MutPtrNull(inner_ty, _) => SolType::MutPtr(Box::new(inner_ty.clone())),

            SolValue::FuncDef(ident, ty_args) => SolType::FuncDef(ident.clone(), ty_args.clone()),
            SolValue::FuncSym(ident, ty_args) => SolType::FuncSym(ident.clone(), ty_args.clone()),
            SolValue::AdtCtor(ident, ty_args) => SolType::AdtCtor(ident.clone(), ty_args.clone()),
            SolValue::Virtual(ident, ty_args) => SolType::Virtual(ident.clone(), ty_args.clone()),
            SolValue::Libfunc(ident, ty_args) => SolType::Libfunc(ident.clone(), ty_args.clone()),
            SolValue::External(ident, ty_args) => SolType::External(ident.clone(), ty_args.clone()),
            SolValue::Intrinsic(ident, ty_args) => {
                SolType::Intrinsic(ident.clone(), ty_args.clone())
            }
            SolValue::Closure(ident, ty_args) => SolType::Closure(ident.clone(), ty_args.clone()),
            SolValue::FnPtr(fn_sig, _, _) => SolType::FnPtr(fn_sig.clone()),
            SolValue::FnPtrNull(fn_sig) => SolType::FnPtr(fn_sig.clone()),
        }
    }

    pub(crate) fn type_of_const(&self, constant: &SolConst) -> SolType {
        match constant {
            SolConst::Value(val) => self.type_of_value(val),
            SolConst::Param(ident) => self
                .generics
                .iter()
                .find(|param| ident == &param.ident)
                .map(|param| match param.kind {
                    SolGenericKind::Type => SolType::Param(ident.clone()),
                    SolGenericKind::Const => self
                        .cp_types
                        .get(ident)
                        .unwrap_or_else(|| {
                            bug!("[invariant] const generic used but type not found")
                        })
                        .clone(),
                    SolGenericKind::Lifetime => bug!("[invariant] lifetime generic used as const"),
                })
                .unwrap_or_else(|| bug!("[invariant] generic parameter not found")),
            SolConst::Unevaluated(ty, _, _) => ty.clone(),
        }
    }

    /// Try to parse the origin of a function
    fn try_parse_function_origin(&self, def_id: DefId) -> Option<ResolvedStdlib> {
        let crate_name = self.tcx.crate_name(def_id.krate);
        let stdlib = match crate_name.as_str() {
            "core" => ResolvedStdlib::Core,
            "alloc" => ResolvedStdlib::Alloc,
            "std" => ResolvedStdlib::Std,
            "proc_macro" => ResolvedStdlib::ProcMacro,
            "test" => ResolvedStdlib::Test,
            _ => return None,
        };
        Some(stdlib)
    }

    fn try_resolve_function_favor_stdlib(
        &mut self,
        def_id: DefId,
        ty_args: GenericArgsRef<'tcx>,
        default: impl Fn(SolIdent, Vec<SolGenericArg>) -> ResolvedFunction,
    ) -> ResolvedFunction {
        let ident = self.mk_ident(def_id);
        let generics = self.mk_generic_args(ty_args);
        match self.try_parse_function_origin(def_id) {
            None => default(ident, generics),
            Some(_) => ResolvedFunction::Libfunc(ident, generics),
        }
    }

    /// Try to resolve a function instance from a type
    fn try_resolve_instance(&mut self, ty: Ty<'tcx>) -> ResolvedFunction {
        // try to resolve the instance
        let instance = match ty.kind() {
            ty::FnDef(def_id, ty_args) => {
                // instance resolution normalizes associated-item arguments internally using
                // infallible normalization, and yet, THIR can contain valid projections that
                // are not normalizable in the current body, so keep those calls symbolic.
                let ty_args = match self
                    .tcx
                    .try_normalize_erasing_regions(self.typing_env, Unnormalized::new_wip(*ty_args))
                {
                    Ok(ty_args) => ty_args,
                    Err(_) => {
                        let ty_args = self.tcx.erase_and_anonymize_regions(*ty_args);
                        return self.try_resolve_function_favor_stdlib(
                            *def_id,
                            ty_args,
                            ResolvedFunction::FuncSym,
                        );
                    }
                };

                // NOTE: this `try_resolve` has to be after the normalization attempt above
                match Instance::try_resolve(self.tcx, self.typing_env, *def_id, ty_args) {
                    Ok(None) => {
                        return self.try_resolve_function_favor_stdlib(
                            *def_id,
                            ty_args,
                            ResolvedFunction::FuncSym,
                        );
                    }
                    Ok(Some(instance)) => instance,
                    Err(_) => bug!("[invariant] unable to resolve function instance"),
                }
            }
            _ => bug!("[invariant] expected a function definition type after normalization"),
        };

        // handle constructors early
        let def_id = instance.def_id();
        if self.tcx.is_constructor(def_id) {
            let adt_ty = self
                .tcx_try_normalize(self.tcx.fn_sig(def_id).instantiate(self.tcx, instance.args))
                .no_bound_vars()
                .unwrap_or_else(|| bug!("[invariant] unable to instantiate constructor signature"))
                .output();
            match self.mk_type(adt_ty) {
                SolType::Struct(adt_ident, adt_ty_args) | SolType::Enum(adt_ident, adt_ty_args) => {
                    return ResolvedFunction::AdtCtor(adt_ident, adt_ty_args);
                }
                _ => bug!("[invariant] constructor does not return a tuple-like type"),
            }
        }

        // aggresively mark libcalls before further looking at the instance
        let tcx = self.tcx;
        self.try_resolve_function_favor_stdlib(def_id, instance.args, |ident, generics| {
            match instance.def {
                InstanceKind::Item(..) => {
                    if tcx.is_foreign_item(def_id) {
                        ResolvedFunction::External(ident, generics)
                    } else {
                        ResolvedFunction::FuncDef(ident, generics)
                    }
                }
                InstanceKind::Virtual(..) => ResolvedFunction::Virtual(ident, generics),
                InstanceKind::Intrinsic(..)
                | InstanceKind::FnPtrShim { .. }
                | InstanceKind::DropGlue(..)
                | InstanceKind::CloneShim { .. } => ResolvedFunction::Intrinsic(ident, generics),
                _ => {
                    bug!("[invariant] unexpected instance kind for function resolution: {instance}")
                }
            }
        })
    }

    /// Check whether a type has any feasible value
    fn has_feasible_value(&self, ty: Ty<'tcx>) -> bool {
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
            ty::Pat(base_ty, _) => self.has_feasible_value(*base_ty),
            ty::Tuple(elems) => elems.iter().all(|e| self.has_feasible_value(e)),
            ty::Adt(adt_def, generics) => match adt_def.adt_kind() {
                AdtKind::Struct => adt_def
                    .non_enum_variant()
                    .fields
                    .iter()
                    .all(|f| self.has_feasible_value(self.tcx_field_ty(f, generics))),
                AdtKind::Union => adt_def
                    .non_enum_variant()
                    .fields
                    .iter()
                    .any(|f| self.has_feasible_value(self.tcx_field_ty(f, generics))),
                AdtKind::Enum => adt_def.variants().iter().any(|variant| {
                    variant
                        .fields
                        .iter()
                        .all(|f| self.has_feasible_value(self.tcx_field_ty(f, generics)))
                }),
            },
            ty::Array(sub, len) => {
                // a zero-length array is always inhabited regardless of the element type.
                // only require a feasible element when the length is non-zero or symbolic.
                match len.try_to_target_usize(self.tcx) {
                    Some(0) => true,
                    None | Some(_) => self.has_feasible_value(*sub),
                }
            }
            ty::Slice(sub) => self.has_feasible_value(*sub),

            // unsupported
            ty::Dynamic(..) => bug!("[unsupported] feasibility query for dynamic type"),

            // should not appear
            _ => bug!("[invariant] unexpected type in feasibility query: {ty}"),
        }
    }

    /// Get a uniquely feasible field for a union
    fn get_uniquely_feasible_field(
        &self,
        def: AdtDef<'tcx>,
        generics: GenericArgsRef<'tcx>,
    ) -> Option<FieldIdx> {
        let mut feasible_field = None;
        for (field_idx, field_def) in def.non_enum_variant().fields.iter_enumerated() {
            if self.has_feasible_value(self.tcx_field_ty(field_def, generics)) {
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
    fn get_uniquely_feasible_variant(
        &self,
        def: AdtDef<'tcx>,
        generics: GenericArgsRef<'tcx>,
    ) -> Option<VariantIdx> {
        let mut feasible_variant = None;
        for (variant_idx, variant_def) in def.variants().iter_enumerated() {
            if variant_def
                .fields
                .iter()
                .all(|f| self.has_feasible_value(self.tcx_field_ty(f, generics)))
            {
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

        // build typing env
        let typing_env = tcx.typing_env_normalized_for_post_analysis(def_id);

        // check if the owner is a closure
        let is_closure = match tcx.def_kind(def_id) {
            DefKind::SyntheticCoroutineBody => continue,
            DefKind::Closure => true,
            _ => false,
        };

        // skip coroutine-related owners
        if is_closure
            && matches!(
                tcx.type_of(def_id).instantiate_identity().skip_norm_wip().kind(),
                ty::Coroutine(..) | ty::CoroutineClosure(..) | ty::CoroutineWitness(..)
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
        let mut const_param_types = vec![];
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
            let param_name = SolGenericName(param_symbol.to_ident_string());
            let param_kind = match param_def_kind {
                GenericParamDefKind::Lifetime => SolGenericKind::Lifetime,
                GenericParamDefKind::Type { has_default: _, synthetic: _ } => {
                    // skip injected closure-related type parameters, which are not `synthetic`
                    // they used to be named as below, but is unnamed now.
                    // * <closure_kind>: I16,
                    // * <closure_signature>: fn(..) -> ..
                    // * <upvars>: (..) (a.k.a, a tuple)
                    //
                    // FIXME: we should ideally have a more robust way to identify these parameters
                    // instead of relying on the naming convention, and these parameters also appear
                    // in nested bodies (e.g., a constant in a closure also carries these parameters),
                    // so we may need to track the parent-child relationships of bodies as well.
                    if param_symbol.is_empty() {
                        continue;
                    }
                    SolGenericKind::Type
                }
                GenericParamDefKind::Const { has_default: _ } => {
                    // pre-collect the types of const parameters
                    let const_ty = tcx.normalize_erasing_regions(
                        typing_env,
                        tcx.type_of(*param_def_id).instantiate_identity(),
                    );
                    const_param_types.push((param_ident.clone(), const_ty));

                    // still return the proper generic kind
                    SolGenericKind::Const
                }
            };
            bundle_generics.push(SolGenericParam {
                ident: param_ident,
                index: SolGenericIndex(*param_index as usize),
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

        // collect the const param types
        for (const_ident, const_ty) in const_param_types.into_iter() {
            let parsed_ty = exec_builder.mk_type(const_ty);
            exec_builder.cp_types.insert(const_ident.clone(), parsed_ty);
        }

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
            mut cp_types,
            adt_defs,
            trait_defs,
            dyn_types,
            static_inits,
            upvar_decls: _,
            var_scope: _,
            inst_ids: _,
            log_stack,
        } = exec_builder;
        assert!(log_stack.is_empty(), "[invariant] log stack is not empty");

        // collect the generic declarations
        let mut flat_generics = vec![];
        for param in generics.into_iter() {
            let param_meta = match param.kind {
                SolGenericKind::Lifetime => SolGenericMeta::Lifetime,
                SolGenericKind::Type => SolGenericMeta::Type,
                SolGenericKind::Const => SolGenericMeta::Const(
                    cp_types
                        .remove(&param.ident)
                        .unwrap_or_else(|| bug!("[invariant] missing const generic type")),
                ),
            };
            flat_generics.push((param.ident, param.name, param_meta));
        }
        assert!(cp_types.is_empty(), "[invariant] leftover const param types");

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

        // collect the dynamic types
        let mut flat_dyn_types = vec![];
        for (ident, (_, dyn_ty)) in dyn_types.into_iter() {
            match dyn_ty {
                DynTypeStatus::Resolving(_) => bug!("[invariant] missing dynamic type clauses"),
                DynTypeStatus::Processed(clauses) => {
                    flat_dyn_types.push((ident, clauses));
                }
            }
        }

        // collect static initializers
        let mut flat_static_inits = vec![];
        for (ident, init) in static_inits.into_iter() {
            flat_static_inits.push((
                ident,
                init.unwrap_or_else(|| bug!("[invariant] missing static initializer")),
            ));
        }

        // complete the bundle construction
        let hir_id = tcx.local_def_id_to_hir_id(owner_id);
        let exec_full = base.mk_mir(owner_id, hir_id, tcx.hir_span_with_body(hir_id), exec);
        bundles.push(SolBundle {
            generics: flat_generics,
            adt_defs: flat_adt_defs,
            trait_defs: flat_trait_defs,
            dyn_types: flat_dyn_types,
            static_inits: flat_static_inits,
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
    for (ident, desc, span) in id_cache.into_values() {
        id_desc.push((ident, desc, span));
    }
    id_desc.sort_by(|(ident_a, desc_a, _), (ident_b, desc_b, _)| {
        desc_a.cmp(desc_b).then_with(|| ident_a.cmp(ident_b))
    });

    // construct the crate
    SolCrate { root: module_mir, bundles, id_desc }
}

/* --- BEGIN OF SYNC --- */

/// A trait alias for all sorts IR elements
pub(crate) trait SolIR =
    Debug + Clone + PartialEq + Eq + PartialOrd + Ord + Hash + Serialize + DeserializeOwned;

/*
* Common
 */

/// The base information associated with anything that has an hir_id, span, but no def_id
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(bound = "T: Serialize + DeserializeOwned")]
pub(crate) struct SolHIR<T: SolIR> {
    pub(crate) doc_comments: Vec<SolDocComment>,
    pub(crate) data: T,
}

/// The base information associated with anything that has a def_id, span, and maybe hir_id
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolCrate {
    pub(crate) root: SolMIR<SolModule>,
    pub(crate) bundles: Vec<SolBundle>,
    pub(crate) id_desc: Vec<(SolIdent, SolPathDesc, SolSpan)>,
}

/// A complete execution context
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolBundle {
    pub(crate) generics: Vec<(SolIdent, SolGenericName, SolGenericMeta)>,
    pub(crate) adt_defs: Vec<(SolIdent, Vec<SolGenericArg>, SolAdtDef)>,
    pub(crate) trait_defs: Vec<(SolIdent, Vec<SolGenericArg>, SolTraitDef)>,
    pub(crate) dyn_types: Vec<(SolDynTypeIndex, Vec<SolClause>)>,
    pub(crate) static_inits: Vec<(SolIdent, SolValue)>,
    pub(crate) executable: SolMIR<SolExec>,
}

/*
 * Module
 */

/// A module
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolModule {
    pub(crate) name: SolModuleName,
    pub(crate) scope: SolSpan,
    pub(crate) items: Vec<SolMIR<SolItem>>,
}

/*
 * Item
 */

/// Details associated with an item
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) enum SolItem {
    Module(SolModule),
}

/*
 * Exec (THIR)
 */

/// The main body of a THIR (e.g., a function or a constant)
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) enum SolExec {
    Function(SolFnDef),
    Closure(SolClosure),
    ConstEval(SolCEval),
}

/// THIR of a function
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolFnDef {
    pub(crate) abi: SolExternAbi,
    pub(crate) vis: SolVisibility,
    pub(crate) ret_ty: SolType,
    pub(crate) params: Vec<(SolType, Option<SolPattern>)>,
    pub(crate) body: SolExpr,
}

/// THIR of a closure
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolClosure {
    pub(crate) ret_ty: SolType,
    pub(crate) params: Vec<(SolType, Option<SolPattern>)>,
    pub(crate) upvars: Vec<(SolLocalVarIndex, SolType, SolLocalVarName)>,
    pub(crate) body: SolExpr,
}

/// THIR of a constant evaluation
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolCEval {
    pub(crate) ty: SolType,
    pub(crate) body: SolExpr,
}

/// Visibility of a function
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) enum SolVisibility {
    Public,
    Restricted(SolIdent),
}

/// External ABI of a function
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) enum SolExternAbi {
    C { variadic: bool },
    Rust { safety: bool },
    System,
}

/*
 * Typing
 */

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
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
    // algebraic data types
    Struct(SolIdent, Vec<SolGenericArg>),
    Union(SolIdent, Vec<SolGenericArg>),
    Enum(SolIdent, Vec<SolGenericArg>),
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
    FuncDef(SolIdent, Vec<SolGenericArg>),
    FuncSym(SolIdent, Vec<SolGenericArg>),
    AdtCtor(SolIdent, Vec<SolGenericArg>),
    Virtual(SolIdent, Vec<SolGenericArg>),
    Libfunc(SolIdent, Vec<SolGenericArg>),
    External(SolIdent, Vec<SolGenericArg>),
    Intrinsic(SolIdent, Vec<SolGenericArg>),
    Closure(SolIdent, Vec<SolGenericArg>),
    FnPtr(SolFnSig),
    // dynamic types
    Dynamic(SolDynTypeIndex),
    Assoc {
        trait_ident: SolIdent,
        trait_ty_args: Vec<SolGenericArg>,
        item_ident: SolIdent,
        item_ty_args: Vec<SolGenericArg>,
    },
    // alias types
    // MAYFIX: currently we always resolve (i.e., normalize) alias/opaque types
    // to their underlying types. We might want to keep them around in some cases.
}

/// A function signature
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolFnSig {
    pub(crate) abi: SolExternAbi,
    pub(crate) param_tys: Vec<SolType>,
    pub(crate) output_ty: Box<SolType>,
}

/// A pattern type
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) enum SolTyPat {
    NotNull,
    Range(SolConst, SolConst),
    Or(Vec<SolTyPat>),
}

/// A projection term
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) enum SolProjTerm {
    Type(SolType),
    Const(SolConst),
}

/// Differentiate kinds of generics
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) enum SolGenericKind {
    Lifetime,
    Const,
    Type,
}

/// Like `SolGenericKind`, but enhanced with constant type
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) enum SolGenericMeta {
    Lifetime,
    Const(SolType),
    Type,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolGenericParam {
    pub(crate) ident: SolIdent,
    pub(crate) index: SolGenericIndex,
    pub(crate) name: SolGenericName,
    pub(crate) kind: SolGenericKind,
}

/// A generic argument
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) enum SolGenericArg {
    Type(SolType),
    Const(SolConst),
    Lifetime,
}

/// User-defined type, i.e., an algebraic data type (ADT)
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) enum SolAdtDef {
    Struct { fields: Vec<SolField> },
    Union { fields: Vec<SolField> },
    Enum { variants: Vec<SolVariant> },
}

/// A field definition in an ADT
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolField {
    pub(crate) index: SolFieldIndex,
    pub(crate) name: SolFieldName,
    pub(crate) ty: SolType,
    pub(crate) default: Option<SolIdent>,
}

/// A field definition in an ADT
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolVariant {
    pub(crate) index: SolVariantIndex,
    pub(crate) name: SolVariantName,
    pub(crate) discr: SolVariantDiscr,
    pub(crate) fields: Vec<SolField>,
}

/// Trait definition
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolTraitDef {
    pub(crate) clauses: Vec<SolClause>,
}

/// A clause in a predicate
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) enum SolConst {
    Param(SolIdent),
    Value(SolValue),
    Unevaluated(SolType, SolIdent, Vec<SolGenericArg>),
}

/// A constant with concrete value and type
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
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
    ImmRefStatic(SolType, SolIdent, usize),
    MutRefStatic(SolType, SolIdent, usize),
    ImmRefNull(SolType, usize),
    MutRefNull(SolType, usize),
    ImmPtr(Box<SolValue>),
    MutPtr(Box<SolValue>),
    ImmPtrStatic(SolType, SolIdent, usize),
    MutPtrStatic(SolType, SolIdent, usize),
    ImmPtrNull(SolType, usize),
    MutPtrNull(SolType, usize),
    // function pointers
    FuncDef(SolIdent, Vec<SolGenericArg>),
    FuncSym(SolIdent, Vec<SolGenericArg>),
    AdtCtor(SolIdent, Vec<SolGenericArg>),
    Virtual(SolIdent, Vec<SolGenericArg>),
    Libfunc(SolIdent, Vec<SolGenericArg>),
    External(SolIdent, Vec<SolGenericArg>),
    Intrinsic(SolIdent, Vec<SolGenericArg>),
    Closure(SolIdent, Vec<SolGenericArg>),
    FnPtr(SolFnSig, SolIdent, Vec<SolGenericArg>),
    FnPtrNull(SolFnSig),
}

/*
 * Expression
 */

/// An expression in THIR
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolExpr {
    pub(crate) id: SolInstIndex,
    pub(crate) ty: SolType,
    pub(crate) op: Box<SolOp>,
    pub(crate) span: SolSpan,
    pub(crate) scope: SolScope,
}

/// Details of the operation in an expression
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) enum SolOp {
    // markers
    Scope(SolHIR<SolExpr>, SolScope),
    TypeAscribe(SolExpr),
    // use
    Use(SolExpr),
    LocalVar(SolLocalVarIndex),
    UpVar(SolLocalVarIndex),
    ConstParam(SolIdent),
    ConstBlock(SolIdent, Vec<SolGenericArg>),
    NamedConst(SolIdent, Vec<SolGenericArg>),
    StaticRef(SolIdent),
    // intrisics
    Deref(SolExpr),
    // operators
    Not(SolExpr),
    Neg(SolExpr),
    Add(SolExpr, SolExpr),
    Sub(SolExpr, SolExpr),
    Mul(SolExpr, SolExpr),
    Div(SolExpr, SolExpr),
    Rem(SolExpr, SolExpr),
    BitXor(SolExpr, SolExpr),
    BitAnd(SolExpr, SolExpr),
    BitOr(SolExpr, SolExpr),
    Shl(SolExpr, SolExpr),
    Shr(SolExpr, SolExpr),
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
        scope: SolScope,
    },
    Loop(SolExpr),
    Break(SolScope, Option<SolExpr>),
    Continue(SolScope),
    Return(Option<SolExpr>),
    // function calls
    Call {
        target: SolType,
        callee: SolExpr,
        args: Vec<SolExpr>,
    },
    // pattern
    Let(SolExpr, SolPattern),
    Match(SolExpr, Vec<SolHIR<SolMatchArm>>),
    // compound
    Block(SolBlock),
    // literals
    BaseLiteral(bool, SolValue, SolSpan),
    ConstValue(SolValue),
    // closure
    Closure(SolIdent, Vec<SolGenericArg>, Vec<SolExpr>),
}

/// A pattern matcher in THIR
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolPattern {
    pub(crate) ty: SolType,
    pub(crate) rule: SolPatRule,
    pub(crate) span: SolSpan,
}

/// A pattern matching rule in THIR
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
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
    Range {
        lo: Option<SolValue>,
        hi: Option<SolValue>,
        end_inclusive: bool,
    },
    Constant(SolValue),
    Or(Vec<SolPattern>),
}

/// Binding mode
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) enum SolBindMode {
    ImmByValue,
    MutByValue,
    ImmByImmRef,
    ImmByMutRef,
    MutByImmRef,
    MutByMutRef,
}

/// A match arm in THIR
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolMatchArm {
    pub(crate) scope: SolScope,
    pub(crate) pat: SolPattern,
    pub(crate) guard: Option<SolExpr>,
    pub(crate) body: SolExpr,
    pub(crate) span: SolSpan,
}

/// A block of expressiosn in THIR
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolBlock {
    pub(crate) scope: SolScope,
    pub(crate) safety: bool,
    pub(crate) stmts: Vec<SolStmt>,
    pub(crate) expr: Option<SolExpr>,
    pub(crate) span: SolSpan,
}

/// A statement in THIR
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) enum SolStmt {
    Expr(SolInstIndex, SolScope, SolExpr),
    Bind(SolInstIndex, SolHIR<SolLetBinding>),
}

/// A scope in THIR
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolScope {
    pub(crate) index: SolScopeIndex,
}

/// Let binding in a statement in THIR
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolLetBinding {
    pub(crate) pat: SolPattern,
    pub(crate) init: Option<(SolExpr, Option<SolBlock>)>,
    pub(crate) span: SolSpan,
}

/// Base for ADT construction
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) enum SolAdtBase {
    None,
    Overlay(SolExpr, Vec<SolType>),
    Default(Vec<SolType>),
}

/*
 * Comment
 */

/// A documentation comment
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) enum SolDocComment {
    Outer(String),
    Inner(String),
}

/*
 * Naming
 */

/// An identifier in the crate
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolIdent {
    pub(crate) krate: SolHash64,
    pub(crate) local: SolHash64,
}

/// A 64-bit hash
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolHash64(pub(crate) u64);

/// A 128-bit hash
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolHash128(pub(crate) u128);

/// A module name
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolModuleName(pub(crate) String);

/// A generic parameter name
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolGenericName(pub(crate) String);

/// A generic parameter name
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolGenericIndex(pub(crate) usize);

/// A uniquely identifier for a dynamic type
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolDynTypeIndex(pub(crate) usize);

/// A name to a local variable
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolLocalVarName(pub(crate) String);

/// A index to a local variable
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolLocalVarIndex(pub(crate) usize, pub(crate) usize);

/// A field name
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolFieldName(pub(crate) String);

/// A field index
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolFieldIndex(pub(crate) usize);

/// A variant name
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolVariantName(pub(crate) String);

/// A variant index
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolVariantIndex(pub(crate) usize);

/// A variant discrimanant
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolVariantDiscr(pub(crate) u128);

/// A scope index
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolScopeIndex(pub(crate) usize);

/// An instruction index
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolInstIndex(pub(crate) usize);

/// A description of a definition path
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolPathDesc(pub(crate) String);

/// A span description
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct SolSpan {
    pub(crate) file_id: SolHash128,
    pub(crate) start_line: u32,
    pub(crate) start_column: u32,
    pub(crate) end_line: u32,
    pub(crate) end_column: u32,
}

/* --- END OF SYNC --- */

/// Status of a dyn type
enum DynTypeStatus {
    Resolving(UniverseIndex),
    Processed(Vec<SolClause>),
}

/// Known crates in the rust standard library
enum ResolvedStdlib {
    Core,
    Alloc,
    Std,
    ProcMacro,
    Test,
}

/// Differentiate resolved function kinds
enum ResolvedFunction {
    FuncDef(SolIdent, Vec<SolGenericArg>),
    FuncSym(SolIdent, Vec<SolGenericArg>),
    AdtCtor(SolIdent, Vec<SolGenericArg>),
    Virtual(SolIdent, Vec<SolGenericArg>),
    Libfunc(SolIdent, Vec<SolGenericArg>),
    External(SolIdent, Vec<SolGenericArg>),
    Intrinsic(SolIdent, Vec<SolGenericArg>),
}

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
        self.stack.push_back((tag, msg));
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
fn util_ident_string<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> String {
    util_debug_symbol(tcx, def_id, &List::empty())
}

/// Create a fully qualified path string with crate name
#[inline]
fn util_debug_symbol<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    ty_args: GenericArgsRef<'tcx>,
) -> String {
    with_no_visible_paths!(with_resolve_crate_name!(with_no_trimmed_paths!(
        tcx.def_path_str_with_args(def_id, ty_args)
    )))
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
    .unwrap_or_else(|_| bug!("[unsupported] non utf-8 string"))
}

/// Lookup the variant index by discriminant value
#[inline]
fn get_variant_by_discriminant<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: AdtDef<'tcx>,
    scalar: ScalarInt,
) -> VariantIdx {
    let mut variant_idx_found = None;
    for (variant_idx, discr) in def.discriminants(tcx) {
        if scalar_eq_discriminant(scalar, discr) {
            variant_idx_found = Some(variant_idx);
            break;
        }
    }
    variant_idx_found.unwrap_or_else(|| {
        bug!(
            "[invariant] no discriminant matches {scalar} for {}",
            util_ident_string(tcx, def.did())
        );
    })
}

/// Compare a scalar int and a discriminant
#[inline]
fn scalar_eq_discriminant(scalar: ScalarInt, Discr { val, ty }: Discr<'_>) -> bool {
    let scalar_size = scalar.size().bytes();
    match ty.kind() {
        ty::Int(sub_ty) => match (sub_ty, scalar_size) {
            (IntTy::I8, 1) => scalar.to_i8() == val as i8,
            (IntTy::I16, 1) => scalar.to_i8() == val as i8,
            (IntTy::I16, 2) => scalar.to_i16() == val as i16,
            (IntTy::I32, 1) => scalar.to_i8() == val as i8,
            (IntTy::I32, 2) => scalar.to_i16() == val as i16,
            (IntTy::I32, 4) => scalar.to_i32() == val as i32,
            (IntTy::I64, 1) => scalar.to_i8() == val as i8,
            (IntTy::I64, 2) => scalar.to_i16() == val as i16,
            (IntTy::I64, 4) => scalar.to_i32() == val as i32,
            (IntTy::I64, 8) => scalar.to_i64() == val as i64,
            (IntTy::I128, 1) => scalar.to_i8() == val as i8,
            (IntTy::I128, 2) => scalar.to_i16() == val as i16,
            (IntTy::I128, 4) => scalar.to_i32() == val as i32,
            (IntTy::I128, 8) => scalar.to_i64() == val as i64,
            (IntTy::I128, 16) => scalar.to_i128() == val as i128,
            (IntTy::Isize, 1) => scalar.to_i8() == val as i8,
            (IntTy::Isize, 2) => scalar.to_i16() == val as i16,
            (IntTy::Isize, 4) => scalar.to_i32() == val as i32,
            (IntTy::Isize, 8) => scalar.to_i64() == val as i64,
            (IntTy::Isize, 16) => scalar.to_i128() == val as i128,
            _ => bug!("[invariant] unexpected ({scalar_size}, {ty}) pair for discriminant"),
        },
        ty::Uint(_) => scalar.to_uint(scalar.size()) == val,
        _ => scalar.to_bits_unchecked() == val,
    }
}
