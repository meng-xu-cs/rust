use std::collections::{BTreeMap, VecDeque};
use std::fmt::Debug;
use std::path::PathBuf;

use rustc_abi::ExternAbi;
use rustc_ast::{AttrStyle, FloatTy, IntTy, Mutability, UintTy};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::attrs::AttributeKind;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_hir::{Attribute, CRATE_HIR_ID, HirId, Item, ItemKind, Mod, Safety};
use rustc_middle::thir::{BodyTy, ExprId, Thir};
use rustc_middle::ty::{
    AdtDef, AdtKind, Const, ConstKind, FnSig, GenericArg, GenericArgKind, GenericArgsRef,
    ParamConst, ParamTy, Pattern, PatternKind, ScalarInt, Ty, TyCtxt, ValTreeKind, Value,
    VariantDiscr,
};
use rustc_middle::{bug, ty};
use rustc_span::{DUMMY_SP, RemapPathScopeComponents, Span, StableSourceFileId, Symbol};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tracing::info;

/// A builder for creating a nlai module
pub(crate) struct Builder<'tcx> {
    /// compiler context
    tcx: TyCtxt<'tcx>,

    /// source directory
    src_dir: PathBuf,

    /// source cache
    src_cache: FxHashSet<StableSourceFileId>,

    /// collected definitions of datatypes
    ty_defs: BTreeMap<SolIdent, BTreeMap<Vec<SolGenericArg>, Option<SolAdtDef>>>,

    /// a cache of id to identifier mappings
    id_cache: FxHashMap<DefId, (SolIdent, SolPathDesc)>,

    /// log stack
    log_stack: LogStack,
}

impl<'tcx> Builder<'tcx> {
    /// Create a new builder
    pub(crate) fn new(tcx: TyCtxt<'tcx>, src_dir: PathBuf) -> Self {
        Self {
            tcx,
            src_dir,
            src_cache: FxHashSet::default(),
            ty_defs: BTreeMap::new(),
            id_cache: FxHashMap::default(),
            log_stack: LogStack::new(),
        }
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

    #[allow(dead_code)]
    /// Record an AST node in the THIR body with its metadata
    fn mk_ast<T: SolIR>(&mut self, span: Span, data: T) -> SolAST<T> {
        SolAST { span: self.mk_span(span), data }
    }

    /// Record a MIR node in the THIR body with its metadata
    fn mk_mir<T: SolIR>(&mut self, hir_id: HirId, span: Span, data: T) -> SolMIR<T> {
        SolMIR {
            ident: self.mk_ident(hir_id.expect_owner().to_def_id()),
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
            name: SolModuleName(name.to_ident_string()),
            scope: self.mk_span(spans.inner_span),
            items,
        }
    }

    /// Record a function ABI
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

    /// Record a generic argument
    pub(crate) fn mk_generic_arg(&mut self, ty_arg: GenericArg<'tcx>) -> SolGenericArg {
        match ty_arg.kind() {
            GenericArgKind::Type(ty) => SolGenericArg::Type(self.mk_type(ty)),
            GenericArgKind::Const(val) => SolGenericArg::Const(self.mk_const(val)),
            GenericArgKind::Lifetime(region) => {
                if !region.is_erased() {
                    bug!("[invariant] regions not erased in THIR: {region}");
                }
                SolGenericArg::Lifetime
            }
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
        if self.ty_defs.get(&ident).map_or(false, |inner| inner.contains_key(&generic_args)) {
            return (ident, generic_args);
        }

        // mark start
        let def_desc = Self::debug_symbol(self.tcx, def_id, ty_args);
        self.log_stack.push("|ADT|", def_desc.clone());

        // first update the entry to mark that type definition in progress
        self.ty_defs.entry(ident.clone()).or_default().insert(generic_args.clone(), None);

        // now create the type definition
        let ty_def = match adt_def.adt_kind() {
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
        self.ty_defs.entry(ident.clone()).or_default().insert(generic_args.clone(), Some(ty_def));

        // mark end
        self.log_stack.pop();

        // return the result
        (ident, generic_args)
    }

    /// Record a type in MIR/THIR context
    pub(crate) fn mk_type(&mut self, ty: Ty<'tcx>) -> SolType {
        match ty.kind() {
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
                if !region.is_erased() {
                    bug!("[invariant] regions not erased in THIR: {region}");
                }
                let sub_ty = self.mk_type(*referent_ty);
                match mutability {
                    Mutability::Not => SolType::ImmRef(Box::new(sub_ty)),
                    Mutability::Mut => SolType::MutRef(Box::new(sub_ty)),
                }
            }

            // type parameter
            ty::Param(ParamTy { index, name }) => {
                SolType::Param(SolParamIndex(*index as usize), SolParamName(name.to_ident_string()))
            }

            // compound types
            ty::Tuple(sub_tys) => {
                SolType::Tuple(sub_tys.iter().map(|sub_ty| self.mk_type(sub_ty)).collect())
            }
            ty::Slice(elem_ty) => SolType::Slice(Box::new(self.mk_type(*elem_ty))),
            ty::Array(elem_ty, size) => {
                SolType::Array(Box::new(self.mk_type(*elem_ty)), Box::new(self.mk_const(*size)))
            }

            // unsupported
            ty::Coroutine(..) | ty::CoroutineClosure(..) | ty::CoroutineWitness(..) => {
                bug!("[unsupported] coroutine type {ty}")
            }

            // unexpected
            ty::Infer(..) | ty::Bound(..) | ty::Placeholder(..) | ty::Error(..) => {
                bug!("[invaraint] unexpected type {ty}")
            }

            _ => todo!("[todo] unhandled type {ty}"),
        }
    }

    /// Record a constant in MIR/THIR context
    pub(crate) fn mk_const(&mut self, cval: Const<'tcx>) -> SolConst {
        match cval.kind() {
            ConstKind::Param(ParamConst { index, name }) => {
                SolConst::Param(SolParamIndex(index as usize), SolParamName(name.to_ident_string()))
            }
            ConstKind::Value(val) => SolConst::Value(self.mk_value(val)),

            // unsupported
            ConstKind::Expr(..) => {
                bug!("[unsupported] const expr {cval}")
            }

            // unexpected
            ConstKind::Infer(..)
            | ConstKind::Bound(..)
            | ConstKind::Unevaluated(..)
            | ConstKind::Placeholder(..)
            | ConstKind::Error(..) => {
                bug!("[invaraint] unexpected const {cval}")
            }
        }
    }

    /// Record a (constant) value in MIR/THIR context
    pub(crate) fn mk_value(&mut self, val: Value<'tcx>) -> SolValue {
        let Value { ty, valtree } = val;

        match *valtree {
            ValTreeKind::Leaf(leaf) => self.mk_value_from_scalar(val.ty, *leaf),
            ValTreeKind::Branch(box []) => self.mk_value_from_zst(ty),
            ValTreeKind::Branch(box _) => {
                todo!("[todo] multi-branch constant {val} for type {ty}")
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
            _ => todo!(),
        }
    }

    /// Record a (constant) value for a ZST
    pub(crate) fn mk_value_from_zst(&mut self, ty: Ty<'tcx>) -> SolValue {
        self.mk_value_when_zst(ty).unwrap_or_else(|| {
            bug!("[invariant] failed to create a value for ZST type {ty}");
        })
    }

    /// Try to record a (constant) value for a ZST or None of the type is not zero-sized
    fn mk_value_when_zst(&mut self, ty: Ty<'tcx>) -> Option<SolValue> {
        let zst_val = match ty.kind() {
            ty::Tuple(elems) => {
                // a tuple is a ZST if all its elements are ZSTs (including the empty tuple)
                let mut elem_vals = vec![];
                for elem_ty in elems.iter() {
                    let elem_val = self.mk_value_when_zst(elem_ty)?;
                    elem_vals.push(elem_val);
                }
                SolValue::Tuple(elem_vals)
            }

            // FIXME: handle more ZST types
            _ => return None,
        };
        Some(zst_val)
    }

    /// Record a executable, i.e., a THIR body (for a function or a constant/static)
    pub(crate) fn mk_exec(&mut self, thir: &Thir<'tcx>, _expr: ExprId) -> SolExec {
        // switch-case on the body type
        match thir.body_type {
            BodyTy::Fn(sig) => {
                let FnSig { abi, c_variadic, safety, inputs_and_output: _ } = sig;

                // parse function signature
                let parsed_abi = self.mk_abi(abi, c_variadic, safety);
                let ret_ty = self.mk_type(sig.output());
                let params: Vec<_> =
                    sig.inputs().iter().map(|input_ty| self.mk_type(*input_ty)).collect();

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
                SolExec::Constant(SolCEval { ty: const_ty })
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
            let thir_lock = thir_body.borrow();
            let exec = self.mk_exec(&*thir_lock, thir_expr);
            let hir_id = HirId::make_owner(owner_id);
            executables.push(self.mk_mir(hir_id, self.tcx.hir_span_with_body(hir_id), exec));
        }

        // unpack the builder
        let Self { tcx: _, src_dir: _, src_cache: _, ty_defs, id_cache, log_stack } = self;

        // sanity check
        if !log_stack.is_empty() {
            bug!("[invariant] log stack is not empty");
        }

        // collect the datatype definitions
        let mut flat_ty_defs = vec![];
        for (ident, l1) in ty_defs.into_iter() {
            for (mono, def) in l1.into_iter() {
                match def {
                    None => bug!("[invariant] missing type definition"),
                    Some(val) => flat_ty_defs.push((ident.clone(), mono, val)),
                }
            }
        }

        // collect the id to description mappings
        let mut id_desc = vec![];

        // we don't care about the ordering of the values
        #[allow(rustc::potential_query_instability)]
        for (ident, desc) in id_cache.into_values() {
            id_desc.push((ident, desc));
        }

        // construct the crate
        SolCrate { root: module_mir, execs: executables, ty_defs: flat_ty_defs, id_desc }
    }

    /// Create a fully qualified path string with crate name
    #[inline]
    fn debug_symbol(tcx: TyCtxt<'tcx>, def_id: DefId, ty_args: GenericArgsRef<'tcx>) -> String {
        let path_str = tcx.def_path_str_with_args(def_id, ty_args);
        if def_id.is_local() {
            let krate = tcx.crate_name(LOCAL_CRATE).to_ident_string();
            format!("{krate}::{path_str}")
        } else {
            path_str
        }
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
    pub(crate) execs: Vec<SolMIR<SolExec>>,
    pub(crate) ty_defs: Vec<(SolIdent, Vec<SolGenericArg>, SolAdtDef)>,
    pub(crate) id_desc: Vec<(SolIdent, SolPathDesc)>,
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
}

/// THIR of a constant evaluation
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolCEval {
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
    Param(SolParamIndex, SolParamName),
    // compound types
    Tuple(Vec<SolType>),
    Slice(Box<SolType>),
    Array(Box<SolType>, Box<SolConst>),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolTyPat {
    NotNull,
    Range(SolConst, SolConst),
    Or(Vec<SolTyPat>),
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
}

/// A field definition in an ADT
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolVariant {
    pub(crate) index: SolVariantIndex,
    pub(crate) name: SolVariantName,
    pub(crate) discr: SolVariantDiscr,
    pub(crate) fields: Vec<SolField>,
}

/*
 * Constant
 */

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolConst {
    Param(SolParamIndex, SolParamName),
    Value(SolValue),
}

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
    // composite
    Tuple(Vec<SolValue>),
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

/// A parameter index
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolParamIndex(pub(crate) usize);

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
