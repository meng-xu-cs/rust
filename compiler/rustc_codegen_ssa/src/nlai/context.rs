use std::fmt::Debug;
use std::path::PathBuf;

use rustc_abi::ExternAbi;
use rustc_ast::{AttrStyle, FloatTy, IntTy, LitFloatType, LitIntType, LitKind, Mutability, UintTy};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::attrs::AttributeKind;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{CRATE_DEF_ID, DefId, LOCAL_CRATE, LocalDefId};
use rustc_hir::{
    Attribute, CRATE_HIR_ID, ConstArg, ConstArgKind, FnDecl, FnPtrTy, FnRetTy, GenericParam,
    GenericParamKind, Generics, HirId, ImplicitSelfKind, ItemKind, Lifetime, LifetimeKind,
    LifetimeParamKind, MissingLifetimeKind, Mod, MutTy, OwnerId, ParamName, PrimTy, Safety,
    Ty as HirTy, TyKind as HirTyKind, TyPat, TyPatKind,
};
use rustc_middle::bug;
use rustc_middle::ty::TyCtxt;
use rustc_span::{DUMMY_SP, Ident, RemapPathScopeComponents, Span, StableSourceFileId, Symbol};
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

    /// hir stack
    hir_stack: Vec<OwnerId>,

    /// a cache of id to identifier mappings
    id_cache: FxHashMap<DefId, (SolIdent, SolPathDesc)>,
}

impl<'tcx> Builder<'tcx> {
    /// Create a new builder
    pub(crate) fn new(tcx: TyCtxt<'tcx>, src_dir: PathBuf) -> Self {
        Self {
            tcx,
            src_dir,
            src_cache: FxHashSet::default(),
            hir_stack: Vec::new(),
            id_cache: FxHashMap::default(),
        }
    }

    fn mk_symbol(&self, ident: Ident) -> SolSymbol {
        SolSymbol(ident.name.to_ident_string())
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

    fn mk_spanned<T: SolIR>(&mut self, span: Span, data: T) -> SolSpanned<T> {
        SolSpanned { span: self.mk_span(span), data }
    }

    #[allow(unused)]
    fn mk_hir<T: SolIR>(&mut self, hir_id: HirId, span: Span, data: T) -> SolHIR<T> {
        // sanity check
        if hir_id.is_owner() {
            bug!("[invariant] owner hir_id should be used to build SolMIR instead of SolHIR");
        }
        if !self.hir_stack.last().is_some_and(|owner| *owner == hir_id.owner) {
            bug!(
                "[invariant] hir_id owner {:?} does not match current HIR stack top {:?}",
                hir_id.owner,
                self.hir_stack.last()
            );
        }

        // pack the information
        SolHIR { span: self.mk_span(span), doc_comments: self.mk_doc_comments(hir_id), data }
    }

    fn mk_mir<T: SolIR>(&mut self, hir_id: HirId, span: Span, data: T) -> SolMIR<T> {
        // sanity check
        if !hir_id.is_owner() {
            bug!("[invariant] non-owner hir_id should be used to build SolHIR instead of SolMIR");
        }

        // pack the information
        SolMIR {
            ident: self.mk_ident(hir_id.expect_owner().to_def_id()),
            span: self.mk_span(span),
            doc_comments: self.mk_doc_comments(hir_id),
            data,
        }
    }

    fn mk_mir_from_hir<T: SolIR>(
        &mut self,
        hir_id: HirId,
        def_id: LocalDefId,
        span: Span,
        data: T,
    ) -> SolMIR<T> {
        // sanity check
        if hir_id.is_owner() {
            bug!("[invariant] owner hir_id should be used to build SolMIR instead of SolHIR");
        }
        if !self.hir_stack.last().is_some_and(|owner| *owner == hir_id.owner) {
            bug!(
                "[invariant] hir_id owner {:?} does not match current HIR stack top {:?}",
                hir_id.owner,
                self.hir_stack.last()
            );
        }
        if hir_id.owner.def_id == def_id {
            bug!("[invariant] hir_id owner def_id should be different from provided def_id");
        }

        // pack the information
        SolMIR {
            ident: self.mk_ident(def_id.to_def_id()),
            span: self.mk_span(span),
            doc_comments: self.mk_doc_comments(hir_id),
            data,
        }
    }

    fn mk_generic_param<'hir>(&mut self, param: GenericParam<'hir>) -> SolMIR<SolGenericParam> {
        let GenericParam {
            hir_id,
            def_id,
            name,
            span,
            pure_wrt_drop: _,
            kind,
            colon_span: _,
            source: _,
        } = param;

        // switch case by kind
        let param_data = match kind {
            GenericParamKind::Lifetime { kind } => match kind {
                LifetimeParamKind::Elided(reason) => {
                    match name {
                        ParamName::Plain(_) => {
                            bug!("[invariant] elided lifetime param has a user-defined name");
                        }
                        ParamName::Fresh => {}
                        ParamName::Error(_) => unreachable!(),
                    }
                    match reason {
                        MissingLifetimeKind::Underscore => {
                            SolGenericParam::Lifetime { name: SolLifetimeName::ElidedExplicit }
                        }
                        MissingLifetimeKind::Ampersand => {
                            SolGenericParam::Lifetime { name: SolLifetimeName::ElidedImplicit }
                        }
                        MissingLifetimeKind::Comma => {
                            bug!("[unsupported] MissingLifetimeKind::Comma");
                        }
                        MissingLifetimeKind::Brackets => {
                            bug!("[unsupported] MissingLifetimeKind::Brackets");
                        }
                    }
                }
                LifetimeParamKind::Explicit => match name {
                    ParamName::Plain(ident) => SolGenericParam::Lifetime {
                        name: SolLifetimeName::Named(self.mk_symbol(ident)),
                    },
                    ParamName::Fresh => {
                        bug!("[invariant] explicit lifetime param has a fresh name");
                    }
                    ParamName::Error(_) => unreachable!(),
                },
                LifetimeParamKind::Error => unreachable!(),
            },
            GenericParamKind::Type { default, synthetic } => {
                if synthetic {
                    bug!("[unsupported] type param is synthetic");
                }
                match name {
                    ParamName::Plain(ident) => SolGenericParam::Type {
                        name: self.mk_symbol(ident),
                        default: default.map(|ty| self.mk_hir_type(*ty)),
                    },
                    ParamName::Fresh => bug!("[invariant] generic type param has a fresh name"),
                    ParamName::Error(_) => unreachable!(),
                }
            }
            GenericParamKind::Const { ty, default } => match name {
                ParamName::Plain(ident) => SolGenericParam::Const {
                    name: self.mk_symbol(ident),
                    ty: self.mk_hir_type(*ty),
                    default: default.map(|arg| self.mk_const_arg(*arg)),
                },
                ParamName::Fresh => bug!("[invariant] generic const param has a fresh name"),
                ParamName::Error(_) => unreachable!(),
            },
        };

        // construct the generic param
        self.mk_mir_from_hir(hir_id, def_id, span, param_data)
    }

    fn mk_generics<'hir>(&mut self, generics: Generics<'hir>) -> SolSpanned<SolGenerics> {
        let Generics {
            params,
            predicates: _,
            has_where_clause_predicates: _,
            where_clause_span: _,
            span,
        } = generics;

        // convert params
        let mut parsed_params = vec![];
        for param in params {
            parsed_params.push(self.mk_generic_param(*param));
        }

        // convert predicates (FIXME)

        // pack the generics
        self.mk_spanned(span, SolGenerics { params: parsed_params })
    }

    fn mk_struct<'hir>(
        &mut self,
        owner: OwnerId,
        name: Symbol,
        generics: Generics<'hir>,
    ) -> SolStruct {
        // prepare the stack
        self.hir_stack.push(owner);

        // convert generics
        let generics = self.mk_generics(generics);

        // construct the struct after popping the stack
        let last_owner = self.hir_stack.pop();
        assert_eq!(Some(owner), last_owner, "[invariant] HIR stack corrupted when building struct");
        SolStruct { name: SolSymbol(name.to_ident_string()), generics }
    }

    fn mk_lifetime(&mut self, lifetime: Lifetime) -> SolHIR<SolLifetime> {
        let Lifetime { hir_id, ident, kind, source: _, syntax: _ } = lifetime;

        // switch case by name
        let lifetime_data = match kind {
            LifetimeKind::Param(def_id) => SolLifetime::Param(self.mk_ident(def_id.to_def_id())),
            LifetimeKind::Static => SolLifetime::Static,
            LifetimeKind::Infer | LifetimeKind::ImplicitObjectLifetimeDefault => {
                bug!("[invariant] inferred lifetime should not appear in THIR context");
            }
            LifetimeKind::Error => unreachable!(),
        };

        // pack all the information about the lifetime
        self.mk_hir(hir_id, ident.span, lifetime_data)
    }

    fn mk_literal(&mut self, lit: LitKind) -> SolLiteral {
        match lit {
            LitKind::Bool(val) => SolLiteral::Bool(val),
            LitKind::Byte(val) => SolLiteral::Byte(val),
            LitKind::Char(val) => SolLiteral::Char(val),
            LitKind::Str(val, _) => SolLiteral::String(val.to_string()),
            LitKind::CStr(val, _) => SolLiteral::CString(val.as_byte_str().to_vec()),
            LitKind::ByteStr(val, _) => SolLiteral::Bytes(val.as_byte_str().to_vec()),
            LitKind::Int(val, ty) => match ty {
                LitIntType::Signed(IntTy::I8) => SolLiteral::I8(val.get() as i8),
                LitIntType::Signed(IntTy::I16) => SolLiteral::I16(val.get() as i16),
                LitIntType::Signed(IntTy::I32) => SolLiteral::I32(val.get() as i32),
                LitIntType::Signed(IntTy::I64) => SolLiteral::I64(val.get() as i64),
                LitIntType::Signed(IntTy::I128) => SolLiteral::I128(val.get() as i128),
                LitIntType::Signed(IntTy::Isize) => SolLiteral::Isize(val.get() as isize),
                LitIntType::Unsigned(UintTy::U8) => SolLiteral::U8(val.get() as u8),
                LitIntType::Unsigned(UintTy::U16) => SolLiteral::U16(val.get() as u16),
                LitIntType::Unsigned(UintTy::U32) => SolLiteral::U32(val.get() as u32),
                LitIntType::Unsigned(UintTy::U64) => SolLiteral::U64(val.get() as u64),
                LitIntType::Unsigned(UintTy::U128) => SolLiteral::U128(val.get() as u128),
                LitIntType::Unsigned(UintTy::Usize) => SolLiteral::Usize(val.get() as usize),
                LitIntType::Unsuffixed => SolLiteral::Int(val.to_string()),
            },
            LitKind::Float(symbol, ty) => match ty {
                LitFloatType::Suffixed(FloatTy::F64) => SolLiteral::F64(symbol.to_string()),
                LitFloatType::Suffixed(FloatTy::F32) => SolLiteral::F32(symbol.to_string()),
                LitFloatType::Suffixed(FloatTy::F16) => SolLiteral::F16(symbol.to_string()),
                LitFloatType::Suffixed(FloatTy::F128) => SolLiteral::F128(symbol.to_string()),
                LitFloatType::Unsuffixed => SolLiteral::Float(symbol.to_string()),
            },
            LitKind::Err(..) => unreachable!(),
        }
    }

    fn mk_const_arg<'hir>(&mut self, arg: ConstArg<'hir>) -> SolHIR<SolConstArg> {
        let ConstArg { hir_id, kind, span } = arg;

        // switch case by value
        let arg_data = match kind {
            ConstArgKind::Tup(elems) => {
                let mut inner_args = vec![];
                for &elem in elems {
                    inner_args.push(self.mk_const_arg(*elem));
                }
                SolConstArg::Tuple(inner_args)
            }
            ConstArgKind::Literal(lit) => SolConstArg::Literal(self.mk_literal(lit)),

            ConstArgKind::Path(..) | ConstArgKind::Struct(..) | ConstArgKind::TupleCall(..) => {
                todo!()
            }

            ConstArgKind::Anon(..) => bug!("[unsupported] anonymous const arg"),
            ConstArgKind::Infer(..) => {
                bug!("[invariant] inferred const arg should not appear in THIR context");
            }
            ConstArgKind::Error(..) => unreachable!(),
        };

        // pack all the information about the const arg
        self.mk_hir(hir_id, span, arg_data)
    }

    fn mk_ty_pat<'hir>(&mut self, pat: TyPat<'hir>) -> SolHIR<SolTyPat> {
        let TyPat { hir_id, kind, span } = pat;

        // construct the pattern type
        let pat_data = match kind {
            TyPatKind::NotNull => SolTyPat::NotNull,
            TyPatKind::Range(start, end) => {
                SolTyPat::Range(self.mk_const_arg(*start), self.mk_const_arg(*end))
            }
            TyPatKind::Or(pats) => {
                let mut inner_pats = vec![];
                for pat in pats {
                    inner_pats.push(self.mk_ty_pat(*pat));
                }
                SolTyPat::Or(inner_pats)
            }
            TyPatKind::Err(..) => unreachable!(),
        };

        // pack all the information about the pattern type
        self.mk_hir(hir_id, span, pat_data)
    }

    fn mk_qpath_res_for_type(&mut self, res: Res<HirId>) -> SolHirType {
        match res {
            Res::Def(kind, def_id) => match kind {
                DefKind::Struct
                | DefKind::Union
                | DefKind::Enum
                | DefKind::ForeignTy
                | DefKind::TyAlias
                | DefKind::Trait
                | DefKind::TraitAlias
                | DefKind::AssocTy
                | DefKind::TyParam => SolHirType::Defined(self.mk_ident(def_id)),
                _ => bug!("[invariant] expect type kind in type resolution only"),
            },
            Res::PrimTy(prim_ty) => {
                let converted = match prim_ty {
                    PrimTy::Bool => SolPrimTy::Bool,
                    PrimTy::Int(IntTy::I8) => SolPrimTy::I8,
                    PrimTy::Int(IntTy::I16) => SolPrimTy::I16,
                    PrimTy::Int(IntTy::I32) => SolPrimTy::I32,
                    PrimTy::Int(IntTy::I64) => SolPrimTy::I64,
                    PrimTy::Int(IntTy::I128) => SolPrimTy::I128,
                    PrimTy::Int(IntTy::Isize) => SolPrimTy::Isize,
                    PrimTy::Uint(UintTy::U8) => SolPrimTy::U8,
                    PrimTy::Uint(UintTy::U16) => SolPrimTy::U16,
                    PrimTy::Uint(UintTy::U32) => SolPrimTy::U32,
                    PrimTy::Uint(UintTy::U64) => SolPrimTy::U64,
                    PrimTy::Uint(UintTy::U128) => SolPrimTy::U128,
                    PrimTy::Uint(UintTy::Usize) => SolPrimTy::Usize,
                    PrimTy::Float(FloatTy::F16) => SolPrimTy::F16,
                    PrimTy::Float(FloatTy::F32) => SolPrimTy::F32,
                    PrimTy::Float(FloatTy::F64) => SolPrimTy::F64,
                    PrimTy::Float(FloatTy::F128) => SolPrimTy::F128,
                    PrimTy::Char => SolPrimTy::Char,
                    PrimTy::Str => SolPrimTy::Str,
                };
                SolHirType::Primitive(converted)
            }
            Res::SelfTyParam { .. } => todo!(),
            Res::SelfTyAlias { .. } => todo!(),

            Res::Local(..) | Res::SelfCtor(..) | Res::NonMacroAttr(..) | Res::ToolMod => {
                bug!("[invariant] expect type namespace in type resolution only")
            }
            Res::Err => unreachable!(),
        }
    }

    fn mk_hir_type<'hir>(&mut self, ty: HirTy<'hir>) -> SolHIR<SolHirType> {
        let HirTy { hir_id, span, kind } = ty;

        // switch case by kind
        let data = match kind {
            // basics
            HirTyKind::Never => SolHirType::Never,

            // pattern
            HirTyKind::Pat(base, pat) => {
                SolHirType::Pattern(Box::new(self.mk_hir_type(*base)), self.mk_ty_pat(*pat))
            }

            // resolved
            HirTyKind::Path(path) => {
                let res = self.tcx.typeck(hir_id.owner.def_id).qpath_res(&path, hir_id);
                self.mk_qpath_res_for_type(res)
            }

            // composite
            HirTyKind::Slice(inner_ty) => SolHirType::Slice(Box::new(self.mk_hir_type(*inner_ty))),
            HirTyKind::Tup(tys) => {
                let mut inner_tys = vec![];
                for ty in tys {
                    inner_tys.push(self.mk_hir_type(*ty));
                }
                SolHirType::Tuple(inner_tys)
            }
            HirTyKind::Array(inner_ty, len) => {
                SolHirType::Array(Box::new(self.mk_hir_type(*inner_ty)), self.mk_const_arg(*len))
            }

            // references
            HirTyKind::Ptr(MutTy { ty: inner_ty, mutbl: Mutability::Not }) => {
                SolHirType::ImmPtr(Box::new(self.mk_hir_type(*inner_ty)))
            }
            HirTyKind::Ptr(MutTy { ty: inner_ty, mutbl: Mutability::Mut }) => {
                SolHirType::MutPtr(Box::new(self.mk_hir_type(*inner_ty)))
            }
            HirTyKind::Ref(lifetime, MutTy { ty: inner_ty, mutbl: Mutability::Not }) => {
                SolHirType::ImmRef(
                    self.mk_lifetime(*lifetime),
                    Box::new(self.mk_hir_type(*inner_ty)),
                )
            }
            HirTyKind::Ref(lifetime, MutTy { ty: inner_ty, mutbl: Mutability::Mut }) => {
                SolHirType::MutRef(
                    self.mk_lifetime(*lifetime),
                    Box::new(self.mk_hir_type(*inner_ty)),
                )
            }

            // function pointer
            HirTyKind::FnPtr(FnPtrTy {
                safety,
                abi,
                generic_params,
                decl:
                    FnDecl { inputs, output, c_variadic, implicit_self, lifetime_elision_allowed: _ },
                param_idents,
            }) => {
                if !matches!(implicit_self, ImplicitSelfKind::None) {
                    bug!(
                        "[invariant] function pointer should not have an implicit self in THIR context"
                    );
                }
                if param_idents.len() != inputs.len() {
                    bug!(
                        "[invariant] function pointer param idents length {} does not match inputs length {}",
                        param_idents.len(),
                        inputs.len()
                    );
                }

                // parse abi
                let abi_data = match abi {
                    ExternAbi::Rust => {
                        if *c_variadic {
                            bug!("[invariant] Rust ABI function pointer should not be variadic");
                        }
                        match safety {
                            Safety::Safe => SolExternAbi::Rust { safety: true },
                            Safety::Unsafe => SolExternAbi::Rust { safety: false },
                        }
                    }
                    ExternAbi::C { unwind: _ } => SolExternAbi::C { variadic: *c_variadic },
                    ExternAbi::System { unwind: _ } => {
                        if *c_variadic {
                            bug!("[invariant] System ABI function pointer should not be variadic");
                        }
                        SolExternAbi::System
                    }
                    _ => bug!("[unsupported] function pointer with ABI {:?}", abi),
                };

                // parse generics
                let mut generic_params_data = vec![];
                for param in generic_params.iter() {
                    generic_params_data.push(self.mk_generic_param(*param));
                }

                // parse inputs
                let mut input_tys = vec![];
                for (input_ty, input_ident) in inputs.iter().zip(param_idents.iter()) {
                    input_tys.push((
                        self.mk_hir_type(*input_ty),
                        input_ident.map(|ident| self.mk_symbol(ident)),
                    ));
                }

                // parse output
                let output_ty = match *output {
                    FnRetTy::DefaultReturn(_) => None,
                    FnRetTy::Return(ret_ty) => Some(Box::new(self.mk_hir_type(*ret_ty))),
                };

                // construct the function pointer type
                SolHirType::FnPtr {
                    abi: abi_data,
                    generics: generic_params_data,
                    inputs: input_tys,
                    output: output_ty,
                }
            }

            // inference
            HirTyKind::Infer(..) | HirTyKind::InferDelegation(..) => {
                bug!("[invariant] inferred type should not appear in THIR context");
            }

            // unsupported
            HirTyKind::UnsafeBinder(..) => bug!("[unsupported] unsafe binder type"),

            // unexpected
            HirTyKind::Err(_) => unreachable!(),

            HirTyKind::OpaqueDef(..)
            | HirTyKind::TraitAscription(..)
            | HirTyKind::TraitObject(..) => todo!(),
        };

        // pack all the information about the type
        self.mk_hir(hir_id, span, data)
    }

    fn mk_module(&mut self, owner: OwnerId, name: Symbol, module: Mod<'tcx>) -> SolModule {
        let Mod { spans, item_ids } = module;

        // prepare the stack
        self.hir_stack.push(owner);

        // collect items defined in the module
        let mut items = vec![];

        // iterate over all items in the module
        for item_id in item_ids {
            let item = self.tcx.hir_item(*item_id);
            let item_mir = match item.kind {
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
                    let struct_data = self.mk_struct(item.owner_id, ident.name, *generics);
                    self.mk_mir(item.hir_id(), item.span, SolItem::Struct(struct_data))
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
                ItemKind::Mod(mod_ident, mod_content) => {
                    let module_data = self.mk_module(item.owner_id, mod_ident.name, *mod_content);
                    self.mk_mir(item.hir_id(), item.span, SolItem::Module(module_data))
                }

                // unsupported items
                ItemKind::GlobalAsm { .. } => bug!("[unsupported] global assembly"),
            };

            // make the final item
            items.push(item_mir);
        }

        // construct the module after popping the stack
        let last_owner = self.hir_stack.pop();
        assert_eq!(Some(owner), last_owner, "[invariant] HIR stack corrupted when building module");
        SolModule {
            name: SolSymbol(name.to_ident_string()),
            scope: self.mk_span(spans.inner_span),
            items,
        }
    }

    /// Build the crate
    pub(crate) fn build(mut self) -> SolCrate {
        // recursively build the modules starting from the root module
        let module_data = self.mk_module(
            OwnerId { def_id: CRATE_DEF_ID },
            self.tcx.crate_name(LOCAL_CRATE),
            *self.tcx.hir_root_module(),
        );
        let module_mir = self.mk_mir(CRATE_HIR_ID, DUMMY_SP, module_data);

        // unpack the builder and sanity check
        let Self { tcx: _, src_dir: _, src_cache: _, hir_stack, id_cache } = self;
        assert!(hir_stack.is_empty(), "[invariant] HIR stack is not empty after building crate");

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

/// Anything that has a span without an hir_id
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(bound = "T: Serialize + DeserializeOwned")]
pub(crate) struct SolSpanned<T: SolIR> {
    pub(crate) span: SolSpan,
    pub(crate) data: T,
}

/// The base information associated with anything that has an hir_id (and span)
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(bound = "T: Serialize + DeserializeOwned")]
pub(crate) struct SolHIR<T: SolIR> {
    pub(crate) span: SolSpan,
    pub(crate) doc_comments: Vec<SolDocComment>,
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
    Struct(SolStruct),
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
pub(crate) enum SolGenericParam {
    Lifetime { name: SolLifetimeName },
    Type { name: SolSymbol, default: Option<SolHIR<SolHirType>> },
    Const { name: SolSymbol, ty: SolHIR<SolHirType>, default: Option<SolHIR<SolConstArg>> },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolGenerics {
    pub(crate) params: Vec<SolMIR<SolGenericParam>>,
}

/*
* Type defs
*/

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct SolStruct {
    pub(crate) name: SolSymbol,
    pub(crate) generics: SolSpanned<SolGenerics>,
}

/*
 * Type uses
 */

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolExternAbi {
    C { variadic: bool },
    Rust { safety: bool },
    System,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolLifetime {
    Static,
    Param(SolIdent),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolTyPat {
    NotNull,
    Range(SolHIR<SolConstArg>, SolHIR<SolConstArg>),
    Or(Vec<SolHIR<SolTyPat>>),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolPrimTy {
    Bool,
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
    Char,
    Str,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolHirType {
    Never,
    Primitive(SolPrimTy),
    Defined(SolIdent),
    Slice(Box<SolHIR<SolHirType>>),
    Tuple(Vec<SolHIR<SolHirType>>),
    Array(Box<SolHIR<SolHirType>>, SolHIR<SolConstArg>),
    ImmPtr(Box<SolHIR<SolHirType>>),
    MutPtr(Box<SolHIR<SolHirType>>),
    ImmRef(SolHIR<SolLifetime>, Box<SolHIR<SolHirType>>),
    MutRef(SolHIR<SolLifetime>, Box<SolHIR<SolHirType>>),
    Pattern(Box<SolHIR<SolHirType>>, SolHIR<SolTyPat>),
    FnPtr {
        abi: SolExternAbi,
        generics: Vec<SolMIR<SolGenericParam>>,
        inputs: Vec<(SolHIR<SolHirType>, Option<SolSymbol>)>,
        output: Option<Box<SolHIR<SolHirType>>>,
    },
}

/*
* Constants
*/

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolLiteral {
    Bool(bool),
    Byte(u8),
    Char(char),
    String(String),
    CString(Vec<u8>),
    Bytes(Vec<u8>),
    F16(String),
    F32(String),
    F64(String),
    F128(String),
    Float(String),
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
    Int(String),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) enum SolConstArg {
    Literal(SolLiteral),
    Tuple(Vec<SolHIR<SolConstArg>>),
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
