use crate::solana::context::{SolGenericArg, SolIdent, SolType};

#[derive(Debug, Clone, Copy)]
pub(crate) enum BuiltinFunction {
    /// core::intrinsics::abort
    IntrinsicsAbort,
    /// core::intrinsics::cold_path
    IntrinsicsColdPath,
    /// core::intrinsics::raw_eq
    IntrinsicsRawEq,

    /// alloc::alloc::Global::alloc_impl
    AllocImpl,
    /// alloc::string::SpecToString::spec_to_string
    SpecToString,

    /// core::hint::assert_unchecked::precondition_check
    HintAssertPreconditionCheck,
    /// core::alloc::layout::Layout::from_size_align_unchecked::precondition_check
    AllocLayoutFromSizeAlignPreconditionCheck,
    /// core::ptr::copy_nonoverlapping::precondition_check
    CopyNonOverlappingPreconditionCheck,
    /// alloc::vec::|Any|::set_len:::precondition_check
    VecSetLenPreconditionCheck,
}

impl BuiltinFunction {
    fn ident(&self) -> SolIdent {
        match self {
            Self::IntrinsicsAbort => {
                SolIdent::from_root("core").with_type_ns("intrinsics").with_func_ns("abort")
            }
            Self::IntrinsicsColdPath => {
                SolIdent::from_root("core").with_type_ns("intrinsics").with_func_ns("cold_path")
            }
            Self::IntrinsicsRawEq => {
                SolIdent::from_root("core").with_type_ns("intrinsics").with_func_ns("raw_eq")
            }

            Self::AllocImpl => SolIdent::from_root("alloc")
                .with_type_ns("alloc")
                .with_type_impl(SolType::Adt(
                    SolIdent::from_root("alloc").with_type_ns("alloc").with_type_ns("Global"),
                    vec![],
                ))
                .with_func_ns("alloc_impl"),
            Self::SpecToString => SolIdent::from_root("alloc")
                .with_type_ns("string")
                .with_trait_impl(
                    SolIdent::from_root("alloc")
                        .with_type_ns("string")
                        .with_type_ns("SpecToString"),
                    vec![],
                )
                .with_func_ns("spec_to_string"),

            Self::HintAssertPreconditionCheck => SolIdent::from_root("core")
                .with_type_ns("hint")
                .with_func_ns("assert_unchecked")
                .with_func_ns("precondition_check"),
            Self::AllocLayoutFromSizeAlignPreconditionCheck => SolIdent::from_root("core")
                .with_type_ns("alloc")
                .with_type_ns("layout")
                .with_type_impl(SolType::Adt(
                    SolIdent::from_root("core")
                        .with_type_ns("alloc")
                        .with_type_ns("layout")
                        .with_type_ns("Layout"),
                    vec![],
                ))
                .with_func_ns("from_size_align_unchecked")
                .with_func_ns("precondition_check"),
            Self::CopyNonOverlappingPreconditionCheck => SolIdent::from_root("core")
                .with_type_ns("ptr")
                .with_func_ns("copy_nonoverlapping")
                .with_func_ns("precondition_check"),
            Self::VecSetLenPreconditionCheck => SolIdent::from_root("alloc")
                .with_type_ns("vec")
                .with_type_impl(SolType::Never)
                .with_func_ns("set_len")
                .with_func_ns("precondition_check"),
        }
    }

    fn all_items() -> Vec<Self> {
        vec![
            Self::IntrinsicsAbort,
            Self::IntrinsicsColdPath,
            Self::IntrinsicsRawEq,
            Self::AllocImpl,
            Self::SpecToString,
            Self::HintAssertPreconditionCheck,
            Self::AllocLayoutFromSizeAlignPreconditionCheck,
            Self::CopyNonOverlappingPreconditionCheck,
            Self::VecSetLenPreconditionCheck,
        ]
    }
}

impl BuiltinFunction {
    fn string_match(ident: &str, target: &str) -> bool {
        target == "*" || ident == target
    }

    fn type_match(ty: &SolType, target: &SolType) -> bool {
        /* FIXME: fuzzy match of idents in types */
        matches!(target, SolType::Never) || ty == target
    }

    fn trait_match(
        ident: &SolIdent,
        ty_args: &[SolGenericArg],
        target_ident: &SolIdent,
        target_ty_args: &[SolGenericArg],
    ) -> bool {
        if matches!(target_ident, SolIdent::CrateRoot(symbol) if symbol.0 == "*") {
            return true;
        }
        Self::ident_match(ident, target_ident) && ty_args == target_ty_args
    }

    fn ident_match(ident: &SolIdent, target: &SolIdent) -> bool {
        if matches!(target, SolIdent::CrateRoot(symbol) if symbol.0 == "**") {
            return true;
        }
        match (ident, target) {
            (SolIdent::CrateRoot(name), SolIdent::CrateRoot(target_name)) => {
                Self::string_match(&name.0, &target_name.0)
            }
            (
                SolIdent::FuncNs { parent, name },
                SolIdent::FuncNs { parent: target_parent, name: target_name },
            ) => {
                Self::string_match(&name.0, &target_name.0)
                    && Self::ident_match(parent, target_parent)
            }
            (
                SolIdent::TypeNs { parent, name },
                SolIdent::TypeNs { parent: target_parent, name: target_name },
            ) => {
                Self::string_match(&name.0, &target_name.0)
                    && Self::ident_match(parent, target_parent)
            }
            (
                SolIdent::TraitImpl { parent, ident, ty_args },
                SolIdent::TraitImpl {
                    parent: target_parent,
                    ident: target_ident,
                    ty_args: target_ty_args,
                },
            ) => {
                Self::trait_match(ident, ty_args, target_ident, target_ty_args)
                    && Self::ident_match(parent, target_parent)
            }
            (
                SolIdent::TypeImpl { parent, ty },
                SolIdent::TypeImpl { parent: target_parent, ty: target_ty },
            ) => Self::type_match(ty, target_ty) && Self::ident_match(parent, target_parent),
            (SolIdent::Extern { parent }, SolIdent::Extern { parent: target_parent }) => {
                Self::ident_match(parent, target_parent)
            }
            _ => false,
        }
    }

    pub(crate) fn try_to_resolve(ident: &SolIdent) -> Option<Self> {
        for item in Self::all_items() {
            if Self::ident_match(ident, &item.ident()) {
                return Some(item);
            }
        }
        None
    }
}
