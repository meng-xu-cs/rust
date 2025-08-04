use tracing::warn;

use crate::solana::context::SolIdent;

#[derive(Debug, Clone, Copy)]
pub(crate) enum BuiltinFunction {
    /// core::intrinsics::abort
    IntrinsicsAbort,
    /// core::intrinsics::cold_path
    IntrinsicsColdPath,
    /// core::intrinsics::raw_eq
    IntrinsicsRawEq,

    /// core::hint::assert_unchecked::precondition_check
    HintAssertPreconditionCheck,
    /// core::alloc::layout::|self|::from_size_align_unchecked::precondition_check
    /// AllocFromSizeAlignPreconditionCheck,
    /// core::ptr::non_null::|self|::new_unchecked::precondition_check
    /// PtrNonNullNewPreconditionCheck,
    /// core::ptr::copy_nonoverlapping::precondition_check
    /// CopyNonOverlappingPreconditionCheck,
    /// alloc::vec::|self|::set_len:::precondition_check
    /// VecSetLenPreconditionCheck,
    #[allow(dead_code)] // TODO: remove this
    X,
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

            Self::HintAssertPreconditionCheck => SolIdent::from_root("core")
                .with_type_ns("hint")
                .with_func_ns("assert_unchecked")
                .with_func_ns("precondition_check"),
            Self::X => SolIdent::from_root("<X>"), // TODO: remove
        }
    }

    fn all_items() -> Vec<Self> {
        vec![
            Self::IntrinsicsAbort,
            Self::IntrinsicsColdPath,
            Self::IntrinsicsRawEq,
            Self::HintAssertPreconditionCheck,
        ]
    }
}

// TODO: remove the mark
#[allow(dead_code)]
impl BuiltinFunction {
    fn string_match(ident: &str, target: &str) -> bool {
        target == "*" || ident == target
    }

    fn ident_match(ident: &SolIdent, target: &SolIdent) -> bool {
        if matches!(target, SolIdent::CrateRoot(symbol) if symbol == "**") {
            return true;
        }
        match (ident, target) {
            (SolIdent::CrateRoot(name), SolIdent::CrateRoot(target_name)) => {
                Self::string_match(name, target_name)
            }
            (
                SolIdent::FuncNs { parent, name },
                SolIdent::FuncNs { parent: target_parent, name: target_name },
            ) => Self::string_match(name, target_name) && Self::ident_match(parent, target_parent),
            (
                SolIdent::TypeNs { parent, name },
                SolIdent::TypeNs { parent: target_parent, name: target_name },
            ) => Self::string_match(name, target_name) && Self::ident_match(parent, target_parent),
            (
                SolIdent::SelfImpl { parent, self_ident },
                SolIdent::SelfImpl { parent: target_parent, self_ident: target_self_ident },
            ) => {
                Self::ident_match(parent, target_parent)
                    && Self::ident_match(self_ident, target_self_ident)
            }
            (
                SolIdent::TraitImpl { parent, trait_ident },
                SolIdent::TraitImpl { parent: target_parent, trait_ident: target_trait_ident },
            ) => {
                Self::ident_match(parent, target_parent)
                    && Self::ident_match(trait_ident, target_trait_ident)
            }
            (SolIdent::Extern { parent }, SolIdent::Extern { parent: target_parent }) => {
                Self::ident_match(parent, target_parent)
            }
            _ => false,
        }
    }

    pub(crate) fn try_to_resolve(ident: &SolIdent) -> Option<Self> {
        warn!("trying to resolve builtin function: {ident:#?}");
        for item in Self::all_items() {
            if Self::ident_match(ident, &item.ident()) {
                return Some(item);
            }
        }
        None
    }
}
