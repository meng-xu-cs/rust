use std::fmt::Display;

use tracing::warn;

use crate::solana::context::SolIdent;

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
    AllocFromSizeAlignPreconditionCheck,
    /// core::ptr::non_null::|self|::new_unchecked::precondition_check
    PtrNonNullNewPreconditionCheck,
    /// core::ptr::copy_nonoverlapping::precondition_check
    CopyNonOverlappingPreconditionCheck,
}

impl BuiltinFunction {
    fn try_to_match(ident: &SolIdent, mut segments: Vec<&'static str>) -> bool {
        let segment = match segments.pop() {
            None => return false,
            Some(s) => s,
        };
        match ident {
            SolIdent::CrateRoot(name) => segments.is_empty() && name == segment,
            SolIdent::FuncNs { parent, name } if name == segment => {
                Self::try_to_match(parent, segments)
            }
            SolIdent::TypeNs { parent, name } if name == segment => {
                Self::try_to_match(parent, segments)
            }
            SolIdent::SelfImpl { parent } if segment == "|self|" => {
                Self::try_to_match(parent, segments)
            }
            _ => false,
        }
    }

    pub(crate) fn try_to_resolve(ident: &SolIdent) -> Option<Self> {
        warn!("trying to resolve builtin function: {ident:#?}");

        if Self::try_to_match(ident, vec!["core", "intrinsics", "abort"]) {
            Some(Self::IntrinsicsAbort)
        } else if Self::try_to_match(ident, vec!["core", "intrinsics", "cold_path"]) {
            Some(Self::IntrinsicsColdPath)
        } else if Self::try_to_match(ident, vec!["core", "intrinsics", "raw_eq"]) {
            Some(Self::IntrinsicsRawEq)
        } else if Self::try_to_match(
            ident,
            vec!["core", "hint", "assert_unchecked", "precondition_check"],
        ) {
            Some(Self::HintAssertPreconditionCheck)
        } else if Self::try_to_match(
            ident,
            vec![
                "core",
                "alloc",
                "layout",
                "|self|",
                "from_size_align_unchecked",
                "precondition_check",
            ],
        ) {
            Some(Self::AllocFromSizeAlignPreconditionCheck)
        } else if Self::try_to_match(
            ident,
            vec!["core", "ptr", "non_null", "|self|", "new_unchecked", "precondition_check"],
        ) {
            Some(Self::PtrNonNullNewPreconditionCheck)
        } else if Self::try_to_match(
            ident,
            vec!["core", "ptr", "copy_nonoverlapping", "precondition_check"],
        ) {
            Some(Self::CopyNonOverlappingPreconditionCheck)
        } else {
            None
        }
    }
}

impl Display for BuiltinFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IntrinsicsAbort => write!(f, "abort"),
            Self::IntrinsicsColdPath => write!(f, "cold_path"),
            Self::IntrinsicsRawEq => write!(f, "raw_eq"),
            Self::HintAssertPreconditionCheck => {
                write!(f, "hint::assert::precondition_check")
            }
            Self::AllocFromSizeAlignPreconditionCheck => {
                write!(f, "alloc::from_size_align::precondition_check")
            }
            Self::PtrNonNullNewPreconditionCheck => {
                write!(f, "non_null::new::precondition_check")
            }
            Self::CopyNonOverlappingPreconditionCheck => {
                write!(f, "copy_nonoverlapping::precondition_check")
            }
        }
    }
}
