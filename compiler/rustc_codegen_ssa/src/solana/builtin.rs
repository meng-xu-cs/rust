use std::fmt::Display;

use crate::solana::context::{SolGenericArg, SolIdent};

pub(crate) enum BuiltinFunction {
    /// deref mut from std::ops
    DerefMut,
    /// core::intrinsics::abort
    IntrinsicsAbort,
    /// core::intrinsics::cold_path
    IntrinsicsColdPath,
    /// core::hint::assert_unchecked::precondition_check
    HintAssertPreconditionCheck,
}

enum InternalNs {
    DerefMut,
    IntrinsicsAbort,
    IntrinsicsColdPath,
    HintAssertPreconditionCheck,
}

impl InternalNs {
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
            _ => false,
        }
    }

    fn try_to_resolve(ident: &SolIdent) -> Option<Self> {
        if Self::try_to_match(ident, vec!["core", "ops", "deref", "DerefMut"]) {
            Some(Self::DerefMut)
        } else if Self::try_to_match(ident, vec!["core", "intrinsics", "abort"]) {
            Some(Self::IntrinsicsAbort)
        } else if Self::try_to_match(ident, vec!["core", "intrinsics", "cold_path"]) {
            Some(Self::IntrinsicsColdPath)
        } else if Self::try_to_match(
            ident,
            vec!["core", "hint", "assert_unchecked", "precondition_check"],
        ) {
            Some(Self::HintAssertPreconditionCheck)
        } else {
            None
        }
    }
}

enum InternalTrait {
    DerefMut,
}

impl InternalTrait {
    fn try_to_resolve(ident: &SolIdent) -> Option<Self> {
        match ident {
            SolIdent::TraitImpl { parent: _, trait_ident } => {
                match InternalNs::try_to_resolve(trait_ident)? {
                    InternalNs::DerefMut => Some(Self::DerefMut),
                    _ => return None,
                }
            }
            _ => return None,
        }
    }
}

impl BuiltinFunction {
    /// Try to resolve an instance into a builtin function
    pub(crate) fn try_to_resolve(ident: &SolIdent, _ty_args: &[SolGenericArg]) -> Option<Self> {
        if let Some(internal_ns) = InternalNs::try_to_resolve(ident) {
            let resolved = match internal_ns {
                InternalNs::IntrinsicsAbort => Self::IntrinsicsAbort,
                InternalNs::IntrinsicsColdPath => Self::IntrinsicsColdPath,
                InternalNs::HintAssertPreconditionCheck => Self::HintAssertPreconditionCheck,
                InternalNs::DerefMut => return None,
            };
            return Some(resolved);
        }

        match ident {
            SolIdent::FuncNs { parent, name } => {
                match name.as_str() {
                    "deref_mut" => {
                        if matches!(InternalTrait::try_to_resolve(parent)?, InternalTrait::DerefMut)
                        {
                            return Some(Self::DerefMut);
                        }
                    }
                    _ => return None,
                }
                return None;
            }
            _ => return None,
        }
    }
}

impl Display for BuiltinFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DerefMut => write!(f, "deref_mut"),
            Self::IntrinsicsAbort => write!(f, "abort"),
            Self::IntrinsicsColdPath => write!(f, "cold_path"),
            Self::HintAssertPreconditionCheck => write!(f, "precondition_check"),
        }
    }
}
