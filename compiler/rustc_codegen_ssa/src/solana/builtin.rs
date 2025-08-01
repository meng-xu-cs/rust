use std::fmt::Display;

use crate::solana::context::{SolGenericArg, SolIdent};

pub(crate) enum BuiltinFunction {
    /// deref mut from std::ops
    DerefMut,
}

enum InternalTypeNs {
    DerefMut,
}

impl InternalTypeNs {
    fn try_to_match(ident: &SolIdent, mut segments: Vec<&'static str>) -> bool {
        let segment = match segments.pop() {
            None => return false,
            Some(s) => s,
        };
        match ident {
            SolIdent::CrateRoot(name) => segments.is_empty() && name == segment,
            SolIdent::TypeNs { parent, name } if name == segment => {
                Self::try_to_match(parent, segments)
            }
            _ => false,
        }
    }

    fn try_to_resolve(ident: &SolIdent) -> Option<Self> {
        if Self::try_to_match(ident, vec!["core", "ops", "deref", "DerefMut"]) {
            Some(Self::DerefMut)
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
                match InternalTypeNs::try_to_resolve(trait_ident)? {
                    InternalTypeNs::DerefMut => Some(Self::DerefMut),
                }
            }
            _ => return None,
        }
    }
}

impl BuiltinFunction {
    /// Try to resolve an instance into a builtin function
    pub(crate) fn try_to_resolve(ident: &SolIdent, _ty_args: &[SolGenericArg]) -> Option<Self> {
        match ident {
            SolIdent::FuncNs { parent, name } => {
                if name == "deref_mut" {
                    match InternalTrait::try_to_resolve(parent)? {
                        InternalTrait::DerefMut => return Some(Self::DerefMut),
                    }
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
        }
    }
}
