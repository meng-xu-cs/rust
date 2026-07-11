use std::ffi::OsStr;

use super::environment_key_has_prefix;

#[test]
fn git_environment_prefix_is_ascii_case_insensitive() {
    assert!(environment_key_has_prefix(OsStr::new("gIt_COMMON_DIR"), "GIT_"));
    assert!(!environment_key_has_prefix(OsStr::new("NOT_GIT_DIR"), "GIT_"));
}

#[cfg(windows)]
#[test]
fn windows_git_prefix_and_forced_alias_use_os_unicode_folding() {
    use super::{maximal_windows_environment_alias, windows_environment_keys_equal};

    assert!(environment_key_has_prefix(OsStr::new("GıT_COMMON_DIR"), "GIT_"));

    let canonical = "NLAITESTPROVENANCE";
    let maximal = maximal_windows_environment_alias(canonical);
    assert!(windows_environment_keys_equal(OsStr::new(canonical), OsStr::new(&maximal)));
    assert!(maximal.as_str() >= "NLAıTESTPROVENANCE");
}
