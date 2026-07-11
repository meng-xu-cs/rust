use std::ffi::OsString;
use std::fs;
use std::process::{Command, Output};

use super::{forced_env_config_args, forced_env_entries, insert_forced_env_config_args};

const KEY: &str = "NLAITESTPROVENANCE";

#[test]
fn forced_env_config_precedes_program_separator() {
    let mut args = vec!["run".into(), "--quiet".into(), "--".into(), "program-argument".into()];
    let forced_env = vec![(KEY.to_owned(), "trusted".to_owned())];
    insert_forced_env_config_args(&mut args, &forced_env);

    let separator = args.iter().position(|arg| arg == "--").unwrap();
    let expected = forced_env_config_args(KEY, "trusted").map(OsString::from);
    assert_eq!(&args[separator - expected.len()..separator], &expected);
    assert_eq!(args[separator + 1], "program-argument");
}

#[test]
fn forced_env_overrides_table_workspace_and_cli_configuration() {
    let output = run_nested_cargo(
        concat!(
            "[env.NLAITESTPROVENANCE]\n",
            "value = \"hostile-workspace\"\n",
            "force = true\n",
            "relative = false\n",
        ),
        &[
            "--config",
            "env.NLAITESTPROVENANCE.value=\"hostile-cli\"",
            "--config",
            "env.NLAITESTPROVENANCE.force=true",
            "--config",
            "env.NLAITESTPROVENANCE.relative=false",
        ],
        &"b".repeat(64),
    );

    assert_nested_success(&output, &"b".repeat(64));
}

#[test]
fn forced_env_quotes_an_all_digit_value_as_a_toml_string() {
    let trusted = "0".repeat(64);
    let output = run_nested_cargo("", &[], &trusted);
    assert_nested_success(&output, &trusted);
}

#[test]
fn incompatible_scalar_configuration_fails_before_compilation() {
    let output =
        run_nested_cargo("[env]\nNLAITESTPROVENANCE = \"hostile\"\n", &[], &"c".repeat(64));

    assert!(!output.status.success(), "scalar configuration unexpectedly compiled the crate");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("failed to merge") || stderr.contains("expected a table"),
        "Cargo did not report the expected fail-closed configuration merge error:\n{stderr}"
    );
}

#[cfg(not(windows))]
#[test]
fn forced_env_has_the_literal_freshness_key() {
    assert_eq!(forced_env_entries(KEY, "trusted"), vec![(KEY.to_owned(), "trusted".to_owned())]);
}

#[cfg(windows)]
#[test]
fn forced_env_has_literal_and_maximal_windows_keys() {
    use crate::utils::exec::maximal_windows_environment_alias;

    let maximal = maximal_windows_environment_alias(KEY);
    let entries = forced_env_entries(KEY, "trusted");
    assert_eq!(entries.first().unwrap(), &(KEY.to_owned(), "trusted".to_owned()));
    if maximal == KEY {
        assert_eq!(entries.len(), 1);
    } else {
        assert_eq!(entries.last().unwrap(), &(maximal, "trusted".to_owned()));
        assert_eq!(entries.len(), 2);
    }
}

fn run_nested_cargo(workspace_config: &str, hostile_cli: &[&str], trusted: &str) -> Output {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir_all(root.path().join("src")).unwrap();
    fs::create_dir_all(root.path().join(".cargo")).unwrap();
    fs::create_dir_all(root.path().join("cargo-home")).unwrap();
    fs::write(
        root.path().join("Cargo.toml"),
        "[package]\nname = \"forced-env-test\"\nversion = \"0.0.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    fs::write(
        root.path().join("src/main.rs"),
        "fn main() { print!(\"{}\", env!(\"NLAITESTPROVENANCE\")); }\n",
    )
    .unwrap();
    if !workspace_config.is_empty() {
        fs::write(root.path().join(".cargo/config.toml"), workspace_config).unwrap();
    }

    let mut args = vec!["run".into(), "--quiet".into(), "--offline".into()];
    args.extend(hostile_cli.iter().map(OsString::from));
    args.extend(["--".into(), "program-argument".into()]);
    insert_forced_env_config_args(&mut args, &forced_env_entries(KEY, trusted));

    Command::new(env!("CARGO"))
        .current_dir(root.path())
        .env("CARGO_HOME", root.path().join("cargo-home"))
        .env(KEY, trusted)
        .env_remove("RUSTC_WRAPPER")
        .env_remove("RUSTC_WORKSPACE_WRAPPER")
        .args(args)
        .output()
        .unwrap()
}

fn assert_nested_success(output: &Output, trusted: &str) {
    assert!(
        output.status.success(),
        "nested Cargo failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, trusted.as_bytes());
}
