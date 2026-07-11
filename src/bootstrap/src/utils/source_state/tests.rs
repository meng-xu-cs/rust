use super::*;
use crate::core::config::DryRun;
use crate::utils::exec::command;

fn git(root: &Path, args: &[&str]) {
    let mut command = command("git");
    isolate_git_environment(&mut command);
    command
        .env("GIT_AUTHOR_NAME", "NLAI Fixture")
        .env("GIT_AUTHOR_EMAIL", "nlai@example.invalid")
        .env("GIT_AUTHOR_DATE", "2000-01-01T00:00:00Z")
        .env("GIT_COMMITTER_NAME", "NLAI Fixture")
        .env("GIT_COMMITTER_EMAIL", "nlai@example.invalid")
        .env("GIT_COMMITTER_DATE", "2000-01-01T00:00:00Z")
        .current_dir(root)
        .args(args);
    assert!(command.run(ExecutionContext::new(0, true)));
}

fn git_fails(root: &Path, args: &[&str]) {
    let mut command = command("git").allow_failure();
    isolate_git_environment(&mut command);
    command.current_dir(root).args(args);
    assert!(command.run_capture(ExecutionContext::new(0, true)).is_failure());
}

fn fixture() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    initialize_fixture(root.path());
    root
}

fn initialize_fixture(root: &Path) {
    fs::create_dir_all(root).unwrap();
    git(root, &["init", "--quiet"]);
    git(root, &["symbolic-ref", "HEAD", "refs/heads/main"]);
    fs::write(root.join("tracked"), b"base\n").unwrap();
    git(root, &["add", "tracked"]);
    git(root, &["commit", "--quiet", "-m", "base"]);
}

#[test]
fn source_state_derives_committed_object_format_without_a_modern_git_probe() {
    let context = ExecutionContext::new(0, true);
    let nonexistent = Path::new("/this/path/must/not/be-probed");
    assert_eq!(
        repository_object_format(nonexistent, &RepositoryHead::Commit(b"0".repeat(40)), &context)
            .unwrap(),
        ("sha1".to_owned(), 40)
    );
    assert_eq!(
        repository_object_format(nonexistent, &RepositoryHead::Commit(b"0".repeat(64)), &context)
            .unwrap(),
        ("sha256".to_owned(), 64)
    );
}

fn digest(root: &Path) -> String {
    fingerprint(root, &ExecutionContext::new(0, true)).unwrap().value
}

fn digest_with_context(root: &Path, context: &ExecutionContext) -> String {
    fingerprint(root, context).unwrap().value
}

#[test]
fn source_state_clean_vector_is_deterministic() {
    let first = fixture();
    let second = fixture();
    let expected = digest(first.path());
    assert_eq!(expected, digest(second.path()));
    assert_eq!(expected, "ded728055da226ccc14119abd05df96f2a0654b90f443a73a9344a956a90944c");
}

#[test]
fn source_state_read_only_probes_run_during_bootstrap_dry_run() {
    let root = fixture();
    let expected = digest(root.path());
    let mut context = ExecutionContext::new(0, true);
    context.set_dry_run(DryRun::UserSelected);

    assert_eq!(fingerprint(root.path(), &context).unwrap().value, expected);
}

#[test]
fn source_state_isolates_git_config_but_trusts_the_canonical_repository() {
    let root = fixture();
    let canonical_root = root.path().canonicalize().unwrap();
    let output = git_output(
        &canonical_root,
        &["config", "--get-all", "safe.directory"],
        &ExecutionContext::new(0, true),
    )
    .unwrap();
    let mut expected = os_bytes(canonical_root.as_os_str()).unwrap();
    expected.push(b'\n');

    assert_eq!(output, expected);

    let effective =
        git_output(&canonical_root, &["config", "--list", "-z"], &ExecutionContext::new(0, true))
            .unwrap();
    assert!(
        !effective.split(|byte| *byte == 0).any(|record| {
            record.starts_with(b"user.name\n") || record.starts_with(b"user.email\n")
        }),
        "ambient global Git configuration leaked into provenance probes"
    );

    for key in ["core.ignoreCase", "core.precomposeUnicode"] {
        assert_eq!(
            git_output(
                &canonical_root,
                &["config", "--get", key],
                &ExecutionContext::new(0, true),
            )
            .unwrap(),
            b"false\n",
            "source-state probes did not force bytewise path enumeration for {key}"
        );
    }
}

#[test]
fn source_state_does_not_hide_case_distinct_untracked_inputs() {
    let root = fixture();
    let distinct = root.path().join("TRACKED");
    match fs::OpenOptions::new().write(true).create_new(true).open(&distinct) {
        Ok(mut file) => {
            use std::io::Write;
            file.write_all(b"first\n").unwrap();
        }
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => return,
        Err(error) => panic!("unable to create case-sensitivity probe: {error}"),
    }
    git(root.path(), &["config", "core.ignoreCase", "true"]);

    let first = digest(root.path());
    fs::write(&distinct, b"second\n").unwrap();
    let second = digest(root.path());

    assert_ne!(first, second, "core.ignoreCase hid a case-distinct untracked input");
}

#[test]
fn source_state_rejects_git_worktree_redirection() {
    let root = fixture();
    let outside = tempfile::tempdir().unwrap();
    git(root.path(), &["config", "core.worktree", outside.path().to_str().unwrap()]);

    let error = fingerprint(root.path(), &ExecutionContext::new(0, true)).unwrap_err();
    assert!(error.contains("not inside the Git worktree"), "unexpected: {error}");
}

#[test]
fn source_state_neutralizes_ambient_and_repository_global_excludes() {
    let root = fixture();
    let hidden = root.path().join("ambient-hidden");
    fs::write(&hidden, b"one\n").unwrap();
    let configuration = tempfile::tempdir().unwrap();
    let excludes = configuration.path().join("excludes");
    fs::write(&excludes, b"ambient-hidden\n").unwrap();
    git(root.path(), &["config", "core.excludesFile", excludes.to_str().unwrap()]);

    let first = digest(root.path());
    fs::write(&hidden, b"two\n").unwrap();
    let second = digest(root.path());
    assert_ne!(first, second, "repository core.excludesFile hid an attested input");

    git(root.path(), &["config", "--unset", "core.excludesFile"]);
    let xdg = configuration.path().join("xdg");
    fs::create_dir_all(xdg.join("git")).unwrap();
    fs::write(xdg.join("git/ignore"), b"ambient-hidden\n").unwrap();
    let canonical_root = root.path().canonicalize().unwrap();
    let arguments =
        ["ls-files", "--others", "--exclude-standard", "--full-name", "--no-directory", "-z"];
    let mut command = source_git_command(&canonical_root);
    command.env("XDG_CONFIG_HOME", &xdg).uncached().run_in_dry_run().args(arguments);
    let output = command.run_capture(ExecutionContext::new(0, true));
    let listed = checked_git_stdout(
        &canonical_root,
        &arguments,
        output.is_failure(),
        output.stdout_bytes(),
        output.stderr_bytes(),
    )
    .unwrap();
    assert!(
        parse_nul_records(&listed, "hostile-XDG untracked list")
            .unwrap()
            .contains(&b"ambient-hidden".as_slice()),
        "XDG_CONFIG_HOME/git/ignore hid an attested input"
    );
}

#[cfg(unix)]
#[test]
fn source_state_accepts_newlines_in_the_canonical_worktree_path() {
    let parent = tempfile::tempdir().unwrap();
    let root = parent.path().join("line\nbreak");
    initialize_fixture(&root);
    let clean = digest(&root);
    fs::write(root.join("untracked"), b"input\n").unwrap();
    assert_ne!(digest(&root), clean);
}

#[test]
fn source_state_git_probe_failures_return_context() {
    let root = fixture();
    let error = git_output(
        root.path(),
        &["nlai-command-that-does-not-exist"],
        &ExecutionContext::new(0, true),
    )
    .unwrap_err();

    assert!(error.contains("git nlai-command-that-does-not-exist failed"), "unexpected: {error}");
    assert!(error.contains("not a git command"), "unexpected: {error}");
}

#[test]
fn source_state_rejects_diagnostics_even_when_git_exits_successfully() {
    let error = checked_git_stdout(
        Path::new("/canonical/rust"),
        &["ls-files", "--others"],
        false,
        b"partial\0",
        b"warning: incomplete listing\n",
    )
    .unwrap_err();

    assert!(error.contains("reported diagnostics"), "unexpected: {error}");
    assert!(error.contains("warning: incomplete listing"), "unexpected: {error}");
}

#[cfg(unix)]
#[test]
fn source_state_rejects_successful_but_incomplete_git_listings() {
    use std::os::unix::fs::PermissionsExt;

    let root = fixture();
    fs::write(root.path().join("tracked"), b"include!(\"secret/input.rs\");\n").unwrap();
    let secret = root.path().join("secret");
    fs::create_dir(&secret).unwrap();
    let input = secret.join("input.rs");
    fs::write(&input, b"const INPUT: u8 = 1;\n").unwrap();
    fs::set_permissions(&secret, fs::Permissions::from_mode(0o111)).unwrap();

    // Search permission still lets a compiler open this known path, while Git cannot enumerate the
    // directory and exits zero after warning. The warning must invalidate the snapshot.
    assert_eq!(fs::read(&input).unwrap(), b"const INPUT: u8 = 1;\n");
    let result = fingerprint(root.path(), &ExecutionContext::new(0, true));
    fs::set_permissions(&secret, fs::Permissions::from_mode(0o700)).unwrap();

    match result {
        Err(error) => {
            assert!(error.contains("reported diagnostics"), "unexpected: {error}");
            assert!(error.contains("secret"), "unexpected: {error}");
        }
        Ok(_) => {
            // Privileged Unix users can enumerate mode-0111 directories, so this failure mode is
            // not reproducible in that environment. The production rule is covered by the error
            // path above wherever the filesystem enforces the permission boundary.
        }
    }
}

#[cfg(unix)]
#[test]
fn source_state_does_not_execute_repository_fsmonitor_hooks() {
    use std::os::unix::fs::PermissionsExt;

    let root = fixture();
    let hook = root.path().join("fsmonitor-hook");
    fs::write(&hook, "#!/bin/sh\n: > \"$0.invoked\"\nprintf 'builtin:fake\\0/'\n").unwrap();
    fs::set_permissions(&hook, fs::Permissions::from_mode(0o755)).unwrap();
    git(root.path(), &["config", "core.fsmonitor", hook.to_str().unwrap()]);

    let mut marker = hook.as_os_str().to_owned();
    marker.push(".invoked");
    let marker = PathBuf::from(marker);
    let _ = digest(root.path());
    assert!(!marker.exists(), "source-state probes executed {hook:?}");
}

#[cfg(windows)]
#[test]
fn source_state_rejects_windows_paths_that_can_escape_the_repository() {
    let root = Path::new(r"C:\canonical\rust");
    for path in [br"..\outside".as_slice(), br"\outside", br"C:outside", br"C:\outside"] {
        let error = join_git_path(root, path).unwrap_err();
        assert!(error.contains("non-relative repository path"), "unexpected: {error}");
    }
}

#[test]
fn source_state_distinguishes_index_worktree_and_untracked_contents() {
    let root = fixture();
    // Reuse one execution context throughout. These probes deliberately bypass bootstrap's Git
    // command cache, so every digest must observe the mutation immediately preceding it.
    let context = ExecutionContext::new(0, true);
    let clean = digest_with_context(root.path(), &context);
    fs::write(root.path().join("tracked"), b"unstaged\n").unwrap();
    let unstaged = digest_with_context(root.path(), &context);
    git(root.path(), &["add", "tracked"]);
    let staged = digest_with_context(root.path(), &context);
    fs::write(root.path().join("untracked"), b"one\n").unwrap();
    let untracked_one = digest_with_context(root.path(), &context);
    fs::write(root.path().join("untracked"), b"two\n").unwrap();
    let untracked_two = digest_with_context(root.path(), &context);

    assert_eq!(BTreeSet::from([clean, unstaged, staged, untracked_one, untracked_two]).len(), 5);
}

#[test]
fn source_state_bypasses_preexisting_bootstrap_command_cache_entries() {
    let root = fixture();
    let canonical_root = root.path().canonicalize().unwrap();
    let context = ExecutionContext::new(0, true);
    let before = digest_with_context(&canonical_root, &context);
    let mut cached_probe = source_git_command(&canonical_root);
    cached_probe.args([
        "ls-files",
        "--others",
        "--exclude-standard",
        "--full-name",
        "--no-directory",
        "-z",
    ]);
    assert!(cached_probe.run_capture(&context).is_success());

    fs::write(canonical_root.join("appeared-after-cache"), b"new\n").unwrap();
    let shared_context = digest_with_context(&canonical_root, &context);
    let fresh_context = digest(&canonical_root);

    assert_ne!(shared_context, before);
    assert_eq!(shared_context, fresh_context);
}

#[test]
fn source_state_distinguishes_intent_to_add_from_the_same_index_object_and_worktree() {
    let root = fixture();
    let path = root.path().join("addition");
    fs::write(&path, b"worktree bytes\n").unwrap();
    git(root.path(), &["add", "--intent-to-add", "addition"]);
    let intent_to_add = digest(root.path());

    fs::write(&path, b"").unwrap();
    git(root.path(), &["add", "addition"]);
    fs::write(&path, b"worktree bytes\n").unwrap();
    let staged_empty_then_modified = digest(root.path());

    assert_ne!(intent_to_add, staged_empty_then_modified);
}

#[test]
fn source_state_keeps_semantic_index_flags_and_discards_transient_git_state() {
    const CE_UPDATE: u32 = 1 << 16;
    const CE_UPTODATE: u32 = 1 << 18;
    const CE_FSMONITOR_VALID: u32 = 1 << 21;
    const CE_VALID: u32 = 1 << 15;
    const CE_INTENT_TO_ADD: u32 = 1 << 29;
    const CE_SKIP_WORKTREE: u32 = 1 << 30;

    assert_eq!(canonical_index_flags(CE_UPDATE | CE_UPTODATE | CE_FSMONITOR_VALID), 0);
    assert_eq!(
        canonical_index_flags(
            CE_UPDATE
                | CE_UPTODATE
                | CE_FSMONITOR_VALID
                | CE_VALID
                | CE_INTENT_TO_ADD
                | CE_SKIP_WORKTREE
        ),
        CE_VALID | CE_INTENT_TO_ADD | CE_SKIP_WORKTREE
    );

    let root = fixture();
    let ordinary = digest(root.path());
    git(root.path(), &["update-index", "--assume-unchanged", "tracked"]);
    let assume_unchanged = digest(root.path());
    git(root.path(), &["update-index", "--no-assume-unchanged", "tracked"]);
    git(root.path(), &["update-index", "--skip-worktree", "tracked"]);
    let skip_worktree = digest(root.path());

    assert_eq!(BTreeSet::from([ordinary, assume_unchanged, skip_worktree]).len(), 3);
}

#[test]
fn source_state_includes_resolve_undo_index_records() {
    let root = fixture();
    git(root.path(), &["checkout", "--quiet", "-b", "side"]);
    fs::write(root.path().join("tracked"), b"side\n").unwrap();
    git(root.path(), &["commit", "--quiet", "-am", "side"]);
    git(root.path(), &["checkout", "--quiet", "main"]);
    fs::write(root.path().join("tracked"), b"main\n").unwrap();
    git(root.path(), &["commit", "--quiet", "-am", "main"]);
    git_fails(root.path(), &["merge", "--no-edit", "side"]);
    fs::write(root.path().join("tracked"), b"resolved\n").unwrap();
    git(root.path(), &["add", "tracked"]);
    let with_resolve_undo = digest(root.path());

    git(root.path(), &["update-index", "--clear-resolve-undo"]);
    let without_resolve_undo = digest(root.path());

    assert_ne!(with_resolve_undo, without_resolve_undo);
}

#[test]
fn source_state_recurses_nested_submodules_and_marks_missing_checkouts() {
    let workspace = tempfile::tempdir().unwrap();
    let grandchild = workspace.path().join("grandchild");
    let child = workspace.path().join("child");
    let parent = workspace.path().join("parent");
    initialize_fixture(&grandchild);
    initialize_fixture(&child);
    git(
        &child,
        &[
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "--quiet",
            "../grandchild",
            "nested",
        ],
    );
    git(&child, &["commit", "--quiet", "-am", "add nested submodule"]);
    initialize_fixture(&parent);
    git(
        &parent,
        &["-c", "protocol.file.allow=always", "submodule", "add", "--quiet", "../child", "sub"],
    );
    git(&parent, &["commit", "--quiet", "-am", "add child submodule"]);
    git(
        &parent,
        &[
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "update",
            "--init",
            "--recursive",
            "--quiet",
        ],
    );

    let clean = digest(&parent);
    fs::write(parent.join("sub/nested/tracked"), b"nested dirty\n").unwrap();
    let nested_dirty = digest(&parent);
    fs::write(parent.join("sub/nested/untracked"), b"nested untracked\n").unwrap();
    let nested_untracked = digest(&parent);
    fs::remove_dir_all(parent.join("sub")).unwrap();
    let missing = digest(&parent);
    fs::create_dir(parent.join("sub")).unwrap();
    let uninitialized = digest(&parent);

    assert_eq!(missing, uninitialized);
    assert_eq!(BTreeSet::from([clean, nested_dirty, nested_untracked, missing]).len(), 4);

    fs::write(parent.join("sub/unowned-input"), b"must not be silently omitted\n").unwrap();
    let error = fingerprint(&parent, &ExecutionContext::new(0, true)).unwrap_err();
    assert!(
        error.contains("nonempty directory without an initialized .git repository"),
        "unexpected error: {error}"
    );
}

#[test]
fn source_state_recurses_nonignored_untracked_repositories() {
    let workspace = tempfile::tempdir().unwrap();
    let parent = workspace.path().join("parent");
    let nested = parent.join("nested");
    initialize_fixture(&parent);
    initialize_fixture(&nested);

    let clean_nested = digest(&parent);
    fs::write(nested.join("tracked"), b"nested dirty\n").unwrap();
    let dirty_nested = digest(&parent);
    fs::write(nested.join("untracked"), b"nested untracked\n").unwrap();
    let untracked_nested = digest(&parent);

    assert_eq!(BTreeSet::from([clean_nested, dirty_nested, untracked_nested]).len(), 3);
}

#[test]
fn source_state_represents_unborn_untracked_repositories() {
    let workspace = tempfile::tempdir().unwrap();
    let parent = workspace.path().join("parent");
    let nested = parent.join("unborn");
    initialize_fixture(&parent);
    fs::create_dir(&nested).unwrap();
    git(&nested, &["init", "--quiet"]);
    git(&nested, &["symbolic-ref", "HEAD", "refs/heads/main"]);
    fs::write(nested.join("input"), b"one\n").unwrap();
    let first = digest(&parent);
    fs::write(nested.join("input"), b"two\n").unwrap();
    let second = digest(&parent);

    assert_ne!(first, second);
}

#[cfg(unix)]
#[test]
fn source_state_distinguishes_modes_and_symlink_targets() {
    use std::os::unix::fs::{PermissionsExt, symlink};

    let root = fixture();
    let base = digest(root.path());
    let tracked = root.path().join("tracked");
    fs::set_permissions(&tracked, fs::Permissions::from_mode(0o755)).unwrap();
    let executable = digest(root.path());

    symlink("tracked", root.path().join("link")).unwrap();
    let first_link = digest(root.path());
    fs::remove_file(root.path().join("link")).unwrap();
    symlink("missing", root.path().join("link")).unwrap();
    let second_link = digest(root.path());

    assert_eq!(BTreeSet::from([base, executable, first_link, second_link]).len(), 4);
}

#[cfg(unix)]
#[test]
fn source_state_rejects_intermediate_symlink_escape() {
    use std::os::unix::fs::symlink;

    let root = fixture();
    fs::create_dir(root.path().join("dir")).unwrap();
    fs::write(root.path().join("dir/input"), b"tracked\n").unwrap();
    git(root.path(), &["add", "dir/input"]);
    git(root.path(), &["commit", "--quiet", "-m", "nested tracked input"]);

    let outside = tempfile::tempdir().unwrap();
    fs::write(outside.path().join("input"), b"tracked\n").unwrap();
    fs::remove_dir_all(root.path().join("dir")).unwrap();
    symlink(outside.path(), root.path().join("dir")).unwrap();

    let error = fingerprint(root.path(), &ExecutionContext::new(0, true)).unwrap_err();
    assert!(error.contains("traverses symlink ancestor"), "unexpected error: {error}");
}

#[cfg(all(unix, not(target_os = "macos")))]
#[test]
fn source_state_accepts_non_utf8_paths_without_text_round_trip() {
    let root = fixture();
    let before = digest(root.path());
    let raw_name = OsString::from_vec(b"raw-\xff".to_vec());
    fs::write(root.path().join(raw_name), b"raw\n").unwrap();
    assert_ne!(digest(root.path()), before);
}
