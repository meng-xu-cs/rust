//! Canonical fingerprinting of the Git source state used to build rustc.
//!
//! The fingerprint is independent of absolute checkout paths and Git's human-facing formatting.
//! Every record is tagged and length-framed, paths come from `git ls-files -z`, and file contents
//! are read without applying attributes or working-tree filters.

use std::collections::BTreeSet;
use std::env;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, Read};
#[cfg(unix)]
use std::os::unix::ffi::{OsStrExt, OsStringExt};
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};

use super::exec::{BootstrapCommand, ExecutionContext, environment_key_has_prefix};
use super::helpers;

pub(crate) const SOURCE_STATE_FINGERPRINT_CONTEXT: &str = "nlai.rust-source-state.blake3.v1";

// Git's public `ls-files --debug` projection prints the complete in-memory `ce_flags` word. Only
// these three bits are persistent logical index state not already represented by the staged-entry
// record: CE_VALID (`assume-unchanged`), CE_INTENT_TO_ADD, and CE_SKIP_WORKTREE. The name length,
// stage, and CE_EXTENDED storage marker are derived from other fields, while bits 16..=28 are
// process-local implementation state such as CE_UPTODATE and CE_FSMONITOR_VALID. Hashing that
// transient state would make one unchanged index acquire different identities after unrelated Git
// operations.
const SEMANTIC_INDEX_FLAGS: u32 = 0x0000_8000 | 0x2000_0000 | 0x4000_0000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SourceStateFingerprint {
    pub(crate) commit: String,
    pub(crate) value: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct IndexEntry {
    raw: Vec<u8>,
    mode: Vec<u8>,
    stage: Vec<u8>,
    path: Vec<u8>,
    flags: u32,
}

#[derive(Clone, Copy)]
enum HeadPolicy {
    RequireCommit,
    AllowUnborn,
}

enum RepositoryHead {
    Commit(Vec<u8>),
    Unborn(Vec<u8>),
}

/// Fingerprint one repository and every initialized Git submodule reachable from its index.
pub(crate) fn fingerprint(
    root: &Path,
    exec_ctx: &ExecutionContext,
) -> Result<SourceStateFingerprint, String> {
    let canonical_root = root
        .canonicalize()
        .map_err(|error| format!("unable to canonicalize source root {root:?}: {error}"))?;
    let mut hasher = blake3::Hasher::new_derive_key(SOURCE_STATE_FINGERPRINT_CONTEXT);
    let mut visited = BTreeSet::new();
    let commit = fingerprint_repository(
        &canonical_root,
        &[],
        HeadPolicy::RequireCommit,
        exec_ctx,
        &mut visited,
        &mut hasher,
    )?
    .ok_or_else(|| "top-level Rust repository unexpectedly has an unborn HEAD".to_owned())?;
    Ok(SourceStateFingerprint {
        commit: String::from_utf8(commit)
            .expect("canonical hexadecimal Git commit must always be valid UTF-8"),
        value: hasher.finalize().to_hex().to_string(),
    })
}

fn fingerprint_repository(
    root: &Path,
    scope: &[u8],
    head_policy: HeadPolicy,
    exec_ctx: &ExecutionContext,
    visited: &mut BTreeSet<PathBuf>,
    hasher: &mut blake3::Hasher,
) -> Result<Option<Vec<u8>>, String> {
    let canonical_root = root
        .canonicalize()
        .map_err(|error| format!("unable to canonicalize repository {root:?}: {error}"))?;
    verify_canonical_worktree(&canonical_root, exec_ctx)?;
    if !visited.insert(canonical_root.clone()) {
        return Err(format!("source-state recursion revisited repository {canonical_root:?}"));
    }

    frame(hasher, b"repository", scope);
    let head = repository_head(&canonical_root, head_policy, exec_ctx)?;
    match &head {
        RepositoryHead::Commit(commit) => frame(hasher, b"head", commit),
        RepositoryHead::Unborn(reference) => frame(hasher, b"head-unborn", reference),
    }
    let (object_format, object_hash_length) =
        repository_object_format(&canonical_root, &head, exec_ctx)?;
    frame(hasher, b"object-format", object_format.as_bytes());

    let index_output = git_output(
        &canonical_root,
        &["ls-files", "--cached", "--stage", "--debug", "--full-name", "--no-abbrev", "-z"],
        exec_ctx,
    )?;
    // This is the complete logical entry projection: every path/stage, object identity, Git mode,
    // and persisted entry flag. `--debug` is parsed but its host- and time-dependent stat fields are
    // deliberately discarded. Hashing the index file itself would also include cache-tree,
    // fsmonitor, split-index, and untracked-cache storage details that do not alter this mapping.
    let mut index_entries = parse_index_entries(&index_output, object_hash_length)?;
    index_entries.sort_by(|left, right| {
        (&left.path, &left.stage, &left.mode, &left.raw).cmp(&(
            &right.path,
            &right.stage,
            &right.mode,
            &right.raw,
        ))
    });
    let mut index_slots = BTreeSet::new();
    let mut worktree_paths = BTreeSet::<Vec<u8>>::new();
    let mut submodules = BTreeSet::<Vec<u8>>::new();
    for entry in &index_entries {
        if !index_slots.insert((entry.path.clone(), entry.stage.clone())) {
            return Err(format!(
                "Git index contains duplicate stage {:?} for path {:?}",
                String::from_utf8_lossy(&entry.stage),
                entry.path
            ));
        }
        framed_scoped_record(hasher, b"index", scope, &entry.raw);
        frame(hasher, b"index-flags", &entry.flags.to_le_bytes());
        worktree_paths.insert(entry.path.clone());
        if entry.mode == b"160000" {
            submodules.insert(entry.path.clone());
        }
    }

    let resolve_undo_output = git_output(
        &canonical_root,
        &["ls-files", "--resolve-undo", "--full-name", "--no-abbrev", "-z"],
        exec_ctx,
    )?;
    let mut resolve_undo_entries =
        parse_stage_entries(&resolve_undo_output, object_hash_length, "resolve-undo index")?;
    resolve_undo_entries.sort_by(|left, right| {
        (&left.path, &left.stage, &left.mode, &left.raw).cmp(&(
            &right.path,
            &right.stage,
            &right.mode,
            &right.raw,
        ))
    });
    let mut resolve_undo_slots = BTreeSet::new();
    for entry in resolve_undo_entries {
        if !resolve_undo_slots.insert((entry.path.clone(), entry.stage.clone())) {
            return Err(format!(
                "resolve-undo index contains duplicate stage {:?} for path {:?}",
                String::from_utf8_lossy(&entry.stage),
                entry.path
            ));
        }
        framed_scoped_record(hasher, b"resolve-undo", scope, &entry.raw);
    }

    for path in worktree_paths {
        if !submodules.contains(&path) {
            fingerprint_worktree_path(&canonical_root, scope, b"tracked", &path, false, hasher)?;
        }
    }

    let untracked_output = git_output(
        &canonical_root,
        &["ls-files", "--others", "--exclude-standard", "--full-name", "--no-directory", "-z"],
        exec_ctx,
    )?;
    let mut untracked_paths = parse_nul_records(&untracked_output, "untracked path list")?;
    untracked_paths.sort_unstable();
    if let Some(path) = adjacent_duplicate(&untracked_paths) {
        return Err(format!("Git returned duplicate untracked path {path:?}"));
    }
    for path in untracked_paths {
        if let Some(repository_path) = path.strip_suffix(b"/") {
            fingerprint_untracked_repository(
                &canonical_root,
                scope,
                repository_path,
                exec_ctx,
                visited,
                hasher,
            )?;
        } else {
            fingerprint_worktree_path(&canonical_root, scope, b"untracked", path, true, hasher)?;
        }
    }

    for submodule_path in submodules {
        fingerprint_submodule(&canonical_root, scope, &submodule_path, exec_ctx, visited, hasher)?;
    }

    visited.remove(&canonical_root);
    frame(hasher, b"repository-end", scope);
    Ok(match head {
        RepositoryHead::Commit(commit) => Some(commit),
        RepositoryHead::Unborn(_) => None,
    })
}

fn verify_canonical_worktree(root: &Path, exec_ctx: &ExecutionContext) -> Result<(), String> {
    let inside = git_output(root, &["rev-parse", "--is-inside-work-tree"], exec_ctx)?;
    if !matches!(inside.as_slice(), b"true\n" | b"true") {
        return Err(format!(
            "canonical source {root:?} is not inside the Git worktree selected from that directory: {:?}",
            String::from_utf8_lossy(&inside)
        ));
    }
    // `--show-toplevel` has no NUL-delimited form and emits repository paths verbatim, so a valid
    // checkout path containing LF cannot be distinguished from its record terminator. At the
    // already-canonical candidate directory, an empty raw prefix proves the same root property
    // without parsing any path bytes.
    let prefix = git_output(root, &["rev-parse", "--show-prefix"], exec_ctx)?;
    if !matches!(prefix.as_slice(), b"\n" | b"") {
        return Err(format!(
            "canonical source {root:?} is below Git's top-level worktree with prefix {:?}",
            String::from_utf8_lossy(&prefix)
        ));
    }
    Ok(())
}

fn repository_head(
    root: &Path,
    policy: HeadPolicy,
    exec_ctx: &ExecutionContext,
) -> Result<RepositoryHead, String> {
    match git_output(root, &["rev-parse", "--verify", "HEAD^{commit}"], exec_ctx) {
        Ok(output) => {
            Ok(RepositoryHead::Commit(one_canonical_hash_line(&output, "HEAD commit")?.to_vec()))
        }
        Err(commit_error) if matches!(policy, HeadPolicy::AllowUnborn) => {
            match unborn_head_reference(root, exec_ctx)? {
                Some(reference) => Ok(RepositoryHead::Unborn(reference)),
                None => Err(commit_error),
            }
        }
        Err(error) => Err(error),
    }
}

fn unborn_head_reference(
    root: &Path,
    exec_ctx: &ExecutionContext,
) -> Result<Option<Vec<u8>>, String> {
    let symbolic_output = match git_output(root, &["symbolic-ref", "--quiet", "HEAD"], exec_ctx) {
        Ok(output) => output,
        Err(_) => return Ok(None),
    };
    let symbolic = one_nonempty_line(&symbolic_output, "symbolic HEAD")?;
    let references = git_output(root, &["for-each-ref", "--format=%(refname)"], exec_ctx)?;
    if references
        .split(|byte| *byte == b'\n')
        .filter(|reference| !reference.is_empty())
        .any(|reference| reference == symbolic)
    {
        return Ok(None);
    }
    Ok(Some(symbolic.to_vec()))
}

fn repository_object_format(
    root: &Path,
    head: &RepositoryHead,
    exec_ctx: &ExecutionContext,
) -> Result<(String, usize), String> {
    match head {
        // A canonical object name is encoded in the repository's storage object format. Inferring
        // that format from a committed HEAD avoids `rev-parse --show-object-format`, which is not
        // available in Git 1.8.3 on Rust's supported CentOS 7 dist builders.
        RepositoryHead::Commit(commit) => object_format_from_name_length(root, commit.len()),
        RepositoryHead::Unborn(_) => {
            // An unborn repository has no object name to inspect. SHA-1 is Git's historical
            // default; SHA-256 repositories record their non-default storage format in the
            // repository extension. `git config --get-all` and NUL output predate Git 1.8.3.
            let configured = git_optional_config(root, "extensions.objectFormat", exec_ctx)?;
            match configured.as_deref() {
                None | Some(b"sha1") => Ok(("sha1".to_owned(), 40)),
                Some(b"sha256") => Ok(("sha256".to_owned(), 64)),
                Some(format) => Err(format!(
                    "repository {root:?} reports unsupported object format {:?}",
                    String::from_utf8_lossy(format)
                )),
            }
        }
    }
}

fn object_format_from_name_length(root: &Path, length: usize) -> Result<(String, usize), String> {
    match length {
        40 => Ok(("sha1".to_owned(), 40)),
        64 => Ok(("sha256".to_owned(), 64)),
        _ => Err(format!(
            "repository {root:?} returned a canonical HEAD with unsupported hexadecimal length {length}"
        )),
    }
}

fn git_optional_config(
    root: &Path,
    key: &str,
    exec_ctx: &ExecutionContext,
) -> Result<Option<Vec<u8>>, String> {
    let args = ["config", "--null", "--get-all", key];
    let mut command = source_git_command(root);
    command.uncached().run_in_dry_run().args(args);
    let output = command.run_capture(exec_ctx);
    if output.is_success() {
        if !output.stderr_bytes().is_empty() {
            return Err(format!(
                "git {} reported diagnostics while fingerprinting {root:?}: {}",
                args.join(" "),
                String::from_utf8_lossy(output.stderr_bytes()).trim_end()
            ));
        }
        let records = parse_nul_records(output.stdout_bytes(), "Git configuration value")?;
        return match records.as_slice() {
            [value] => Ok(Some(value.to_vec())),
            _ => Err(format!(
                "repository {root:?} contains {} values for required-singleton Git setting {key}",
                records.len()
            )),
        };
    }

    if output.status().and_then(|status| status.code()) == Some(1)
        && output.stdout_bytes().is_empty()
        && output.stderr_bytes().is_empty()
    {
        return Ok(None);
    }
    Err(format!(
        "git {} failed in {root:?}: {}",
        args.join(" "),
        String::from_utf8_lossy(output.stderr_bytes()).trim_end()
    ))
}

fn git_output(root: &Path, args: &[&str], exec_ctx: &ExecutionContext) -> Result<Vec<u8>, String> {
    let mut command = source_git_command(root);
    // Source identity is a read-only prerequisite, including for bootstrap's supported dry-run
    // planning mode. A synthetic empty `CommandOutput` must never be parsed as provenance.
    command.uncached().run_in_dry_run().args(args);
    let output = command.run_capture(exec_ctx);
    checked_git_stdout(
        root,
        args,
        output.is_failure(),
        output.stdout_bytes(),
        output.stderr_bytes(),
    )
}

fn checked_git_stdout(
    root: &Path,
    args: &[&str],
    failed: bool,
    stdout: &[u8],
    stderr: &[u8],
) -> Result<Vec<u8>, String> {
    if failed {
        return Err(format!(
            "git {} failed in {root:?}: {}",
            args.join(" "),
            String::from_utf8_lossy(stderr).trim_end()
        ));
    }
    if !stderr.is_empty() {
        return Err(format!(
            "git {} reported diagnostics while fingerprinting {root:?}: {}",
            args.join(" "),
            String::from_utf8_lossy(stderr).trim_end()
        ));
    }
    Ok(stdout.to_vec())
}

fn source_git_command(root: &Path) -> BootstrapCommand {
    let mut command = helpers::git(Some(root)).allow_failure();
    isolate_git_environment(&mut command);
    // `safe.directory` is a protected Git setting. Removing ambient Git configuration without
    // replacing it would make a checkout that its owner explicitly trusted unusable in mounted
    // CI/container workspaces. Trust only the already-canonical repository being attested.
    let mut safe_directory = OsString::from("safe.directory=");
    safe_directory.push(root.as_os_str());
    let mut excludes_file = OsString::from("core.excludesFile=");
    excludes_file.push(if cfg!(windows) { "NUL" } else { "/dev/null" });
    command
        .arg("-c")
        .arg(safe_directory)
        // `GIT_CONFIG_GLOBAL=/dev/null` does not disable the default
        // `$XDG_CONFIG_HOME/git/ignore`, and repository configuration can name another global
        // excludes file. Neither ambient source may decide which worktree bytes are attested.
        .arg("-c")
        .arg(excludes_file)
        // Read-only provenance probes must not execute a repository-configured fsmonitor hook or
        // trust a cached untracked-directory answer maintained outside this snapshot. Use an empty
        // fsmonitor value rather than `false`: Git before 2.36 interpreted `false` as a hook path.
        // Enumerate paths bytewise even if a repository was copied from a case-folding or Unicode-
        // normalizing filesystem; otherwise Git can hide a distinct untracked compiler input.
        // Suppress the one expected advisory caused by projecting a sparse index into complete
        // logical entries; every remaining stderr byte is a fail-closed provenance diagnostic.
        .args([
            "-c",
            "core.ignoreCase=false",
            "-c",
            "core.precomposeUnicode=false",
            "-c",
            "core.fsmonitor=",
            "-c",
            "core.untrackedCache=false",
            "-c",
            "advice.sparseIndexExpanded=false",
        ]);
    command
}

fn isolate_git_environment(command: &mut BootstrapCommand) {
    for (name, _) in env::vars_os() {
        if environment_key_has_prefix(&name, "GIT_") {
            command.env_remove(name);
        }
    }
    command
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", if cfg!(windows) { "NUL" } else { "/dev/null" })
        // Git before 2.32 ignores `GIT_CONFIG_GLOBAL`. Point both historical global-config roots
        // at the platform null device as well, so supported legacy dist builders cannot read
        // ambient `~/.gitconfig`, XDG config, or includes reached from either file.
        .env("HOME", if cfg!(windows) { "NUL" } else { "/dev/null" })
        .env("XDG_CONFIG_HOME", if cfg!(windows) { "NUL" } else { "/dev/null" })
        .env("GIT_NO_REPLACE_OBJECTS", "1")
        .env("GIT_OPTIONAL_LOCKS", "0")
        .env("LC_ALL", "C")
        .env("LANG", "C");
}

fn one_canonical_hash_line<'a>(output: &'a [u8], description: &str) -> Result<&'a [u8], String> {
    let value = one_nonempty_line(output, description)?;
    if value.contains(&b'\n')
        || !matches!(value.len(), 40 | 64)
        || !value.iter().all(u8::is_ascii_hexdigit)
        || value.iter().any(u8::is_ascii_uppercase)
    {
        return Err(format!(
            "git returned noncanonical {description}: {:?}",
            String::from_utf8_lossy(value)
        ));
    }
    Ok(value)
}

fn one_nonempty_line<'a>(output: &'a [u8], description: &str) -> Result<&'a [u8], String> {
    let value = output.strip_suffix(b"\n").unwrap_or(output);
    if value.is_empty() || value.contains(&b'\n') || value.contains(&0) {
        return Err(format!(
            "git returned noncanonical {description}: {:?}",
            String::from_utf8_lossy(value)
        ));
    }
    Ok(value)
}

fn parse_index_entries(
    output: &[u8],
    object_hash_length: usize,
) -> Result<Vec<IndexEntry>, String> {
    let mut entries = Vec::new();
    let mut remaining = output;
    while !remaining.is_empty() {
        let nul = remaining
            .iter()
            .position(|byte| *byte == 0)
            .ok_or_else(|| "Git index entry is not NUL-terminated".to_owned())?;
        let raw = &remaining[..nul];
        remaining = &remaining[nul + 1..];
        let (flags, consumed) = parse_index_debug_block(remaining)?;
        remaining = &remaining[consumed..];
        entries.push(parse_stage_entry(raw, object_hash_length, canonical_index_flags(flags))?);
    }
    Ok(entries)
}

fn canonical_index_flags(flags: u32) -> u32 {
    flags & SEMANTIC_INDEX_FLAGS
}

fn parse_stage_entries(
    output: &[u8],
    object_hash_length: usize,
    description: &str,
) -> Result<Vec<IndexEntry>, String> {
    parse_nul_records(output, description)?
        .into_iter()
        .map(|raw| parse_stage_entry(raw, object_hash_length, 0))
        .collect()
}

fn parse_stage_entry(
    raw: &[u8],
    object_hash_length: usize,
    flags: u32,
) -> Result<IndexEntry, String> {
    let tab = raw
        .iter()
        .position(|byte| *byte == b'\t')
        .ok_or_else(|| "Git index entry has no path separator".to_owned())?;
    let metadata = &raw[..tab];
    let path = &raw[tab + 1..];
    if path.is_empty() {
        return Err("Git index entry has an empty path".to_owned());
    }
    let mut fields = metadata.split(|byte| *byte == b' ');
    let mode = fields.next().unwrap_or_default();
    let object = fields.next().unwrap_or_default();
    let stage = fields.next().unwrap_or_default();
    if fields.next().is_some()
        || mode.len() != 6
        || !mode.iter().all(|byte| matches!(byte, b'0'..=b'7'))
        || object.len() != object_hash_length
        || !object.iter().all(u8::is_ascii_hexdigit)
        || object.iter().any(u8::is_ascii_uppercase)
        || !matches!(stage, b"0" | b"1" | b"2" | b"3")
    {
        return Err(format!(
            "Git index entry has noncanonical metadata {:?}",
            String::from_utf8_lossy(metadata)
        ));
    }
    Ok(IndexEntry {
        raw: raw.to_vec(),
        mode: mode.to_vec(),
        stage: stage.to_vec(),
        path: path.to_vec(),
        flags,
    })
}

fn parse_index_debug_block(output: &[u8]) -> Result<(u32, usize), String> {
    const PREFIXES: [&[u8]; 5] = [b"  ctime: ", b"  mtime: ", b"  dev: ", b"  uid: ", b"  size: "];
    let mut consumed = 0;
    let mut flags = None;
    for (index, prefix) in PREFIXES.into_iter().enumerate() {
        let rest = &output[consumed..];
        let newline = rest
            .iter()
            .position(|byte| *byte == b'\n')
            .ok_or_else(|| "Git index debug record is not newline-terminated".to_owned())?;
        let line = &rest[..newline];
        if !line.starts_with(prefix) {
            return Err(format!(
                "Git index debug record has unexpected line {:?}",
                String::from_utf8_lossy(line)
            ));
        }
        if index == PREFIXES.len() - 1 {
            let separator = b"\tflags: ";
            let position = line
                .windows(separator.len())
                .position(|window| window == separator)
                .ok_or_else(|| "Git index debug record has no flags field".to_owned())?;
            let value = &line[position + separator.len()..];
            if value.is_empty() || value.len() > 8 || !value.iter().all(u8::is_ascii_hexdigit) {
                return Err(format!("Git index entry has noncanonical flags {value:?}"));
            }
            flags = Some(
                u32::from_str_radix(
                    std::str::from_utf8(value)
                        .expect("ASCII hexadecimal index flags must be valid UTF-8"),
                    16,
                )
                .expect("validated hexadecimal index flags must fit in u32"),
            );
        }
        consumed += newline + 1;
    }
    Ok((flags.expect("flags parsed from final debug line"), consumed))
}

fn adjacent_duplicate<'a>(records: &[&'a [u8]]) -> Option<&'a [u8]> {
    records.windows(2).find_map(|pair| (pair[0] == pair[1]).then_some(pair[0]))
}

fn fingerprint_submodule(
    root: &Path,
    scope: &[u8],
    git_path: &[u8],
    exec_ctx: &ExecutionContext,
    visited: &mut BTreeSet<PathBuf>,
    hasher: &mut blake3::Hasher,
) -> Result<(), String> {
    let child_root = join_git_path(root, git_path)?;
    let child_scope = scoped_path(scope, git_path);
    let metadata = match fs::symlink_metadata(&child_root) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            frame(hasher, b"submodule-missing", &child_scope);
            return Ok(());
        }
        Err(error) => {
            return Err(format!("unable to inspect submodule path {child_root:?}: {error}"));
        }
    };

    // A type-changing conflict can leave a regular file or symlink at a path for which one index
    // stage is a gitlink. That worktree object is still a build input and must not be discarded.
    if !metadata.is_dir() {
        return fingerprint_worktree_path(
            root,
            scope,
            b"tracked-gitlink-conflict",
            git_path,
            false,
            hasher,
        );
    }

    match fs::symlink_metadata(child_root.join(".git")) {
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            let mut entries = fs::read_dir(&child_root).map_err(|error| {
                format!("unable to enumerate uninitialized submodule {child_root:?}: {error}")
            })?;
            if entries
                .next()
                .transpose()
                .map_err(|error| {
                    format!("unable to enumerate uninitialized submodule {child_root:?}: {error}")
                })?
                .is_some()
            {
                return Err(format!(
                    "gitlink path {child_root:?} is a nonempty directory without an initialized .git repository"
                ));
            }
            frame(hasher, b"submodule-missing", &child_scope);
            return Ok(());
        }
        Err(error) => {
            return Err(format!(
                "unable to inspect submodule repository marker {:?}: {error}",
                child_root.join(".git")
            ));
        }
    }

    let inside = git_output(&child_root, &["rev-parse", "--is-inside-work-tree"], exec_ctx)?;
    if !matches!(inside.as_slice(), b"true\n" | b"true") {
        return Err(format!(
            "initialized submodule {child_root:?} is not a Git worktree: {:?}",
            String::from_utf8_lossy(&inside)
        ));
    }
    let prefix = git_output(&child_root, &["rev-parse", "--show-prefix"], exec_ctx)?;
    if !matches!(prefix.as_slice(), b"\n" | b"") {
        return Err(format!(
            "submodule path {child_root:?} resolved inside an enclosing Git worktree with prefix {:?}",
            String::from_utf8_lossy(&prefix)
        ));
    }
    fingerprint_repository(
        &child_root,
        &child_scope,
        HeadPolicy::RequireCommit,
        exec_ctx,
        visited,
        hasher,
    )
    .map(|_| ())
}

fn fingerprint_untracked_repository(
    root: &Path,
    scope: &[u8],
    git_path: &[u8],
    exec_ctx: &ExecutionContext,
    visited: &mut BTreeSet<PathBuf>,
    hasher: &mut blake3::Hasher,
) -> Result<(), String> {
    let child_root = join_git_path(root, git_path)?;
    let metadata = fs::symlink_metadata(&child_root).map_err(|error| {
        format!("unable to inspect untracked nested repository {child_root:?}: {error}")
    })?;
    if !metadata.is_dir() {
        return Err(format!(
            "Git reported untracked nested repository {child_root:?}, but it is not a directory"
        ));
    }
    fs::symlink_metadata(child_root.join(".git")).map_err(|error| {
        format!(
            "Git reported untracked nested repository {child_root:?}, but its .git marker is unavailable: {error}"
        )
    })?;

    let inside = git_output(&child_root, &["rev-parse", "--is-inside-work-tree"], exec_ctx)?;
    if !matches!(inside.as_slice(), b"true\n" | b"true") {
        return Err(format!(
            "untracked nested repository {child_root:?} is not a Git worktree: {:?}",
            String::from_utf8_lossy(&inside)
        ));
    }
    let prefix = git_output(&child_root, &["rev-parse", "--show-prefix"], exec_ctx)?;
    if !matches!(prefix.as_slice(), b"\n" | b"") {
        return Err(format!(
            "untracked nested repository {child_root:?} resolved inside an enclosing Git worktree with prefix {:?}",
            String::from_utf8_lossy(&prefix)
        ));
    }

    let child_scope = scoped_path(scope, git_path);
    frame(hasher, b"untracked-repository", &child_scope);
    fingerprint_repository(
        &child_root,
        &child_scope,
        HeadPolicy::AllowUnborn,
        exec_ctx,
        visited,
        hasher,
    )
    .map(|_| ())
}

fn parse_nul_records<'a>(output: &'a [u8], description: &str) -> Result<Vec<&'a [u8]>, String> {
    if output.is_empty() {
        return Ok(Vec::new());
    }
    if !output.ends_with(&[0]) {
        return Err(format!("{description} is not NUL-terminated"));
    }
    output[..output.len() - 1]
        .split(|byte| *byte == 0)
        .map(|record| {
            if record.is_empty() {
                Err(format!("{description} contains an empty record"))
            } else {
                Ok(record)
            }
        })
        .collect()
}

fn fingerprint_worktree_path(
    root: &Path,
    scope: &[u8],
    kind: &[u8],
    git_path: &[u8],
    disappearance_is_error: bool,
    hasher: &mut blake3::Hasher,
) -> Result<(), String> {
    let full_path = join_git_path(root, git_path)?;
    let scoped = scoped_path(scope, git_path);
    frame(hasher, b"worktree-kind", kind);
    frame(hasher, b"worktree-path", &scoped);
    let metadata = match fs::symlink_metadata(&full_path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound && !disappearance_is_error => {
            frame(hasher, b"missing", &scoped);
            return Ok(());
        }
        Err(error) => {
            return Err(format!("unable to inspect source path {full_path:?}: {error}"));
        }
    };

    if metadata.file_type().is_symlink() {
        let target = fs::read_link(&full_path)
            .map_err(|error| format!("unable to read source symlink {full_path:?}: {error}"))?;
        let after = fs::symlink_metadata(&full_path)
            .map_err(|error| format!("unable to restat source symlink {full_path:?}: {error}"))?;
        let after_target = fs::read_link(&full_path)
            .map_err(|error| format!("unable to reread source symlink {full_path:?}: {error}"))?;
        if !after.file_type().is_symlink()
            || file_stamp(&metadata) != file_stamp(&after)
            || target != after_target
        {
            return Err(format!("source symlink {full_path:?} changed while being fingerprinted"));
        }
        frame(hasher, b"symlink", &os_bytes(target.as_os_str())?);
        return Ok(());
    }
    if !metadata.is_file() {
        return Err(format!("source path {full_path:?} is neither a regular file nor a symlink"));
    }

    frame(hasher, b"regular-mode", &[u8::from(executable_bit(&metadata))]);
    let mut file = File::open(&full_path)
        .map_err(|error| format!("unable to open source file {full_path:?}: {error}"))?;
    let before = file
        .metadata()
        .map_err(|error| format!("unable to stat open source file {full_path:?}: {error}"))?;
    if !before.is_file() || file_stamp(&metadata) != file_stamp(&before) {
        return Err(format!(
            "source file {full_path:?} changed type or identity while being opened"
        ));
    }
    frame_header(hasher, b"regular-content", before.len());
    let mut read = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("unable to read source file {full_path:?}: {error}"))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
        read = read
            .checked_add(count as u64)
            .ok_or_else(|| format!("source file length overflow for {full_path:?}"))?;
    }
    let after = file
        .metadata()
        .map_err(|error| format!("unable to restat open source file {full_path:?}: {error}"))?;
    let path_after = fs::symlink_metadata(&full_path)
        .map_err(|error| format!("unable to restat source path {full_path:?}: {error}"))?;
    if read != before.len()
        || !path_after.is_file()
        || file_stamp(&before) != file_stamp(&after)
        || file_stamp(&after) != file_stamp(&path_after)
    {
        return Err(format!("source file {full_path:?} changed while being fingerprinted"));
    }
    Ok(())
}

#[cfg(unix)]
fn file_stamp(metadata: &fs::Metadata) -> (u64, u64, u64, i64, i64, i64, i64, u32) {
    (
        metadata.dev(),
        metadata.ino(),
        metadata.len(),
        metadata.mtime(),
        metadata.mtime_nsec(),
        metadata.ctime(),
        metadata.ctime_nsec(),
        metadata.mode(),
    )
}

#[cfg(not(unix))]
fn file_stamp(metadata: &fs::Metadata) -> (u64, Option<std::time::SystemTime>, bool) {
    (metadata.len(), metadata.modified().ok(), metadata.permissions().readonly())
}

#[cfg(unix)]
fn executable_bit(metadata: &fs::Metadata) -> bool {
    metadata.permissions().mode() & 0o111 != 0
}

#[cfg(not(unix))]
fn executable_bit(_metadata: &fs::Metadata) -> bool {
    false
}

fn join_git_path(root: &Path, path: &[u8]) -> Result<PathBuf, String> {
    if path.is_empty()
        || path.starts_with(b"/")
        || path
            .split(|byte| *byte == b'/')
            .any(|part| part.is_empty() || part == b"." || part == b"..")
    {
        return Err(format!("Git returned unsafe repository-relative path {path:?}"));
    }
    let path = PathBuf::from(os_string(path)?);
    let components = path.components().collect::<Vec<_>>();
    if components.iter().any(|component| !matches!(component, Component::Normal(_))) {
        return Err(format!("Git returned non-relative repository path {:?}", path.as_os_str()));
    }

    // `root.join(path)` must never traverse an intermediate symlink. Besides escaping the source
    // root, that would let a tracked path attest one external file while relative includes consume
    // additional external bytes that Git never enumerated. A final-component symlink is safe: its
    // link text is explicitly fingerprinted by `fingerprint_worktree_path`.
    let mut ancestor = root.to_path_buf();
    for component in &components[..components.len() - 1] {
        ancestor.push(component.as_os_str());
        match fs::symlink_metadata(&ancestor) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(format!(
                    "repository path {:?} traverses symlink ancestor {ancestor:?}",
                    path.as_os_str()
                ));
            }
            Ok(metadata) if !metadata.is_dir() => {
                return Err(format!(
                    "repository path {:?} traverses non-directory ancestor {ancestor:?}",
                    path.as_os_str()
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => break,
            Err(error) => {
                return Err(format!(
                    "unable to inspect repository path ancestor {ancestor:?}: {error}"
                ));
            }
        }
    }
    Ok(root.join(path))
}

fn scoped_path(scope: &[u8], path: &[u8]) -> Vec<u8> {
    if scope.is_empty() {
        path.to_vec()
    } else {
        let mut result = Vec::with_capacity(scope.len() + 1 + path.len());
        result.extend_from_slice(scope);
        result.push(b'/');
        result.extend_from_slice(path);
        result
    }
}

#[cfg(unix)]
fn os_string(bytes: &[u8]) -> Result<OsString, String> {
    Ok(OsString::from_vec(bytes.to_vec()))
}

#[cfg(not(unix))]
fn os_string(bytes: &[u8]) -> Result<OsString, String> {
    String::from_utf8(bytes.to_vec())
        .map(OsString::from)
        .map_err(|_| format!("Git returned a non-UTF-8 path on this platform: {bytes:?}"))
}

#[cfg(unix)]
fn os_bytes(value: &std::ffi::OsStr) -> Result<Vec<u8>, String> {
    Ok(value.as_bytes().to_vec())
}

#[cfg(not(unix))]
fn os_bytes(value: &std::ffi::OsStr) -> Result<Vec<u8>, String> {
    value
        .to_str()
        .map(str::as_bytes)
        .map(ToOwned::to_owned)
        .ok_or_else(|| "path is not valid UTF-8 on this platform".to_owned())
}

fn framed_scoped_record(hasher: &mut blake3::Hasher, tag: &[u8], scope: &[u8], value: &[u8]) {
    frame(hasher, tag, scope);
    frame(hasher, b"value", value);
}

fn frame(hasher: &mut blake3::Hasher, tag: &[u8], value: &[u8]) {
    frame_header(hasher, tag, value.len() as u64);
    hasher.update(value);
}

fn frame_header(hasher: &mut blake3::Hasher, tag: &[u8], value_len: u64) {
    hasher.update(&(tag.len() as u64).to_le_bytes());
    hasher.update(tag);
    hasher.update(&value_len.to_le_bytes());
}

#[cfg(test)]
mod tests;
