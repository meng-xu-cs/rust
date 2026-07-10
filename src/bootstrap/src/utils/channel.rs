//! Build configuration for Rust's release channels.
//!
//! Implements the stable/beta/nightly channel distinctions by setting various
//! flags like the `unstable_features`, calculating variables like `release` and
//! `package_vers`, and otherwise indicating to the compiler what it should
//! print out as part of its version information.

use std::fs;
use std::path::Path;

use super::exec::ExecutionContext;
use super::helpers;
use crate::Build;
use crate::utils::helpers::t;

#[derive(Clone, Default)]
pub enum GitInfo {
    /// This is not a git repository.
    #[default]
    Absent,
    /// This is a git repository.
    /// If the info should be used (`omit_git_hash` is false), this will be
    /// `Some`, otherwise it will be `None`.
    Present(Option<Info>),
    /// This is not a git repository, but the info can be fetched from the
    /// `git-commit-info` file.
    RecordedForTarball(Info),
}

#[derive(Clone)]
pub struct Info {
    pub commit_date: String,
    pub sha: String,
    pub short_sha: String,
}

impl GitInfo {
    pub fn new(omit_git_hash: bool, dir: &Path, exec_ctx: impl AsRef<ExecutionContext>) -> GitInfo {
        // See if this even begins to look like a git dir
        if !dir.join(".git").exists() {
            match read_commit_info_file(dir) {
                Some(info) => return GitInfo::RecordedForTarball(info),
                None => return GitInfo::Absent,
            }
        }

        let mut git_command = helpers::git(Some(dir));
        git_command.arg("rev-parse");
        let output = git_command.allow_failure().run_capture(&exec_ctx);

        if output.is_failure() {
            return GitInfo::Absent;
        }

        // If we're ignoring the git info, we don't actually need to collect it, just make sure this
        // was a git repo in the first place.
        if omit_git_hash {
            return GitInfo::Present(None);
        }

        // Ok, let's scrape some info
        // We use the command's spawn API to execute these commands concurrently, which leads to performance improvements.
        let mut git_log_cmd = helpers::git(Some(dir));
        let ver_date = git_log_cmd
            .arg("log")
            .arg("-1")
            .arg("--date=short")
            .arg("--pretty=format:%cd")
            .run_in_dry_run()
            .start_capture_stdout(&exec_ctx);

        let mut git_hash_cmd = helpers::git(Some(dir));
        let ver_hash = git_hash_cmd
            .arg("rev-parse")
            // HEAD may legally name an annotated tag object. Compiler identity is commit identity,
            // so embed the peeled commit exactly like downstream source attestation does.
            .arg("HEAD^{commit}")
            .run_in_dry_run()
            .start_capture_stdout(&exec_ctx);

        let mut git_short_hash_cmd = helpers::git(Some(dir));
        let short_ver_hash = git_short_hash_cmd
            .arg("rev-parse")
            .arg("--short=9")
            .arg("HEAD^{commit}")
            .run_in_dry_run()
            .start_capture_stdout(&exec_ctx);

        GitInfo::Present(Some(Info {
            commit_date: ver_date.wait_for_output(&exec_ctx).stdout().trim().to_string(),
            sha: ver_hash.wait_for_output(&exec_ctx).stdout().trim().to_string(),
            short_sha: short_ver_hash.wait_for_output(&exec_ctx).stdout().trim().to_string(),
        }))
    }

    pub fn info(&self) -> Option<&Info> {
        match self {
            GitInfo::Absent => None,
            GitInfo::Present(info) => info.as_ref(),
            GitInfo::RecordedForTarball(info) => Some(info),
        }
    }

    pub fn sha(&self) -> Option<&str> {
        self.info().map(|s| &s.sha[..])
    }

    pub fn sha_short(&self) -> Option<&str> {
        self.info().map(|s| &s.short_sha[..])
    }

    pub fn commit_date(&self) -> Option<&str> {
        self.info().map(|s| &s.commit_date[..])
    }

    pub fn version(&self, build: &Build, num: &str) -> String {
        let mut version = build.release(num);
        if let Some(inner) = self.info() {
            version.push_str(" (");
            version.push_str(&inner.short_sha);
            version.push(' ');
            version.push_str(&inner.commit_date);
            version.push(')');
        }
        version
    }

    /// Returns whether this directory has a `.git` directory which should be managed by bootstrap.
    pub fn is_managed_git_subrepository(&self) -> bool {
        match self {
            GitInfo::Absent | GitInfo::RecordedForTarball(_) => false,
            GitInfo::Present(_) => true,
        }
    }

    /// Returns whether this is being built from a tarball.
    pub fn is_from_tarball(&self) -> bool {
        match self {
            GitInfo::Absent | GitInfo::Present(_) => false,
            GitInfo::RecordedForTarball(_) => true,
        }
    }
}

/// Read the commit information from the `git-commit-info` file given the
/// project root.
pub fn read_commit_info_file(root: &Path) -> Option<Info> {
    if let Ok(contents) = fs::read_to_string(root.join("git-commit-info")) {
        let mut lines = contents.lines();
        let sha = lines.next();
        let short_sha = lines.next();
        let commit_date = lines.next();
        let info = match (commit_date, sha, short_sha) {
            (Some(commit_date), Some(sha), Some(short_sha)) => Info {
                commit_date: commit_date.to_owned(),
                sha: sha.to_owned(),
                short_sha: short_sha.to_owned(),
            },
            _ => panic!("the `git-commit-info` file is malformed"),
        };
        Some(info)
    } else {
        None
    }
}

/// Write the commit information to the `git-commit-info` file given the project
/// root.
pub fn write_commit_info_file(root: &Path, info: &Info) {
    let commit_info = format!("{}\n{}\n{}\n", info.sha, info.short_sha, info.commit_date);
    t!(fs::write(root.join("git-commit-info"), commit_info));
}

/// Write the commit hash to the `git-commit-hash` file given the project root.
pub fn write_commit_hash_file(root: &Path, sha: &str) {
    t!(fs::write(root.join("git-commit-hash"), sha));
}

#[cfg(test)]
mod tests {
    use super::GitInfo;
    use crate::utils::exec::ExecutionContext;
    use crate::utils::tests::git::git_test;

    #[test]
    fn git_identity_peels_an_annotated_tag_head_to_its_commit() {
        git_test(|git| {
            let expected_commit = git.get_current_commit();
            git.run_git(&["tag", "--annotate", "--message", "fixture", "fixture-tag"]);
            git.run_git(&["symbolic-ref", "HEAD", "refs/tags/fixture-tag"]);
            assert_ne!(git.run_git(&["rev-parse", "HEAD"]), expected_commit);
            let expected_short = git.run_git(&["rev-parse", "--short=9", "HEAD^{commit}"]);

            let context = ExecutionContext::new(0, true);
            let identity = GitInfo::new(false, git.get_path(), &context);
            assert_eq!(identity.sha(), Some(expected_commit.as_str()));
            assert_eq!(identity.sha_short(), Some(expected_short.as_str()));
        });
    }
}
