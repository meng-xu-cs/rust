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
