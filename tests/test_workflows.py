"""The three workflows, read as YAML. No GitHub is involved; these tests keep the pipeline's SHAPE true:

  * ingest.yml: build -> eval-gate -> publish -> promote-dev -> sit -> pre-prod (approval) -> prod (approval)
  * the gate sits before publish and publish consumes the EVALUATED index
  * every environment promote job has its own environment; build/eval/publish have none (OIDC sub = ref:refs/heads/main)
  * no secret is echoed or pasted into a script; every AWS login masks the account id
"""

import re
from pathlib import Path

import pytest
import yaml

from tests.conftest import GENERATED_DIRS, repo_files

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / ".github" / "workflows"
NAMES = ("kb-ci", "ingest", "promote-index")
PINNED = {
    "actions/checkout": "v7", "actions/setup-python": "v7", "actions/upload-artifact": "v7",
    "actions/download-artifact": "v8", "aws-actions/configure-aws-credentials": "v6",
}


def load(name):
    data = yaml.safe_load((WF / f"{name}.yml").read_text())
    data["on"] = data.pop(True, data.get("on"))                  # YAML reads a bare `on:` as the boolean True
    return data


def steps(job):
    return job.get("steps", [])


def run_scripts(workflow):
    return [s["run"] for j in workflow["jobs"].values() for s in steps(j) if "run" in s]


def chain(workflow):
    """Jobs in dependency order."""
    jobs, order = workflow["jobs"], []

    def needs(j):
        n = jobs[j].get("needs", [])
        return [n] if isinstance(n, str) else n

    def visit(j):
        for d in needs(j):
            visit(d)
        if j not in order:
            order.append(j)

    for j in jobs:
        visit(j)
    return order


@pytest.mark.parametrize("name", NAMES)
def test_every_workflow_parses_and_has_jobs(name):
    wf = load(name)
    assert wf["name"] == name and wf["jobs"] and wf["on"]


@pytest.mark.parametrize("name", NAMES)
def test_every_job_runs_on_the_pinned_runner_image(name):
    for job_id, job in load(name)["jobs"].items():
        assert job["runs-on"] == "ubuntu-24.04", f"{name} job {job_id}: pin the runner (ubuntu-24.04), not a moving label like ubuntu-latest"


def test_kb_ci_is_the_required_check_and_needs_no_aws():
    wf = load("kb-ci")
    assert set(wf["on"]) == {"pull_request", "push"} and wf["on"]["push"]["branches"] == ["main"]
    assert list(wf["jobs"]) == ["kb-check"]
    text = (WF / "kb-ci.yml").read_text()
    assert "secrets." not in text and "id-token" not in text and "configure-aws-credentials" not in text
    scripts = "\n".join(run_scripts(wf))
    for needle in ("pytest", "ingest lint", "--embeddings tfidf", "--embeddings hash", "--previous build/main-hash",
                   "eval.evaluate", "ingest summary", "_main/docs"):
        assert needle in scripts, needle
    checkouts = [s for s in steps(wf["jobs"]["kb-check"]) if s.get("uses", "").startswith("actions/checkout")]
    assert len(checkouts) == 2 and checkouts[1]["with"]["ref"] == "main"       # main's docs, for the comparison
    assert wf["permissions"] == {"contents": "read"}


def test_ingest_chain_order_and_triggers():
    wf = load("ingest")
    assert chain(wf) == ["build", "eval-gate", "publish", "promote-dev", "promote-sit", "promote-pre-prod", "promote-prod"]
    assert wf["on"]["push"]["branches"] == ["main"]
    assert wf["on"]["push"]["paths"] == ["docs/**", "ingest/**", "eval/**"]
    assert "workflow_dispatch" in wf["on"]
    assert wf["concurrency"]["cancel-in-progress"] is False


def test_the_eval_gate_comes_before_publish_and_publish_uses_the_evaluated_index():
    wf = load("ingest")
    assert wf["jobs"]["publish"]["needs"] == "eval-gate" and wf["jobs"]["eval-gate"]["needs"] == "build"
    gate = "\n".join(s["run"] for s in steps(wf["jobs"]["eval-gate"]) if "run" in s)
    assert "eval.evaluate --index build/index --baseline ssm:/finrisk/prod/index" in gate
    publish = wf["jobs"]["publish"]
    downloads = [s for s in steps(publish) if s.get("uses", "").startswith("actions/download-artifact")]
    assert downloads[0]["with"]["name"] == "index-evaluated-${{ github.sha }}"
    uploads = [s["with"]["name"] for s in steps(wf["jobs"]["eval-gate"]) if s.get("uses", "").startswith("actions/upload-artifact")]
    assert uploads == ["index-evaluated-${{ github.sha }}"]
    assert "ingest publish" in "\n".join(s["run"] for s in steps(publish) if "run" in s)


def test_build_is_titan_and_incremental_on_the_latest_published_index():
    build = load("ingest")["jobs"]["build"]
    step = next(s for s in steps(build) if s.get("id") == "build")
    assert step["env"]["EMBEDDINGS"] == "titan"
    assert "--previous latest" in step["run"] and step["env"]["KB_BUCKET"] == "${{ secrets.KB_BUCKET }}"


def test_environments_and_roles():
    wf = load("ingest")
    jobs = wf["jobs"]
    for env in ("dev", "sit", "pre-prod", "prod"):
        job = jobs[f"promote-{env}"]
        assert job["environment"] == env
        login = next(s for s in steps(job) if s.get("uses", "").startswith("aws-actions/configure-aws-credentials"))
        assert login["with"]["role-to-assume"] == "${{ secrets.AWS_ROLE_ARN }}"     # an Environment secret
        cmd = next(s for s in steps(job) if s.get("name", "").startswith("Promote"))
        assert f"--env {env} " in cmd["run"] and "--version \"$VERSION\"" in cmd["run"]
        assert cmd["env"]["VERSION"] == "${{ needs.publish.outputs.version }}"
        assert job["concurrency"]["group"] == f"kb-promote-{env}"
    for name in ("build", "eval-gate", "publish"):
        assert "environment" not in jobs[name]                    # OIDC sub stays ref:refs/heads/main for finrisk-kb-build
        login = next(s for s in steps(jobs[name]) if s.get("uses", "").startswith("aws-actions/configure-aws-credentials"))
        assert login["with"]["role-to-assume"] == "${{ secrets.AWS_KB_BUILD_ROLE_ARN }}"
    assert jobs["build"]["if"] == "github.ref == 'refs/heads/main'"
    assert jobs["promote-sit"]["needs"] == ["publish", "promote-dev"]
    assert jobs["promote-pre-prod"]["needs"] == ["publish", "promote-sit"]
    assert jobs["promote-prod"]["needs"] == ["publish", "promote-pre-prod"]


def test_promote_index_is_the_same_chain_for_an_existing_version():
    wf = load("promote-index")
    assert chain(wf) == ["promote-dev", "promote-sit", "promote-pre-prod", "promote-prod"]
    version = wf["on"]["workflow_dispatch"]["inputs"]["version"]
    assert version["required"] is True and version["type"] == "string"
    assert [wf["jobs"][f"promote-{e}"]["environment"] for e in ("dev", "sit", "pre-prod", "prod")] == ["dev", "sit", "pre-prod", "prod"]
    for job in wf["jobs"].values():
        cmd = next(s for s in steps(job) if s.get("name", "").startswith("Promote"))
        assert cmd["env"]["VERSION"] == "${{ inputs.version }}"
        assert '--version "$VERSION"' in cmd["run"]
    text = (WF / "promote-index.yml").read_text()
    assert "publish" not in "\n".join(wf["jobs"]) and "ingest build" not in text


@pytest.mark.parametrize("name", NAMES)
def test_every_aws_login_masks_the_account_id_and_only_those_jobs_get_a_token(name):
    wf = load(name)
    for job_name, job in wf["jobs"].items():
        items = steps(job)
        logins = [i for i, s in enumerate(items) if s.get("uses", "").startswith("aws-actions/configure-aws-credentials")]
        perms = job.get("permissions", {})
        if not logins:
            assert perms.get("id-token") != "write", f"{job_name} asks for a token but never logs in"
            continue
        assert perms.get("id-token") == "write" and perms.get("contents") == "read"
        follow = items[logins[0] + 1]["run"]
        assert follow == 'echo "::add-mask::$(aws sts get-caller-identity --query Account --output text)"'


@pytest.mark.parametrize("name", NAMES)
def test_no_secret_or_input_is_pasted_into_a_script_or_echoed(name):
    wf = load(name)
    for script in run_scripts(wf):
        assert "${{" not in script, f"expression inside a run script: {script!r}"
        assert not re.search(r"echo[^\n]*(KB_BUCKET|ROLE_ARN|secrets\.)", script)
    text = (WF / f"{name}.yml").read_text()
    assert not re.search(r"AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|aws-access-key-id", text)
    # secrets only appear as `with:` / `env:` values
    for line in text.splitlines():
        if "secrets." in line and not line.lstrip().startswith("#"):
            assert re.match(r"\s+[\w-]+: \$\{\{ secrets\.[A-Z_]+ \}\}$", line), line


@pytest.mark.parametrize("name", NAMES)
def test_actions_are_pinned_to_the_course_majors(name):
    for line in (WF / f"{name}.yml").read_text().splitlines():
        match = re.search(r"uses: ([\w./-]+)@(\S+)", line)
        if match:
            assert PINNED[match.group(1)] == match.group(2), line


def test_ingest_cli_commands_used_by_the_workflows_exist():
    from ingest.cli import parser
    choices = set(parser()._subparsers._group_actions[0].choices)
    used = set()
    for name in NAMES:
        for script in run_scripts(load(name)):
            used |= set(re.findall(r"python -m ingest (\w[\w-]*)", script))
    assert used and used <= choices >= {"build", "publish", "promote", "latest", "diff"}


@pytest.mark.parametrize("name", NAMES)
def test_role_secret_rule_environment_jobs_read_AWS_ROLE_ARN_others_read_a_repo_secret(name):
    """One rule for all four repos: environment -> secrets.AWS_ROLE_ARN; no environment -> a repo secret named after the role."""
    for job_name, job in load(name)["jobs"].items():
        for step in steps(job):
            if step.get("uses", "").startswith("aws-actions/configure-aws-credentials"):
                role = step["with"]["role-to-assume"]
                expected = "${{ secrets.AWS_ROLE_ARN }}" if "environment" in job else "${{ secrets.AWS_KB_BUILD_ROLE_ARN }}"
                assert role == expected, job_name
    assert "AWS_KB_PROMOTE_ROLE_ARN" not in (WF / f"{name}.yml").read_text()


@pytest.mark.parametrize("name", ("ingest", "promote-index"))
def test_the_region_is_the_repository_variable_AWS_REGION(name):
    """github_bootstrap.sh sets AWS_REGION as a variable in every repo; a workflow that hard-codes it leaves it unused."""
    assert load(name)["env"]["AWS_REGION"] == "${{ vars.AWS_REGION }}"


def test_the_secrets_table_is_at_the_top_of_ingest_yml():
    head = "\n".join((WF / "ingest.yml").read_text().splitlines()[:25])
    for name in ("AWS_REGION", "AWS_KB_BUILD_ROLE_ARN", "KB_BUCKET", "AWS_ROLE_ARN"):
        assert name in head


ACCOUNT_ID = re.compile(r"(?<!\d)\d{12}(?!\d)")
LONG_HEX = re.compile(r"[0-9a-f]{32,}")             # a digest or a commit sha can contain 12 digits in a row


def account_ids(root=ROOT):
    return [path.relative_to(root).as_posix() for path in repo_files(root)
            if path.suffix in {".md", ".yml", ".yaml", ".py", ".txt", ".toml", ".json"}
            and ACCOUNT_ID.search(LONG_HEX.sub("", path.read_text(errors="ignore")))]


def test_no_twelve_digit_account_id_anywhere_in_the_repo():
    """Only files git tracks (or would) are scanned: not .git/ (a runner's .git holds fetched refs), .venv/, build/ or _main/."""
    assert account_ids() == []


def test_the_scan_ignores_generated_folders_and_digests_but_still_sees_a_leak(tmp_path):
    fake = "123456" + "789012"                          # written in two pieces: this file is scanned too
    for folder in sorted(GENERATED_DIRS):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "manifest.json").write_text('{"account": "%s"}\n' % fake)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "ok.md").write_text("digest sha256:ab" + fake + "cd" + "e" * 48 + "\n")        # 12 digits inside a digest
    assert account_ids(tmp_path) == []
    assert [p.name for p in repo_files(tmp_path)] == ["ok.md"]
    (tmp_path / "docs" / "leak.md").write_text(f"role arn:aws:iam::{fake}:role/x\n")
    assert account_ids(tmp_path) == ["docs/leak.md"]