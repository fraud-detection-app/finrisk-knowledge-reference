"""Eval metrics and the publish gate."""

import json
from pathlib import Path

import pytest

from eval.evaluate import (Baseline, EXIT_GATE_FAILED, Question, apply_gate, evaluate_index, load_baseline,
                           load_questions, main, reciprocal_rank)
from eval.policy import OUT_OF_POLICY_PATTERNS, POLICY_PATTERNS, is_out_of_policy
from ingest.build import build_index
from ingest.reader import IndexReader
from tests.conftest import FakeS3, FakeSSM, write_doc

ROOT = Path(__file__).resolve().parents[1]


# --- the question set ------------------------------------------------------------------------------------------

def test_shipped_questions_are_well_formed():
    questions = load_questions(ROOT / "eval" / "questions.yaml")
    answer = [q for q in questions if not q.must_refuse]
    refuse = [q for q in questions if q.must_refuse]
    assert len(questions) >= 12 and len(refuse) >= 3
    docs = {p.relative_to(ROOT).as_posix() for p in (ROOT / "docs").glob("*.md")}
    assert all(set(q.expected_docs) <= docs for q in answer), "every expected doc must exist in docs/"
    assert all(is_out_of_policy(q.question) for q in refuse)
    assert not any(is_out_of_policy(q.question) for q in answer)


def test_policy_patterns_carry_the_sync_warning():
    text = (ROOT / "eval" / "policy.py").read_text()
    assert "KEEP THEM IN SYNC" in text and "COPIED FROM THE RISK SERVICE" in text
    assert len(OUT_OF_POLICY_PATTERNS) == 11 and len(POLICY_PATTERNS) == 2


@pytest.mark.parametrize("question", [
    "Should this customer receive a loan?", "Which stocks should I buy with my savings?",
    "Can you waive the fraud flag on my account?", "Will it guarantee a profit?",
])
def test_out_of_scope_questions_are_refused(question):
    assert is_out_of_policy(question)


def test_a_question_needs_exactly_one_of_expected_docs_or_must_refuse(tmp_path):
    bad = tmp_path / "q.yaml"
    bad.write_text("questions:\n  - {id: a, question: x}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_questions(bad)


# --- metrics -----------------------------------------------------------------------------------------------------

def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], ("a",)) == 1.0
    assert reciprocal_rank(["a", "b", "c"], ("c", "z")) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a"], ("z",)) == 0.0


@pytest.fixture
def reader(docs_dir, tmp_path):
    build_index(docs_dir, tmp_path / "i", "tfidf", source_sha="d" * 40)
    return IndexReader.open(tmp_path / "i")


def test_recall_mrr_and_refusals(reader):
    questions = [
        Question("a", "Why is is_flagged_fraud excluded from the features?", ("docs/model_card.md",)),
        Question("b", "who must review high risk transactions", ("docs/fraud_policy.md",)),
        Question("c", "how do I roll back", ("docs/runbook.md",)),
        Question("d", "what is the colour of the moon", ("docs/runbook.md",)),          # not answerable -> a miss
        Question("r", "Which stocks should I buy?", (), must_refuse=True),
    ]
    metrics = evaluate_index(reader, questions)
    assert metrics["answerable"] == 4 and metrics["recall_at_5"] == 0.75
    assert metrics["mrr"] == pytest.approx(0.75, abs=0.01)
    assert metrics["refusals"] == {"expected": 1, "refused": 1, "false_refusals": 0}
    assert [r["ok"] for r in metrics["per_question"]] == [True, True, True, False, True]


def test_an_answerable_question_that_trips_a_refusal_pattern_is_flagged(reader):
    metrics = evaluate_index(reader, [Question("x", "Should I read the fraud policy?", ("docs/fraud_policy.md",))])
    assert metrics["refusals"]["false_refusals"] == 1 and metrics["per_question"][0]["ok"] is False


# --- the gate ----------------------------------------------------------------------------------------------------

def metrics(recall, refused=3, expected=3, false=0):
    return {"recall_at_5": recall, "refusals": {"expected": expected, "refused": refused, "false_refusals": false}}


def ok_baseline(recall):
    return Baseline({"recall_at_5": recall, "index_version": "v-prod"}, "ok", source="prod")


def test_gate_passes_at_the_floor_and_with_a_small_drop():
    gate = apply_gate(metrics(0.80), ok_baseline(0.85))              # exactly the floor, exactly the allowed drop
    assert gate["passed"] and gate["baseline"]["recall_at_5"] == 0.85


def test_gate_fails_below_the_floor_even_without_a_baseline():
    gate = apply_gate(metrics(0.79), None)
    assert not gate["passed"] and not gate["checks"][0]["ok"]


def test_gate_fails_on_a_drop_bigger_than_five_points():
    gate = apply_gate(metrics(0.86), ok_baseline(0.95))             # above the floor, but down 0.09
    assert not gate["passed"]
    assert [c["name"] for c in gate["checks"] if not c["ok"]] == ["no_drop_vs_prod"]


def test_gate_fails_when_a_must_refuse_question_is_answered():
    gate = apply_gate(metrics(1.0, refused=2), ok_baseline(0.9))
    assert not gate["passed"] and [c["name"] for c in gate["checks"] if not c["ok"]] == ["refusals"]


def test_missing_baseline_is_skipped_unreadable_is_a_warning_unless_required():
    first_release = Baseline(None, "none", "points at no index yet")
    assert apply_gate(metrics(0.9), first_release)["passed"]
    unreadable = Baseline(None, "unreadable", "AccessDeniedException")
    soft = apply_gate(metrics(0.9), unreadable)
    assert soft["passed"] and "WARNING" in soft["checks"][1]["detail"]
    assert not apply_gate(metrics(0.9), unreadable, require_baseline=True)["passed"]


def test_baseline_from_the_prod_pointer(s3):
    s3.put_object("kb", "indexes/v1/eval_report.json", json.dumps({"recall_at_5": 0.91, "index_version": "v1"}).encode())
    ssm = FakeSSM(**{"/finrisk/prod/index": json.dumps({"version": "v1", "uri": "s3://kb/indexes/v1/"})})
    baseline = load_baseline("ssm:/finrisk/prod/index", ssm=ssm, s3=s3)
    assert baseline.status == "ok" and baseline.report["recall_at_5"] == 0.91
    empty = FakeSSM(**{"/finrisk/prod/index": json.dumps({"version": None})})
    assert load_baseline("ssm:/finrisk/prod/index", ssm=empty, s3=s3).status == "none"
    assert load_baseline("ssm:/finrisk/prod/index", ssm=FakeSSM(), s3=s3).status == "none"      # parameter not created yet


def test_baseline_without_a_report_or_with_denied_access(s3):
    ssm = FakeSSM(**{"/finrisk/prod/index": json.dumps({"version": "v0", "uri": "s3://kb/indexes/v0/"})})
    assert load_baseline("ssm:/finrisk/prod/index", ssm=ssm, s3=s3).status == "none"            # old index, no report

    class Denied:
        def get_parameter(self, Name):
            from tests.conftest import client_error
            raise client_error("AccessDeniedException")

    assert load_baseline("ssm:/finrisk/prod/index", ssm=Denied(), s3=s3).status == "unreadable"


# --- the CLI: exit code 3 and the report next to the index ---------------------------------------------------------

def test_cli_writes_the_report_and_exits_3_on_failure(docs_dir, tmp_path, capsys):
    build_index(docs_dir, tmp_path / "i", "tfidf", source_sha="e" * 40)
    good = tmp_path / "good.yaml"
    good.write_text("questions:\n  - {id: a, question: why is is_flagged_fraud excluded, expected_docs: [docs/model_card.md]}\n"
                    "  - {id: r, question: 'Which stocks should I buy?', must_refuse: true}\n", encoding="utf-8")
    assert main(["--index", str(tmp_path / "i"), "--questions", str(good)]) == 0
    report = json.loads((tmp_path / "i" / "eval_report.json").read_text())
    assert report["passed"] is True and report["recall_at_5"] == 1.0 and report["embeddings"] == "tfidf"
    assert report["gate"]["min_recall_at_5"] == 0.8 and report["gate"]["max_drop"] == 0.05

    bad = tmp_path / "bad.yaml"
    bad.write_text("questions:\n  - {id: a, question: why is is_flagged_fraud excluded, expected_docs: [docs/runbook.md]}\n", encoding="utf-8")
    assert main(["--index", str(tmp_path / "i"), "--questions", str(bad)]) == EXIT_GATE_FAILED == 3
    assert json.loads((tmp_path / "i" / "eval_report.json").read_text())["passed"] is False
    assert "GATE FAILED" in capsys.readouterr().out


def test_cli_drop_against_a_baseline_report(docs_dir, tmp_path):
    build_index(docs_dir, tmp_path / "i", "tfidf", source_sha="e" * 40)
    q = tmp_path / "q.yaml"
    q.write_text("questions:\n  - {id: a, question: why is is_flagged_fraud excluded, expected_docs: [docs/model_card.md]}\n"
                 "  - {id: b, question: how do I roll back, expected_docs: [docs/runbook.md]}\n", encoding="utf-8")
    prod = tmp_path / "prod_report.json"
    prod.write_text(json.dumps({"recall_at_5": 1.0, "index_version": "prod"}), encoding="utf-8")
    assert main(["--index", str(tmp_path / "i"), "--questions", str(q), "--baseline", str(prod)]) == 0
    prod.write_text(json.dumps({"recall_at_5": 1.0 + 0.0, "index_version": "prod"}), encoding="utf-8")
    # make the candidate worse: one of two questions misses -> 0.5, a drop of 0.5
    q.write_text("questions:\n  - {id: a, question: why is is_flagged_fraud excluded, expected_docs: [docs/model_card.md]}\n"
                 "  - {id: b, question: how do I roll back, expected_docs: [docs/model_card.md]}\n", encoding="utf-8")
    assert main(["--index", str(tmp_path / "i"), "--questions", str(q), "--baseline", str(prod), "--min-recall", "0.1"]) == 3
