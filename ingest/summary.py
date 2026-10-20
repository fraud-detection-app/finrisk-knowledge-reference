"""Markdown for the GitHub job summary: what changed, what was reused, how the eval compares with main."""

from __future__ import annotations

import json
from pathlib import Path


def _load(path: str | Path | None) -> dict | None:
    if not path or not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def render(plan: dict | None, report: dict | None, baseline: dict | None, title: str = "Knowledge base check",
           baseline_label: str = "main") -> str:
    lines = [f"### {title}", ""]
    if plan:
        lines += [
            "| Documents | | Chunks | |", "|---|---|---|---|",
            f"| added (+) | {len(plan['added'])} | reused | {plan['reused']} |",
            f"| changed (~) | {len(plan['changed'])} | embedded | {plan['embedded']} |",
            f"| deleted (-) | {len(plan['removed'])} | dropped | {plan['dropped']} |",
            "", f"`{plan['line']}`", "",
        ]
        for label, key in (("added", "added"), ("changed", "changed"), ("deleted", "removed")):
            if plan[key]:
                lines.append(f"- {label}: " + ", ".join(f"`{p}`" for p in plan[key]))
        if plan.get("reason"):
            lines += ["", f"> {plan['reason']}"]
        if plan.get("provider") == "hash":
            lines += ["", "> Planned with offline `hash` vectors: Titan reuses chunks the same way, without calling Bedrock here."]
        lines.append("")
    if report:
        gate = report["gate"]
        lines += [f"| Eval | this change | {baseline_label} |", "|---|---|---|"]
        gated = (gate.get("baseline") or {}).get("recall_at_5")
        base_recall = baseline.get("recall_at_5") if baseline else gated
        base_mrr = baseline.get("mrr") if baseline else None
        lines.append(f"| recall@5 | {report['recall_at_5']:.4f} | {'-' if base_recall is None else f'{base_recall:.4f}'} |")
        lines.append(f"| MRR | {report['mrr']:.4f} | {'-' if base_mrr is None else f'{base_mrr:.4f}'} |")
        refusals = report["refusals"]
        lines.append(f"| must-refuse refused | {refusals['refused']}/{refusals['expected']} | |")
        lines += ["", f"**Gate: {'PASSED' if report['passed'] else 'FAILED'}** "
                      f"(recall@5 >= {gate['min_recall_at_5']:.2f}, drop <= {gate['max_drop']:.2f})", ""]
        for check in gate["checks"]:
            lines.append(f"- {'ok' if check['ok'] else 'FAIL'}: {check['detail']}")
        missed = [r for r in report["per_question"] if not r["ok"]]
        if missed:
            lines += ["", "Questions that failed:"]
            for row in missed:
                lines.append(f"- `{row['id']}` {row['question']}")
    return "\n".join(lines) + "\n"


def render_files(plan_path: str | None, report_path: str | None, baseline_path: str | None, title: str,
                 baseline_label: str = "main") -> str:
    return render(_load(plan_path), _load(report_path), _load(baseline_path), title, baseline_label)
