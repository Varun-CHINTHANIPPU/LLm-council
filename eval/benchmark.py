"""Benchmark summary: Council vs. individual models and baselines (#10).

Reads the results.jsonl written by eval.execute (no API calls here) and
reports, per system: correctness, unresolved rate, API calls and latency.
For Council it adds auditor effectiveness: how often the independent auditor
raised concerns on answers that turned out wrong vs. right.

Comparisons are paired by task with bootstrap confidence intervals. Two kinds
are kept apart:
* call-matched: baselines that used the same number of API calls as Council
  (fair on cost);
* per-model: each single model's one-call answer (fair on "does the council
  beat its own members?", but Council spends many more calls).

Usage:
    python -m eval.benchmark runs/eval_dir [--json]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from eval.metrics import bootstrap_ci


def _read(path) -> list[dict]:
    path = Path(path)
    source = path / "results.jsonl" if path.is_dir() else path
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _mean(values):
    values = [float(v) for v in values if isinstance(v, (int, float))]
    return statistics.fmean(values) if values else None


def _median(values):
    values = [float(v) for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


def per_system(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("system")].append(row)
    out = {}
    for system, group in sorted(groups.items(), key=lambda kv: str(kv[0])):
        scored = [r for r in group if r.get("correct") is not None]
        out[system] = {
            "tasks": len({r.get("task_id") for r in group}),
            "scored": len(scored),
            "accuracy": _mean([r["correct"] for r in scored]),
            "unresolved_rate": _mean([bool(r.get("unresolved")) for r in group]),
            "mean_calls": _mean([r.get("calls") for r in group]),
            "mean_latency_s": _mean([r.get("latency") for r in group]),
            "median_latency_s": _median([r.get("latency") for r in group]),
        }
    return out


def auditor_effectiveness(rows: list[dict]) -> dict:
    """Did the auditor flag the answers that were actually wrong?

    flag = the auditor raised at least one concern. catch_rate = P(flag |
    wrong); false_alarm_rate = P(flag | right). A useful auditor has a catch
    rate clearly above its false-alarm rate. Also reported for "any open
    issue" (reviewer objections not addressed, or audit concerns).
    """
    council = [r for r in rows if r.get("system") == "council"
               and r.get("correct") is not None]
    audited = [r for r in council if r.get("audit_completed")]

    def rates(flag):
        wrong = [r for r in audited if not r["correct"]]
        right = [r for r in audited if r["correct"]]
        catch = _mean([flag(r) for r in wrong]) if wrong else None
        alarm = _mean([flag(r) for r in right]) if right else None
        return {"wrong_answers": len(wrong), "right_answers": len(right),
                "catch_rate": catch, "false_alarm_rate": alarm,
                "discrimination": (catch - alarm if catch is not None
                                   and alarm is not None else None)}

    return {
        "scored_council_tasks": len(council),
        "audit_completion_rate": (len(audited) / len(council) if council else None),
        "auditor_concern": rates(lambda r: (r.get("auditor_issue_count") or 0) > 0),
        "any_open_issue": rates(lambda r: (r.get("open_issue_count") or 0) > 0),
        "note": "Small samples make these rates noisy; read them with the "
                "task counts.",
    }


def paired(rows: list[dict], baseline: str, *, call_matched: bool) -> dict | None:
    by_task = defaultdict(dict)
    for row in rows:
        if row.get("correct") is None or row.get("oracle_assisted"):
            continue
        if call_matched and not row.get("equal_budget", True):
            continue
        by_task[row["task_id"]].setdefault(row["system"], []).append(float(row["correct"]))
    deltas = [statistics.fmean(s["council"]) - statistics.fmean(s[baseline])
              for _, s in sorted(by_task.items())
              if "council" in s and baseline in s]
    if not deltas:
        return None
    mean, low, high = bootstrap_ci(deltas) if len(deltas) > 1 else (deltas[0],) * 3
    return {"paired_tasks": len(deltas), "council_minus_baseline": mean,
            "ci95": [low, high],
            "verdict": ("council better" if low > 0 else
                        "council worse" if high < 0 else "no clear difference")}


def summarize(rows: list[dict]) -> dict:
    systems = per_system(rows)
    baselines = [s for s in systems if s not in {"council", None}]
    return {
        "systems": systems,
        "auditor": auditor_effectiveness(rows),
        "per_model_comparison": {
            s: paired(rows, s, call_matched=False)
            for s in baselines if str(s).startswith("single")},
        "call_matched_comparison": {
            s: paired(rows, s, call_matched=True)
            for s in baselines if not str(s).startswith("single:")},
        "notes": [
            "per_model_comparison ignores cost: Council uses many calls, each "
            "single model one.",
            "call_matched_comparison only uses rows marked equal_budget.",
            "best_of_n rows that used the gold-answer checker are excluded "
            "(oracle_assisted).",
        ],
    }


def render(summary: dict) -> str:
    def pct(x):
        return "-" if x is None else f"{100 * x:.0f}%"

    def num(x, digits=1):
        return "-" if x is None else f"{x:.{digits}f}"

    lines = ["| system | scored | accuracy | unresolved | calls | latency (s, mean/median) |",
             "|---|---|---|---|---|---|"]
    for name, s in summary["systems"].items():
        lines.append(f"| {name} | {s['scored']} | {pct(s['accuracy'])} | "
                     f"{pct(s['unresolved_rate'])} | {num(s['mean_calls'])} | "
                     f"{num(s['mean_latency_s'])} / {num(s['median_latency_s'])} |")
    a = summary["auditor"]
    lines += ["", "Auditor: completed on " + pct(a["audit_completion_rate"])
              + " of scored Council tasks."]
    for key, label in (("auditor_concern", "auditor raised a concern"),
                       ("any_open_issue", "any open issue")):
        r = a[key]
        lines.append(f"- {label}: catch rate {pct(r['catch_rate'])} on "
                     f"{r['wrong_answers']} wrong answers, false alarms "
                     f"{pct(r['false_alarm_rate'])} on {r['right_answers']} right ones")
    for title, key in (("Council vs single models (not cost-matched)",
                        "per_model_comparison"),
                       ("Council vs call-matched baselines", "call_matched_comparison")):
        lines += ["", title + ":"]
        for name, c in summary[key].items():
            lines.append(f"- {name}: " + ("no paired tasks" if c is None else
                         f"{c['council_minus_baseline']:+.2f} "
                         f"[{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] over "
                         f"{c['paired_tasks']} tasks, {c['verdict']}"))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    summary = summarize(_read(args.results))
    print(json.dumps(summary, indent=2) if args.json else render(summary))


if __name__ == "__main__":
    main()