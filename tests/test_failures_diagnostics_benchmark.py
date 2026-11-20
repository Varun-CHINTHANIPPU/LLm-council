"""Regressions: network failures, malformed and incomplete model
output, recovery, provider availability, benchmark summary and report.

Synthetic models only; zero external API calls.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from council.budget import RunBudget
from council.events import EventStore
from council.full import FullCouncilRunner
from council.providers import provider_coverage, quota_availability
from council.replay import ResumeTransport
from council.report import render_markdown
from council.transport import CallResult
from eval.benchmark import summarize, render

from test_answers_fallbacks_resume import Models, run_council

SOLVERS = ("gpt-oss", "command", "nemotron")


class Raw(Models):
    """Models whose replies for chosen (family, role) are raw text queues."""

    def __init__(self, raw, **kw):
        super().__init__(**kw)
        self.raw = {k: list(v) for k, v in raw.items()}

    async def complete(self, family, message, budget, *, role, synthesis=False):
        queue = self.raw.get((family, role)) or self.raw.get(("*", role))
        text = (queue.pop(0) if len(queue) > 1 else queue[0]) if queue else None
        if text is not None:  # None (or the last item None) = normal reply
            self.calls.append((family, role))
            self.budgets.append((family, role, budget.remaining()))
            return CallResult(family, "mock", family, text, 0.0, "h")
        return await super().complete(family, message, budget, role=role,
                                      synthesis=synthesis)


def assert_sane(test, result):
    """Invariants every run must keep, whatever failed."""
    test.assertEqual(result["verdict"]["status"], "unresolved")
    test.assertFalse(result["answer_selection"]["verified"])
    test.assertEqual(result["research_result"]["supported_findings"], [])
    test.assertIn(result["answer_selection"]["status"], {
        "model_reviewed_unverified", "model_agreement_unverified",
        "unjudged_unverified", "single_model_unverified",
        "degraded_single_model_unverified", "quorum_stopped", "unresolved"})
    test.assertIsInstance(render_markdown(result), str)


class NetworkFailures(unittest.IsolatedAsyncioTestCase):
    async def test_outage_at_every_stage_still_finishes(self):
        stages = {
            "one proposer": {("command", "proposal")},
            "all reviewers": {(f, "review") for f in SOLVERS},
            "all revisions": {(f, "revision") for f in SOLVERS},
            "all re-reviews": {(f, "rereview") for f in SOLVERS},
            "judge": {("gemini", "synthesis")},
            "auditor": {("gemma", "audit")},
            "everything after proposals": {
                (f, r) for f in SOLVERS + ("gemini", "gemma")
                for r in ("review", "revision", "rereview", "synthesis", "audit")},
        }
        for label, fail in stages.items():
            with self.subTest(stage=label):
                result = await run_council(Models(fail=fail,
                                                   answers={"gpt-oss": "42"}))
                assert_sane(self, result)
                self.assertIsNotNone(result["final_answer"])

    async def test_audit_outage_is_reported_not_hidden(self):
        result = await run_council(Models(fail={("gemma", "audit")}))
        self.assertFalse(result["audit"]["completed"])
        self.assertIsNone(result["answer_selection"]["auditor_issue_count"])
        self.assertIn("The independent audit did not complete",
                      render_markdown(result))

    async def test_tiny_deadline_finishes_with_honest_labels(self):
        result = await run_council(Models(), seconds=6)
        assert_sane(self, result)


class MalformedOutput(unittest.IsolatedAsyncioTestCase):
    async def test_bad_json_proposal_is_retried_once_then_accepted(self):
        models = Raw({("command", "proposal"): ["not json at all", None]})
        result = await run_council(models)
        command_calls = [r for f, r in models.calls if f == "command" and r == "proposal"]
        self.assertEqual(len(command_calls), 2)  # exactly one format retry
        self.assertIn("command", result["role_coverage"]["valid_proposer_families"])
        assert_sane(self, result)

    async def test_proposal_garbage_twice_is_dropped(self):
        models = Raw({("command", "proposal"): ["{}", "{}"]})
        result = await run_council(models)
        self.assertEqual(result["role_coverage"]["valid_proposer_families"],
                         ["gpt-oss", "nemotron"])
        self.assertTrue(any("command" in w for w in result["warnings"]))

    async def test_proposal_without_claims_is_incomplete_and_rejected(self):
        body = json.dumps({"approach": "An answer.", "final_answer": "",
                           "claims": [], "hypotheses": [], "experiments": []})
        result = await run_council(Raw({("nemotron", "proposal"): [body]}))
        self.assertNotIn("nemotron", result["role_coverage"]["valid_proposer_families"])
        assert_sane(self, result)

    async def test_review_with_unknown_claim_ids_is_dropped(self):
        bad = json.dumps({"objections": [{"claim_id": "c99", "issue": "x",
                                          "consequence": "y",
                                          "suggested_check": "z"}]})
        result = await run_council(Raw({("*", "review"): [bad]}))
        self.assertEqual(result["objections"], [])
        self.assertTrue(any("unknown claim id" in w for w in result["warnings"]))

    async def test_judge_garbage_falls_back_to_deterministic_answer(self):
        models = Raw({("gemini", "synthesis"): ["[]"]},
                     answers={"gpt-oss": "7", "command": "7", "nemotron": "8"})
        result = await run_council(models)
        self.assertEqual(result["answer_selection"]["basis"],
                         "proposal_majority_unjudged")
        assert_sane(self, result)

    async def test_rereview_with_bad_ids_or_verdicts_is_ignored(self):
        bad = json.dumps({"items": [{"objection_id": "o404", "verdict": "resolved",
                                     "reason": "x"}]})
        result = await run_council(Raw({("*", "rereview"): [bad]}))
        peer = [i for i in result["issues"] if i["source"] == "peer_review"]
        self.assertTrue(all(i["rereview"] is None for i in peer))
        self.assertEqual({i["status"] for i in peer}, {"unreviewed", "contested"})

    async def test_malformed_audit_is_not_counted_as_completed(self):
        result = await run_council(Raw({("gemma", "audit"): ['{"issues": 5}']}))
        self.assertFalse(result["audit"]["completed"])


class Recovery(unittest.IsolatedAsyncioTestCase):
    async def test_resume_retries_a_judge_that_was_down(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            with EventStore(root, "a") as events:
                first = Models(fail={("gemini", "synthesis")}, record=events.record)
                runner = FullCouncilRunner(first, events,
                                           RunBudget.start(120, reserve=20))
                original = await runner.run("Q?", mode="full")
                rows = events.replay()
            self.assertNotEqual(original["answer_selection"]["basis"], "judge_synthesis")
            with EventStore(root, "b") as events:
                live = Models(record=events.record)
                runner = FullCouncilRunner(
                    ResumeTransport(rows, live, record=events.record), events,
                    RunBudget.start(120, reserve=20))
                resumed = await runner.run("Q?", mode="full")
            self.assertEqual(resumed["answer_selection"]["basis"], "judge_synthesis")
            self.assertNotIn("proposal", [r for _, r in live.calls])


class ProviderAvailability(unittest.TestCase):
    NOW = 1_000_000.0

    def coverage(self):
        env = {"GROQ_API_KEY": "k", "GEMINI_API_KEY": "k", "LLM7_API_KEY": "k",
               "OPENROUTER_API_KEY": "k", "COHERE_API_KEY": "k"}
        return provider_coverage(env=env, active_families=("gpt-oss", "gemini"))

    def test_key_alone_is_not_reported_as_available_quota(self):
        report = quota_availability(self.coverage(), now=self.NOW, today="2026-11-20")
        p = report["providers"]
        self.assertEqual(p["Groq"]["availability"], "key_only_never_observed")
        self.assertEqual(p["Cohere"]["availability"], "free_confirmation_required")
        self.assertEqual(p["Cloudflare"]["availability"], "no_key")
        self.assertIn("Groq", report["unknown_quota"])
        self.assertEqual(report["usable_now"], [])

    def test_evidence_classifies_providers(self):
        key = lambda p, m: json.dumps([p, m])
        health = {"models": {
            key("Groq", "gpt-oss"): {"healthy_until": self.NOW + 100},
            key("Gemini", "flash"): {"cooldown_until": self.NOW + 50,
                                     "reason": "rate_limit"}},
            "providers": {"OpenRouter": {"until": self.NOW + 999,
                                         "reason": "billing_required"}}}
        daily = {"providers": {"LLM7.io": {"day": "2026-11-20", "requests": 40}}}
        report = quota_availability(self.coverage(), health=health, daily=daily,
                                    daily_caps={"LLM7.io": 40}, now=self.NOW,
                                    today="2026-11-20")
        p = report["providers"]
        self.assertEqual(p["Groq"]["availability"], "recently_succeeded")
        self.assertEqual(p["Gemini"]["availability"], "cooling_down")
        self.assertIn("rate_limit", " ".join(p["Gemini"]["evidence"]))
        self.assertEqual(p["OpenRouter"]["availability"], "blocked_billing_or_auth")
        self.assertEqual(p["LLM7.io"]["availability"], "daily_cap_reached")
        self.assertEqual(report["usable_now"], ["Groq"])

    def test_expired_cooldowns_and_old_usage_do_not_count(self):
        key = json.dumps(["Gemini", "flash"])
        health = {"models": {key: {"cooldown_until": self.NOW - 1, "reason": "x",
                                   "healthy_until": self.NOW - 1}}}
        daily = {"providers": {"Gemini": {"day": "2026-11-19", "requests": 999}}}
        report = quota_availability(self.coverage(), health=health, daily=daily,
                                    daily_caps={"Gemini": 10}, now=self.NOW,
                                    today="2026-11-20")
        self.assertEqual(report["providers"]["Gemini"]["availability"],
                         "key_only_never_observed")

    def test_live_probe_outranks_cache(self):
        report = quota_availability(
            self.coverage(), probes={"Groq": {"ok": True},
                                     "Gemini": {"ok": False, "kind": "rate_limit"}},
            now=self.NOW, today="2026-11-20")
        self.assertEqual(report["providers"]["Groq"]["availability"], "probe_ok")
        self.assertEqual(report["providers"]["Gemini"]["availability"], "probe_failed")


class Benchmark(unittest.TestCase):
    def rows(self):
        rows = []
        for task, (council, single_a, single_b, flagged) in enumerate([
                (1, 1, 0, False), (1, 0, 0, False), (0, 0, 1, True),
                (1, 1, 1, True), (0, 0, 0, True), (1, 0, 1, False)]):
            tid = "t" + str(task)
            rows.append({"task_id": tid, "system": "council", "correct": council,
                         "calls": 16, "latency": 30.0, "audit_completed": True,
                         "auditor_issue_count": int(flagged),
                         "open_issue_count": int(flagged), "equal_budget": True,
                         "unresolved": True})
            rows.append({"task_id": tid, "system": "single:gpt-oss",
                         "correct": single_a, "calls": 1, "latency": 2.0,
                         "equal_budget": False})
            rows.append({"task_id": tid, "system": "single:gemini",
                         "correct": single_b, "calls": 1, "latency": 3.0,
                         "equal_budget": False})
            rows.append({"task_id": tid, "system": "self_consistency",
                         "correct": single_a, "calls": 16, "equal_budget": True})
        return rows

    def test_summary_metrics(self):
        s = summarize(self.rows())
        self.assertAlmostEqual(s["systems"]["council"]["accuracy"], 4 / 6)
        self.assertEqual(s["systems"]["council"]["mean_calls"], 16)
        self.assertAlmostEqual(s["systems"]["single:gpt-oss"]["mean_latency_s"], 2.0)
        a = s["auditor"]["auditor_concern"]
        self.assertEqual((a["wrong_answers"], a["right_answers"]), (2, 4))
        self.assertAlmostEqual(a["catch_rate"], 1.0)
        self.assertAlmostEqual(a["false_alarm_rate"], 0.25)
        per_model = s["per_model_comparison"]["single:gpt-oss"]
        self.assertEqual(per_model["paired_tasks"], 6)
        self.assertAlmostEqual(per_model["council_minus_baseline"], 2 / 6)
        # Single models are not call-matched, so not in the cost-fair table.
        self.assertNotIn("single:gpt-oss", s["call_matched_comparison"])
        self.assertEqual(s["call_matched_comparison"]["self_consistency"]
                         ["paired_tasks"], 6)
        self.assertIn("| council | 6 | 67% |", render(s))

    def test_execute_no_longer_counts_unverified_answers_as_verified(self):
        from eval import execute

        class Eval(Models):
            async def complete(self, family, message, budget, *, role, synthesis=False):
                if role.startswith("baseline"):
                    self.calls.append((family, role))
                    return CallResult(family, "mock", family, "42", 0.0, "h")
                return await super().complete(family, message, budget, role=role,
                                              synthesis=synthesis)

        task = {"id": "T1", "kind": "numeric", "prompt": "6*7?", "expected": "42"}
        with TemporaryDirectory() as folder, EventStore(Path(folder), "e") as events:
            rows = asyncio.run(execute.run_task(task, Eval(), events, deadline=120,
                                                baseline_samples=2))
        council = rows[0]
        self.assertEqual(council["verified"], 0)
        self.assertEqual(council["false_verified"], 0)
        self.assertEqual(council["answer_basis"], "judge_synthesis")
        self.assertTrue(council["audit_completed"])
        singles = {r["system"] for r in rows if r.get("individual_model")}
        self.assertTrue({"single:gpt-oss", "single:gemini"} <= singles)
        self.assertTrue(all(r["correct"] == 1 for r in rows
                            if r.get("individual_model")))


class Report(unittest.IsolatedAsyncioTestCase):
    async def test_degraded_and_stopped_runs_say_so(self):
        down = {("command", "proposal"), ("nemotron", "proposal")}
        degraded = render_markdown(await run_council(Models(fail=down)))
        self.assertIn("Degraded run", degraded)
        self.assertIn("quorum not met", degraded)
        with TemporaryDirectory() as d, EventStore(Path(d), "s") as events:
            runner = FullCouncilRunner(Models(fail=down), events,
                                       RunBudget.start(120, reserve=20),
                                       quorum_policy="stop")
            stopped = render_markdown(await runner.run("Q?", mode="full"))
        self.assertIn("_No answer was produced._", stopped)

    async def test_report_sections(self):
        text = render_markdown(await run_council(Models()), run_id="r1")
        for heading in ("## Answer", "## Competing answers", "## Reviewer objections",
                        "## Independent auditor", "## Run"):
            self.assertIn(heading, text)
        self.assertIn("not independently verified", text)
        self.assertIn("`r1`", text)


if __name__ == "__main__":
    unittest.main()