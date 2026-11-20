"""Regressions: final answer, judge/auditor fallbacks, resume.

Synthetic models only; zero external API calls. Nothing here verifies an
answer: every selected answer is labelled with how it was chosen.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from council.budget import RunBudget
from council.events import EventStore
from council.full import FullCouncilRunner, QuickCouncilRunner
from council.replay import ResumeTransport
from council.transport import CallResult, FamilyUnavailable

QUESTION = "What is the answer to the abstract research problem?"


class Interrupted(Exception):
    """Simulates a crash (e.g. Ctrl-C or a killed process) mid-run."""


class Models:
    """Configurable synthetic families.

    fail: {(family, role)} raise FamilyUnavailable every time.
    fail_once: {(family, role)} raise only on the first call.
    crash_on: (family, role) raises Interrupted, simulating a dead process.
    answers: family -> proposal final_answer.
    synthesis_answer: final_answer the judge returns ("" -> omitted).
    wait: family -> seconds reported by soonest_ready (None = unusable);
          omit to model a transport without availability data.
    record: optional event recorder, mimicking FamilyTransport.
    """

    def __init__(self, *, fail=(), fail_once=(), crash_on=None, answers=None,
                 synthesis_answer="Consolidated: 42.", wait=None, record=None,
                 delay=None):
        self.fail = set(fail)
        self.fail_once = set(fail_once)
        self.crash_on = crash_on
        self.answers = answers or {}
        self.synthesis_answer = synthesis_answer
        self.wait = wait
        self.record = record or (lambda *a, **k: None)
        self.delay = delay or {}
        self.calls = []
        self.budgets = []
        self.role_timeouts = {"audit": 30.0}
        if wait is not None:
            self.soonest_ready = lambda family: self.wait.get(family, 0.0)

    async def complete(self, family, message, budget, *, role, synthesis=False):
        self.calls.append((family, role))
        self.budgets.append((family, role, budget.remaining()))
        if (family, role) in self.delay:
            await asyncio.sleep(self.delay[(family, role)])
        if (family, role) == self.crash_on:
            raise Interrupted(family + "/" + role)
        if (family, role) in self.fail:
            raise FamilyUnavailable(family, ["synthetic outage"])
        if (family, role) in self.fail_once:
            self.fail_once.discard((family, role))
            if self.wait is not None:
                self.wait[family] = 0.01  # short cooldown after the failure
            raise FamilyUnavailable(family, ["synthetic transient outage"])
        digest = hashlib.sha256(message.encode()).hexdigest()
        self.record("model_call_started", role=role, family=family,
                    prompt_hash=digest)
        task = json.loads(message.split("\nDATA_JSON:\n", 1)[1])
        if role == "proposal":
            payload = {"approach": family + " reasons about the problem.",
                       "final_answer": self.answers.get(family, ""),
                       "claims": [{"id": "c1", "statement": family + " claim.",
                                   "assumptions": [], "limitations": []}],
                       "hypotheses": [], "experiments": []}
        elif role == "review":
            claim = task["claims"][0]["id"]
            payload = {"objections": [{"claim_id": claim,
                       "issue": "Unclear step.", "consequence": "May be wrong.",
                       "suggested_check": "Recheck it."}]}
        elif role == "revision":
            payload = {"revised_approach": None, "revised_claims": [],
                       "responses": [{"objection_id": o["id"],
                                      "disposition": "accept"
                                      if family == "command" else "reject",
                                      "explanation": "Considered."}
                                     for o in task["objections"]]}
        elif role == "synthesis":
            payload = {"proposed_approach": "Long consolidated reasoning.",
                       "hypotheses": [], "unresolved_objections": [],
                       "proof_obligations": [], "proposed_experiments": [],
                       "next_actions": [], "completion_status": "partial"}
            if self.synthesis_answer:
                payload["final_answer"] = self.synthesis_answer
            payload["answer_comparison"] = [
                {"proposal_id": pid, "relation": "adopted",
                 "key_difference": "", "reason": "same"}
                for pid in task["candidate_answers"]]
        elif role == "rereview":
            # Re-review disagrees with authors who rejected their objection.
            payload = {"items": [{"objection_id": o["id"],
                                  "verdict": "still_open"
                                  if (o.get("author_response") or {})
                                  .get("disposition") == "reject" else "resolved",
                                  "reason": "Checked against the current text."}
                                 for o in task["objections"]]}
        elif role == "compare":
            payload = {"answer_comparison": [
                {"proposal_id": pid, "relation": "adopted",
                 "key_difference": "", "reason": "same"}
                for pid in task["candidate_answers"]],
                "unresolved_differences": []}
        elif role == "audit":
            payload = {"issues": ["One concern."], "suggested_checks": []}
        else:
            raise AssertionError(role)
        text = json.dumps(payload)
        self.record("model_call_succeeded", role=role, family=family,
                    provider="mock", model=family, prompt_hash=digest,
                    response=text, seconds=0.0)
        return CallResult(family, "mock", family, text, 0.0, digest)


def roles_of(models, role):
    return [f for f, r in models.calls if r == role]


async def run_council(transport, *, mode="full", judge_fallbacks=(),
                      auditor_fallbacks=(), folder=None, run_id="t",
                      seconds=120, **options):
    options.setdefault("review_depth", "thorough")
    options.setdefault("task_type", "answer")
    with TemporaryDirectory() as tmp:
        root = Path(folder or tmp)
        with EventStore(root, run_id) as events:
            if mode == "quick":
                runner = QuickCouncilRunner(transport, events,
                                            RunBudget.start(seconds, reserve=20))
            else:
                runner = FullCouncilRunner(
                    transport, events, RunBudget.start(seconds, reserve=20),
                    judge_fallbacks=judge_fallbacks,
                    auditor_fallbacks=auditor_fallbacks, **options)
            return await runner.run(QUESTION, mode=mode)


class FinalAnswer(unittest.IsolatedAsyncioTestCase):
    async def test_judge_final_answer_is_returned_and_labelled(self):
        result = await run_council(Models())
        self.assertEqual(result["final_answer"], "Consolidated: 42.")
        self.assertEqual(result["selected_answer"], "Consolidated: 42.")
        sel = result["answer_selection"]
        self.assertEqual(sel["basis"], "judge_synthesis")
        self.assertEqual(sel["family"], "gemini")
        self.assertFalse(sel["verified"])
        self.assertIn("not independently verified", sel["label"])
        self.assertEqual(sel["auditor_issue_count"], 1)
        self.assertEqual(len(sel["alternatives"]), 3)
        self.assertEqual(result["verdict"]["status"], "unresolved")

    async def test_synthesis_without_final_answer_uses_consolidation(self):
        result = await run_council(Models(synthesis_answer=""))
        self.assertEqual(result["final_answer"], "Long consolidated reasoning.")

    async def test_quick_mode_returns_single_unreviewed_answer(self):
        result = await run_council(Models(answers={"gpt-oss": "7"}), mode="quick")
        self.assertTrue(result["final_answer"].startswith("7"))
        self.assertEqual(result["answer_selection"]["basis"], "single_proposal")
        self.assertIn("not cross-checked", result["answer_selection"]["label"])

    async def test_no_proposals_gives_no_answer(self):
        fail = {(f, "proposal") for f in ("gpt-oss", "command", "nemotron")}
        result = await run_council(Models(fail=fail))
        self.assertIsNone(result["final_answer"])
        self.assertEqual(result["answer_selection"]["status"], "unresolved")


class JudgeFailureRecovery(unittest.IsolatedAsyncioTestCase):
    judges_down = {("gemini", "synthesis"), ("glm", "synthesis")}

    async def test_majority_answer_when_all_judges_fail(self):
        models = Models(fail=self.judges_down,
                        answers={"gpt-oss": "42", "command": "42.",
                                 "nemotron": "41"})
        result = await run_council(models, judge_fallbacks=("glm",))
        sel = result["answer_selection"]
        self.assertEqual(sel["basis"], "proposal_majority_unjudged")
        self.assertEqual(sel["agreeing_proposals"], ["p1", "p2"])
        self.assertTrue(result["final_answer"].startswith("42"))
        self.assertIn("Final answer selected without a judge "
                      "(proposal_majority_unjudged)",
                      result["research_result"]["limitations"])
        self.assertTrue(result["audit"]["completed"])

    async def test_least_contested_answer_without_agreement(self):
        # command accepts its objection in revision; others reject theirs.
        models = Models(fail=self.judges_down,
                        answers={"gpt-oss": "1", "command": "2",
                                 "nemotron": "3"})
        result = await run_council(models, judge_fallbacks=("glm",))
        sel = result["answer_selection"]
        self.assertEqual(sel["basis"], "least_contested_proposal_unjudged")
        self.assertEqual(sel["family"], "command")
        self.assertEqual(sel["open_objections"], 0)
        self.assertFalse(sel["verified"])

    async def test_judge_fallback_used_before_deterministic_choice(self):
        models = Models(fail={("gemini", "synthesis")})
        result = await run_council(models, judge_fallbacks=("glm",))
        self.assertEqual(result["answer_selection"]["basis"], "judge_synthesis")
        self.assertEqual(result["answer_selection"]["family"], "glm")


class AuditorAvailability(unittest.IsolatedAsyncioTestCase):
    async def test_ready_fallback_goes_before_primary_in_cooldown(self):
        models = Models(wait={"gemma": 500.0})
        result = await run_council(models, auditor_fallbacks=("minimax",))
        self.assertEqual([f for f, r in models.calls if r == "audit"],
                         ["minimax"])
        self.assertEqual(result["audit"]["family"], "minimax")

    async def test_unusable_family_is_skipped_without_a_call(self):
        models = Models(wait={"gemma": None})
        result = await run_council(models, auditor_fallbacks=("minimax",))
        self.assertNotIn(("gemma", "audit"), models.calls)
        self.assertTrue(result["audit"]["completed"])

    async def test_transient_outage_is_retried_once_after_cooldown(self):
        models = Models(wait={}, fail_once={("gemma", "audit")})
        result = await run_council(models)
        self.assertEqual([f for f, r in models.calls if r == "audit"],
                         ["gemma", "gemma"])
        self.assertTrue(result["audit"]["completed"])

    async def test_persistent_outage_is_retried_at_most_once(self):
        models = Models(wait={"gemma": 0.0}, fail={("gemma", "audit")})
        result = await run_council(models)
        self.assertLessEqual(
            len([1 for f, r in models.calls if r == "audit"]), 2)
        self.assertFalse(result["audit"]["completed"])
        self.assertIn("Independent auditor unavailable",
                      result["research_result"]["limitations"])

    async def test_hung_primary_auditor_is_hedged_by_fallback(self):
        models = Models(delay={("gemma", "audit"): 30.0})
        started = asyncio.get_running_loop().time()
        result = await run_council(models, auditor_fallbacks=("minimax",),
                                   hedge_seconds=0.2)
        self.assertLess(asyncio.get_running_loop().time() - started, 5.0)
        self.assertEqual(result["audit"]["family"], "minimax")
        self.assertEqual(roles_of(models, "audit"), ["gemma", "minimax"])


class ResumeEndToEnd(unittest.IsolatedAsyncioTestCase):
    async def test_interrupted_runs_resume_without_repeating_calls(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            # Run A dies during the judge step.
            with EventStore(root, "a") as events:
                live = Models(crash_on=("gemini", "synthesis"),
                              record=events.record)
                runner = FullCouncilRunner(live, events,
                                           RunBudget.start(120, reserve=20),
                                           review_depth="thorough")
                with self.assertRaises(Interrupted):
                    await runner.run(QUESTION, mode="full")
                rows_a = events.replay()
            done_a = sum(r["event"] == "model_call_succeeded" for r in rows_a)
            self.assertEqual(done_a, 16)  # 3 proposals, reviews, revisions;
            # 3 re-reviews, 2 targeted repairs, 2 confirming re-reviews

            # Resume B replays those 16 and dies again during the audit.
            with EventStore(root, "b") as events:
                live = Models(crash_on=("gemma", "audit"), record=events.record)
                transport = ResumeTransport(rows_a, live, record=events.record)
                runner = FullCouncilRunner(transport, events,
                                           RunBudget.start(120, reserve=20),
                                           review_depth="thorough")
                with self.assertRaises(Interrupted):
                    await runner.run(QUESTION, mode="full")
                rows_b = events.replay()
            self.assertEqual(live.calls, [("gemini", "synthesis"),
                                          ("gemma", "audit")])
            self.assertEqual(transport.replayed, 16)

            # Resume C from B: everything before the audit must come from B.
            with EventStore(root, "c") as events:
                live = Models(record=events.record)
                transport = ResumeTransport(rows_b, live, record=events.record)
                runner = FullCouncilRunner(transport, events,
                                           RunBudget.start(120, reserve=20),
                                           review_depth="thorough")
                result = await runner.run(QUESTION, mode="full")
            self.assertEqual(live.calls, [("gemma", "audit")])
            self.assertEqual(transport.replayed, 17)
            self.assertEqual(result["final_answer"], "Consolidated: 42.")
            self.assertTrue(result["audit"]["completed"])


if __name__ == "__main__":
    unittest.main()