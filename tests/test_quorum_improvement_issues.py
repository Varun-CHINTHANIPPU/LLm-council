"""Regressions: quorum policy, bounded improvement with independent
re-review, issue-resolution tracking, and judge comparison of alternatives.

Synthetic models only; zero external API calls. In the default fake, the
`command` author accepts its objection and the other authors reject theirs;
the re-reviewer keeps objections open whenever the author rejected them.
"""
from __future__ import annotations

import json

import unittest

from council.full import FullCouncilRunner
from council.transport import CallResult, FamilyUnavailable

from test_answers_fallbacks_resume import Models, run_council, RunBudget

SOLVERS = ("gpt-oss", "command", "nemotron")


def roles(models, role):
    return [f for f, r in models.calls if r == role]


class Overrides(Models):
    """Models with per-role replacement payloads: role -> fn(task, family)."""

    def __init__(self, overrides, **kw):
        super().__init__(**kw)
        self.overrides = overrides

    async def complete(self, family, message, budget, *, role, synthesis=False):
        if role in self.overrides:
            self.calls.append((family, role))
            self.budgets.append((family, role, budget.remaining()))
            if (family, role) in self.fail:
                raise FamilyUnavailable(family, ["synthetic outage"])
            task = json.loads(message.split("\nDATA_JSON:\n", 1)[1])
            body = self.overrides[role](task, family)
            return CallResult(family, "mock", family, json.dumps(body), 0.0, "h")
        return await super().complete(family, message, budget, role=role,
                                      synthesis=synthesis)


class Quorum(unittest.IsolatedAsyncioTestCase):
    two_down = {("command", "proposal"), ("nemotron", "proposal")}

    async def test_degrade_skips_cross_review_and_judge_but_audits(self):
        models = Models(fail=self.two_down, answers={"gpt-oss": "5"})
        result = await run_council(models)
        self.assertEqual(result["quorum"]["action"],
                         "degraded_no_cross_review_or_judge")
        self.assertFalse(result["quorum"]["met"])
        self.assertEqual(roles(models, "review"), [])
        self.assertEqual(roles(models, "synthesis"), [])
        self.assertEqual(roles(models, "audit"), ["gemma"])
        sel = result["answer_selection"]
        self.assertEqual(sel["status"], "degraded_single_model_unverified")
        self.assertTrue(result["final_answer"].startswith("5"))
        self.assertTrue(any(x.startswith("Quorum not met: 1 of 2")
                            for x in result["research_result"]["limitations"]))

    async def test_stop_spends_no_more_calls_and_gives_no_answer(self):
        models = Models(fail=self.two_down, answers={"gpt-oss": "5"})
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from council.events import EventStore
        with TemporaryDirectory() as d, EventStore(Path(d), "s") as events:
            runner = FullCouncilRunner(models, events,
                                       RunBudget.start(120, reserve=20),
                                       quorum_policy="stop")
            result = await runner.run("Q?", mode="full")
        self.assertEqual([r for _, r in models.calls].count("proposal"), 3)
        self.assertEqual(len(models.calls), 3)  # nothing after the proposals
        self.assertIsNone(result["final_answer"])
        self.assertEqual(result["answer_selection"]["status"], "quorum_stopped")
        self.assertEqual(result["research_result"]["completion_status"], "unresolved")
        self.assertEqual(result["answer_selection"]["alternatives"][0]["family"],
                         "gpt-oss")  # the lone answer is still preserved

    async def test_two_of_three_meets_default_quorum(self):
        models = Models(fail={("nemotron", "proposal")})
        result = await run_council(models)
        self.assertTrue(result["quorum"]["met"])
        self.assertEqual(result["answer_selection"]["basis"], "judge_synthesis")

    async def test_stricter_minimum_degrades(self):
        models = Models(fail={("nemotron", "proposal")})
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from council.events import EventStore
        with TemporaryDirectory() as d, EventStore(Path(d), "m") as events:
            runner = FullCouncilRunner(models, events,
                                       RunBudget.start(120, reserve=20),
                                       min_proposals=3)
            result = await runner.run("Q?", mode="full")
        self.assertFalse(result["quorum"]["met"])
        self.assertEqual(roles(models, "synthesis"), [])

    def test_invalid_policy_rejected(self):
        with self.assertRaises(ValueError):
            FullCouncilRunner(Models(), None, RunBudget.start(10, reserve=1),
                              quorum_policy="maybe")
        with self.assertRaises(ValueError):
            FullCouncilRunner(Models(), None, RunBudget.start(10, reserve=1),
                              min_proposals=1)


class ImprovementAndIssues(unittest.IsolatedAsyncioTestCase):
    async def test_rereview_repair_rereview_and_statuses(self):
        models = Models()
        result = await run_council(models)
        # 3 initial revisions + 2 targeted repairs for the rejected objections.
        self.assertEqual(len(roles(models, "revision")), 5)
        self.assertEqual(len(roles(models, "rereview")), 5)
        by_id = {i["id"]: i for i in result["issues"]}
        # p2 (command) accepted -> independently confirmed fixed.
        p2 = next(i for i in result["issues"] if i["proposal_id"] == "p2")
        self.assertEqual(p2["status"], "addressed")
        # p1, p3 authors reject; the re-reviewer maintains -> contested.
        for pid in ("p1", "p3"):
            issue = next(i for i in result["issues"] if i["proposal_id"] == pid)
            self.assertEqual(issue["status"], "contested")
            self.assertIn("maintains it", issue["reason"])
        self.assertEqual(by_id["a1"]["source"], "audit")
        self.assertEqual(by_id["a1"]["status"], "unreviewed")
        self.assertEqual(result["issue_summary"],
                         {"addressed": 1, "contested": 2, "unresolved": 0,
                          "unreviewed": 1})
        self.assertEqual(result["answer_selection"]["open_issue_count"], 3)

    async def test_rereviewer_is_neither_author_nor_critic(self):
        models = Models()
        result = await run_council(models)
        for issue in result["issues"]:
            if issue["source"] != "peer_review":
                continue
            author = next(p["family"] for p in result["proposals"]
                          if p["id"] == issue["proposal_id"])
            self.assertNotIn(issue["rereview"]["family"],
                             {author, issue["raised_by"]})

    async def test_zero_rounds_rereviews_without_repair(self):
        models = Models()
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from council.events import EventStore
        with TemporaryDirectory() as d, EventStore(Path(d), "z") as events:
            runner = FullCouncilRunner(models, events,
                                       RunBudget.start(120, reserve=20),
                                       improvement_rounds=0,
                                       review_depth="thorough")
            await runner.run("Q?", mode="full")
        self.assertEqual(len(roles(models, "revision")), 3)
        self.assertEqual(len(roles(models, "rereview")), 3)

    async def test_repair_that_works_is_marked_addressed(self):
        seen = {}

        def rereview(task, family):
            items = []
            for o in task["objections"]:
                first = o["id"] not in seen
                seen[o["id"]] = True
                items.append({"objection_id": o["id"],
                              "verdict": "still_open" if first else "resolved",
                              "reason": "missing bound" if first else "bound added"})
            return {"items": items}

        def revision(task, family):
            # Second-round repairs carry the re-reviewer's reason.
            for o in task["objections"]:
                if seen.get(o["id"]):
                    assert o["still_open_because"] == "missing bound"
            return {"revised_approach": "Now with the bound.",
                    "revised_claims": [], "responses": [
                        {"objection_id": o["id"], "disposition": "accept",
                         "explanation": "Added."} for o in task["objections"]]}

        models = Overrides({"rereview": rereview, "revision": revision})
        result = await run_council(models)
        peer = [i for i in result["issues"] if i["source"] == "peer_review"]
        self.assertTrue(all(i["status"] == "addressed" for i in peer))
        self.assertTrue(all(i["rereview"]["round"] == 1 for i in peer))

    async def test_rereview_outage_falls_back_to_author_dispositions(self):
        fail = {(f, "rereview") for f in SOLVERS}
        result = await run_council(Models(fail=fail))
        statuses = {i["proposal_id"]: i for i in result["issues"]
                    if i["source"] == "peer_review"}
        self.assertEqual(statuses["p2"]["status"], "unreviewed")
        self.assertIn("not independently re-reviewed", statuses["p2"]["reason"])
        self.assertEqual(statuses["p1"]["status"], "contested")
        self.assertIn("no independent re-review", statuses["p1"]["reason"])

    async def test_unresolved_disposition_is_tracked(self):
        def revision(task, family):
            return {"revised_approach": None, "revised_claims": [], "responses": [
                {"objection_id": o["id"], "disposition": "unresolved",
                 "explanation": "Needs an experiment."} for o in task["objections"]]}
        fail = {(f, "rereview") for f in SOLVERS}
        result = await run_council(Overrides({"revision": revision}, fail=fail))
        peer = [i for i in result["issues"] if i["source"] == "peer_review"]
        self.assertTrue(all(i["status"] == "unresolved" for i in peer))


class JudgeComparison(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def synthesis(compare_ids, extra=None):
        def body(task, family):
            out = {"proposed_approach": "Consolidated.", "final_answer": "42",
                   "answer_comparison": [
                       {"proposal_id": pid, "relation": "rejected"
                        if pid == "p3" else "adopted",
                        "key_difference": "p3 says 41" if pid == "p3" else "",
                        "reason": "r"} for pid in compare_ids],
                   "unresolved_differences": ["Whether the bound is tight."]}
            out.update(extra or {})
            return out
        return body

    async def test_rejected_alternative_is_preserved_as_difference(self):
        models = Overrides({"synthesis": self.synthesis(["p1", "p2", "p3"])})
        result = await run_council(models)
        self.assertEqual(roles(models, "compare"), [])
        rel = {c["proposal_id"]: c["relation"] for c in result["answer_comparison"]}
        self.assertEqual(rel, {"p1": "adopted", "p2": "adopted", "p3": "rejected"})
        self.assertIn("Whether the bound is tight.", result["unresolved_differences"])
        self.assertIn("p3 (nemotron): p3 says 41", result["unresolved_differences"])

    async def test_missing_comparison_triggers_one_follow_up(self):
        def compare(task, family):
            return {"answer_comparison": [
                {"proposal_id": pid, "relation": "partly_adopted",
                 "key_difference": "uses a weaker bound", "reason": "r"}
                for pid in task["candidate_answers"]],
                "unresolved_differences": []}
        models = Overrides({"synthesis": self.synthesis(["p1"]),
                            "compare": compare})
        result = await run_council(models)
        self.assertEqual(roles(models, "compare"), ["gemini"])
        rel = {c["proposal_id"]: c["relation"] for c in result["answer_comparison"]}
        self.assertEqual(rel["p2"], "partly_adopted")
        self.assertFalse(any("did not compare" in x for x in
                             result["research_result"]["limitations"]))

    async def test_never_compared_answers_are_preserved_and_flagged(self):
        def compare(task, family):
            return {"answer_comparison": [], "unresolved_differences": []}
        models = Overrides({"synthesis": self.synthesis(["p1"]),
                            "compare": compare},
                           answers={"command": "40"})
        result = await run_council(models)
        rel = {c["proposal_id"]: c["relation"] for c in result["answer_comparison"]}
        self.assertEqual(rel["p2"], "not_compared_by_judge")
        self.assertIn("Judge did not compare proposal(s) p2, p3; their answers "
                      "are preserved", result["research_result"]["limitations"])
        self.assertTrue(any(d.startswith("p2 (command) was not compared")
                            and "40" in d
                            for d in result["unresolved_differences"]))

    async def test_judge_sees_issue_status_and_candidate_answers(self):
        captured = {}

        def synthesis(task, family):
            captured.update(task)
            return self.synthesis(["p1", "p2", "p3"])(task, family)
        await run_council(Overrides({"synthesis": synthesis}))
        self.assertEqual(sorted(captured["candidate_answers"]), ["p1", "p2", "p3"])
        self.assertEqual({i["status"] for i in captured["issue_status"]},
                         {"addressed", "contested"})


if __name__ == "__main__":
    unittest.main()