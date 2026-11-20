"""Execution-efficiency and research-output regressions (Nov 20 LWE run).

Covers: document mode (outline from instructions, parallel drafting,
continuation, cross-family review, targeted revision, audit), the fast answer
pipeline, stage timing, the per-run circuit breaker, local repair of
malformed output, hedged judge calls and JSON-mode exemption for prose roles.
Synthetic models only; zero external API calls.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

from council.budget import RunBudget
from council.document import extract_sections, split_sections
from council.events import EventStore
from council.full import FullCouncilRunner
from council.report import render_markdown
from council.structured import response_format
from council.transport import CallResult, FamilyUnavailable

from test_answers_fallbacks_resume import Models

TITLES = ["Problem statement", "Background on LWE", "Noise distributions",
          "Decryption failure model", "Attack landscape", "Hypotheses",
          "Parameter sets", "Experimental design", "Metrics", "Baselines",
          "Statistical tests", "Implementation", "Reproducibility", "Threats",
          "Ethics", "Timeline", "Resources", "Expected results",
          "Limitations", "References policy"]
INSTRUCTIONS = ("Produce a complete research design with these sections:\n"
                + "\n".join(f"{n}. {t}" for n, t in enumerate(TITLES, 1))
                + "\n\nResearch idea (included at the end): investigate how LWE "
                  "noise distributions affect decryption failures and attack cost.")
BODY = ("Substantive content with parameters, metrics and reasoning. " * 6).strip()


class DocModels(Models):
    """Document-mode fakes on top of the answer-mode fakes.

    truncate: writer -> how many of its requested sections it returns.
    major_every: sections whose number is divisible by this get a major finding.
    short_revision: revisions come back truncated (must be rejected).
    """

    def __init__(self, *, truncate=None, major_every=5, short_revision=False, **kw):
        super().__init__(**kw)
        self.truncate = truncate or {}
        self.major_every = major_every
        self.short_revision = short_revision
        self.reviewed = []  # (reviewer, section numbers)

    async def complete(self, family, message, budget, *, role, synthesis=False):
        if role not in {"draft_sections", "section_review", "section_revise", "outline"}:
            return await super().complete(family, message, budget, role=role,
                                          synthesis=synthesis)
        self.calls.append((family, role))
        if (family, role) in self.delay:
            await asyncio.sleep(self.delay[(family, role)])
        if (family, role) in self.fail:
            raise FamilyUnavailable(family, ["synthetic outage"])
        task = json.loads(message.split("\nDATA_JSON:\n", 1)[1])
        if role == "draft_sections":
            wanted = task["write_sections"][:self.truncate.get(family, 99)]
            text = "\n\n".join(f"## {s['number']}. {s['title']}\n\n{BODY} "
                               f"(by {family})" for s in wanted)
        elif role == "section_review":
            self.reviewed.append((family, [s["number"] for s in task["sections"]]))
            text = json.dumps({"findings": [
                {"section": s["number"], "severity": "major"
                 if s["number"] % self.major_every == 0 else "minor",
                 "issue": "Missing concrete parameter ranges.", "fix": "Add them."}
                for s in task["sections"]]})
        elif role == "section_revise":
            extra = "" if self.short_revision else " Revised with parameter ranges."
            text = "\n\n".join(
                f"## {s['number']}. {s['title']}\n\n"
                + (BODY[:90] if self.short_revision else s["text"] + extra)
                for s in task["sections"])
        else:
            text = json.dumps({"sections": ["Intro", "Method", "Results"]})
        return CallResult(family, "mock", family, text, 0.0, "h")


async def run(models, question=INSTRUCTIONS, *, seconds=300, **options):
    with TemporaryDirectory() as folder, EventStore(Path(folder), "d") as events:
        runner = FullCouncilRunner(models, events, RunBudget.start(seconds, reserve=20),
                                   **options)
        result = await runner.run(question, mode="full")
        return result, [e["event"] for e in events.replay()]


def roles(models, role):
    return [f for f, r in models.calls if r == role]


class SectionParsing(unittest.TestCase):
    def test_extracts_required_sections_from_instructions(self):
        self.assertEqual(extract_sections(INSTRUCTIONS), TITLES)

    def test_heading_styles(self):
        text = "\n".join(["## 1. A", "## 2) B", "Section 3: C", "**4. D**", "5. E"])
        self.assertEqual(extract_sections(text), ["A", "B", "C", "D", "E"])

    def test_short_lists_and_subsections_do_not_trigger_document_mode(self):
        self.assertEqual(extract_sections("Steps:\n1. a\n2. b\n3. c"), [])
        text = "\n".join(["1. A", "1.1 sub", "2. B", "3. C", "3.2 sub", "4. D", "5. E"])
        self.assertEqual(extract_sections(text), ["A", "B", "C", "D", "E"])

    def test_split_keeps_only_requested_substantive_sections(self):
        md = f"## 1. A\n\n{BODY}\n\n## 2. B\n\ntoo short\n\n## 3. C\n\n{BODY}"
        self.assertEqual(sorted(split_sections(md, [1, 2])), [1])

    def test_prose_roles_never_forced_into_json_mode(self):
        for provider, model in (("Cohere", "command-a"), ("Groq", "openai/gpt-oss-120b")):
            self.assertIsNone(response_format(provider, model, "draft_sections"))
            self.assertIsNotNone(response_format(provider, model, "review"))


class DocumentMode(unittest.IsolatedAsyncioTestCase):
    async def test_full_document_is_delivered_reviewed_and_audited(self):
        models = DocModels()
        result, events = await run(models)
        doc = result["document"]
        self.assertEqual((doc["required_sections"], doc["delivered"]), (20, 20))
        self.assertEqual(result["task_type"], "document")
        answer = result["final_answer"]
        positions = [answer.index(f"## {n}. {t}") for n, t in enumerate(TITLES, 1)]
        self.assertEqual(positions, sorted(positions))
        # 20 sections in blocks of 4 = 5 draft calls; one review per writer.
        self.assertEqual(len(roles(models, "draft_sections")), 5)
        self.assertEqual(len(roles(models, "section_review")), 3)
        self.assertEqual(roles(models, "outline"), [])  # read from instructions
        for s in doc["sections"]:
            self.assertNotEqual(s["writer"], s["reviewer"])
        # Only sections 5, 10, 15, 20 had major findings.
        self.assertEqual(doc["revised"], 4)
        self.assertIn("Revised with parameter ranges", answer)
        self.assertTrue(result["audit"]["completed"])
        self.assertEqual(result["answer_selection"]["status"],
                         "document_reviewed_unverified")
        self.assertFalse(result["answer_selection"]["verified"])
        self.assertIn("document_assembled", events)
        stages = [s["stage"] for s in result["timing"]["stages"]]
        self.assertEqual(stages, ["outline", "drafting", "review", "revision", "audit"])
        report = render_markdown(result)
        self.assertIn("Sections delivered: 20 of 20", report)
        self.assertIn("| drafting |", report)

    async def test_failed_writer_is_continued_by_others(self):
        models = DocModels(fail={("command", "draft_sections")})
        result, events = await run(models)
        doc = result["document"]
        self.assertEqual(doc["delivered"], 20)
        self.assertIn("draft_continuation", events)
        self.assertNotIn("command", {s["writer"] for s in doc["sections"]})
        self.assertIn("command", result["timing"]["open_circuits"])
        for s in doc["sections"]:
            self.assertNotEqual(s["writer"], s["reviewer"])

    async def test_truncated_reply_is_continued_not_restarted(self):
        models = DocModels(truncate={"gpt-oss": 2})
        result, _ = await run(models)
        self.assertEqual(result["document"]["delivered"], 20)
        first = len([1 for f in roles(models, "draft_sections")])
        self.assertGreater(first, 5)  # continuation calls were added
        self.assertLessEqual(first, 5 + 6)  # only for the missing sections

    async def test_review_outage_still_returns_the_document(self):
        fail = {(f, "section_review") for f in ("gpt-oss", "command", "nemotron")}
        result, _ = await run(DocModels(fail=fail))
        self.assertEqual(result["document"]["delivered"], 20)
        self.assertEqual(result["answer_selection"]["status"],
                         "document_unreviewed_unverified")
        self.assertTrue(any("not reviewed by another model family" in x
                            for x in result["research_result"]["limitations"]))

    async def test_truncated_revision_is_rejected(self):
        result, _ = await run(DocModels(short_revision=True))
        self.assertEqual(result["document"]["revised"], 0)
        statuses = {i["status"] for i in result["issues"]
                    if i["source"] == "section_review" and i["severity"] == "major"}
        self.assertEqual(statuses, {"unresolved"})

    async def test_all_writers_down_gives_no_document(self):
        fail = {(f, "draft_sections") for f in ("gpt-oss", "command", "nemotron")}
        result, _ = await run(DocModels(fail=fail))
        self.assertIsNone(result["final_answer"])
        self.assertEqual(result["answer_selection"]["status"], "unresolved")
        self.assertIn("Sections not delivered: 1, 2", " ".join(
            result["research_result"]["limitations"]))

    async def test_forced_document_mode_drafts_an_outline(self):
        models = DocModels()
        result, _ = await run(models, "Write a short study design.",
                              task_type="document")
        self.assertEqual(len(roles(models, "outline")), 1)
        self.assertEqual(result["document"]["delivered"], 3)

    async def test_answer_questions_stay_in_answer_mode(self):
        models = DocModels()
        result, _ = await run(models, "What is the LWE problem?")
        self.assertEqual(result["task_type"], "answer")
        self.assertEqual(roles(models, "draft_sections"), [])


class FastAnswerMode(unittest.IsolatedAsyncioTestCase):
    async def test_fast_default_makes_eight_calls_in_four_stages(self):
        models = Models()
        result, _ = await run(models, "What is the answer?")
        self.assertEqual([r for _, r in models.calls].count("revision"), 0)
        self.assertEqual([r for _, r in models.calls].count("rereview"), 0)
        self.assertEqual(len(models.calls), 8)
        self.assertEqual([s["stage"] for s in result["timing"]["stages"]],
                         ["proposals", "review", "judge", "audit"])
        self.assertEqual(result["review_depth"], "fast")
        # Objections are tracked even without a revision round.
        self.assertEqual(result["issue_summary"]["unreviewed"], 4)

    async def test_reviews_run_concurrently(self):
        delay = {(f, "review"): 0.4 for f in ("gpt-oss", "command", "nemotron")}
        started = time.monotonic()
        await run(Models(delay=delay), "What is the answer?")
        self.assertLess(time.monotonic() - started, 1.0)  # not 3 x 0.4 s

    async def test_timing_counts_network_attempts_and_failures(self):
        models = Models(fail={("gemini", "synthesis")})
        models.record = lambda *a, **k: None
        result, _ = await run(models, "What is the answer?", judge_fallbacks=("glm",))
        stages = {s["stage"]: s for s in result["timing"]["stages"]}
        self.assertEqual(stages["proposals"]["network_attempts"], 3)
        self.assertEqual(stages["judge"]["role_calls_failed"], 1)
        self.assertEqual(stages["judge"]["role_calls_ok"], 1)


class CircuitBreaker(unittest.IsolatedAsyncioTestCase):
    async def test_failed_family_is_not_called_again(self):
        models = Models(fail={("command", "review")})
        result, events = await run(models, "What is the answer?",
                                   review_depth="thorough")
        self.assertEqual(roles(models, "revision").count("command"), 0)
        self.assertEqual(roles(models, "rereview").count("command"), 0)
        self.assertIn("command", result["timing"]["open_circuits"])
        self.assertIn("circuit_opened", events)

    async def test_can_be_disabled(self):
        models = Models(fail={("command", "review")})
        await run(models, "What is the answer?", review_depth="thorough",
                  circuit_breaker=False)
        self.assertGreater(roles(models, "revision").count("command"), 0)


class Salvage(Models):
    def __init__(self, replies, **kw):
        super().__init__(**kw)
        self.replies = replies

    async def complete(self, family, message, budget, *, role, synthesis=False):
        if (family, role) in self.replies:
            self.calls.append((family, role))
            return CallResult(family, "mock", family, self.replies[(family, role)],
                              0.0, "h")
        return await super().complete(family, message, budget, role=role,
                                      synthesis=synthesis)


class LocalRepair(unittest.IsolatedAsyncioTestCase):
    async def test_prose_synthesis_is_kept_without_a_retry(self):
        prose = "The consolidated answer is 42 because... " * 20
        models = Salvage({("gemini", "synthesis"): prose})
        result, events = await run(models, "What is the answer?")
        self.assertEqual(roles(models, "synthesis"), ["gemini"])
        self.assertIn("output_salvaged", events)
        self.assertEqual(result["answer_selection"]["basis"], "judge_synthesis")
        self.assertTrue(result["final_answer"].startswith("The consolidated answer"))

    async def test_sloppy_json_is_repaired(self):
        sloppy = json.dumps({"proposed_approach": "Answer: 42.", "final_answer": "42",
                             "hypotheses": "one hypothesis", "confidence": 0.9,
                             "answer_comparison": [{"proposal_id": "p1",
                                                    "relation": "maybe"}]})
        models = Salvage({("gemini", "synthesis"): sloppy})
        result, _ = await run(models, "What is the answer?")
        self.assertEqual(result["final_answer"], "42")
        self.assertIn("one hypothesis", result["research_result"]["hypotheses"])

    async def test_prose_proposal_becomes_a_flagged_claim(self):
        prose = "Detailed answer. " + "More reasoning. " * 40
        models = Salvage({("command", "proposal"): prose})
        result, _ = await run(models, "What is the answer?")
        self.assertIn("command", result["role_coverage"]["valid_proposer_families"])
        claim = next(c for c in result["claims"] if c["owner_family"] == "command")
        self.assertIn("Recovered from unstructured output", claim["limitations"])

    async def test_short_garbage_still_gets_one_format_retry(self):
        models = Salvage({("gemini", "synthesis"): "oops"})
        await run(models, "What is the answer?")
        self.assertEqual(roles(models, "synthesis"), ["gemini", "gemini"])


class Hedging(unittest.IsolatedAsyncioTestCase):
    async def test_silent_judge_is_hedged(self):
        models = Models(delay={("gemini", "synthesis"): 30.0})
        started = time.monotonic()
        result, events = await run(models, "What is the answer?",
                                   judge_fallbacks=("glm",), hedge_seconds=0.2)
        self.assertLess(time.monotonic() - started, 5.0)
        self.assertEqual(result["answer_selection"]["family"], "glm")
        self.assertIn("role_hedged", events)


if __name__ == "__main__":
    unittest.main()