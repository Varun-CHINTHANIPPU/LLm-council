"""General-purpose Council regressions using only offline fake model calls."""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from council.budget import RunBudget
from council.cli import main
from council.document import extract_sections
from council.events import EventStore
from council.full import FullCouncilRunner
from council.models import CouncilResult, ResearchResult
from council.prompts import prompt
from council.report import render_markdown
from council.schemas import validate
from test_answers_fallbacks_resume import Models


class GeneralPurposeBehavior(unittest.TestCase):
    def test_role_prompts_allow_nonscientific_tasks(self):
        for role in ("proposal", "review", "synthesis", "audit", "draft_sections"):
            content = prompt(role, {"question": "Write a friendly birthday note."})
            self.assertIn("DATA_JSON:", content)
            self.assertIn("birthday note", content)
        self.assertIn("general-purpose", prompt("proposal", {}).lower())
        self.assertIn("creative", prompt("review", {}).lower())

    def test_long_synthesis_no_longer_rejected_by_short_answer_schema(self):
        response = validate("synthesis", {
            "proposed_approach": "A detailed answer",
            "final_answer": "X" * 12000,
        })
        self.assertEqual(len(response["final_answer"]), 12000)

    def test_general_result_with_backward_compatible_alias(self):
        item = CouncilResult(research_question="Write a poem")
        self.assertIs(CouncilResult, ResearchResult)
        self.assertEqual(item.request, "Write a poem")
        self.assertEqual(item.to_dict()["request"],
                         item.to_dict()["research_question"])

    def test_nonresearch_prompt_does_not_trigger_document_mode(self):
        self.assertEqual(extract_sections("Write a haiku about the sea."), [])
        self.assertEqual(extract_sections("1. a\n2. b\n3. c"), [])
        self.assertEqual(
            extract_sections("## 1. A\n## 2. B\n## 3. C\n## 4. D\n## 5. E"),
            ["A", "B", "C", "D", "E"])

    def test_start_is_general_and_keeps_venv_support(self):
        script = (Path(__file__).resolve().parents[1] / "start").read_text()
        self.assertIn(".venv/bin/python", script)
        self.assertIn("--task-type auto", script)

    def test_standard_mode_dry_run_uses_ordinary_answer(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory) / "input"
            folder.mkdir()
            (folder / "01.txt").write_text("Write a birthday note")
            (folder / "02.txt").write_text("Friendly and concise")
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(["--input-dir", str(folder), "--dry-run",
                             "--mode", "standard", "--config",
                             str(Path(directory) / "missing.yaml")])
            payload = json.loads(out.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(payload["mode"], "standard")
            self.assertEqual(payload["task_type"], "answer")
            self.assertEqual(payload["api_calls"], 0)

    def test_report_generic_heading_and_legacy_input(self):
        report = render_markdown({
            "research_result": {"research_question": "Write a haiku"},
            "final_answer": "Morning dew",
            "answer_selection": {"label": "Unverified"},
        })
        self.assertIn("# Council response", report)
        self.assertIn("Write a haiku", report)
        self.assertIn("Morning dew", report)


class GeneralPurposeRunner(unittest.IsolatedAsyncioTestCase):
    async def test_standard_nonresearch_task_uses_answer_pipeline(self):
        with TemporaryDirectory() as directory:
            with EventStore(Path(directory), "general") as events:
                models = Models(synthesis_answer="Happy birthday! Enjoy your day.")
                runner = FullCouncilRunner(
                    models, events, RunBudget.start(120, reserve=20),
                    solvers=("gpt-oss", "command", "nemotron"),
                    judge_family="gemini", auditor_family="gemma",
                    task_type="auto", review_depth="fast")
                result = await runner.run("Write a cheerful birthday note.",
                                          mode="standard")
                self.assertEqual(result["task_type"], "answer")
                self.assertIn("Happy birthday", result["final_answer"])
                self.assertEqual(result["council_result"]["request"],
                                 "Write a cheerful birthday note.")
                self.assertEqual(result["research_result"]["research_question"],
                                 "Write a cheerful birthday note.")
                self.assertNotIn("draft_sections",
                                 [role for _, role in models.calls])


if __name__ == "__main__":
    unittest.main()
