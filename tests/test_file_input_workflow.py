"""Two-file input/output workflow, using offline fake runners only."""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from council.file_io import InputFolderError, atomic_write_text, read_two_text_files
from council.cli import main


class TwoFileWorkflowTests(unittest.TestCase):
    def test_exactly_two_sorted_texts_concatenate_with_blank_line(self):
        with TemporaryDirectory() as root:
            folder = Path(root) / "input"
            folder.mkdir()
            (folder / "z_context.txt").write_text("Second: β\n", encoding="utf-8")
            (folder / "A_question.TXT").write_text("First: α\n", encoding="utf-8")
            (folder / "ignored.md").write_text("not part of the prompt")
            text, paths = read_two_text_files(folder)
            self.assertEqual(text, "First: α\n\nSecond: β\n")
            self.assertEqual([x.name for x in paths], ["A_question.TXT","z_context.txt"])

    def test_missing_extra_and_empty_files_are_rejected(self):
        with TemporaryDirectory() as root:
            folder = Path(root) / "input"
            with self.assertRaises(InputFolderError):
                read_two_text_files(folder)
            folder.mkdir()
            (folder / "one.txt").write_text("First")
            with self.assertRaisesRegex(InputFolderError, "found 1"):
                read_two_text_files(folder)
            (folder / "two.txt").write_text("   ")
            with self.assertRaisesRegex(InputFolderError, "empty"):
                read_two_text_files(folder)
            (folder / "two.txt").write_text("Second")
            (folder / "three.txt").write_text("Third")
            with self.assertRaisesRegex(InputFolderError, "found 3"):
                read_two_text_files(folder)

    def test_invalid_utf8_is_reported_before_any_provider_call(self):
        with TemporaryDirectory() as root:
            folder = Path(root)
            (folder / "a.txt").write_bytes(b"\xff")
            (folder / "b.txt").write_text("Valid")
            with self.assertRaisesRegex(InputFolderError, "UTF-8"):
                read_two_text_files(folder)

    def test_output_overwrites_old_result_and_keeps_no_temp(self):
        with TemporaryDirectory() as root:
            out = Path(root) / "output" / "answer.txt"
            atomic_write_text(out, "first result\n")
            atomic_write_text(out, "new result\n")
            self.assertEqual(out.read_text(), "new result\n")
            self.assertEqual(list(out.parent.iterdir()), [out])

    def test_cli_dry_run_reads_input_without_live_calls_or_output(self):
        with TemporaryDirectory() as root:
            folder = Path(root)/"input"
            folder.mkdir()
            (folder/"01.txt").write_text("Question one\n",encoding="utf-8")
            (folder/"02.txt").write_text("More context\n",encoding="utf-8")
            stream=io.StringIO()
            with redirect_stdout(stream):
                result=main(["--input-dir",str(folder),"--output-dir",
                             str(Path(root)/"output"),"--dry-run",
                             "--config",str(Path(root)/"absent.yaml")])
            payload=json.loads(stream.getvalue())
            self.assertEqual(result,0)
            self.assertEqual(payload["question"],"Question one\n\nMore context\n")
            self.assertEqual(payload["api_calls"],0)
            self.assertEqual(payload["input_mode"],"two_files")
            self.assertFalse((Path(root)/"output").exists())

    def test_file_input_conflict_is_refused(self):
        with TemporaryDirectory() as root:
            stream=io.StringIO()
            with redirect_stderr(stream):
                with self.assertRaises(SystemExit) as error:
                    main(["--input-dir",str(root),"This is another question"])
            self.assertEqual(error.exception.code,2)
            self.assertIn("cannot be combined",stream.getvalue())

    def test_cli_full_file_flow_uses_concatenation_and_overwrites(self):
        with TemporaryDirectory() as root:
            root=Path(root)
            inp=root/"input"
            inp.mkdir()
            (inp/"1.txt").write_text("Explain alpha",encoding="utf-8")
            (inp/"2.txt").write_text("using beta",encoding="utf-8")
            captured=[]
            class FakeRunner:
                async def run(self, question, *, mode):
                    captured.append((question,mode))
                    return {
                      "research_result":{
                        "research_question":question,
                        "proposed_approach":"Answer",
                        "completion_status":"partial",
                        "limitations":[]},
                      "final_answer":"Combined response #" + str(len(captured)),
                      "answer_selection":{
                        "basis":"judge_synthesis",
                        "label":"Model-reviewed, unverified"},
                      "audit":{"completed":False},
                      "quorum":{"met":True,"obtained":2,"requested":2},
                      "role_coverage":{"judge_family":None},
                      "actual_models":[],
                    }
            out=root/"output"
            for count in (1,2):
                stream=io.StringIO()
                with patch("council.cli.build_runner",return_value=FakeRunner()), \
                     patch("council.cli.FamilyTransport",return_value=object()), \
                     redirect_stdout(stream):
                    main(["--mode","full","--input-dir",str(inp),
                          "--output-dir",str(out),
                          "--run-dir",str(root/"runs"),
                          "--config",str(root/"absent.yaml")])
                self.assertIn("Combined response #"+str(count),
                              (out/"answer.txt").read_text(encoding="utf-8"))
                if count==2:
                    self.assertNotIn("Combined response #1",
                                     (out/"answer.txt").read_text())
            self.assertEqual(captured,[("Explain alpha\n\nusing beta\n","full")]*2)
            self.assertEqual((out/"combined_input.txt").read_text(),
                             "Explain alpha\n\nusing beta\n")
            self.assertEqual(len(list(out.iterdir())),2)

    def test_json_stdout_kept_while_text_answer_is_written(self):
        with TemporaryDirectory() as root:
            folder=Path(root)
            inp=folder/"input"
            inp.mkdir()
            (inp/"a.txt").write_text("A")
            (inp/"b.txt").write_text("B")
            class FakeRunner:
                async def run(self, question, *, mode):
                    return {"research_result":{"research_question":question},
                            "final_answer":"A and B",
                            "answer_selection":{"basis":"single_proposal"},
                            "audit":{"completed":False}}
            stream=io.StringIO()
            with patch("council.cli.build_runner",return_value=FakeRunner()), \
                 patch("council.cli.FamilyTransport",return_value=object()), \
                 redirect_stdout(stream):
                main(["--input-dir",str(inp),"--output-dir",str(folder/"output"),
                      "--run-dir",str(folder/"runs"),"--json",
                      "--config",str(folder/"absent.yaml")])
            self.assertEqual(json.loads(stream.getvalue())["final_answer"],"A and B")
            self.assertIn("A and B",(folder/"output"/"answer.txt").read_text())


if __name__=="__main__":
    unittest.main()
