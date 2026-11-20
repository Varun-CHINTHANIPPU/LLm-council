"""General Council audit regressions: synthetic models, zero external API calls."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from council.budget import RunBudget
from council.events import EventStore
from council.full import FullCouncilRunner, QuickCouncilRunner
from council.transport import CallResult, FamilyUnavailable


class ScriptedModels:
    def __init__(self, failures=()):
        self.failures=set(failures)
        self.calls=[]
        self.role_timeouts={"audit":30.0}

    async def complete(self, family, message, budget, *, role, synthesis=False):
        self.calls.append((family,role))
        if (family,role) in self.failures:
            raise FamilyUnavailable(family,["synthetic rate limit"])
        task=json.loads(message.split("\nDATA_JSON:\n",1)[1])
        if role=="proposal":
            payload={
                "approach":f"{family}: propose an explicit, unproved answer.",
                "final_answer":"",
                "claims":[{"id":"c1",
                           "statement":"A research claim requiring review.",
                           "assumptions":["Model assumptions are incomplete"],
                           "limitations":["No external validation"]}],
                "hypotheses":[],
                "experiments":[]}
        elif role=="review":
            payload={"objections":[{
                "claim_id":"c1",
                "issue":"The stated assumptions may be insufficient.",
                "consequence":"The conclusion could be invalid.",
                "suggested_check":"Review the missing assumption."}]}
        elif role=="revision":
            payload={"revised_approach":"Conditional research answer.",
                     "revised_claims":[],
                     "responses":[]}
        elif role=="synthesis":
            payload={"proposed_approach":"A cautious combined research answer.",
                     "supported_findings":[],
                     "hypotheses":[],
                     "unresolved_objections":[],
                     "proof_obligations":[],
                     "proposed_experiments":[],
                     "next_actions":[],
                     "completion_status":"partial",
                     "answer_comparison":[
                         {"proposal_id":pid,"relation":"adopted",
                          "key_difference":"","reason":"same"}
                         for pid in task["candidate_answers"]]}
        elif role=="rereview":
            payload={"items":[{"objection_id":o["id"],"verdict":"resolved",
                               "reason":"The revision states the assumption."}
                              for o in task["objections"]]}
        elif role=="audit":
            self.last_audit=task
            payload={"issues":["The key assumption is untested."],
                     "suggested_checks":["Investigate the assumption."]}
        else:
            raise AssertionError("Obsolete or unexpected role: "+role)
        return CallResult(family,"mock",family,json.dumps(payload),0.0,"synthetic")


class AuditFirstTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self,transport, *,mode="full",fallbacks=(),judge_fallbacks=()):
        with TemporaryDirectory() as folder:
            with EventStore(Path(folder),"test") as events:
                runner=FullCouncilRunner(
                    transport,events,RunBudget.start(120,reserve=20),
                    auditor_fallbacks=fallbacks,
                    judge_fallbacks=judge_fallbacks,review_depth="thorough")
                result=await runner.run("Explain an abstract research problem.",mode=mode)
                event_names=[x["event"] for x in events.replay()]
                return result,event_names

    async def test_full_cross_reviews_and_independent_audit(self):
        transport=ScriptedModels()
        result,events=await self.exercise(transport)
        roles=[role for _,role in transport.calls]
        self.assertEqual(roles[:3],["proposal"]*3)
        self.assertEqual(roles.count("review"),3)
        self.assertEqual(roles.count("revision"),3)
        self.assertIn("synthesis",roles)
        self.assertEqual(roles[-1],"audit")
        self.assertEqual(result["audit"]["family"],"gemma")
        self.assertTrue(result["audit"]["completed"])
        self.assertEqual(result["verdict"]["status"],"unresolved")
        # A usable answer, honestly labelled; the verdict stays unresolved.
        self.assertEqual(result["final_answer"],"A cautious combined research answer.")
        self.assertEqual(result["answer_selection"]["basis"],"judge_synthesis")
        self.assertFalse(result["answer_selection"]["verified"])
        self.assertEqual(result["research_result"]["supported_findings"],[])
        self.assertEqual(result["role_coverage"]["review_objection_count"],3)
        self.assertIn("The key assumption is untested.",
                      result["research_result"]["unresolved_objections"])
        self.assertIn("audit_received",events)
        self.assertEqual(events[-1],"run_finished")

    async def test_failed_primary_auditor_uses_unused_family(self):
        transport=ScriptedModels({("gemma","audit")})
        result,events=await self.exercise(transport,fallbacks=("minimax",))
        self.assertEqual(
            [f for f,r in transport.calls if r=="audit"],["gemma","minimax"])
        self.assertTrue(result["audit"]["completed"])
        self.assertEqual(result["audit"]["family"],"minimax")
        self.assertEqual(result["verdict"]["status"],"unresolved")
        self.assertIn("audit_received",events)

    async def test_failed_judge_still_reaches_reserved_auditor(self):
        transport=ScriptedModels({("gemini","synthesis"),("glm","synthesis")})
        result,_=await self.exercise(transport,judge_fallbacks=("glm",))
        self.assertFalse(any(f=="glm" and r!="synthesis"
                             for f,r in transport.calls))
        self.assertIn("gemma",[
            f for f,r in transport.calls if r=="audit"])
        self.assertTrue(result["audit"]["completed"])
        self.assertEqual(result["verdict"]["status"],"unresolved")
        self.assertIn("The key assumption is untested.",
                      result["research_result"]["unresolved_objections"])

    async def test_audit_outage_cannot_be_misreported_as_clean(self):
        transport=ScriptedModels({("gemma","audit"),("minimax","audit")})
        result,_=await self.exercise(transport,fallbacks=("minimax",))
        self.assertFalse(result["audit"]["completed"])
        self.assertIsNone(result["audit"]["family"])
        self.assertIn("Independent auditor unavailable",
                      result["research_result"]["limitations"])
        self.assertIn("Full-mode audit not completed",
                      result["research_result"]["limitations"])
        self.assertEqual(result["verdict"]["status"],"unresolved")

    async def test_quick_draft_skips_all_other_models(self):
        with TemporaryDirectory() as folder:
            with EventStore(Path(folder),"quick") as events:
                transport=ScriptedModels()
                result=await QuickCouncilRunner(
                    transport,events,RunBudget.start(50,5)
                ).run("Give an unverified draft.")
                self.assertEqual(transport.calls,[("gpt-oss","proposal")])
                self.assertFalse(result["audit"]["completed"])
                self.assertEqual(result["verdict"]["status"],"unresolved")

    def test_judge_and_audit_fallback_cannot_share_family(self):
        with TemporaryDirectory() as folder:
            with EventStore(Path(folder),"roles") as events:
                with self.assertRaises(ValueError):
                    FullCouncilRunner(
                        ScriptedModels(),events,RunBudget.start(40,5),
                        auditor_fallbacks=("minimax",),
                        judge_fallbacks=("minimax",))


if __name__=="__main__":
    unittest.main()
