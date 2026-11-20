"""Council pipelines: answer mode and document mode.

Answer mode (fast, default): sealed parallel proposals -> parallel cross-
review -> judge synthesis (hedged) -> independent audit. review_depth
"thorough" adds author revisions and independent re-review rounds.
Document mode (council.document): outline -> parallel section drafting with
continuation -> cross-family section review -> targeted revision -> audit.

Every stage is timed. A usable answer exists as early as possible; later
stages can only improve it. Model agreement is never treated as proof.
"""
from __future__ import annotations
import asyncio
from contextlib import contextmanager
import json
import re
import time
from dataclasses import asdict
from council.budget import RunBudget
from council.families import FAMILIES, validate_assignment
from council.ledger import SQLiteLedger
from council.models import (Claim, ClaimLedger, Objection, AuthorResponse,
                            CouncilResult)
from council.parse import (OutputError, parse_object, require_string,
                           string_list, item_list)
from council.prompts import prompt
from council.plan_sanity import is_plan_instead_of_answer
from council.schemas import validate as validate_output
from council.structured import TEXT_ROLES
from council.transport import FamilyUnavailable

DEFAULT_SOLVERS = ("gpt-oss", "command", "nemotron")
JUDGE = "gemini"
AUDITOR = "glm"
# degrade: with fewer valid proposals than min_proposals, skip cross-review
#          and the judge, still run the independent audit, and return the
#          single answer labelled as a degraded single-model result.
# stop:    spend no further calls and return no final answer.
QUORUM_POLICIES = ("degrade", "stop")
ISSUE_STATUSES = ("addressed", "contested", "unresolved", "unreviewed")
REVIEW_DEPTHS = ("fast", "thorough")
TASK_TYPES = ("auto", "answer", "document")
# A failure that says nothing about the provider (our own deadline).
_SELF_INFLICTED = ("deadline", "insufficient_time", "budget_timeout")
# Long-generation roles get longer per-request allowances than the default.
LONG_ROLE_SECONDS = {"draft_sections": 150.0, "section_revise": 120.0,
                     "synthesis": 120.0, "section_review": 90.0}


class CouncilRunner:
    def __init__(self, transport, events, budget: RunBudget, *,
                 solvers: tuple[str, ...] = DEFAULT_SOLVERS,
                 judge_family: str = JUDGE,
                 auditor_family: str | None = AUDITOR,
                 auditor_fallbacks: tuple[str,...] = (),
                 judge_fallbacks: tuple[str,...] = (),
                 solver_strategies: dict | None = None,
                 quorum_policy: str = "degrade",
                 min_proposals: int = 2,
                 improvement_rounds: int = 1,
                 review_depth: str = "fast",
                 task_type: str = "auto",
                 hedge_seconds: float = 30.0,
                 circuit_breaker: bool = True):
        validate_assignment(solvers, judge_family, auditor_family)
        if quorum_policy not in QUORUM_POLICIES:
            raise ValueError("quorum_policy must be degrade or stop")
        if type(min_proposals) is not int or not 1 <= min_proposals <= 10:
            raise ValueError("min_proposals must be an integer from 1 to 10")
        if len(solvers) > 1 and min_proposals < 2:
            raise ValueError("Cross-review needs min_proposals of at least 2")
        min_proposals = min(min_proposals, len(solvers))
        if type(improvement_rounds) is not int or not 0 <= improvement_rounds <= 3:
            raise ValueError("improvement_rounds must be 0..3")
        if review_depth not in REVIEW_DEPTHS:
            raise ValueError("review_depth must be fast or thorough")
        if task_type not in TASK_TYPES:
            raise ValueError("task_type must be auto, answer or document")
        if not 0 < float(hedge_seconds) <= 600:
            raise ValueError("hedge_seconds must be in (0, 600]")
        self.quorum_policy = quorum_policy
        self.min_proposals = min_proposals
        self.improvement_rounds = improvement_rounds
        self.review_depth = review_depth
        self.task_type = task_type
        self.hedge_seconds = float(hedge_seconds)
        self.circuit_breaker = bool(circuit_breaker)
        self.transport = transport
        self.events = events
        self.budget = budget
        self.solvers = solvers
        self.judge_family = judge_family
        self.auditor_family = auditor_family
        self.auditor_fallbacks = tuple(auditor_fallbacks)
        self.judge_fallbacks = tuple(judge_fallbacks)
        self.solver_strategies = dict(solver_strategies or {})
        if any(value not in {"direct","code","decompose"}
               for value in self.solver_strategies.values()):
            raise ValueError("Invalid solver strategy")
        reserved=set(solvers)|{judge_family,auditor_family}
        for group in (self.auditor_fallbacks,self.judge_fallbacks):
            if (len(set(group))!=len(group)
                    or any(f not in FAMILIES or f in reserved for f in group)):
                raise ValueError("Fallback families must be distinct and independent")
        if set(self.auditor_fallbacks) & set(self.judge_fallbacks):
            raise ValueError("Judge and auditor may not share a fallback family")
        if self.auditor_fallbacks and auditor_family is None:
            raise ValueError("Audit fallback needs a primary auditor")
        self.ledger = ClaimLedger()
        self.warnings: list[str] = []
        self.actual_models: list[dict] = []
        self.tripped: dict[str, str] = {}
        self._stage = "setup"
        self._timing: dict[str, dict] = {}
        self._run_started = time.monotonic()
        roles = getattr(transport, "role_timeouts", None)
        if isinstance(roles, dict):
            for role, seconds in LONG_ROLE_SECONDS.items():
                roles.setdefault(role, seconds)
        self._wrap_transport_recorder()

    # ------------------------------------------------------------------
    # Stage timing (P9)
    # ------------------------------------------------------------------
    def _stats(self, stage: str | None = None) -> dict:
        return self._timing.setdefault(stage or self._stage, {
            "wall_seconds": 0.0, "role_calls_ok": 0, "role_calls_failed": 0,
            "seconds_in_failed_calls": 0.0, "network_attempts": 0,
            "network_failures": 0, "skipped_open_circuit": 0})

    @contextmanager
    def stage(self, name: str):
        previous, self._stage = self._stage, name
        started = time.monotonic()
        self.events.record("stage_started", stage=name)
        try:
            yield self._stats(name)
        finally:
            stats = self._stats(name)
            stats["wall_seconds"] = round(stats["wall_seconds"]
                                          + time.monotonic() - started, 2)
            self.events.record("stage_finished", stage=name, **stats)
            self._stage = previous

    def _wrap_transport_recorder(self):
        """Count every network attempt (incl. failures) per stage."""
        original = getattr(self.transport, "record", None)
        if not callable(original):
            return
        runner = self

        def counting(event_type, **payload):
            if event_type in {"model_call_started", "model_call_retry"} \
                    and not payload.get("replayed"):
                runner._stats()["network_attempts"] += 1
            elif event_type == "model_call_failed":
                runner._stats()["network_failures"] += 1
            return original(event_type, **payload)
        try:
            self.transport.record = counting
        except AttributeError:
            pass

    def timing_report(self) -> dict:
        stages = [dict(stage=name, **{k: (round(v, 2) if isinstance(v, float) else v)
                                     for k, v in stats.items()})
                  for name, stats in self._timing.items() if name != "setup"]
        return {"stages": stages,
                "total_seconds": round(time.monotonic() - self._run_started, 2),
                "network_attempts": sum(s["network_attempts"] for s in stages),
                "network_failures": sum(s["network_failures"] for s in stages),
                "role_calls_failed": sum(s["role_calls_failed"] for s in stages),
                "seconds_in_failed_calls": round(sum(
                    s["seconds_in_failed_calls"] for s in stages), 2),
                "open_circuits": dict(self.tripped)}

    # ------------------------------------------------------------------
    # One role call: circuit breaker, local repair, one format retry
    # ------------------------------------------------------------------
    def _trip(self, family: str, failures: list[str]) -> None:
        text = "; ".join(failures)
        if not self.circuit_breaker or any(k in text for k in _SELF_INFLICTED):
            return
        if family not in self.tripped:
            self.tripped[family] = text[:200]
            self.events.record("circuit_opened", family=family, stage=self._stage,
                               reason=text[:200])

    async def _call(self, family: str, role: str, data: dict, *,
                    synthesis: bool = False, budget: RunBudget | None = None,
                    ignore_circuit: bool = False) -> dict | None:
        budget = budget or self.budget
        stats = self._stats()
        if family in self.tripped and not ignore_circuit:
            stats["skipped_open_circuit"] += 1
            self.warnings.append(role + "/" + family + ": skipped (failed earlier "
                                 "in this run)")
            return None
        if budget.allowed(30, synthesis=synthesis) <= 1:
            self.warnings.append(role + "/" + family + ": deadline reached")
            return None
        started = time.monotonic()
        body = await self._call_inner(family, role, data, synthesis, budget)
        if body is None:
            stats["role_calls_failed"] += 1
            stats["seconds_in_failed_calls"] += time.monotonic() - started
        else:
            stats["role_calls_ok"] += 1
        return body

    async def _call_inner(self, family, role, data, synthesis, budget):
        original_message = prompt(role, data)
        retry_roles = {"proposal", "judge", "synthesis", "audit"}
        for attempt in range(2 if role in retry_roles else 1):
            message = (original_message if not attempt else
                       self._format_retry_message(role, original_message))
            if message is None:
                self.warnings.append(role + "/" + family + ": format retry unavailable")
                return None
            try:
                response = await self.transport.complete(
                    family, message, budget, role=role, synthesis=synthesis)
            except FamilyUnavailable as exc:
                self.warnings.append(role + "/" + family + ": " + "; ".join(exc.failures))
                self.events.record("role_unavailable", role=role, family=family,
                                   failures=exc.failures)
                self._trip(family, exc.failures)
                return None
            self.actual_models.append({"role": role, "family": family,
                                       "provider": response.provider,
                                       "model": response.model,
                                       "prompt_hash": response.prompt_hash})
            if role in TEXT_ROLES:
                text = (response.text or "").strip()
                if text:
                    return {"text": text}
                self.warnings.append(role + "/" + family + ": empty text output")
                return None
            try:
                return self._validated(role, data, response.text)
            except (OutputError, ValueError) as error:
                reason = str(error).splitlines()[0][:180]
                salvaged = self._salvage(role, response.text)
                if salvaged is not None:
                    self.events.record("output_salvaged", role=role, family=family,
                                       reason=reason)
                    self.warnings.append(role + "/" + family + ": output repaired "
                                         "locally (" + reason + ")")
                    return salvaged
                self.events.record("role_parse_error", role=role, family=family,
                                   provider=response.provider, model=response.model,
                                   reason=reason, attempt=attempt + 1)
                if (role in retry_roles and attempt == 0 and
                        budget.allowed(30, synthesis=synthesis) > 2):
                    self.events.record("format_retry_requested", role=role,
                                       family=family, failed_model=response.model,
                                       reason=reason)
                    continue
                self.warnings.append(role + "/" + family +
                                     ": invalid structured output (" + reason + ")")
                return None
        return None

    def _validated(self, role, data, text):
        validated = validate_output(role, parse_object(text))
        if (role == "proposal" and is_plan_instead_of_answer(
                str(data.get("question", "")), validated["approach"])):
            raise ValueError("plan_not_answer: generated instructions instead of "
                             "fulfilling the user request")
        return validated

    @staticmethod
    def _format_retry_message(role, original):
        separator = "\nDATA_JSON:\n"
        if separator not in original:
            return None
        formatting = {
            "proposal": "Answer the question rather than outlining a plan. Return "
                        "JSON with approach, final_answer, claims (each id, "
                        "statement, assumptions and limitations), hypotheses and "
                        "experiments. Do not invent evidence. ",
            "synthesis": "Return JSON with proposed_approach, final_answer, "
                         "answer_comparison, hypotheses, unresolved_objections, "
                         "proof_obligations, proposed_experiments and next_actions. "
                         "Model agreement is not verification. ",
            "audit": "Return JSON with issues and suggested_checks arrays. ",
        }.get(role, "Return JSON with status 'answer' or 'unresolved', optional "
                    "answer, partial_answer, blocking and basis. ")
        return original.replace(
            separator, "\nFORMAT RETRY: Previous generation did not satisfy this "
            "role's JSON contract. " + formatting + "No markdown, commentary, or "
            "fabricated evidence.\n" + separator, 1)

    # Field name -> (type, default) used for conservative local repair.
    _REPAIR_LISTS = {
        "synthesis": ("supported_findings", "hypotheses", "unresolved_objections",
                      "proof_obligations", "proposed_experiments", "next_actions",
                      "unresolved_differences"),
        "audit": ("issues", "suggested_checks"),
        "proposal": ("hypotheses", "experiments"),
    }

    def _salvage(self, role: str, text: str) -> dict | None:
        """Repair a useful but malformed reply locally instead of discarding it.

        Only fills in or coerces what the model clearly meant: drops unknown
        keys, turns a single string into a one-item list, supplies empty
        optional lists, and for long prose without JSON keeps the prose as
        the answer. Nothing is invented; repaired output is flagged.
        """
        if role not in {"synthesis", "proposal", "audit"}:
            return None
        try:
            obj = parse_object(text)
        except OutputError:
            obj = None
        prose = (text or "").strip()
        if obj is None:
            if len(prose) < 400 or role == "audit":
                return None
            if role == "synthesis":
                obj = {"proposed_approach": prose, "final_answer": prose}
            else:
                first = re.split(r"(?<=[.!?])\s+", prose, maxsplit=1)[0][:600]
                obj = {"approach": prose, "final_answer": "",
                       "claims": [{"id": "c1", "statement": first,
                                   "assumptions": [], "limitations": [
                                       "Recovered from unstructured output"]}]}
        try:
            from council.schemas import SCHEMAS
            fields = SCHEMAS[role].model_fields
        except (ImportError, KeyError):
            return None
        clean = {k: v for k, v in obj.items() if k in fields}
        for name in self._REPAIR_LISTS.get(role, ()):
            value = clean.get(name)
            if value is None:
                clean[name] = []
            elif isinstance(value, str):
                clean[name] = [value] if value.strip() else []
            elif isinstance(value, list):
                clean[name] = [x if isinstance(x, str) else json.dumps(x)
                               for x in value if x not in (None, "")]
        if role == "synthesis":
            if not isinstance(clean.get("proposed_approach"), str) or \
                    not clean["proposed_approach"].strip():
                answer = clean.get("final_answer")
                if not (isinstance(answer, str) and answer.strip()):
                    return None
                clean["proposed_approach"] = answer
            comparison = clean.get("answer_comparison")
            if not isinstance(comparison, list):
                clean.pop("answer_comparison", None)
            else:
                clean["answer_comparison"] = [
                    c for c in comparison if isinstance(c, dict)
                    and c.get("relation") in {"adopted", "partly_adopted",
                                              "rejected", "not_comparable"}
                    and isinstance(c.get("proposal_id"), str)]
            if clean.get("completion_status") not in {"partial", "unresolved"}:
                clean.pop("completion_status", None)
        if role == "proposal":
            claims = clean.get("claims")
            if isinstance(claims, list):
                clean["claims"] = [
                    {"id": str(c.get("id") or "c" + str(n)),
                     "statement": c["statement"],
                     "assumptions": [x for x in c.get("assumptions") or []
                                     if isinstance(x, str)],
                     "limitations": [x for x in c.get("limitations") or []
                                     if isinstance(x, str)]}
                    for n, c in enumerate(claims, 1)
                    if isinstance(c, dict) and isinstance(c.get("statement"), str)
                    and c["statement"].strip()]
            if not isinstance(clean.get("final_answer"), str):
                clean["final_answer"] = ""
        try:
            return validate_output(role, clean)
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Role with independent fallbacks: availability order + hedging
    # ------------------------------------------------------------------
    def _availability_order(self, families: tuple[str, ...], role: str
                            ) -> list[tuple[str, float]]:
        """Order candidate families by what the transport knows, without calls.

        Families usable now keep their configured order first; families in
        a temporary cooldown follow, soonest first. Families no wait can help
        (unconfigured, billing/auth blocks) are dropped and recorded. A
        transport without availability information keeps the given order.
        """
        probe = getattr(self.transport, "soonest_ready", None)
        if not callable(probe):
            return [(family, 0.0) for family in families]
        ready, waiting = [], []
        for position, family in enumerate(families):
            try:
                wait = probe(family)
            except Exception:  # availability data is advisory only
                wait = 0.0
            if wait is None:
                self.events.record("role_candidate_skipped", role=role,
                                   family=family, reason="not_available")
                continue
            (ready if wait <= 0 else waiting).append((position, family, float(wait)))
        waiting.sort(key=lambda item: (item[2], item[0]))
        return [(family, wait) for _, family, wait in ready + waiting]

    async def _call_with_fallbacks(self, role: str, families: tuple[str, ...],
                                   data: dict, deadline: float) -> tuple[dict | None, str | None]:
        """Try independent families for one role within a shared deadline.

        Hedged: the next candidate starts when the running one fails OR has
        not answered within hedge_seconds; the first valid reply wins and the
        rest are cancelled. Families that already played another role are
        skipped (independence), as are families whose circuit is open. If
        every candidate fails and one leaves a short cooldown before the
        deadline, it is retried once.
        """
        primary = families[0] if families else None
        candidates = [
            (family, wait) for family, wait in self._availability_order(families, role)
            if (family == primary or not any(
                m["family"] == family for m in self.actual_models))
            and family not in self.tripped]
        queue = [family for family, wait in candidates if wait <= 0]
        running: dict[asyncio.Task, str] = {}
        tried: list[str] = []

        def launch():
            family = queue.pop(0)
            tried.append(family)
            budget = RunBudget(deadline, 0.0)
            running[asyncio.ensure_future(
                self._call(family, role, data, synthesis=True, budget=budget))] = family

        try:
            while queue or running:
                if not running:
                    if time.monotonic() + 3 >= deadline:
                        break
                    launch()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                done, _ = await asyncio.wait(
                    set(running), timeout=min(self.hedge_seconds, remaining),
                    return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    if queue and time.monotonic() + 3 < deadline:
                        self.events.record("role_hedged", role=role,
                                           waiting_on=list(running.values()),
                                           starting=queue[0])
                        launch()
                    continue
                for task in done:
                    family = running.pop(task)
                    body = task.result()
                    self.events.record(role + "_attempt_completed", family=family,
                                       succeeded=body is not None)
                    if body is not None:
                        return body, family
        finally:
            for task in running:
                task.cancel()
            if running:
                await asyncio.gather(*running, return_exceptions=True)
        probe = getattr(self.transport, "soonest_ready", None)
        if callable(probe):
            for family, _ in candidates:
                try:
                    wait = probe(family)
                except Exception:
                    wait = None
                if wait is None or time.monotonic() + wait + 15 >= deadline:
                    continue
                self.events.record("role_retry_after_cooldown", role=role,
                                   family=family, wait_seconds=round(wait, 1),
                                   previously_tried=family in tried)
                if wait > 0:
                    await asyncio.sleep(wait)
                body = await self._call(family, role, data, synthesis=True,
                                        budget=RunBudget(deadline, 0.0),
                                        ignore_circuit=True)
                self.events.record(role + "_attempt_completed", family=family,
                                   succeeded=body is not None, retry=True)
                if body is not None:
                    return body, family
                break  # one retry only
        return None, None

    async def _timed_call(self, family: str, role: str, data: dict,
                          deadline: float, *, remaining_candidates: int = 1) -> dict | None:
        available = deadline - time.monotonic()
        share = (available if remaining_candidates <= 1
                 else max(8.0, available / remaining_candidates))
        return await self._call(family, role, data, synthesis=True,
                                budget=RunBudget(min(deadline, time.monotonic() + share), 0.0))

    @staticmethod
    def _normalized(text: str) -> str:
        return " ".join(str(text or "").casefold().split()).rstrip(".")

    def _select_answer(self, proposals: list[dict], objections: list,
                       replies: list, synthesis: dict | None,
                       judge_family: str | None) -> dict:
        """Choose the answer to show, with an explicit, honest basis.

        Nothing here verifies anything. Preference order:
        1. the judge's consolidated answer (model-reviewed);
        2. without a judge: an answer two or more independent proposals
           state identically;
        3. without a judge: the proposal with the fewest objections its
           author did not accept;
        4. a single proposal (quick mode or degraded run), unreviewed.
        """
        alternatives = [{"proposal_id": p["id"], "family": p["family"],
                         "answer": self._proposal_answer(p)} for p in proposals]
        if synthesis is not None:
            text = (str(synthesis.get("final_answer") or "").strip()
                    or str(synthesis.get("proposed_approach") or "").strip())
            if text:
                return {"answer": text, "basis": "judge_synthesis",
                        "family": judge_family, "proposal_id": None,
                        "status": "model_reviewed_unverified",
                        "alternatives": alternatives}
        if not proposals:
            return {"answer": None, "basis": "no_valid_proposal",
                    "family": None, "proposal_id": None,
                    "status": "unresolved", "alternatives": []}
        if len(proposals) == 1:
            p = proposals[0]
            return {"answer": self._proposal_answer(p),
                    "basis": "single_proposal", "family": p["family"],
                    "proposal_id": p["id"], "status": "single_model_unverified",
                    "alternatives": alternatives}
        scalar = {}
        for p in proposals:
            key = self._normalized(p.get("final_answer"))
            if key:
                scalar.setdefault(key, []).append(p)
        agreeing = max(scalar.values(), key=len, default=[])
        if len(agreeing) >= 2 and len(agreeing) * 2 > len(proposals):
            p = agreeing[0]
            return {"answer": self._proposal_answer(p),
                    "basis": "proposal_majority_unjudged", "family": p["family"],
                    "proposal_id": p["id"], "agreeing_proposals":
                        [x["id"] for x in agreeing],
                    "status": "model_agreement_unverified",
                    "alternatives": alternatives}
        accepted = {r.objection_id for r in replies if r.disposition == "accept"}
        def open_objections(p):
            return sum(1 for o in objections
                       if o.proposal_id == p["id"] and o.id not in accepted)
        p = min(proposals, key=lambda x: (open_objections(x), x["id"]))
        return {"answer": self._proposal_answer(p),
                "basis": "least_contested_proposal_unjudged",
                "family": p["family"], "proposal_id": p["id"],
                "open_objections": open_objections(p),
                "status": "unjudged_unverified", "alternatives": alternatives}

    @staticmethod
    def _proposal_answer(p: dict) -> str:
        return (p.get("revised_approach") or p.get("approach") or "").strip() \
            if not p.get("final_answer") else (
                p["final_answer"].strip() + "\n\n" +
                (p.get("revised_approach") or p.get("approach") or "").strip()).strip()

    def _proposal(self, family: str, body: dict, index: int) -> dict:
        """Validate the entire proposal before admitting any claim to the ledger."""
        approach = require_string(body, "approach")
        raw_claims = item_list(body, "claims")
        if not raw_claims:
            raise OutputError("Proposal must include at least one claim")
        proposal_id = "p" + str(index)
        claim_map: dict[str, str] = {}
        pending: list[Claim] = []
        for j, raw in enumerate(raw_claims, 1):
            local = require_string(raw, "id")
            statement = require_string(raw, "statement")
            if local in claim_map:
                raise OutputError("Duplicate local claim id")
            cid = proposal_id + ":c" + str(j)
            claim_map[local] = cid
            pending.append(Claim(
                id=cid, version=1, statement=statement, owner_family=family,
                limitations=string_list(raw, "limitations")))
        final_answer = body.get("final_answer") or ""
        if not isinstance(final_answer, str):
            raise OutputError("final_answer must be a string")
        hypotheses = string_list(body, "hypotheses")
        experiments = string_list(body, "experiments")
        artifact = body.get("artifact")
        if artifact is not None and not isinstance(artifact, dict):
            raise OutputError("artifact must be an object")
        if any(claim.id in self.ledger.claims for claim in pending):
            raise OutputError("Proposal claim ID already registered")
        for claim in pending:
            self.ledger.add_claim(claim)
        return {"id": proposal_id, "family": family, "approach": approach,
                "final_answer": final_answer.strip(),
                "claim_ids": list(claim_map.values()), "claim_map": claim_map,
                "hypotheses": hypotheses, "experiments": experiments,
                "artifact": artifact or {}}



    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------
    async def _cross_review(self, question, proposals, objections, budget=None):
        """All reviews run concurrently; each proposal goes to another family."""
        pairs = []
        for index, p in enumerate(proposals):
            reviewer = proposals[(index + 1) % len(proposals)]["family"]
            if reviewer != p["family"]:
                pairs.append((p, reviewer))
        bodies = await asyncio.gather(*(
            self._call(reviewer, "review", {
                "question": question, "proposal": p,
                "claims": [asdict(self.ledger.claims[cid])
                           for cid in p["claim_ids"]]}, budget=budget)
            for p, reviewer in pairs))
        for (p, reviewer), body in zip(pairs, bodies):
            if body is None:
                continue
            try:
                for raw in item_list(body, "objections"):
                    local_id = require_string(raw, "claim_id")
                    cid = p["claim_map"].get(local_id, local_id)
                    if cid not in p["claim_ids"]:
                        self.warnings.append("Dropped review with unknown claim id")
                        continue
                    objection = Objection(
                        id="o" + str(len(objections) + 1),
                        proposal_id=p["id"], claim_id=cid,
                        reviewer_family=reviewer,
                        issue=require_string(raw, "issue"),
                        consequence=require_string(raw, "consequence"),
                        suggested_check=require_string(raw, "suggested_check"))
                    objections.append(objection)
                    self.events.record("review_received", objection=asdict(objection))
            except OutputError:
                self.warnings.append("Rejected malformed review from " + reviewer)

    async def _revise(self, p, related, replies, *, notes=None):
        """The author answers objections; returns True if anything changed."""
        payload = {
            "original_approach": p.get("revised_approach") or p["approach"],
            "claims": [asdict(self.ledger.claims[cid]) for cid in p["claim_ids"]],
            "objections": [dict(asdict(o), **({"still_open_because": notes[o.id]}
                                              if notes and o.id in notes else {}))
                           for o in related]}
        body = await self._call(p["family"], "revision", payload)
        if body is None:
            return False
        changed = False
        try:
            revised = body.get("revised_approach")
            if isinstance(revised, str) and revised.strip():
                p["revised_approach"] = revised.strip()
                changed = True
            for raw in item_list(body, "revised_claims"):
                cid = require_string(raw, "claim_id")
                if cid not in p["claim_ids"]:
                    cid = p["claim_map"].get(cid, cid)
                if cid not in p["claim_ids"]:
                    continue
                claim = self.ledger.revise(cid, require_string(raw, "statement"))
                changed = True
                self.events.record("claim_revised", claim_id=cid,
                                   new_version=claim.version)
            ids = {o.id for o in related}
            for raw in item_list(body, "responses"):
                oid = require_string(raw, "objection_id")
                if oid not in ids:
                    continue
                response = AuthorResponse(
                    objection_id=oid,
                    disposition=require_string(raw, "disposition"),
                    explanation=require_string(raw, "explanation"))
                replies.append(response)
                self.events.record("author_response_received",
                                   response=asdict(response))
        except (OutputError, ValueError):
            self.warnings.append("Rejected malformed revision from " + p["family"])
        return changed

    def _rereviewer(self, p, related, proposals):
        """A solver family that is neither the author nor (if possible) the critic."""
        critics = {o.reviewer_family for o in related}
        others = [x["family"] for x in proposals if x["family"] != p["family"]]
        fresh = [f for f in others if f not in critics]
        return (fresh or others or [None])[0]

    async def _rereview(self, question, p, related, replies, proposals):
        family = self._rereviewer(p, related, proposals)
        if family is None:
            return {}
        latest = self._latest_replies(replies)
        body = await self._call(family, "rereview", {
            "question": question,
            "current_approach": p.get("revised_approach") or p["approach"],
            "claims": [asdict(self.ledger.claims[cid]) for cid in p["claim_ids"]],
            "objections": [dict(asdict(o), author_response=(
                asdict(latest[o.id]) if o.id in latest else None))
                for o in related]})
        if body is None:
            return {}
        ids = {o.id for o in related}
        verdicts = {}
        for item in body.get("items") or []:
            if item.get("objection_id") in ids and item["objection_id"] not in verdicts:
                verdicts[item["objection_id"]] = {
                    "verdict": item["verdict"], "reason": item["reason"][:600],
                    "family": family}
        self.events.record("rereview_received", proposal_id=p["id"],
                           family=family, verdicts={k: v["verdict"]
                                                    for k, v in verdicts.items()})
        return verdicts

    async def _improve(self, question, proposals, objections, replies, rereviews):
        """Bounded improvement: independent re-review, targeted repair, re-review.

        Round structure for improvement_rounds=N: re-review every objection,
        then up to N times repair the still-open ones and re-review those.
        Each re-review comes from a solver family other than the author.
        """
        pending = {p["id"]: [o for o in objections if o.proposal_id == p["id"]]
                   for p in proposals}
        for round_no in range(self.improvement_rounds + 1):
            work = [(p, pending[p["id"]]) for p in proposals if pending[p["id"]]]
            if not work or self.budget.allowed(30) <= 2:
                if work:
                    self.events.record("improvement_stopped", round=round_no,
                                       reason="time")
                return
            if round_no:
                notes = {oid: rereviews[oid]["reason"] for _, rel in work
                         for oid in (o.id for o in rel) if oid in rereviews}
                await asyncio.gather(*(
                    self._revise(p, rel, replies, notes=notes) for p, rel in work))
                self.events.record("improvement_repair_round", round=round_no,
                                   objections=[o.id for _, rel in work for o in rel])
            results = await asyncio.gather(*(
                self._rereview(question, p, rel, replies, proposals)
                for p, rel in work))
            for (p, rel), verdicts in zip(work, results):
                for oid, verdict in verdicts.items():
                    rereviews[oid] = dict(verdict, round=round_no)
                pending[p["id"]] = [o for o in rel
                                    if rereviews.get(o.id, {}).get("verdict") == "still_open"]
            if round_no == self.improvement_rounds:
                return

    @staticmethod
    def _latest_replies(replies):
        latest = {}
        for reply in replies:
            latest[reply.objection_id] = reply
        return latest

    def _issue_ledger(self, objections, replies, rereviews, audit, auditor_family):
        """Every criticism with a resolution status and the reason for it."""
        latest = self._latest_replies(replies)
        issues = []
        for o in objections:
            reply = latest.get(o.id)
            check = rereviews.get(o.id)
            disposition = reply.disposition if reply else None
            if check and check["verdict"] == "resolved":
                status, why = "addressed", ("independent re-review (" + check["family"]
                                            + ") found it fixed: " + check["reason"])
            elif check and check["verdict"] == "objection_invalid":
                status, why = "addressed", ("independent re-review (" + check["family"]
                                            + ") found the objection does not apply: "
                                            + check["reason"])
            elif check and check["verdict"] == "still_open":
                if disposition == "reject":
                    status, why = "contested", ("author rejects it, independent re-review ("
                                                + check["family"] + ") maintains it: "
                                                + check["reason"])
                else:
                    status, why = "unresolved", ("still open after independent re-review ("
                                                 + check["family"] + "): " + check["reason"])
            elif disposition == "reject":
                status, why = "contested", ("author rejects it; no independent "
                                            "re-review: " + reply.explanation)
            elif disposition == "unresolved":
                status, why = "unresolved", "author marked it unresolved: " + reply.explanation
            elif disposition in {"accept", "partial"}:
                status, why = "unreviewed", ("author " + disposition + "ed it; the fix was "
                                             "not independently re-reviewed")
            else:
                status, why = "unreviewed", "author did not respond; not re-reviewed"
            issues.append({"id": o.id, "source": "peer_review",
                           "proposal_id": o.proposal_id, "claim_id": o.claim_id,
                           "raised_by": o.reviewer_family, "issue": o.issue,
                           "author_disposition": disposition,
                           "rereview": check, "status": status, "reason": why})
        for n, text in enumerate((audit or {}).get("issues") or [], 1):
            issues.append({"id": "a" + str(n), "source": "audit",
                           "proposal_id": None, "claim_id": None,
                           "raised_by": auditor_family, "issue": text,
                           "author_disposition": None, "rereview": None,
                           "status": "unreviewed",
                           "reason": "raised by the independent auditor after the "
                                     "synthesis; no response round follows the audit"})
        return issues

    async def _complete_comparison(self, synthesis, judge_family, proposals, deadline):
        """Make sure every competing answer was compared; ask once if not."""
        ids = [p["id"] for p in proposals]
        seen = {c["proposal_id"]: c for c in synthesis.get("answer_comparison") or []
                if c.get("proposal_id") in ids}
        missing = [p for p in proposals if p["id"] not in seen]
        differences = list(synthesis.get("unresolved_differences") or [])
        asked = False
        if missing and deadline - time.monotonic() > 20:
            asked = True
            body = await self._timed_call(judge_family, "compare", {
                "final_answer": synthesis.get("final_answer")
                                or synthesis.get("proposed_approach"),
                "candidate_answers": {p["id"]: self._proposal_answer(p)
                                      for p in missing}},
                deadline, remaining_candidates=1)
            for c in (body or {}).get("answer_comparison") or []:
                if c.get("proposal_id") in {p["id"] for p in missing}:
                    seen.setdefault(c["proposal_id"], c)
            differences += list((body or {}).get("unresolved_differences") or [])
        comparison = []
        for p in proposals:
            c = seen.get(p["id"])
            comparison.append(dict(c, family=p["family"]) if c else {
                "proposal_id": p["id"], "family": p["family"],
                "relation": "not_compared_by_judge", "key_difference": "",
                "reason": "the judge did not compare this answer"
                          + (" even when asked again" if asked else "")})
        not_compared = [c["proposal_id"] for c in comparison
                        if c["relation"] == "not_compared_by_judge"]
        for c in comparison:
            if c["relation"] in {"rejected", "partly_adopted"} and c.get("key_difference"):
                differences.append(c["proposal_id"] + " (" + c["family"] + "): "
                                   + c["key_difference"])
        for pid in not_compared:
            p = next(x for x in proposals if x["id"] == pid)
            differences.append(pid + " (" + p["family"] + ") was not compared by the "
                               "judge; its answer is preserved: "
                               + self._proposal_answer(p)[:400])
        return comparison, list(dict.fromkeys(differences)), not_compared

    async def _judge_stage(self, role, families, data, deadline):
        with self.stage("judge"):
            return await self._call_with_fallbacks(role, families, data, deadline)

    # ------------------------------------------------------------------
    async def run(self, question: str, *, mode: str = "research") -> dict:
        if mode not in {"quick", "standard", "research", "full"}:
            raise ValueError("mode must be quick, standard, full or research (legacy)")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("The request must not be empty")
        active = self.solvers[:1] if mode == "quick" else self.solvers
        if mode != "quick" and self.task_type != "answer":
            from council.document import extract_sections, run_document
            if self.task_type == "document" or extract_sections(question):
                return await run_document(self, question, mode=mode)
        total = self.budget.remaining()
        # A stage ends at its share of the run or where the end-of-run
        # reserve starts, whichever is first; it carries no reserve itself.
        stage_end = lambda fraction: RunBudget(
            min(self.budget.end - self.budget.synthesis_reserve,
                time.monotonic() + fraction * total), 0.0)
        self.events.record("run_started", question=question, mode=mode,
                           task_type="answer", review_depth=self.review_depth,
                           solver_families=list(active), judge_family=self.judge_family,
                           auditor_family=self.auditor_family,
                           auditor_fallbacks=list(self.auditor_fallbacks),
                           judge_fallbacks=list(self.judge_fallbacks),
                           quorum_policy=self.quorum_policy,
                           min_proposals=self.min_proposals,
                           improvement_rounds=self.improvement_rounds)
        proposals: list[dict] = []
        objections: list[Objection] = []
        replies: list[AuthorResponse] = []
        rereviews: dict[str, dict] = {}
        # Sealed parallel first round: no model sees another proposal.
        with self.stage("proposals"):
            replies_first = await asyncio.gather(*(
                self._call(family,"proposal",{"question":question,
                    "independent":True,
                    "strategy":self.solver_strategies.get(family,"direct")},
                    budget=stage_end(0.40) if mode != "quick" else None)
                for family in active))
        for i, (family,body) in enumerate(zip(active,replies_first), 1):
            if body is None:
                continue
            try:
                p = self._proposal(family, body, i)
            except (OutputError, ValueError) as exc:
                self.warnings.append(family + ": invalid proposal schema: " + str(exc))
                continue
            proposals.append(p)
            self.events.record("proposal_received", proposal=p)

        # --- Quorum (#5) ------------------------------------------------
        required = 1 if mode == "quick" else self.min_proposals
        quorum = {"required": required, "obtained": len(proposals),
                  "requested": len(active), "met": len(proposals) >= required,
                  "policy": self.quorum_policy, "action": "full_pipeline"}
        if mode == "quick":
            quorum["action"] = "quick_single_model"
        elif not proposals:
            quorum["action"] = "no_valid_proposal"
        elif not quorum["met"]:
            quorum["action"] = ("stopped_before_judge_and_audit"
                                if self.quorum_policy == "stop" else
                                "degraded_no_cross_review_or_judge")
            self.events.record("quorum_not_met", **quorum)
        degraded = mode != "quick" and bool(proposals) and not quorum["met"]
        stopped = degraded and self.quorum_policy == "stop"
        reviewed = mode != "quick" and quorum["met"] and len(proposals) >= 2

        # --- Cross-review, revision, improvement (#6) -------------------
        if reviewed:
            with self.stage("review"):
                await self._cross_review(question, proposals, objections,
                                         budget=stage_end(0.60))
            if self.review_depth == "thorough" and objections:
                with self.stage("revision"):
                    await asyncio.gather(*(
                        self._revise(p, [o for o in objections
                                         if o.proposal_id == p["id"]], replies)
                        for p in proposals
                        if any(o.proposal_id == p["id"] for o in objections)))
                with self.stage("improvement"):
                    await self._improve(question, proposals, objections,
                                        replies, rereviews)

        brief = CouncilResult(research_question=question,
                               completion_status="partial" if proposals else "unresolved")
        brief.hypotheses = list(dict.fromkeys(
            hypothesis for p in proposals for hypothesis in p["hypotheses"]))
        brief.proposed_experiments = list(dict.fromkeys(
            experiment for p in proposals for experiment in p["experiments"]))
        brief.proposed_approach = " | ".join(
            p.get("revised_approach", p["approach"]) for p in proposals)
        brief.limitations = list(self.warnings)
        pre_audit_issues = self._issue_ledger(objections, replies, rereviews, None, None)
        brief.unresolved_objections = [
            i["issue"] + " [claim " + i["claim_id"] + ", " + i["id"] + ", " + i["status"] + "]"
            for i in pre_audit_issues if i["status"] != "addressed"]

        # --- Judge (#3, #8) ---------------------------------------------
        synthesis=None
        judge_family_used=None
        comparison, differences, not_compared = [], [], []
        audit_reserve=(min(self.budget.remaining()*0.45,
                           float(getattr(self.transport,"role_timeouts",{})
                                 .get("audit",100.0))+10.0)
                       if mode!="quick" and self.auditor_family else 0.0)
        if proposals and reviewed:
            judge_deadline=self.budget.end-audit_reserve
            synthesis,judge_family_used=await self._judge_stage(
                "synthesis",(self.judge_family,*self.judge_fallbacks),{
                    "question":question,"proposals":proposals,
                    "candidate_answers":{p["id"]:self._proposal_answer(p)
                                         for p in proposals},
                    "claims":[asdict(c) for c in self.ledger.claims.values()],
                    "objections":[asdict(o) for o in objections],
                    "author_responses":[asdict(r) for r in replies],
                    "issue_status":[{k:i[k] for k in ("id","proposal_id","issue",
                                                       "status","reason")}
                                    for i in pre_audit_issues],
                    "verified_evidence":[],
                    "rule":"Independent AI opinion only: no source "
                           "grounding or mathematical proof checker",
                },judge_deadline)
            if synthesis:
                try:
                    brief.proposed_approach = require_string(synthesis, "proposed_approach")
                    brief.judge_summary = brief.proposed_approach
                    inferred = string_list(synthesis, "supported_findings")
                    if inferred:
                        brief.limitations.append(
                            "Judge-proposed findings were reclassified as unverified hypotheses")
                    brief.hypotheses = list(dict.fromkeys(
                        brief.hypotheses + string_list(synthesis, "hypotheses") + inferred))
                    brief.unresolved_objections = list(dict.fromkeys(
                        brief.unresolved_objections +
                        string_list(synthesis, "unresolved_objections")))
                    brief.proof_obligations = string_list(synthesis, "proof_obligations")
                    brief.proposed_experiments = list(dict.fromkeys(
                        brief.proposed_experiments +
                        string_list(synthesis, "proposed_experiments")))
                    brief.next_actions = string_list(synthesis, "next_actions")
                    self.events.record("synthesis_received", family=judge_family_used)
                except OutputError:
                    brief.limitations.append("Synthesis schema invalid; deterministic partial preserved")
                    synthesis=None
            if synthesis:
                with self.stage("judge"):
                    comparison, differences, not_compared = await self._complete_comparison(
                        synthesis, judge_family_used, proposals, judge_deadline)
                if not_compared:
                    brief.limitations.append("Judge did not compare proposal(s) "
                                             + ", ".join(not_compared)
                                             + "; their answers are preserved")
            else:
                brief.limitations.append("Judge unavailable; deterministic partial preserved")
        elif not proposals:
            brief.limitations.append("No validated independent proposal obtained")

        # --- Independent audit (#2) -------------------------------------
        audit=None
        auditor_family_used=None
        if (mode!="quick" and proposals and not stopped
                and self.auditor_family is not None):
            audit_payload={
                "question":question,"research_result":brief.to_dict(),
                "claims":[asdict(c) for c in self.ledger.claims.values()],
                "proposals":[{"id":p["id"],"family":p["family"],
                    "approach":p.get("revised_approach") or p["approach"]}
                    for p in proposals],
                "objections":[asdict(o) for o in objections],
                "issue_status":[{k:i[k] for k in ("id","issue","status","reason")}
                                for i in pre_audit_issues],
                "unresolved_differences":differences,
                "judge_family":judge_family_used,
                "judge_summary":brief.judge_summary,
                "rule":"Independently inspect reasoning, assumptions, "
                       "contradictions and unaddressed objections; do not "
                       "treat model agreement as proof.",
            }
            with self.stage("audit"):
                audit,auditor_family_used=await self._call_with_fallbacks(
                    "audit",(self.auditor_family,*self.auditor_fallbacks),
                    audit_payload,self.budget.end)
            if audit:
                try:
                    string_list(audit,"issues"); string_list(audit,"suggested_checks")
                    brief.proposed_experiments.extend(
                        string_list(audit,"suggested_checks"))
                    self.events.record("audit_received",family=auditor_family_used)
                except OutputError:
                    brief.limitations.append("Audit output malformed")
                    audit=None
                    auditor_family_used=None
            if audit is None:
                brief.limitations.append("Independent auditor unavailable")
        elif mode=="full" and self.auditor_family is None:
            brief.limitations.append("No independent auditor configured")

        # --- Issues (#7) -------------------------------------------------
        issues=self._issue_ledger(objections,replies,rereviews,audit,auditor_family_used)
        brief.unresolved_objections=list(dict.fromkeys(
            brief.unresolved_objections+[i["issue"] for i in issues
                                         if i["source"]=="audit"]))
        issue_summary={status:sum(i["status"]==status for i in issues)
                       for status in ISSUE_STATUSES}

        brief.supported_findings = []
        brief.checked_source_ids = []
        brief.completion_status = ("unresolved" if not proposals or stopped
                                   else "partial")
        if degraded:
            brief.limitations.append(
                "Quorum not met: " + str(len(proposals)) + " of " + str(required)
                + " required independent proposals ("
                + ("stopped" if stopped else "degraded single-model run") + ")")
        if mode=="full" and audit is None and not stopped:
            brief.limitations.append("Full-mode audit not completed")
        brief.proposed_experiments=list(dict.fromkeys(brief.proposed_experiments))

        # --- Answer selection (#1) ---------------------------------------
        if stopped:
            selection={"answer":None,"basis":"quorum_not_met","family":None,
                       "proposal_id":None,"status":"quorum_stopped",
                       "alternatives":[{"proposal_id":p["id"],"family":p["family"],
                                        "answer":self._proposal_answer(p)}
                                       for p in proposals]}
        else:
            selection=self._select_answer(proposals,objections,replies,
                                          synthesis,judge_family_used)
            if degraded and selection["status"]=="single_model_unverified":
                selection["status"]="degraded_single_model_unverified"
        selection["verified"]=False
        selection["auditor_issue_count"]=(len(audit.get("issues") or [])
                                          if audit else None)
        selection["open_issue_count"]=sum(i["status"]!="addressed" for i in issues)
        selection["label"]=ANSWER_LABELS[selection["status"]]
        if (selection["basis"] not in {"judge_synthesis","quorum_not_met",
                                       "no_valid_proposal"}
                and proposals and mode!="quick" and not degraded):
            brief.limitations.append("Final answer selected without a judge ("
                                     +selection["basis"]+")")
        brief.limitations = list(dict.fromkeys(brief.limitations + self.warnings))
        self.events.record("answer_selected",basis=selection["basis"],
                           family=selection["family"],
                           proposal_id=selection["proposal_id"])
        result = {
            "research_result": brief.to_dict(), "proposals": proposals,
            "objections": [asdict(o) for o in objections],
            "author_responses": [asdict(r) for r in replies],
            "claims": [asdict(c) for c in self.ledger.claims.values()],
            "actual_models": self.actual_models, "warnings": self.warnings,
            "verdict":{"status":"unresolved","cause":"model_review_only",
                       "basis":["No source or mathematical checker was run"]},
            "audit":{"completed":audit is not None,
                     "family":auditor_family_used,
                     "issues":audit.get("issues",[]) if audit else [],
                     "suggested_checks":audit.get("suggested_checks",[]) if audit else []},
            "quorum":quorum,
            "issues":issues,"issue_summary":issue_summary,
            "answer_comparison":comparison,
            "unresolved_differences":differences,
            "role_coverage":{"requested_solvers":list(active),
                "valid_proposer_families":[p["family"] for p in proposals],
                "review_objection_count":len(objections),
                "rereview_count":len(rereviews),
                "independent_audit_completed":audit is not None,
                "actual_audit_family":auditor_family_used,
                "judge_family":judge_family_used},
            "candidate_answers":{p["id"]:p.get("final_answer","")
                                 for p in proposals},
            "answer_selection":selection,
            "selected_answer":selection["answer"],
            "final_answer":selection["answer"],
            "task_type":"answer","review_depth":self.review_depth,
            "timing":self.timing_report(),
        }
        if hasattr(self.events,"db") and hasattr(self.events,"run_id"):
            SQLiteLedger(self.events.db).save(self.events.run_id,self.ledger)
        result["council_result"] = brief.to_dict()
        self.events.record("run_finished", result=result)
        return result


ANSWER_LABELS = {
    "model_reviewed_unverified": "Model-reviewed consolidated answer "
        "(judge synthesis); not independently verified",
    "model_agreement_unverified": "Independent proposals agree; no judge was "
        "available; not verified",
    "unjudged_unverified": "Least-contested proposal; no judge was available; "
        "not verified",
    "single_model_unverified": "Single model answer; not cross-checked",
    "degraded_single_model_unverified": "Degraded run: only one valid proposal, "
        "so no cross-review or judge; independently audited at most; not verified",
    "quorum_stopped": "No answer: too few independent proposals and the quorum "
        "policy is stop",
    "unresolved": "No answer could be produced",
    "document_reviewed_unverified": "Multi-model drafted document; sections "
        "reviewed by a different model family; not independently verified",
    "document_unreviewed_unverified": "Multi-model drafted document; the "
        "cross-family review did not complete; not verified",
}
