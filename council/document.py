"""Document mode: produce the requested long document first, then review it.

Pipeline (each step timed; the assembled draft exists after step 3):
1. outline   - required section headings are read from the instructions
               (no model call); only without them does one model draft one.
2. drafting  - sections are split into small blocks across the writer
               families and drafted in parallel lanes. Every section is
               recorded the moment it arrives. Missing sections (a failed
               writer or a truncated reply) are re-requested from another
               available writer: continuation, not a restart.
3. assemble  - the full document in section order.
4. review    - each writer's sections go to a DIFFERENT family, one batched
               call per reviewer, returning short findings per section.
5. revision  - only sections with major findings, by their original writer.
6. audit     - the independent auditor sees the outline, every finding and
               the flagged sections, not the whole document again.

Nothing here verifies the document. Independence: no family reviews its own
text, and the auditor is a separate family.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import re
import time

from council.budget import RunBudget

SECTIONS_PER_BLOCK = 4
MIN_SECTION_CHARS = 80
MAX_CONTINUATION_PASSES = 2
STAGE_SHARE = {"drafting": 0.50, "review": 0.20, "revision": 0.13}

_HEADING = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*)?(?:section[ \t]+)?(\d{1,2})[.):][ \t]+"
    r"(?:\*\*)?([^\n]{1,140}?)(?:\*\*)?[ \t]*$", re.I | re.M)


def extract_sections(text: str, *, minimum: int = 5) -> list[str]:
    """Required section titles, if the instructions list numbered sections.

    Finds the longest run of headings numbered 1, 2, 3, ... (each at the start
    of a line). Returns [] unless at least `minimum` consecutive sections are
    listed, so ordinary numbered lists in a question do not trigger document
    mode. Sub-numbering such as 3.1 is ignored.
    """
    best: list[str] = []
    current: list[str] = []
    for match in _HEADING.finditer(text or ""):
        number, title = int(match.group(1)), match.group(2).strip(" *:-")
        if number == 1:
            current = [title]
        elif current and number == len(current) + 1:
            current.append(title)
        else:
            current = []
        if len(current) > len(best):
            best = list(current)
    return best if len(best) >= minimum else []


def split_sections(text: str, wanted: list[int]) -> dict[int, str]:
    """Pull '## n. Title' sections out of Markdown, keeping only wanted ones."""
    # Every heading ends the previous section, wanted or not: otherwise an
    # extra section a writer added would be glued onto the one before it.
    marks = [(m.start(), int(m.group(1))) for m in _HEADING.finditer(text or "")]
    out: dict[int, str] = {}
    for index, (start, number) in enumerate(marks):
        if number not in wanted:
            continue
        end = marks[index + 1][0] if index + 1 < len(marks) else len(text)
        body = text[start:end].split("\n", 1)
        content = body[1].strip() if len(body) > 1 else ""
        if len(content) >= MIN_SECTION_CHARS and number not in out:
            out[number] = content
    return out


@dataclass
class Section:
    number: int
    title: str
    text: str = ""
    writer: str | None = None
    reviewer: str | None = None
    findings: list = field(default_factory=list)
    revised: bool = False
    attempts: list = field(default_factory=list)

    @property
    def done(self) -> bool:
        return bool(self.text)


def _blocks(numbers: list[int], size: int = SECTIONS_PER_BLOCK) -> list[list[int]]:
    return [numbers[i:i + size] for i in range(0, len(numbers), size)]


def _usable(runner, families) -> list[str]:
    return [f for f in families if f not in runner.tripped]


async def _draft_lane(runner, question, sections, writer, blocks, budget):
    outline = [{"number": s.number, "title": s.title} for s in sections.values()]
    for block in blocks:
        wanted = [n for n in block if not sections[n].done]
        if not wanted or writer in runner.tripped:
            continue
        for n in wanted:
            sections[n].attempts.append(writer)
        body = await runner._call(writer, "draft_sections", {
            "request": question, "outline": outline,
            "write_sections": [{"number": n, "title": sections[n].title}
                               for n in wanted]}, budget=budget, synthesis=True)
        if body is None:
            continue
        for number, text in split_sections(body["text"], wanted).items():
            section = sections[number]
            if section.done:
                continue
            section.text, section.writer = text, writer
            runner.events.record("section_drafted", number=number,
                                 title=section.title, writer=writer,
                                 chars=len(text), text=text)


async def _draft(runner, question, sections, writers, budget):
    numbers = sorted(sections)
    lanes = {w: [] for w in writers}
    for index, block in enumerate(_blocks(numbers)):
        lanes[writers[index % len(writers)]].append(block)
    await asyncio.gather(*(_draft_lane(runner, question, sections, w, b, budget)
                           for w, b in lanes.items() if b))
    for attempt in range(MAX_CONTINUATION_PASSES):
        missing = [n for n in numbers if not sections[n].done]
        available = _usable(runner, writers)
        if not missing or not available or budget.remaining() < 20:
            break
        runner.events.record("draft_continuation", pass_number=attempt + 1,
                             missing=missing, writers=available)
        # Prefer a writer that has not already failed on this section.
        lanes = {w: [] for w in available}
        for index, block in enumerate(_blocks(missing)):
            tried = set(sections[block[0]].attempts)
            fresh = [w for w in available if w not in tried] or available
            lanes[fresh[index % len(fresh)]].append(block)
        await asyncio.gather(*(_draft_lane(runner, question, sections, w, b, budget)
                               for w, b in lanes.items() if b))


def _parse_findings(body: dict | None, numbers: set[int]) -> list[dict]:
    findings = []
    for item in (body or {}).get("findings") or []:
        if not isinstance(item, dict):
            continue
        try:
            number = int(str(item.get("section", "")).strip().split(".")[0])
        except ValueError:
            continue
        issue = str(item.get("issue") or "").strip()
        if number not in numbers or not issue:
            continue
        severity = "major" if str(item.get("severity", "")).lower() == "major" else "minor"
        findings.append({"section": number, "severity": severity,
                         "issue": issue[:600], "fix": str(item.get("fix") or "")[:600]})
    return findings


async def _review(runner, question, sections, writers, budget):
    by_writer: dict[str, list[int]] = {}
    for s in sections.values():
        if s.done:
            by_writer.setdefault(s.writer, []).append(s.number)
    jobs = []
    for writer, numbers in by_writer.items():
        others = [f for f in _usable(runner, writers) if f != writer]
        if not others:
            continue
        reviewer = others[(writers.index(writer) if writer in writers else 0) % len(others)]
        for n in numbers:
            sections[n].reviewer = reviewer
        jobs.append((reviewer, numbers))

    async def one(reviewer, numbers):
        reply = await runner._call(reviewer, "section_review", {
            "request": question,
            "outline": [s.title for s in sections.values()],
            "sections": [{"number": n, "title": sections[n].title,
                          "text": sections[n].text} for n in numbers]},
            budget=budget, synthesis=True)
        for finding in _parse_findings(reply, set(numbers)):
            finding["reviewer"] = reviewer
            sections[finding["section"]].findings.append(finding)
        if reply is None:
            for n in numbers:
                sections[n].reviewer = None
    await asyncio.gather(*(one(r, n) for r, n in jobs))


async def _revise(runner, question, sections, budget):
    flagged: dict[str, list[int]] = {}
    for s in sections.values():
        if s.done and any(f["severity"] == "major" for f in s.findings):
            flagged.setdefault(s.writer, []).append(s.number)

    async def one(writer, numbers):
        reply = await runner._call(writer, "section_revise", {
            "request": question,
            "sections": [{"number": n, "title": sections[n].title,
                          "text": sections[n].text,
                          "major_findings": [f for f in sections[n].findings
                                             if f["severity"] == "major"]}
                         for n in numbers]}, budget=budget, synthesis=True)
        if reply is None:
            return
        for number, text in split_sections(reply["text"], numbers).items():
            # Guard against a truncated or summarised "revision".
            if len(text) >= 0.6 * len(sections[number].text):
                sections[number].text, sections[number].revised = text, True
                runner.events.record("section_revised", number=number,
                                     writer=writer, chars=len(text), text=text)
    await asyncio.gather(*(one(w, n) for w, n in flagged.items()))


def assemble(sections: dict[int, Section]) -> str:
    parts = []
    for number in sorted(sections):
        s = sections[number]
        body = s.text or "_This section could not be drafted in this run._"
        parts.append("## " + str(number) + ". " + s.title + "\n\n" + body)
    return "\n\n".join(parts)


def _issues(sections, audit, auditor_family):
    issues = []
    for s in sections.values():
        for n, f in enumerate(s.findings, 1):
            if f["severity"] == "major" and s.revised:
                status, why = "unreviewed", "section revised by its writer; the " \
                    "revision was not re-reviewed"
            elif f["severity"] == "major":
                status, why = "unresolved", "major finding; the section was not revised"
            else:
                status, why = "unreviewed", "minor finding; left to the reader"
            issues.append({"id": "s" + str(s.number) + "f" + str(n),
                           "source": "section_review", "proposal_id": None,
                           "claim_id": "section " + str(s.number),
                           "raised_by": f["reviewer"], "issue": f["issue"],
                           "severity": f["severity"], "suggested_fix": f["fix"],
                           "author_disposition": None, "rereview": None,
                           "status": status, "reason": why})
    for n, text in enumerate((audit or {}).get("issues") or [], 1):
        issues.append({"id": "a" + str(n), "source": "audit", "proposal_id": None,
                       "claim_id": None, "raised_by": auditor_family, "issue": text,
                       "author_disposition": None, "rereview": None,
                       "status": "unreviewed",
                       "reason": "raised by the independent auditor after revision; "
                                 "no response round follows the audit"})
    return issues


async def _outline(runner, question, writers, budget) -> list[str]:
    titles = extract_sections(question)
    if titles:
        return titles
    for family in _usable(runner, writers):
        body = await runner._call(family, "outline", {"request": question},
                                  budget=budget, synthesis=True)
        titles = [str(t).strip() for t in (body or {}).get("sections") or []
                  if str(t).strip()][:30]
        if len(titles) >= 2:
            return titles
    return ["Answer"]


async def run_document(runner, question: str, *, mode: str = "full") -> dict:
    from council.workflow import ANSWER_LABELS, ISSUE_STATUSES
    total = runner.budget.remaining()
    writers = list(runner.solvers)
    runner.events.record("run_started", question=question, mode=mode,
                         task_type="document", solver_families=writers,
                         judge_family=runner.judge_family,
                         auditor_family=runner.auditor_family,
                         auditor_fallbacks=list(runner.auditor_fallbacks))
    start = time.monotonic()

    def until(share_sum):
        return RunBudget(min(runner.budget.end, start + share_sum * total), 0.0)

    with runner.stage("outline"):
        titles = await _outline(runner, question, writers, until(0.05))
    sections = {n: Section(n, t) for n, t in enumerate(titles, 1)}
    with runner.stage("drafting"):
        await _draft(runner, question, sections, writers,
                     until(STAGE_SHARE["drafting"]))
    delivered = [n for n, s in sections.items() if s.done]
    runner.events.record("document_assembled", delivered=len(delivered),
                         required=len(sections),
                         missing=[n for n in sections if n not in delivered])
    if delivered:
        with runner.stage("review"):
            await _review(runner, question, sections, writers,
                          until(STAGE_SHARE["drafting"] + STAGE_SHARE["review"]))
        if any(f["severity"] == "major" for s in sections.values() for f in s.findings):
            with runner.stage("revision"):
                await _revise(runner, question, sections, until(sum(STAGE_SHARE.values())))

    audit, auditor_used = None, None
    if delivered and runner.auditor_family is not None:
        flagged = [s for s in sections.values() if s.findings and s.done]
        with runner.stage("audit"):
            audit, auditor_used = await runner._call_with_fallbacks(
                "audit", (runner.auditor_family, *runner.auditor_fallbacks), {
                    "question": question,
                    "outline": [s.title for s in sections.values()],
                    "section_findings": [dict(f, section_title=sections[f["section"]].title,
                                              revised=sections[f["section"]].revised)
                                         for s in flagged for f in s.findings],
                    "flagged_sections": [{"number": s.number, "title": s.title,
                                          "text": s.text[:4000]} for s in flagged],
                    "other_section_openings": [{"number": s.number, "title": s.title,
                                                "opening": s.text[:400]}
                                               for s in sections.values()
                                               if s.done and not s.findings],
                    "rule": "Independently inspect the document for errors, missing "
                            "requirements and unsupported claims; do not treat model "
                            "agreement as proof."},
                runner.budget.end)

    document = assemble(sections) if delivered else None
    issues = _issues(sections, audit, auditor_used)
    missing = [n for n in sections if not sections[n].done]
    reviewed = [n for n in delivered if sections[n].reviewer]
    limitations = list(dict.fromkeys(runner.warnings))
    if missing:
        limitations.append("Sections not delivered: " + ", ".join(map(str, missing)))
    if delivered and len(reviewed) < len(delivered):
        limitations.append(str(len(delivered) - len(reviewed)) + " of "
                           + str(len(delivered)) + " sections were not reviewed by "
                           "another model family")
    if delivered and audit is None and runner.auditor_family is not None:
        limitations.append("Independent auditor unavailable")
    status = ("unresolved" if not delivered else "document_reviewed_unverified"
              if reviewed else "document_unreviewed_unverified")
    selection = {"answer": document, "basis": "document_assembled", "family": None,
                 "proposal_id": None, "status": status, "verified": False,
                 "alternatives": [],
                 "auditor_issue_count": len(audit.get("issues") or []) if audit else None,
                 "open_issue_count": sum(i["status"] != "addressed" for i in issues),
                 "label": ANSWER_LABELS[status]}
    runner.events.record("answer_selected", basis="document_assembled",
                         family=None, proposal_id=None)
    result = {
        "research_result": {"research_question": question,
                            "completion_status": ("partial" if delivered else "unresolved"),
                            "proposed_approach": "", "judge_summary": "",
                            "limitations": limitations, "supported_findings": [],
                            "hypotheses": [], "unresolved_objections":
                                [i["issue"] for i in issues if i["status"] != "addressed"],
                            "proposed_experiments": list((audit or {}).get(
                                "suggested_checks") or []),
                            "proof_obligations": [], "next_actions": [],
                            "checked_source_ids": []},
        "task_type": "document",
        "document": {"required_sections": len(sections), "delivered": len(delivered),
                     "missing": missing, "reviewed": len(reviewed),
                     "revised": sum(s.revised for s in sections.values()),
                     "sections": [{"number": s.number, "title": s.title,
                                   "writer": s.writer, "reviewer": s.reviewer,
                                   "chars": len(s.text), "revised": s.revised,
                                   "findings": len(s.findings),
                                   "attempted_by": s.attempts}
                                  for s in sections.values()]},
        "proposals": [], "objections": [], "author_responses": [], "claims": [],
        "actual_models": runner.actual_models, "warnings": runner.warnings,
        "verdict": {"status": "unresolved", "cause": "model_review_only",
                    "basis": ["No source or mathematical checker was run"]},
        "audit": {"completed": audit is not None, "family": auditor_used,
                  "issues": (audit or {}).get("issues", []),
                  "suggested_checks": (audit or {}).get("suggested_checks", [])},
        "quorum": {"required": 1, "obtained": len(set(s.writer for s in sections.values()
                                                       if s.writer)),
                   "requested": len(writers), "met": bool(delivered),
                   "policy": runner.quorum_policy, "action": "document_mode"},
        "issues": issues,
        "issue_summary": {k: sum(i["status"] == k for i in issues) for k in ISSUE_STATUSES},
        "answer_comparison": [], "unresolved_differences": [],
        "role_coverage": {"requested_solvers": writers,
                          "valid_proposer_families": sorted({s.writer for s in
                                                             sections.values() if s.writer}),
                          "review_objection_count": sum(len(s.findings)
                                                        for s in sections.values()),
                          "rereview_count": 0,
                          "independent_audit_completed": audit is not None,
                          "actual_audit_family": auditor_used, "judge_family": None},
        "candidate_answers": {}, "answer_selection": selection,
        "selected_answer": document, "final_answer": document,
        "timing": runner.timing_report(),
    }
    result["council_result"] = dict(result["research_result"], request=question)
    runner.events.record("run_finished", result=result)
    return result