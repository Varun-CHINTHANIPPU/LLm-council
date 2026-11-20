"""Concise, honest Markdown report of one Council run (#12).

Order: the answer and how it was chosen, the reasoning behind it, competing
answers and open differences, criticisms by resolution status, independent
auditor concerns, limitations, and run facts. Nothing here upgrades evidence:
every section repeats what the run recorded.
"""
from __future__ import annotations

STATUS_TITLES = (("contested", "Contested"), ("unresolved", "Unresolved"),
                 ("unreviewed", "Not independently re-reviewed"),
                 ("addressed", "Addressed"))


def _clip(text, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def render_markdown(result: dict, *, run_id: str | None = None,
                    reasoning_chars: int = 1500) -> str:
    brief = result.get("council_result") or result.get("research_result") or {}
    sel = result.get("answer_selection") or {}
    audit = result.get("audit") or {}
    quorum = result.get("quorum") or {}
    lines = ["# Council response", "",
             "**Request:** " + _clip(brief.get("request") or
                                    brief.get("research_question") or
                                    result.get("question") or "(not recorded)", 220), ""]

    lines += ["## Answer", ""]
    answer = result.get("final_answer")
    lines.append(answer.strip() if isinstance(answer, str) and answer.strip()
                 else "_No answer was produced._")
    lines += ["", "> **" + str(sel.get("label") or "Unverified model output") + ".**"]
    facts = []
    if sel.get("family"):
        facts.append("chosen by: " + str(sel["family"]))
    facts.append("basis: " + str(sel.get("basis", "unknown")))
    if sel.get("open_issue_count") is not None:
        facts.append("open issues: " + str(sel["open_issue_count"]))
    lines += ["> " + "; ".join(facts) + ".", ""]

    reasoning = brief.get("judge_summary") or brief.get("proposed_approach")
    if reasoning and _clip(reasoning, 10_000) != _clip(answer, 10_000):
        lines += ["## Supporting reasoning", "", _clip(reasoning, reasoning_chars), ""]

    comparison = result.get("answer_comparison") or []
    differences = result.get("unresolved_differences") or []
    alternatives = sel.get("alternatives") or []
    if comparison or differences or len(alternatives) > 1:
        lines += ["## Competing answers", ""]
        relation = {c.get("proposal_id"): c for c in comparison}
        for alt in alternatives:
            c = relation.get(alt.get("proposal_id"), {})
            tag = c.get("relation", "not compared")
            lines.append("- **" + str(alt.get("proposal_id")) + "** ("
                         + str(alt.get("family")) + ", " + str(tag).replace("_", " ")
                         + "): " + _clip(str(alt.get("answer") or "")
                                         .split("\n\n")[0], 220))
        if differences:
            lines += ["", "**Open differences**", ""]
            lines += ["- " + _clip(d, 400) for d in differences]
        lines.append("")

    doc = result.get("document")
    if doc:
        lines += ["## Document coverage", "",
                  "- Sections delivered: " + str(doc["delivered"]) + " of "
                  + str(doc["required_sections"])
                  + ((" (missing: " + ", ".join(map(str, doc["missing"])) + ")")
                     if doc["missing"] else ""),
                  "- Reviewed by a different model family: " + str(doc["reviewed"])
                  + "; revised after major findings: " + str(doc["revised"]), ""]

    issues = result.get("issues") or []
    peer = [i for i in issues if i.get("source") in {"peer_review", "section_review"}]
    if peer:
        lines += ["## Reviewer objections", "",
                  ", ".join(str(sum(i.get("status") == key for i in peer))
                            + " " + title.lower()
                            for key, title in STATUS_TITLES), ""]
        for key, title in STATUS_TITLES:
            group = [i for i in peer if i.get("status") == key]
            if not group:
                continue
            lines += ["**" + title + "**", ""]
            for i in group:
                lines.append("- " + _clip(i.get("issue"), 300) + " _(on "
                             + str(i.get("claim_id")) + "; " + _clip(i.get("reason"), 240)
                             + ")_")
            lines.append("")

    lines += ["## Independent auditor", ""]
    if audit.get("completed"):
        concerns = audit.get("issues") or []
        lines.append("Audited by " + str(audit.get("family")) + ". "
                     + (str(len(concerns)) + " concern(s), not answered by the "
                        "proposers:" if concerns else "No concerns raised."))
        lines += ["- " + _clip(c, 400) for c in concerns]
        checks = audit.get("suggested_checks") or []
        if checks:
            lines += ["", "Suggested checks:"] + ["- " + _clip(c, 300) for c in checks]
    else:
        lines.append("**The independent audit did not complete.** Treat the answer "
                     "as unaudited.")
    lines.append("")

    limitations = brief.get("limitations") or []
    if limitations:
        lines += ["## Limitations", ""] + ["- " + _clip(x, 300) for x in limitations]
        lines.append("")

    timing = result.get("timing") or {}
    if timing.get("stages"):
        lines += ["## Timing", "",
                  "| Stage | Wall time | Calls ok / failed | Time in failed calls "
                  "| Network attempts (failed) |", "|---|---|---|---|---|"]
        for st in timing["stages"]:
            lines.append("| {} | {:.0f} s | {} / {} | {:.0f} s | {} ({}) |".format(
                st["stage"], st["wall_seconds"], st["role_calls_ok"],
                st["role_calls_failed"], st["seconds_in_failed_calls"],
                st["network_attempts"], st["network_failures"]))
        lines.append("")
        lines.append("Total {:.0f} s.".format(timing.get("total_seconds", 0))
                     + (" Skipped after earlier failures: "
                        + ", ".join(sorted(timing["open_circuits"]))
                        if timing.get("open_circuits") else ""))
        lines.append("")

    roles = result.get("role_coverage") or {}
    calls = len(result.get("actual_models") or [])
    lines += ["## Run", "",
              "- Proposals: " + str(quorum.get("obtained", len(result.get("proposals") or [])))
              + " of " + str(quorum.get("requested", len(roles.get("requested_solvers") or [])))
              + " (quorum " + ("met" if quorum.get("met", True) else "not met, "
                               + str(quorum.get("action"))) + ")",
              "- Judge: " + str(roles.get("judge_family") or "none"),
              "- Objections: " + str(roles.get("review_objection_count", 0))
              + ", independent re-reviews: " + str(roles.get("rereview_count", 0)),
              "- Model calls with a response: " + str(calls)
              + (", network attempts incl. failures: "
                 + str(timing["network_attempts"]) if timing else "")]
    if run_id:
        lines.append("- Run ID: `" + run_id + "`")
    return "\n".join(lines).rstrip() + "\n"