"""Provider JSON-format hints; all responses are locally validated."""
from __future__ import annotations
import os

S={"type":"string"}
A={"type":"array","items":S}
CLAIM={"type":"object","properties":{
        "id":S,"statement":S,"assumptions":A,"limitations":A},
       "required":["id","statement","assumptions","limitations"],
       "additionalProperties":False}
PROPOSAL_SCHEMA={"type":"object","properties":{
        "approach":S,"final_answer":S,
        "claims":{"type":"array","items":CLAIM},
        "hypotheses":A,"experiments":A},
    "required":["approach","final_answer","claims","hypotheses","experiments"],
    "additionalProperties":False}

GEMINI_JUDGE_SCHEMA={
    "type":"OBJECT",
    "properties":{
        "status":{"type":"STRING","enum":["answer","unresolved"]},
        "answer":{"type":"STRING"},
        "partial_answer":{"type":"STRING"},
        "blocking":{"type":"ARRAY","items":{"type":"STRING"}},
        "basis":{"type":"ARRAY","items":{"type":"STRING"}}},
    "required":["status"],
    "propertyOrdering":["status","answer","partial_answer","blocking","basis"]}

# Roles that return Markdown prose, never JSON. Forcing JSON mode on them
# squeezes long documents into escaped strings or truncates them.
TEXT_ROLES=frozenset({"draft_sections","section_revise","probe"})

def response_format(provider:str, model:str, role:str)->dict|None:
    if role.startswith("baseline") or role in TEXT_ROLES:
        return None
    if provider=="Cohere":
        if role=="proposal":
            return {"type":"cohere_json","schema":PROPOSAL_SCHEMA}
        return {"type":"cohere_json"}
    if provider=="Gemini":
        if role=="judge" and os.getenv("COUNCIL_GEMINI_JUDGE_SCHEMA","1")!="0":
            return {"type":"gemini_response_schema","schema":GEMINI_JUDGE_SCHEMA}
        return None
    if provider=="Groq" and model.startswith("openai/gpt-oss-"):
        if role=="proposal":
            return {"type":"json_schema","json_schema":{
                "name":"council_proposal","strict":True,
                "schema":PROPOSAL_SCHEMA}}
        return {"type":"json_object"}
    return None
