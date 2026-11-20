"""Strict, small JSON contracts for the active Council roles."""
from __future__ import annotations
from typing import Literal, get_origin, get_args
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

class ModelSchema(BaseModel):
    model_config=ConfigDict(extra="forbid",str_strip_whitespace=True)

    @model_validator(mode="before")
    @classmethod
    def _listify_text(cls, value):
        if not isinstance(value,dict):
            return value
        clean=dict(value)
        for name,field in cls.model_fields.items():
            if (name in clean and isinstance(clean[name],str)
                    and get_origin(field.annotation) is list
                    and get_args(field.annotation)==(str,)):
                clean[name]=[clean[name]] if clean[name].strip() else []
        return clean

class ProposedClaim(ModelSchema):
    id:str=Field(min_length=1,max_length=160)
    statement:str=Field(min_length=1,max_length=10000)
    assumptions:list[str]=Field(default_factory=list)
    limitations:list[str]=Field(default_factory=list)

class Proposal(ModelSchema):
    approach:str=Field(min_length=1)
    final_answer:str=""
    claims:list[ProposedClaim]=Field(min_length=1)
    hypotheses:list[str]=Field(default_factory=list)
    experiments:list[str]=Field(default_factory=list)

class ReviewObjection(ModelSchema):
    claim_id:str=Field(min_length=1)
    issue:str=Field(min_length=1)
    consequence:str=Field(min_length=1)
    suggested_check:str=Field(min_length=1)

class Review(ModelSchema):
    objections:list[ReviewObjection]=Field(default_factory=list)

class RevisedClaim(ModelSchema):
    claim_id:str=Field(min_length=1)
    statement:str=Field(min_length=1)

class AuthorResponseSchema(ModelSchema):
    objection_id:str=Field(min_length=1)
    disposition:Literal["accept","reject","partial","unresolved"]
    explanation:str=Field(min_length=1)

class Revision(ModelSchema):
    revised_approach:str|None=None
    revised_claims:list[RevisedClaim]=Field(default_factory=list)
    responses:list[AuthorResponseSchema]=Field(default_factory=list)

class AnswerComparison(ModelSchema):
    proposal_id:str=Field(min_length=1,max_length=40)
    relation:Literal["adopted","partly_adopted","rejected","not_comparable"]
    key_difference:str=""
    reason:str=""

class Synthesis(ModelSchema):
    proposed_approach:str=Field(min_length=1)
    final_answer:str=Field(default="",max_length=24000)
    answer_comparison:list[AnswerComparison]=Field(default_factory=list)
    unresolved_differences:list[str]=Field(default_factory=list)
    supported_findings:list[str]=Field(default_factory=list)
    hypotheses:list[str]=Field(default_factory=list)
    unresolved_objections:list[str]=Field(default_factory=list)
    proof_obligations:list[str]=Field(default_factory=list)
    proposed_experiments:list[str]=Field(default_factory=list)
    next_actions:list[str]=Field(default_factory=list)
    completion_status:Literal["partial","unresolved"]="partial"

class RereviewItem(ModelSchema):
    objection_id:str=Field(min_length=1)
    verdict:Literal["resolved","objection_invalid","still_open"]
    reason:str=Field(min_length=1)

class Rereview(ModelSchema):
    items:list[RereviewItem]=Field(default_factory=list)

class Compare(ModelSchema):
    answer_comparison:list[AnswerComparison]=Field(default_factory=list)
    unresolved_differences:list[str]=Field(default_factory=list)

class Outline(ModelSchema):
    sections:list[str]=Field(min_length=1,max_length=40)

class SectionReview(BaseModel):
    # Lenient on purpose: findings are parsed field by field afterwards.
    model_config=ConfigDict(extra="ignore")
    findings:list[dict]=Field(default_factory=list)

class Audit(ModelSchema):
    issues:list[str]=Field(default_factory=list)
    suggested_checks:list[str]=Field(default_factory=list)

class Judge(ModelSchema):
    status:Literal["answer","unresolved"]
    answer:str|None=None
    partial_answer:str|None=None
    blocking:list[str]=Field(default_factory=list)
    basis:list[str]=Field(default_factory=list)

SCHEMAS={
    "proposal":Proposal,
    "review":Review,
    "revision":Revision,
    "synthesis":Synthesis,
    "audit":Audit,
    "rereview":Rereview,
    "compare":Compare,
    "outline":Outline,
    "section_review":SectionReview,
    "judge":Judge,
}

def validate(role:str,payload:dict)->dict:
    schema=SCHEMAS.get(role)
    if schema is None:
        raise ValueError("Inactive Council role: "+str(role))
    try:
        obj=schema.model_validate(payload,strict=True)
    except ValidationError as error:
        raise ValueError("Invalid "+role+" output: "+
            "; ".join(".".join(map(str,e["loc"]))+": "+e["type"]
                      for e in error.errors()[:5])) from None
    return obj.model_dump(exclude_none=True)
