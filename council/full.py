"""Compatibility names for the single, audit-focused Council pipeline.

The former RFC-specific/full formal-verification runner was removed.
All modes now use CouncilRunner; model reviews are never proofs.
"""
from council.workflow import CouncilRunner

FULL_SOLVERS=("gpt-oss","command","nemotron")
TRANSLATOR="glm"
JUDGE="gemini"
AUDITOR="gemma"


class FullCouncilRunner(CouncilRunner):
    def __init__(self, transport, events, budget, *,
                 requirements=None, solvers=FULL_SOLVERS,
                 judge_family=JUDGE, auditor_family=AUDITOR,
                 auditor_fallbacks=(), judge_fallbacks=(),
                 solver_strategies=None, **options):
        super().__init__(transport, events, budget,
                         solvers=tuple(solvers), judge_family=judge_family,
                         auditor_family=auditor_family,
                         auditor_fallbacks=tuple(auditor_fallbacks),
                         judge_fallbacks=tuple(judge_fallbacks),
                         solver_strategies=solver_strategies, **options)
        self.requirements=requirements

    async def run(self, question, *, mode="full"):
        return await super().run(question, mode=mode)


class QuickCouncilRunner(FullCouncilRunner):
    def __init__(self, transport, events, budget, *,
                 requirements=None, solver="gpt-oss",
                 judge_family=JUDGE, solver_strategies=None, **options):
        options.setdefault("task_type", "answer")
        super().__init__(transport, events, budget,
                         requirements=requirements, solvers=(solver,),
                         judge_family=judge_family,
                         auditor_family=None,
                         solver_strategies=solver_strategies, **options)

    async def run(self, question, *, mode="quick"):
        return await super().run(question, mode="quick")
