"""Allowlisted, role-reserved model families; endpoints are not independent votes."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Endpoint:
    provider: str
    model: str
    # Per-endpoint overrides used by long-form reasoning providers.
    # None inherits the caller's timeout and role-dependent token allowance.
    timeout_seconds: float | None = None
    max_tokens: int | None = None

@dataclass(frozen=True)
class Family:
    name: str
    endpoints: tuple[Endpoint, ...]
    counts_for_quorum: bool = True

FAMILIES: dict[str, Family] = {
    "gpt-oss": Family("gpt-oss", (Endpoint("Groq", "openai/gpt-oss-120b"),)),
    "command": Family("command", (Endpoint("Cohere", "command-a-03-2025",
                                               timeout_seconds=120, max_tokens=6000),
                                  Endpoint("Cohere", "command-a-plus-05-2026",
                                           timeout_seconds=120, max_tokens=6000))),
    "nemotron": Family("nemotron", (Endpoint("OpenRouter", "nvidia/nemotron-3-ultra-550b-a55b:free",
                                               timeout_seconds=110, max_tokens=8000),
                                    Endpoint("LLM7.io", "nemotron-3-nano:30b",
                                             timeout_seconds=110, max_tokens=8000))),
    # Gemma for the independent audit. LLM7.io lists gemma4:31b and the
    # smaller gemma-3-27b in its free "turbo" tier (Nov 2026); Ollama Cloud
    # answered HTTP 402 (payment required). Order: larger free model, then
    # the lower one, then Ollama. Any endpoint that turns out to need
    # payment fails as model_not_free_eligible / billing_required and is
    # skipped; nothing is ever bought.
    "gemma": Family("gemma", (Endpoint("LLM7.io", "gemma4:31b", timeout_seconds=60),
                              Endpoint("LLM7.io", "gemma-3-27b", timeout_seconds=60),
                              Endpoint("Ollama Cloud", "gemma4:31b"))),
    "gemini": Family("gemini", (Endpoint("Gemini", "gemini-3.5-flash-lite"),
                               Endpoint("Gemini", "gemini-3.6-flash"),
                               Endpoint("Gemini", "gemini-3.8-flash"),
                               Endpoint("Gemini", "gemini-3.7-flash"))),
    # Slow reasoning endpoints get their own request allowance. The global
    # default stays short so flaky endpoints cannot hold a stage for long.
    "glm": Family("glm", (Endpoint("Z.ai", "glm-4.7-flash", timeout_seconds=60),
                         Endpoint("Cloudflare", "@cf/zai-org/glm-4.7-flash",
                                  timeout_seconds=60))),
    # minimax-m2.7 left LLM7's free tier (model_not_free_eligible). Keep it
    # first in case it returns, then any other MiniMax chat model the live
    # catalog marks free ("auto:minimax", newest version first, lower
    # versions after). Paid-only versions are never selected.
    "minimax": Family("minimax", (Endpoint("LLM7.io", "minimax-m2.7",
                                           timeout_seconds=60),
                                  Endpoint("LLM7.io", "auto:minimax",
                                           timeout_seconds=60)),
                      counts_for_quorum=False),
}

def validate_assignment(solvers: tuple[str, ...], judge: str,
                        auditor: str | None) -> None:
    if len(set(solvers)) != len(solvers):
        raise ValueError("Solver families must be distinct")
    roles=(*solvers,judge)+((auditor,) if auditor is not None else ())
    if len(solvers) < 1 or any(f not in FAMILIES for f in roles):
        raise ValueError("Unknown or missing model family")
    if judge in solvers or (auditor is not None and judge == auditor):
        raise ValueError("Judge family must be reserved and isolated")
    if auditor is not None and auditor in solvers:
        raise ValueError("Auditor family must be separate from solvers")

def quorum(families: set[str]) -> int:
    return sum(FAMILIES[f].counts_for_quorum for f in families)

def endpoint_list(family: str) -> tuple[Endpoint, ...]:
    return FAMILIES[family].endpoints
