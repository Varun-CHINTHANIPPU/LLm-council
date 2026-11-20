"""Thin family-aware adapter over model_router.

The upstream API currently exposes normalized text, not raw provider response
JSON/headers. Calls are tracked honestly as text-only until raw transport exists.
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass
import hashlib
import json
import math
import os
import threading
import time
from pathlib import Path

from model_router import Candidate, KEY_NAMES, Router, RAW_CAPTURE, redacted
from router_errors import CandidateError
from router_health import HealthCache
from council.budget import RunBudget
from council.families import endpoint_list
from council.structured import response_format as model_response_format
from council.quota import QuotaScheduler, QuotaContention

class FamilyUnavailable(RuntimeError):
    def __init__(self, family: str, failures: list[str]):
        self.family = family
        self.failures = failures
        super().__init__(family + ": no eligible endpoint succeeded")

@dataclass(frozen=True)
class CallResult:
    family: str
    provider: str
    model: str
    text: str
    seconds: float
    prompt_hash: str
    raw: dict | None = None
    headers: dict | None = None
    status: int | None = None

class FamilyTransport:
    def __init__(self, *, timeout: float = 28.0, health_file: Path | None = None,
                 record=None, strict_timeout: bool = False):
        if timeout<=1 or not math.isfinite(timeout):
            raise ValueError("timeout must be finite and greater than 1 second")
        self.timeout = timeout
        self.strict_timeout=bool(strict_timeout)
        self.router = Router(timeout=timeout)
        self._llm7_lock = threading.Lock()
        self.cache = HealthCache(health_file or Path(".router_health.json"))
        self.record = record or (lambda *a, **k: None)
        # Role-specific request allowances (seconds). The single reserved
        # audit request reads the whole brief and has no sibling endpoint to
        # fall back to, so it may wait longer than an ordinary call. Always
        # capped by the run budget; ignored with an explicit --timeout.
        try:
            audit_seconds=float(os.getenv("COUNCIL_AUDIT_TIMEOUT","100"))
        except ValueError:
            raise ValueError("COUNCIL_AUDIT_TIMEOUT must be numeric") from None
        if not 30 <= audit_seconds <= 180:
            raise ValueError("COUNCIL_AUDIT_TIMEOUT must be 30..180")
        self.role_timeouts: dict[str, float] = {"audit": audit_seconds}
        # Per-process outcomes only. This reorders a fixed free-tier allowlist.
        self._streak: dict[tuple[str,str],int] = {}
        self._answered: set[tuple[str,str]] = set()
        # Gentle unconfigured pacing (30 RPM/account) is a protective
        # floor, NOT a claim about any provider's free-tier allowance.
        # Real RPM/TPM caps must be configured from provider dashboards.
        try:
            safe_rpm=float(os.getenv("COUNCIL_UNCONFIGURED_RPM","30"))
        except ValueError:
            raise ValueError("COUNCIL_UNCONFIGURED_RPM must be numeric") from None
        if not 1 <= safe_rpm <= 600:
            raise ValueError("COUNCIL_UNCONFIGURED_RPM must be 1..600")
        self.quota = QuotaScheduler(
            default_rate=safe_rpm, default_burst=1,
            daily_state_file=self.cache.path.with_name(".council_daily_quotas.json"))
        # Provider limits are user-confirmed settings, not guessed quotas.
        try:
            configured = json.loads(os.getenv('COUNCIL_LIMITS_JSON', '{}'))
            for provider, settings in configured.items():
                if isinstance(settings, dict) and 'shared_daily_requests' in settings:
                    self.quota.shared_daily[provider] = int(settings['shared_daily_requests'])
                if isinstance(settings, dict) and 'shared_daily_tokens' in settings:
                    self.quota.shared_daily_tokens[provider] = int(settings['shared_daily_tokens'])
        except (ValueError, TypeError, AttributeError):
            raise ValueError('Invalid COUNCIL_LIMITS_JSON')

    @staticmethod
    def repeat_cooldown(kind: str, streak: int | None) -> float | None:
        """Cooldown for the n-th consecutive transient failure of an endpoint.

        A timeout already cost a whole request allowance, so repeats back
        off hard (30s, 2 min, 5 min). An overload reply (HTTP 5xx) or a
        dropped connection returns within seconds and often clears quickly,
        so repeats back off gently (30s, 45s, 60s) and the endpoint is not
        locked out for the rest of a run.
        """
        if not streak:
            return None
        ladder = ((30.0, 120.0, 300.0) if kind == "timeout"
                  else (30.0, 45.0, 60.0))
        return ladder[min(streak, len(ladder)) - 1]

    # Blocks that waiting inside one run cannot clear.
    _LASTING_BLOCKS = ("billing_required", "auth", "model_unavailable",
                       "model_not_free_eligible")

    def soonest_ready(self, family: str) -> float | None:
        """Seconds until some endpoint of the family may be tried again.

        0.0 means one is usable now. None means no wait would help (not
        configured, or blocked for billing/access reasons). Makes no call.
        """
        best = None
        for entry in endpoint_list(family):
            if not self._eligible(entry.provider):
                continue
            blocked = self.cache.blocked(entry.provider, entry.model)
            if not blocked:
                return 0.0
            why, seconds = blocked
            if any(label in str(why) for label in self._LASTING_BLOCKS):
                continue
            best = seconds if best is None else min(best, seconds)
        return None if best is None else float(best)

    @staticmethod
    def _eligible(provider: str) -> bool:
        # A key does not prove a free allowance. Restrict risky providers to
        # user-confirmed free modes; all other entries are a fixed allowlist.
        if provider == "Cohere" and os.getenv("COHERE_TRIAL_CONFIRMED") != "1":
            return False
        if provider == "Ollama Cloud" and os.getenv("OLLAMA_FREE_CONFIRMED") != "1":
            return False
        key = KEY_NAMES.get(provider)
        return bool(key and os.getenv(key, "").strip())

    async def _concrete_endpoints(self, family: str, budget: RunBudget):
        """Expand 'auto:<lineage>' endpoints into the catalog's free models.

        Concrete endpoints pass through unchanged. A catalog that cannot be
        read, or has no free model of that lineage, simply contributes no
        endpoint (recorded), so the family falls through to its next one.
        """
        result = []
        for entry in endpoint_list(family):
            if not entry.model.startswith("auto:"):
                result.append(entry)
                continue
            if not self._eligible(entry.provider):
                continue
            try:
                options = await asyncio.wait_for(asyncio.to_thread(
                    self.router.expand, Candidate(entry.provider, entry.model),
                    set(), 4), timeout=max(1.0, min(15.0, budget.remaining())))
            except (CandidateError, asyncio.TimeoutError, TimeoutError,
                    KeyError, TypeError, ValueError) as exc:
                self.record("auto_endpoint_unresolved", family=family,
                            provider=entry.provider, selector=entry.model,
                            kind=getattr(exc, "kind", "unavailable"))
                continue
            known = {(e.provider, e.model) for e in result}
            for option in options:
                if (option.provider, option.model) not in known:
                    result.append(type(entry)(option.provider, option.model,
                        timeout_seconds=entry.timeout_seconds,
                        max_tokens=entry.max_tokens))
            self.record("auto_endpoint_resolved", family=family,
                        provider=entry.provider, selector=entry.model,
                        models=[o.model for o in options])
        return tuple(result)

    async def complete(self, family: str, message: str, budget: RunBudget, *,
                       role: str, synthesis: bool = False) -> CallResult:
        failures = []
        seen: set[tuple[str, str]] = set()
        def rank(item):
            position,endpoint=item
            key=(endpoint.provider,endpoint.model)
            streak=self._streak.get(key,0)
            state=(2+min(streak,3) if streak else
                   0 if key in self._answered else 1)
            return (state,position)
        ordered=[e for _,e in sorted(enumerate(
            await self._concrete_endpoints(family, budget)),key=rank)]
        for position,entry in enumerate(ordered):
            # With an explicit CLI timeout, do not silently extend the
            # request to a provider's longer preferred endpoint timeout.
            endpoint_timeout=(self.timeout if self.strict_timeout else
                              max(self.timeout,entry.timeout_seconds or 0,
                                  self.role_timeouts.get(role,0)))
            if not self._eligible(entry.provider):
                failures.append(entry.provider + ": unconfigured or free mode unconfirmed")
                continue
            blocked = self.cache.blocked(entry.provider, entry.model)
            if blocked:
                # Name the earlier failure kind and the remaining wait so a
                # skipped role is diagnosable. Kinds are fixed local labels;
                # no provider text, URL or key is included.
                why, seconds = blocked
                failures.append(entry.provider + "/" + entry.model +
                                ": cooldown (" + str(why) + ", " +
                                str(seconds) + "s left)")
                self.record("model_call_skipped", role=role, family=family,
                            provider=entry.provider, model=entry.model,
                            reason="cooldown", cooldown_cause=str(why),
                            seconds_left=seconds)
                continue
            allowed = budget.allowed(endpoint_timeout, synthesis=synthesis)
            if allowed <= 1:
                raise FamilyUnavailable(family, failures + ["deadline"])
            minimum = min(10.0, 0.3 * endpoint_timeout)
            if allowed < minimum:
                failures.append(entry.provider + "/" + entry.model +
                                ": insufficient_time (" + str(round(allowed, 1)) + "s)")
                self.record("model_call_skipped", role=role, family=family,
                            provider=entry.provider, model=entry.model,
                            reason="insufficient_time",
                            seconds_available=round(allowed, 1),
                            seconds_minimum=round(minimum, 1))
                continue
            candidate = Candidate(entry.provider, entry.model)
            # Opt-in larger translator output allowance when a reasoning
            # provider truncates responses. Default stays quota-conservative.
            if role == "translate":
                try:
                    output_tokens=int(os.getenv(
                        "COUNCIL_TRANSLATOR_MAX_TOKENS", "1800"))
                except ValueError:
                    raise ValueError(
                        "COUNCIL_TRANSLATOR_MAX_TOKENS must be an integer") from None
                if not 256 <= output_tokens <= 8192:
                    raise ValueError(
                        "COUNCIL_TRANSLATOR_MAX_TOKENS must be 256..8192")
            else:
                role_tokens = {"proposal": 2800, "review": 3000,
                               "revision": 3500, "repair": 2500,
                               "topic_judge": 2100,
                               # A blind judge writes a summary plus blocker
                               # lists; 1800 truncated on reasoning models.
                               "judge": 3200,
                               # Up to eight verdicts, each with a quotation
                               # and a reason, truncated at 1800.
                               "objection_judge": 3000,
                               "quote_repair": 1200,
                               # The judge writes the consolidated answer the
                               # user receives; 1800 forced short summaries.
                               "synthesis": 6000, "audit": 2500,
                               "rereview": 2000, "compare": 1500,
                               # Document mode: long Markdown sections.
                               "outline": 1200, "draft_sections": 8000,
                               "section_review": 3000,
                               "section_revise": 8000, "probe": 16}
                output_tokens = role_tokens.get(role, 1800)
            if entry.max_tokens is not None:
                output_tokens = max(output_tokens, entry.max_tokens)
            # Cohere's long strict-schema proposals repeatedly truncated
            # despite 6000 tokens. With compact source inputs and at most 5
            # claims, default to a smaller output cap. Other Cohere roles
            # retain the higher 6000-token endpoint limit.
            if entry.provider=="Cohere" and role=="proposal":
                try:
                    cap=int(os.getenv("COUNCIL_COHERE_PROPOSAL_MAX_TOKENS","3500"))
                except ValueError:
                    raise ValueError(
                        "COUNCIL_COHERE_PROPOSAL_MAX_TOKENS must be an integer") from None
                if not 2000<=cap<=6000:
                    raise ValueError(
                        "COUNCIL_COHERE_PROPOSAL_MAX_TOKENS must be 2000..6000")
                output_tokens=min(output_tokens,cap)
            estimated_tokens=max(1,len(message)//3)+output_tokens
            retry_after = None
            http_status = None
            timeout = None
            try:
                call_deadline = time.monotonic() + allowed
                await self.quota.acquire(entry.provider, entry.model,
                    deadline=call_deadline,
                    expected_tokens=estimated_tokens)
                allowed = min(call_deadline - time.monotonic(),
                              budget.allowed(endpoint_timeout, synthesis=synthesis))
                if allowed <= 0:
                    raise TimeoutError()
                # Same allowlist/catalog safeguards as the original router.
                options = await asyncio.wait_for(
                    asyncio.to_thread(self.router.expand, candidate, seen),
                    timeout=allowed)
                if not options or options[0].model != entry.model:
                    raise CandidateError("Model not in permitted catalog", kind="model_unavailable")
                seen.add((entry.provider, entry.model))
                t0 = time.monotonic()
                timeout = min(call_deadline - time.monotonic(),
                              budget.allowed(endpoint_timeout, synthesis=synthesis))
                if timeout <= 1:
                    raise TimeoutError()
                sep = chr(10) + 'DATA_JSON:' + chr(10)
                system_text, user_text = (message.split(sep, 1)
                    if message.startswith('SYSTEM ROLE:') and sep in message
                    else (None, message))
                formatting = model_response_format(entry.provider, entry.model, role)
                http_status = None
                self.record("model_call_started", role=role, family=family,
                            provider=entry.provider, model=entry.model,
                            prompt_hash=hashlib.sha256(message.encode()).hexdigest())
                def infer():
                    if os.getenv("COUNCIL_HARD_DEADLINE", "0") == "1":
                        if entry.provider == "LLM7.io":
                            # Persist pacing between isolated child processes.
                            with self._llm7_lock:
                                self.router._pace_llm7()
                        from council.process_infer import generate as generate_process
                        return generate_process(
                            entry.provider, entry.model, user_text,
                            system_text=system_text, timeout=timeout,
                            max_tokens=output_tokens, response_format=formatting,
                            gemini_checked=tuple(getattr(self.router,"_gemini_checked",())))
                    captured = []
                    token = RAW_CAPTURE.set(captured)
                    try:
                        opts = dict(probe=False, request_timeout=max(0.5, timeout - 0.5),
                                    max_output_tokens=output_tokens,
                                    response_format=formatting)
                        if system_text:
                            opts['system_text'] = system_text
                        answer = self.router.generate(candidate, user_text, **opts)
                        return answer, (captured[-1] if captured else None)
                    finally:
                        RAW_CAPTURE.reset(token)
                # One retry only after a *completed* provider request raises
                # CandidateError. An asyncio.wait_for thread cancellation may
                # leave a request running and must never trigger a duplicate.
                for attempt_no in range(2):
                    try:
                        text, raw = await asyncio.wait_for(
                            asyncio.to_thread(infer), timeout=timeout)
                        break
                    except CandidateError as exc:
                        transient=exc.kind in {"timeout", "server", "connection"}
                        truncated=(exc.kind=="output_truncated" and
                                   role in {"proposal","repair","extract",
                                            "review","revision","hunter",
                                            "judge","topic_judge","audit",
                                            "objection_judge","quote_repair"} and
                                   output_tokens < 16384)
                        if attempt_no or not (transient or truncated):
                            raise
                        # A timed-out endpoint has used its allowance; try
                        # other eligible endpoints before retrying this one.
                        if exc.kind=="timeout" and any(
                                self._eligible(other.provider) and
                                not self.cache.blocked(other.provider,other.model)
                                for other in ordered[position+1:]):
                            raise
                        delay=(4.0 if exc.kind=="server"
                               else 2.0 if exc.kind=="connection"
                               else 0.5 if exc.kind=="timeout" else 0.25)
                        remaining=budget.allowed(endpoint_timeout,
                                                 synthesis=synthesis)
                        if remaining <= delay+6:
                            raise
                        if truncated:
                            output_tokens=min(16384,output_tokens*2)
                        elif (exc.kind=="server" and role=="judge"
                              and entry.provider=="Gemini"
                              and formatting is not None):
                            # A schema-triggered 5xx is possible but NOT
                            # established; one plain-JSON retry distinguishes
                            # this from an endpoint outage without logging
                            # response bodies or confidential model content.
                            formatting=None
                        safe_status=None
                        try:
                            import re
                            matched=re.match(r"HTTP ([0-9]{3}):",str(exc))
                            safe_status=int(matched.group(1)) if matched else None
                        except (TypeError,ValueError):
                            pass
                        await asyncio.sleep(delay)
                        remaining=budget.allowed(endpoint_timeout,
                                                 synthesis=synthesis)
                        if remaining<=5:
                            raise
                        # Every retry consumes a separately reserved request.
                        estimated_tokens=max(1,len(message)//3)+output_tokens
                        retry_deadline=time.monotonic()+remaining
                        await self.quota.acquire(
                            entry.provider, entry.model,
                            deadline=retry_deadline,
                            expected_tokens=estimated_tokens)
                        timeout=min(retry_deadline-time.monotonic(),
                                    budget.allowed(endpoint_timeout,synthesis=synthesis))
                        if timeout <= 1:
                            raise TimeoutError()
                        self.record("model_call_retry",role=role,family=family,
                            provider=entry.provider,model=entry.model,
                            kind=exc.kind,http_status=safe_status,
                            attempt=2,delay_seconds=delay,
                            max_output_tokens=output_tokens,
                            formatting="none" if formatting is None else "structured")
                if not text.strip():
                    raise CandidateError("Empty response", kind="invalid_response")
                elapsed = time.monotonic() - t0
                if entry.provider == "Gemini":
                    self.router._gemini_checked.add(entry.model)
                self.cache.success(entry.provider, entry.model)
                self.quota.success(entry.provider, entry.model)
                self._streak.pop((entry.provider,entry.model),None)
                self._answered.add((entry.provider,entry.model))
                usage = (raw.get('body',{}).get('usage',{})
                         if raw and isinstance(raw.get('body'),dict) else {})
                if not usage and raw and isinstance(raw.get('body'),dict):
                    usage=raw['body'].get('usageMetadata',{})
                if not isinstance(usage,dict):usage={}
                used_input = usage.get('prompt_tokens',usage.get('promptTokenCount'))
                used_output = usage.get('completion_tokens',usage.get('candidatesTokenCount'))
                if raw:
                    self.quota.observe(entry.provider, entry.model,
                        raw.get('headers', {}), input_tokens=used_input,
                        output_tokens=used_output,
                        reserved_tokens=estimated_tokens if used_input is not None
                                          and used_output is not None else 0)
                if failures:
                    self.record("fallback_selected", role=role, family=family,
                                provider=entry.provider, model=entry.model,
                                skipped=list(failures))
                self.record("model_call_succeeded", role=role, family=family,
                            provider=entry.provider, model=entry.model,
                            seconds=elapsed, response=text,
                            prompt_hash=hashlib.sha256(message.encode()).hexdigest(),
                            status=raw.get('status') if raw else None,
                            headers=raw.get('headers') if raw else {},
                            raw_body=raw.get('body') if raw else None,
                            input_tokens=used_input,output_tokens=used_output,
                            reported_model=(raw.get('body',{}).get('model')
                                if raw and isinstance(raw.get('body'),dict) else None))
                return CallResult(family, entry.provider, entry.model, text, elapsed,
                                  hashlib.sha256(message.encode()).hexdigest(),
                                  raw.get('body') if raw else None,
                                  raw.get('headers') if raw else {},
                                  raw.get('status') if raw else None)
            except QuotaContention:
                kind = 'quota_contention'
            except (asyncio.TimeoutError, TimeoutError):
                kind = "timeout"
            except CandidateError as exc:
                kind = exc.kind
                retry_after = exc.retry_after
                import re
                status_match = re.match(r"HTTP ([0-9]{3}):", str(exc))
                http_status = int(status_match.group(1)) if status_match else None
            except (KeyError, TypeError, ValueError):
                kind = "invalid_response"
            # A request that timed out because THIS run gave it clearly less
            # than the endpoint's normal allowance (a short stage slice or an
            # expiring deadline) says nothing about the provider's health.
            # Label it separately: no cooldown, no failure streak, so a slow
            # but working model is not locked out by our own scheduling.
            granted=timeout if timeout is not None else allowed
            if (kind=="timeout" and http_status is None
                    and granted<0.8*endpoint_timeout):
                kind="budget_timeout"
            # No underlying exception content, URLs or keys are persisted.
            streak=None
            if kind in {"timeout","server","connection"}:
                key=(entry.provider,entry.model)
                streak=self._streak[key]=self._streak.get(key,0)+1
                self._answered.discard(key)
            self.cache.failure(entry.provider,entry.model,kind,
                retry_after=retry_after,
                min_seconds=FamilyTransport.repeat_cooldown(kind,streak))
            self.quota.failure(entry.provider, entry.model, kind=kind,
                               retry_after=retry_after)
            failures.append(entry.provider + "/" + entry.model + ": " + kind)
            self.record("model_call_failed", role=role, family=family,
                        provider=entry.provider, model=entry.model, kind=kind,
                        http_status=http_status,
                        retry_after=retry_after)
            if budget.allowed(endpoint_timeout, synthesis=synthesis) <= 1:
                break
        raise FamilyUnavailable(family, failures)
