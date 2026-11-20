"""Non-network diagnostics: eight providers, model-family independence and free confirmation.

Configured keys do not certify live connectivity, free billing or remaining quota.
"""
from __future__ import annotations
import json
import os
from council.families import FAMILIES
from model_router import KEY_NAMES

CONFIRM = {"Cohere":"COHERE_TRIAL_CONFIRMED",
           "Ollama Cloud":"OLLAMA_FREE_CONFIRMED"}

def provider_coverage(*, env=None, active_families=()) -> dict:
    variables = env if env is not None else os.environ
    active = set(active_families)
    if not active <= set(FAMILIES):
        raise ValueError("Unknown active model families")
    providers = {}
    for family_name, family in FAMILIES.items():
        for endpoint in family.endpoints:
            provider = endpoint.provider
            if provider not in KEY_NAMES or provider == "NVIDIA NIM":
                continue
            entry = providers.setdefault(provider, {
                "status":"not_configured", "families":[],
                "in_run":False, "free_confirmed":False, "live_checked":False,
                "eligible_for_attempt":False})
            if family_name not in entry["families"]:
                entry["families"].append(family_name)
            entry["in_run"] |= family_name in active
            key_ok = bool(str(variables.get(KEY_NAMES[provider],"")).strip())
            if provider == "Cloudflare":
                key_ok &= bool(str(variables.get("CLOUDFLARE_ACCOUNT_ID","")).strip())
            flag = CONFIRM.get(provider)
            confirmed = flag is None or str(variables.get(flag,"")) == "1"
            entry["free_confirmed"] = bool(confirmed)
            entry["eligible_for_attempt"] |= bool(key_ok and confirmed)
            if key_ok:
                entry["status"] = ("configured_unverified" if confirmed
                                   else "free_confirmation_required")
    active_eligible=sorted(
        family for family in active
        if any(providers.get(ep.provider,{}).get("eligible_for_attempt",False)
               for ep in FAMILIES[family].endpoints))
    return {
        "providers":dict(sorted(providers.items())),
        "active_attempt_eligible_families":active_eligible,
        "active_not_ready_families":sorted(active - set(active_eligible)),
        "provider_count":len(providers),
        "family_count":len(FAMILIES),
        "active_family_count":len(active),
        "quorum_eligible_families":sorted(
            k for k,v in FAMILIES.items() if v.counts_for_quorum),
        "limitations":[
            "Eligible for attempt means an API key and required free-mode "
            "confirmation exist; it does NOT confirm provider uptime or quota.",
            "No live request was made; actual free eligibility and quotas remain unverified.",
            "A model lineage served by two providers counts as one family.",
            "Role reservations prevent calling all endpoints in every run.",
        ]
    }


# ----------------------------------------------------------------------
# Key configured vs. quota actually available (#9)
# ----------------------------------------------------------------------
LASTING = ("billing_required", "auth", "model_unavailable", "model_not_free_eligible")
AVAILABILITY_ORDER = ("probe_ok", "recently_succeeded", "daily_cap_reached",
                      "blocked_billing_or_auth", "cooling_down", "probe_failed",
                      "used_today_headroom_unknown", "key_only_never_observed",
                      "free_confirmation_required", "no_key")


def _until(value) -> float:
    if isinstance(value, dict):
        value = value.get("until", value.get("cooldown_until", 0))
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def quota_availability(coverage: dict, *, health: dict | None = None,
                       daily: dict | None = None, daily_caps: dict | None = None,
                       probes: dict | None = None, now: float, today: str) -> dict:
    """Classify each provider by EVIDENCE of usable quota, not by key presence.

    health:  the router health-cache data (successes and cooldowns).
    daily:   the shared daily-usage file ({"providers": {p: {day, requests}}}).
    daily_caps: user-configured daily request caps per provider.
    probes:  optional live probe outcomes {provider: {"ok": bool, "kind": str}}.
    Statuses (strongest evidence first): probe_ok, recently_succeeded,
    daily_cap_reached, blocked_billing_or_auth, cooling_down, probe_failed,
    used_today_headroom_unknown, key_only_never_observed,
    free_confirmation_required, no_key.
    """
    health = health or {}
    models = health.get("models") or {}
    blocks = health.get("providers") or {}
    usage = (daily or {}).get("providers") or {}
    caps = daily_caps or {}
    probes = probes or {}
    out = {}
    for provider, entry in coverage["providers"].items():
        evidence = []
        status = None
        if entry["status"] == "not_configured":
            status = "no_key"
        elif entry["status"] == "free_confirmation_required":
            status = "free_confirmation_required"
        probe = probes.get(provider)
        if status is None and probe is not None:
            status = "probe_ok" if probe.get("ok") else "probe_failed"
            evidence.append("live probe: " + ("succeeded" if probe.get("ok")
                                              else str(probe.get("kind") or "failed")))
        used = usage.get(provider)
        used_today = (used.get("requests", 0) if isinstance(used, dict)
                      and used.get("day") == today else 0)
        cap = caps.get(provider)
        if used_today:
            evidence.append(str(used_today) + " request(s) today"
                            + (" of " + str(cap) if cap else ""))
        block = blocks.get(provider)
        provider_block = (block.get("reason"), _until(block) - now) if (
            isinstance(block, dict) and _until(block) > now) else None
        healthy, cooling = [], []
        for key, value in models.items():
            try:
                p, model = json.loads(key)
            except (ValueError, TypeError):
                continue
            if p != provider or not isinstance(value, dict):
                continue
            if _until(value.get("healthy_until", 0)) > now:
                healthy.append(model)
            if _until(value.get("cooldown_until", 0)) > now:
                cooling.append((model, value.get("reason"),
                                round(_until(value["cooldown_until"]) - now)))
        if status is None:
            if cap and used_today >= cap:
                status = "daily_cap_reached"
            elif provider_block and any(k in str(provider_block[0]) for k in LASTING):
                status = "blocked_billing_or_auth"
                evidence.append("provider blocked: " + str(provider_block[0]))
            elif healthy:
                status = "recently_succeeded"
            elif provider_block or cooling:
                status = "cooling_down"
            elif used_today:
                status = "used_today_headroom_unknown"
            else:
                status = "key_only_never_observed"
        if healthy:
            evidence.append("recent success: " + ", ".join(sorted(healthy)))
        for model, reason, seconds in cooling:
            evidence.append(model + " cooling down (" + str(reason) + ", "
                            + str(seconds) + "s left)")
        if provider_block and status != "blocked_billing_or_auth":
            evidence.append("provider cooldown: " + str(provider_block[0]))
        out[provider] = {"availability": status, "families": entry["families"],
                         "in_run": entry["in_run"], "evidence": evidence}
    usable = {"probe_ok", "recently_succeeded", "used_today_headroom_unknown"}
    unknown = {"key_only_never_observed"}
    return {
        "providers": out,
        "usable_now": sorted(p for p, v in out.items() if v["availability"] in usable),
        "unknown_quota": sorted(p for p, v in out.items() if v["availability"] in unknown),
        "not_usable_now": sorted(p for p, v in out.items()
                                 if v["availability"] not in usable | unknown),
        "notes": [
            "A configured key alone is 'key_only_never_observed': it says nothing "
            "about remaining free quota.",
            "'recently_succeeded' means a real call succeeded within the health "
            "cache window; quota can still run out on the next call.",
            "Use --probe for one tiny live call per active family.",
        ],
    }


async def probe_families(transport, families, *, seconds: float = 45.0) -> dict:
    """One minimal live call per family; returns per-provider outcomes."""
    from council.budget import RunBudget
    from council.transport import FamilyUnavailable
    outcomes = {}
    for family in sorted(families):
        budget = RunBudget.start(seconds, reserve=0)
        try:
            result = await transport.complete(
                family, "Reply with the single word OK.", budget,
                role="probe", synthesis=True)
            outcomes.setdefault(result.provider, {"ok": True, "family": family})
        except FamilyUnavailable as exc:
            for failure in exc.failures:
                provider = failure.split("/", 1)[0].split(":", 1)[0].strip()
                kind = (failure.split(": ", 1)[1] if ": " in failure
                        else failure).strip()[:120]
                outcomes.setdefault(provider, {"ok": False, "kind": kind,
                                               "family": family})
    return outcomes
