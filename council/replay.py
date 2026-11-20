"""Zero-API response replay and resumable role-call cache.

Only recorded responses are replayed. Missing calls fail visibly instead of
silently sending requests to providers; a separate ResumeTransport may fall through.
"""
from __future__ import annotations
from collections import defaultdict, deque
import hashlib
from council.transport import CallResult, FamilyUnavailable

def _key(family,role,prompt_hash):
    return (family,role,prompt_hash)

class ReplayTransport:
    def __init__(self,events:list[dict]):
        self.recorded=defaultdict(deque)
        self.unavailable=defaultdict(deque)
        starts=defaultdict(deque)
        for event in events:
            payload=event.get("payload",{})
            family=payload.get("family")
            role=payload.get("role")
            if event["event"]=="model_call_started":
                starts[(family,role)].append(payload.get("prompt_hash"))
            elif event["event"]=="model_call_succeeded":
                key=_key(family,role,payload.get("prompt_hash"))
                if key[-1] is None and starts[(family,role)]:
                    key=_key(family,role,starts[(family,role)].popleft())
                self.recorded[key].append(payload)
            elif event["event"]=="role_unavailable":
                self.unavailable[(family,role)].append(payload)

    async def complete(self,family,message,budget,*,role,synthesis=False):
        digest=hashlib.sha256(message.encode()).hexdigest()
        key=_key(family,role,digest)
        if self.recorded[key]:
            payload=self.recorded[key].popleft()
            return CallResult(family,payload.get("provider",family),
                        payload.get("model","recorded"),
                        payload["response"],float(payload.get("seconds",0)),
                        digest,payload.get("raw_body"),payload.get("headers"),
                        payload.get("status"))
        if self.unavailable[(family,role)]:
            payload=self.unavailable[(family,role)].popleft()
            raise FamilyUnavailable(family,payload.get("failures",["role failed in original run"]))
        raise FamilyUnavailable(family,["No recorded response for role and exact prompt"])

class ResumeTransport(ReplayTransport):
    def __init__(self,events,live,record=None):
        super().__init__(events)
        # Live resume retries old unavailable roles instead of replaying failure.
        self.unavailable.clear()
        self.live=live
        # Replayed answers are written into the NEW run too. Otherwise a
        # resumed run that is interrupted again would have lost them, and a
        # second resume would pay for those API calls once more.
        self.record=record or (lambda *a,**k:None)
        self.replayed=0

    def soonest_ready(self,family):
        probe=getattr(self.live,"soonest_ready",None)
        return probe(family) if callable(probe) else 0.0

    @property
    def role_timeouts(self):
        return getattr(self.live,"role_timeouts",{})

    async def complete(self,family,message,budget,*,role,synthesis=False):
        try:
            result=await super().complete(family,message,budget,role=role,synthesis=synthesis)
            self.replayed+=1
            self.record("model_call_started",role=role,family=family,
                        prompt_hash=result.prompt_hash,replayed=True)
            self.record("model_call_succeeded",role=role,family=family,
                        provider=result.provider,model=result.model,
                        prompt_hash=result.prompt_hash,response=result.text,
                        seconds=0.0,replayed=True)
            return result
        except FamilyUnavailable as exc:
            if not exc.failures or "No recorded response" not in exc.failures[0]:
                raise
            return await self.live.complete(family,message,budget,role=role,
                                            synthesis=synthesis)
