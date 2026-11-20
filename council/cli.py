"""Command line for the general-purpose multi-model LLM council.

No RFC index, proof assistant or symbolic verifier is part of the runtime.
--provider-report and --dry-run are strictly offline unless --probe is given.
"""
from __future__ import annotations
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import uuid

from council.budget import RunBudget
from council.events import EventStore
from council.families import FAMILIES
from council.transport import FamilyTransport
from council.workflow import CouncilRunner, DEFAULT_SOLVERS
from council.full import FullCouncilRunner, QuickCouncilRunner

def final_stage_reserve(mode:str, seconds:float)->float:
    """Reserve judge/auditor opportunities in a shared finite deadline."""
    if mode=="quick":
        return min(15.0,seconds*0.15)
    return min(120.0,seconds*0.40)

def build_runner(mode,transport,events,budget,cfg=None,*,metadata=None,
                 quorum_override=None,depth_override=None,task_override=None):
    metadata=metadata or {}
    old_roles=metadata.get("solver_families") or metadata.get("role_assignment",{}).get("solvers")
    solvers=tuple(old_roles or (cfg.solvers if cfg else DEFAULT_SOLVERS))
    judge=metadata.get("judge_family") or (
        cfg.judge if cfg else "gemini")
    auditor=metadata.get("auditor_family") if "auditor_family" in metadata else (
        cfg.auditor if cfg else "gemma")
    opt=cfg.optional if cfg else {}
    af=tuple(metadata.get("auditor_fallbacks") or opt.get("audit_fallbacks",()))
    jf=tuple(metadata.get("judge_fallbacks") or opt.get("judge_fallbacks",()))
    strategies=metadata.get("solver_strategies") or (
        cfg.solver_strategies if cfg else {})
    # A resumed run keeps its original policy so replayed prompts match.
    policy=dict(
        quorum_policy=metadata.get("quorum_policy") or quorum_override
            or opt.get("quorum_policy","degrade"),
        min_proposals=metadata.get("min_proposals") or opt.get("min_proposals",2),
        improvement_rounds=(metadata["improvement_rounds"]
            if "improvement_rounds" in metadata
            else opt.get("improvement_rounds",1)),
        review_depth=(metadata.get("review_depth") or depth_override
                      or opt.get("review_depth","fast")),
        task_type=(task_override or metadata.get("task_type")
                   or opt.get("task_type","auto")),
        hedge_seconds=float(opt.get("hedge_seconds",30)))
    if mode=="quick":
        return QuickCouncilRunner(transport,events,budget,
            solver=solvers[0],judge_family=judge,
            solver_strategies=strategies,
            hedge_seconds=policy["hedge_seconds"])
    if mode=="full":
        return FullCouncilRunner(transport,events,budget,
            solvers=solvers,judge_family=judge,auditor_family=auditor,
            auditor_fallbacks=af,judge_fallbacks=jf,
            solver_strategies=strategies,**policy)
    return CouncilRunner(transport,events,budget,
        solvers=solvers,judge_family=judge,auditor_family=auditor,
        auditor_fallbacks=af,judge_fallbacks=jf,
        solver_strategies=strategies,**policy)

def configure_transport(transport,cfg):
    if cfg:
        setting=cfg.optional.get("audit_timeout_seconds")
        if setting is not None:
            transport.role_timeouts["audit"]=float(setting)
        for provider,settings in cfg.rate_limits.items():
            transport.quota.configure_provider(
                provider,
                rate_per_minute=float(settings["requests_per_minute"]),
                burst=int(settings.get("burst",1)),
                daily_requests=settings.get("daily_requests"),
                daily_tokens=settings.get("daily_tokens"),
                tokens_per_minute=settings.get("tokens_per_minute"))

def _read_json(path:Path)->dict:
    try:
        data=json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data,dict) else {}
    except (OSError,ValueError,UnicodeError):
        return {}

def provider_availability(coverage,cfg,active,*,probe=False,
                          health_file:Path=Path(".router_health.json")):
    """Key configured vs. evidence of usable quota. Offline unless probe."""
    import os
    import time
    from datetime import datetime,timezone
    from council.providers import quota_availability,probe_families
    caps={p:s["daily_requests"] for p,s in (cfg.rate_limits if cfg else {}).items()
          if isinstance(s,dict) and s.get("daily_requests")}
    try:
        for p,s in json.loads(os.getenv("COUNCIL_LIMITS_JSON","{}")).items():
            if isinstance(s,dict) and "shared_daily_requests" in s:
                caps.setdefault(p,int(s["shared_daily_requests"]))
    except (ValueError,TypeError,AttributeError):
        pass
    probes=None
    if probe:
        live=FamilyTransport(health_file=health_file)
        configure_transport(live,cfg)
        probes=asyncio.run(probe_families(live,active))
    return quota_availability(
        coverage,health=_read_json(health_file),
        daily=_read_json(health_file.with_name(".council_daily_quotas.json")),
        daily_caps=caps,probes=probes,now=time.time(),
        today=datetime.now(timezone.utc).date().isoformat())

def output_result(result,args,run_id):
    # File mode always writes one stable human-readable result, including
    # limitations and audit failures; command-line --json remains unchanged.
    if getattr(args,"file_input_mode",False):
        from council.file_io import atomic_write_text
        from council.report import render_markdown
        report=render_markdown(result,run_id=run_id,
                               reasoning_chars=8000 if args.verbose else 3000)
        atomic_write_text(args.output_dir/"answer.txt",report)
    if args.json:
        print(json.dumps(dict(result,run_id=run_id),
                         ensure_ascii=False,indent=2))
        if getattr(args,"report_out",None):
            from council.report import render_markdown
            args.report_out.write_text(render_markdown(result,run_id=run_id),
                                       encoding="utf-8")
        return
    from council.report import render_markdown
    report=render_markdown(result,run_id=run_id,
                           reasoning_chars=8000 if args.verbose else 1500)
    if getattr(args,"report_out",None):
        args.report_out.parent.mkdir(parents=True,exist_ok=True)
        args.report_out.write_text(report,encoding="utf-8")
    print(report)
    print("\nRecorded in",args.run_dir/(run_id+".sqlite3"))
    if getattr(args,"file_input_mode",False):
        print("Overwritten text report:",args.output_dir/"answer.txt")

def main(argv=None)->int:
    parser=argparse.ArgumentParser(description="General-purpose multi-model LLM council with independent model review")
    parser.add_argument("question",nargs="*")
    parser.add_argument("--mode",choices=("quick","standard","full","research"),
                        help="quick: single model; standard/full: council; research: legacy alias")
    parser.add_argument("--deadline",type=float)
    parser.add_argument("--timeout",type=float)
    parser.add_argument("--config",type=Path,default=Path("council.yaml"))
    parser.add_argument("--requirements-file",type=Path)
    parser.add_argument("--input-dir",type=Path,
                        help="Read and concatenate exactly two .txt files in this folder; "
                             "also the default when no positional question is supplied")
    parser.add_argument("--output-dir",type=Path,default=Path("output"),
                        help="For file-input mode: overwrite answer.txt and "
                             "combined_input.txt in this folder (default: output)")
    parser.add_argument("--run-dir",type=Path,default=Path("runs"))
    parser.add_argument("--run-id")
    parser.add_argument("--replay",metavar="RUN_ID")
    parser.add_argument("--recompute",metavar="RUN_ID")
    parser.add_argument("--resume",metavar="RUN_ID")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--list-families",action="store_true")
    parser.add_argument("--provider-report",action="store_true")
    parser.add_argument("--json",action="store_true")
    parser.add_argument("--quorum-policy",choices=("degrade","stop"),
                        help="What to do when fewer than min_proposals "
                             "independent proposals are valid")
    parser.add_argument("--task-type",choices=("auto","answer","document"),
                        help="auto: answers for ordinary requests; document for 5+ numbered "
                             "sections; override with answer or document")
    parser.add_argument("--review-depth",choices=("fast","thorough"),
                        help="fast: proposals, parallel review, judge, audit; "
                             "thorough adds revisions and re-review rounds")
    parser.add_argument("--report-out",type=Path,
                        help="Also write the concise Markdown report here")
    parser.add_argument("--probe",action="store_true",
                        help="With --provider-report: make ONE tiny live call "
                             "per active family to test actual availability")
    parser.add_argument("--verbose",action="store_true")
    args=parser.parse_args(argv)
    if args.input_dir is not None and (args.question or args.requirements_file
                                        or args.replay or args.resume or args.recompute):
        parser.error("--input-dir cannot be combined with a positional question, "
                     "--requirements-file, --replay, --resume or --recompute")
    args.file_input_mode=False
    cfg=None
    if args.config.is_file():
        from council.config import load_config
        cfg=load_config(args.config)
    args.mode=args.mode or (cfg.mode if cfg else "full")
    if ((args.deadline is not None and args.deadline<=0)
            or (args.timeout is not None and args.timeout<=1)):
        parser.error("Deadline must be positive; per-call timeout greater than 1")
    if args.list_families:
        for name,f in FAMILIES.items():
            print(name+": "+", ".join(e.provider+"/"+e.model for e in f.endpoints))
        return 0
    if args.provider_report:
        from council.providers import provider_coverage
        active=set(cfg.solvers if cfg else DEFAULT_SOLVERS)
        if args.mode=="quick":
            active=set(list(active)[:1])
        else:
            active.add(cfg.judge if cfg else "gemini")
            auditor=cfg.auditor if cfg else "gemma"
            if auditor:
                active.add(auditor)
            active.update(cfg.optional.get("audit_fallbacks",()) if cfg else ())
            active.update(cfg.optional.get("judge_fallbacks",()) if cfg else ())
        coverage=provider_coverage(active_families=active)
        print(json.dumps(dict(coverage,availability=provider_availability(
            coverage,cfg,active,probe=args.probe)),indent=2))
        return 0
    if args.replay:
        with EventStore(args.run_dir,args.replay) as events:
            rows=[e for e in events.replay() if e["event"]=="run_finished"]
        if not rows:
            parser.error("No completed result for run "+args.replay)
        print(json.dumps(rows[-1]["payload"]["result"],
                         ensure_ascii=False,indent=2))
        return 0
    if args.recompute and args.resume:
        parser.error("Use at most one of --recompute or --resume")
    replay_rows=None
    metadata=None
    original=args.recompute or args.resume
    if original:
        with EventStore(args.run_dir,original) as old:
            replay_rows=old.replay()
        metadata=next((e["payload"] for e in replay_rows
                       if e["event"]=="run_started"),None)
        if metadata is None:
            parser.error("No start metadata in checkpoint")
        if metadata.get("pipeline")=="evidence":
            parser.error("This is an old verifier-based run. Use --replay "
                         "to inspect it; recomputation under the new "
                         "pipeline is not compatible.")
        args.mode=metadata.get("mode","full")
    question=" ".join(args.question).strip()
    if metadata:
        question=metadata["question"]
    elif args.requirements_file:
        from council.requirements import Requirements
        req=Requirements.from_file(args.requirements_file)
        question=req.objective
        if req.constraints or req.questions:
            question+=("\nConstraints:\n"+"\n".join(req.constraints)
                       if req.constraints else "")
            question+=("\nSpecific questions:\n"+"\n".join(req.questions)
                       if req.questions else "")
    if args.input_dir is not None or (not question and not original and
                                      not args.requirements_file):
        from council.file_io import InputFolderError, read_two_text_files
        root=args.input_dir if args.input_dir is not None else Path("input")
        try:
            question,_=read_two_text_files(root)
        except InputFolderError as exc:
            parser.error(str(exc))
        args.file_input_mode=True
    if not question:
        parser.error("Provide a prompt or place exactly two .txt files in input/")
    deadline=args.deadline or (cfg.deadline_seconds if cfg else 600)
    if args.dry_run:
        from council.document import extract_sections
        configured_type = args.task_type or (cfg.optional.get("task_type", "auto")
                                             if cfg else "auto")
        effective_type = ("answer" if args.mode == "quick" else
                          "document" if configured_type == "document" or
                          (configured_type == "auto" and bool(extract_sections(question)))
                          else "answer")
        solvers=(metadata.get("solver_families") if metadata else None) or (
            list(cfg.solvers) if cfg else list(DEFAULT_SOLVERS))
        print(json.dumps({"mode":args.mode,"question":question,
                          "solvers":solvers[:1] if args.mode=="quick" else solvers,
                          "judge":cfg.judge if cfg else "gemini",
                          "auditor":None if args.mode=="quick" else (
                              cfg.auditor if cfg else "gemma"),
                          "deadline":deadline,"api_calls":0,
                          "input_mode":"two_files" if args.file_input_mode else "direct",
                          "task_type":effective_type,
                          "output_file":str(args.output_dir/"answer.txt")
                                        if args.file_input_mode else None},indent=2))
        return 0
    run_id=args.run_id or (original+(".resumed" if args.resume else ".recomputed")
                              if original else uuid.uuid4().hex[:16])
    if run_id==original:
        parser.error("Destination run ID must differ from source checkpoint")
    if args.file_input_mode:
        # Persist the exact concatenation before spending any API quota.
        from council.file_io import atomic_write_text
        try:
            atomic_write_text(args.output_dir/"combined_input.txt",question)
        except OSError as exc:
            parser.error("Cannot write combined input to output directory: "+str(exc))
    with EventStore(args.run_dir,run_id) as events:
        live=FamilyTransport(timeout=args.timeout or (
            cfg.per_call_timeout if cfg else 28.0),
            strict_timeout=args.timeout is not None,
            record=events.record)
        configure_transport(live,cfg)
        transport=live
        if replay_rows is not None:
            from council.replay import ReplayTransport,ResumeTransport
            transport=(ResumeTransport(replay_rows,live,record=events.record) if args.resume else
                       ReplayTransport(replay_rows))
        budget=RunBudget.start(deadline,reserve=final_stage_reserve(
            args.mode,deadline))
        runner=build_runner(args.mode,transport,events,budget,cfg,
                            metadata=metadata,
                            quorum_override=args.quorum_policy,
                            depth_override=args.review_depth,
                            task_override=args.task_type)
        events.record("run_configuration",
                      config=asdict(cfg) if cfg else None,
                      deadline_seconds=deadline,
                      per_call_timeout=getattr(live,"timeout",None))
        result=asyncio.run(runner.run(question,mode=args.mode))
    output_result(result,args,run_id)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
