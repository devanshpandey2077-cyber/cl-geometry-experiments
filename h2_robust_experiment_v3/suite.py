"""H2 v3: build -> CPU prepare -> GPU calibrate -> CPU freeze -> submit -> analyze.

Planning/submission/status use only the standard library. Intensive commands require PBS.
"""
from __future__ import annotations
import argparse
import copy
from collections import defaultdict
import getpass
from itertools import combinations
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import subprocess
from h2.common import (ROOT,canonical_hash,digest_file,load_config,model_asset_path,
    model_slug,now,read_json,require_pbs,source_inventory,token_dir,data_dir,write_json)

MODELS = {
 "qwen1.7b": {"repo":"Qwen/Qwen3-1.7B","layers":[7,14,21],"hidden":2048,"billions":1.7,"floor":18},
 "qwen4b": {"repo":"Qwen/Qwen3-4B","layers":[9,18,27],"hidden":2560,"billions":4,"floor":24},
 "qwen8b": {"repo":"Qwen/Qwen3-8B","layers":[9,18,27],"hidden":4096,"billions":8,"floor":36},
 "qwen14b": {"repo":"Qwen/Qwen3-14B","layers":[10,20,30],"hidden":5120,"billions":14,"floor":72},
 "smollm1.7b": {"repo":"HuggingFaceTB/SmolLM2-1.7B","layers":[6,12,18],"hidden":2048,"billions":1.7,"floor":18},
}
SEEDS = [101,202,303,404,505,606,707,808,909,1010]
VARIANTS = {
 "reference":{},
 "lr_low":{"training":{"learning_rate":3e-5}},
 "lr_high":{"training":{"learning_rate":3e-4}},
 "adapter8":{"training":{"lora":{"rank":8,"alpha":16}}},
 "adapter32":{"training":{"lora":{"rank":32,"alpha":64}}},
 "no_dropout":{"training":{"lora":{"dropout":0.0}}},
 "dropout10":{"training":{"lora":{"dropout":0.1}}},
 "length256":{"training":{"max_length":256}},
 "length1024":{"training":{"max_length":1024}},
 "one_epoch":{"training":{"epochs":1}},
 "three_epochs":{"training":{"epochs":3}},
}

def merge(a,b):
    out=copy.deepcopy(a)
    for k,v in b.items(): out[k]=merge(out.get(k,{}),v) if isinstance(v,dict) else v
    return out

def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+",value): raise ValueError("Use letters, digits, underscores or hyphens")
    return value

def plan_path(name): return ROOT/"plans"/safe_name(name)/"plan.json"
def read_plan(name): return read_json(plan_path(name))

def shape_signature(cfg,mode):
    # Resource and learning calibration must cover the actual model/adapter/input shape.
    return canonical_hash({"model":cfg["model"],"mode":mode,
        "training":{k:cfg["training"][k] for k in
        ("max_length","microbatch","gradient_accumulation","learning_rate","lora","epochs")}})

def build(name,profile,models=None,like="primary"):
    safe_name(name)
    root=ROOT/"plans"/name
    if root.exists(): raise FileExistsError("Plan name already exists; use a new name")
    base=load_config(ROOT/"configs"/"full.json")
    family_profile=like if profile=="calibration" else profile
    defaults={"smoke":["qwen1.7b"],"core":["qwen4b"],
        "primary":["qwen4b","qwen8b"],"stress":["qwen4b"],
        "replication":["smollm1.7b"],"scale14b":["qwen14b"],"full_tuning":["qwen1.7b"]}
    chosen=models or defaults[family_profile]
    variants=list(VARIANTS) if family_profile=="stress" else ["reference"]
    samples=[20261031,20261107] if family_profile=="primary" else [20261031]
    seeds=SEEDS if family_profile in ("primary","replication") else SEEDS[:5]
    order_count=5 if family_profile in ("primary","replication") else 3
    if family_profile=="stress": seeds=SEEDS[:3]
    if profile in ("smoke","calibration"): seeds=[17]; order_count=1; samples=[20261031]
    mode="full" if family_profile=="full_tuning" else "lora"
    tasks=base["data"]["tasks"]
    orders={"O1":tasks,"O2":list(reversed(tasks)),"O3":[tasks[i] for i in (2,0,4,1,3)],
            "O4":[tasks[i] for i in (1,3,0,4,2)],"O5":[tasks[i] for i in (3,4,1,2,0)]}
    orders=dict(list(orders.items())[:order_count])
    cells=[]; runs=[]
    for model_key in chosen:
      model=MODELS[model_key]
      for sample in samples:
       for variant in variants:
        cfg=merge(base,VARIANTS[variant])
        cfg.pop("extends",None)
        cfg["purpose"]="functionality" if profile=="smoke" else (
            "calibration" if profile=="calibration" else "confirmation")
        cfg["model"]={"repo":model["repo"],"revision":None,"precision":"bfloat16",
                      "attention":"sdpa","minimum_gpu_gib":model["floor"]}
        if mode=="full": cfg["model"].update(precision="float32",minimum_gpu_gib=72)
        cfg["orders"],cfg["seeds"]=orders,seeds
        cfg["training"].update(microbatch=1,gradient_accumulation=16,eval_batch=1,
            checkpoint_every_updates=100,deterministic=True,eval_splits=["validation","test"])
        cfg["features"].update(layers=model["layers"],primary_layer=model["layers"][1],
            poolings=["content","answer","last_content"],probe_splits=["probe_a","probe_b"],
            auxiliary_endpoints_only=True)
        cfg["analysis"].update(bootstrap=2000,rank=16,min_gap=0.01,
            relative_cutoff_gap_floor=0.001,minimum_learning_tasks=2,
            learning_nll_improvement=0.01,minimum_accuracy_above_chance=0.05)
        sizes={"train":20000,"validation":1000,"probe_a":1000,"probe_b":1000,"test":2000}
        partition="confirmation"; cap=120000
        if family_profile=="core": sizes.update(train=10000)
        if profile=="calibration":
            sizes={"train":1500,"validation":200,"probe_a":200,"probe_b":200,"test":200}
            partition="development"; cap=15000
            cfg["training"].update(max_steps_per_task=100,eval_splits=["validation"])
            cfg["analysis"].update(bootstrap=100)
        if profile=="smoke":
            sizes={"train":100,"validation":20,"probe_a":40,"probe_b":40,"test":40}
            partition="development"; cap=5000
            cfg["training"].update(max_steps_per_task=5,max_length=256)
            cfg["analysis"].update(bootstrap=40,rank=8)
        cfg["data"].update(partition=partition,seed=sample,sizes=sizes,candidate_cap=cap,
            candidate_seed=41027,stable_split_seed=93271)
        cfg["data"]["name"]="v3_"+canonical_hash({k:v for k,v in cfg["data"].items() if k!="name"})[:16]
        cell_id=f"{model_key}_d{sample}_{variant}"
        cfg["name"]=cell_id
        relative=(root/"cells"/(cell_id+".json")).relative_to(ROOT).as_posix()
        cell={"id":cell_id,"model_key":model_key,"variant":variant,"sample_seed":sample,
              "config_path":relative,"config":cfg,"mode":mode,
              "shape_signature":shape_signature(cfg,mode)}
        cells.append(cell)
        jobs=[(m,"O1",seeds[0]) for m in ("frozen","head")]
        training=[(mode,o,s) for s in seeds for o in orders]
        random.Random(6271).shuffle(training)
        for m,o,s in jobs+training:
            rid=f"{cell_id}_{m}_{o}_s{s}"
            runs.append({"id":rid,"cell":cell_id,"mode":m,"order":o,"seed":s,
                "run_path":f"runs/{name}/{cell_id}/{m}_{o}_s{s}",
                "job_name":"h2"+canonical_hash([name,rid])[:12]})
    plan={"schema":3,"name":name,"profile":profile,"like":family_profile,"created_at":now(),
          "cells":cells,"runs":runs,"primary_alpha":0.05,"primary_families":len(cells),
          "model_notes":MODELS,"source_sha256":source_inventory(),
          "claim":"Endpoint representation-affinity rank inversion; no causal/semantic/conflict claim",
          "data_sampling_note":"Cohorts share source pools and can overlap; they are not independent datasets"}
    plan["identity"]=canonical_hash(plan)
    for cell in cells: write_json(ROOT/cell["config_path"],cell["config"])
    write_json(root/"plan.json",plan)
    print(json.dumps({"plan":name,"cells":len(cells),"gpu_runs":len(runs),
                      "trained_runs":sum(r["mode"] not in ("head","frozen") for r in runs)},indent=2))
    return plan

def verify_plan(plan):
    obj=dict(plan); identity=obj.pop("identity")
    if canonical_hash(obj)!=identity: raise ValueError("Plan content was edited")
    if plan["source_sha256"]!=source_inventory(): raise ValueError("Code differs from plan; build a new plan")
    for cell in plan["cells"]:
        if load_config(ROOT/cell["config_path"])!=cell["config"]:
            raise ValueError("Resolved cell config differs from plan")

def prepare(plan):
    require_pbs("cpuq"); verify_plan(plan)
    from h2.prepare import prepare_model,prepare_data,prepare_tokens
    from h2.checkpoints import exclusive_lock
    with exclusive_lock(ROOT/"assets"/"prepare.lock"):
      for cell in plan["cells"]:
        cfg=cell["config"]
        asset=prepare_model(cfg)
        if max(cfg["features"]["layers"])>=asset["num_hidden_layers"]:
            raise ValueError("Feature layer is outside this model's intermediate layers")
        data=prepare_data(cfg); prepare_tokens(cfg,asset,data)
    print("SUITE PREPARATION COMPLETE",flush=True)

def learning_audit(run):
    meta=read_json(run/"run.json"); cfg=meta["config"]
    baseline=read_json(run/"stages"/"stage_000"/"metrics.json")
    improvements=[]
    for i,task in enumerate(meta["identity"]["order"]):
        current=read_json(run/"stages"/f"stage_{i+1:03d}"/"metrics.json")
        previous=read_json(run/"stages"/f"stage_{i:03d}"/"metrics.json")
        before=previous[task]["validation"]; after=current[task]["validation"]
        from h2.tasks import TASKS
        chance=1/len(TASKS[task].labels)
        improvements.append({"task":task,"nll_gain":before["mean_target_nll"]-after["mean_target_nll"],
            "accuracy_gain":after["accuracy"]-before["accuracy"],"post_task_accuracy":after["accuracy"],
            "accuracy_above_chance":after["accuracy"]-chance,
            "nll_gain_from_sequence_baseline":baseline[task]["validation"]["mean_target_nll"]-after["mean_target_nll"]})
    floor=cfg["analysis"]["learning_nll_improvement"]
    competence=cfg["analysis"].get("minimum_accuracy_above_chance",.05)
    return {"tasks":improvements,"pass":sum(x["nll_gain"]>floor and x["accuracy_above_chance"]>=competence
            for x in improvements)>=cfg["analysis"]["minimum_learning_tasks"],
            "threshold_nll_gain":floor,"minimum_accuracy_above_chance":competence,
            "comparison":"immediately before versus after training each task"}

def cross_data_audit(plan):
    from h2.data import read_jsonl,normalize
    ownership={}; sets={}; unique={}
    for c in plan["cells"]: unique[c["config"]["data"]["name"]]=c["config"]
    for name,cfg in unique.items():
        sets[name]={s:set() for s in cfg["data"]["sizes"]}
        for split in cfg["data"]["sizes"]:
          for task in cfg["data"]["tasks"]:
           for row in read_jsonl(data_dir(cfg)/task/(split+".jsonl")):
            sets[name][split].add(row["id"])
            for text in row["texts"]:
                key=canonical_hash(normalize(text))
                if key in ownership and ownership[key]!=split:
                    raise ValueError("A repeated source field changes split role across data cohorts")
                ownership[key]=split
    overlap=[]
    for a,b in combinations(sorted(sets),2):
        overlap.append({"a":a,"b":b,"shared_rows_by_split":{
            s:len(sets[a][s]&sets[b][s]) for s in sets[a]}})
    return {"exact_field_cross_role_overlap":0,"cohort_overlap":overlap,
            "note":"Exact/grouped source protection is not a semantic or pretraining contamination audit"}

def freeze(plan,calibration_name=None):
    from h2.checkpoints import exclusive_lock
    with exclusive_lock(plan_path(plan["name"]).parent/"freeze.lock"):
        return _freeze(plan,calibration_name)

def _freeze(plan,calibration_name=None):
    require_pbs("cpuq"); verify_plan(plan)
    receipt=plan_path(plan["name"]).parent/"FROZEN.json"
    if receipt.exists(): raise FileExistsError("This plan already has a freeze receipt")
    calibration={}
    if plan["profile"] not in ("smoke","calibration"):
        if not calibration_name: raise ValueError("Supply --calibration-plan from completed matching GPU calibrations")
        cal=read_plan(calibration_name); verify_plan(cal)
        if cal["profile"]!="calibration": raise ValueError("Expected a calibration plan")
        for cell in cal["cells"]:
            rs=[r for r in cal["runs"] if r["cell"]==cell["id"] and r["mode"]==cell["mode"]]
            for row in rs:
                run=ROOT/row["run_path"]
                if (run/"COMPLETE.json").exists():
                    audit=learning_audit(run)
                    if audit["pass"]: calibration[cell["shape_signature"]]={"run":row["run_path"],"learning":audit,
                        "completion":read_json(run/"COMPLETE.json")}
        for cell in plan["cells"]:
            if cell["shape_signature"] not in calibration:
                raise ValueError("Missing successful model/shape/optimizer learning calibration: "+cell["id"])
        # Both fixed-encoder controls must also have passed on the actual hardware.
        for row in cal["runs"]:
            if row["mode"] in ("frozen","head"):
                run=ROOT/row["run_path"]
                if not (run/"COMPLETE.json").exists(): raise ValueError("Missing calibration control")
                validate_control(run)
    assets={}; tokens={}; data={}
    for cell in plan["cells"]:
        cfg=cell["config"]
        for path,registry in ((model_asset_path(cfg),assets),(token_dir(cfg)/"manifest.json",tokens),
                              (data_dir(cfg)/"manifest.json",data)):
            if not path.exists(): raise FileNotFoundError("Prepare every cell first: "+str(path))
            value=read_json(path)
            base=Path(value["local_path"]) if registry is assets else path.parent
            for item in value.get("files",[]):
                if digest_file(base/item["path"])!=item["sha256"]: raise ValueError("Prepared artifact changed")
            registry[path.relative_to(ROOT).as_posix()]=digest_file(path)
    record={"created_at":now(),"plan_identity":plan["identity"],"source_sha256":source_inventory(),
        "assets":assets,"tokens":tokens,"data":data,"calibration":calibration,
        "cross_data_audit":cross_data_audit(plan),"analysis_alpha_per_cell":plan["primary_alpha"]/len(plan["cells"]),
        "policy":"All registered cells/runs/comparisons; conditional approximate probe intervals; nulls retained"}
    write_json(receipt,record)
    print("PROTOCOL FROZEN",receipt,flush=True)

def validate_control(run):
    from h2.analyze import load_feature
    import numpy as np
    meta=read_json(run/"run.json"); cfg=meta["config"]
    if meta["identity"]["mode"] not in ("frozen","head"): raise ValueError("Expected fixed-encoder control")
    max_change=0.0
    for split in cfg["features"]["probe_splits"]:
     for layer in cfg["features"]["layers"]:
      for pooling in cfg["features"]["poolings"]:
       for task in cfg["data"]["tasks"]:
        base=load_feature(run/"stages"/"stage_000",split,pooling,layer,task)
        for i in range(1,len(cfg["data"]["tasks"])+1):
            if (cfg["features"].get("auxiliary_endpoints_only",False) and
                i<len(cfg["data"]["tasks"]) and
                (split!=cfg["analysis"]["probe"] or layer!=cfg["features"]["primary_layer"] or
                 pooling!=cfg["features"]["primary_pooling"])): continue
            new=load_feature(run/"stages"/f"stage_{i:03d}",split,pooling,layer,task)
            for key in ("ids","groups","labels"):
                if not np.array_equal(base[key],new[key]): raise ValueError("Control probe identity changed")
            max_change=max(max_change,float(np.max(np.abs(base["features"]-new["features"]))))
    if max_change>cfg["analysis"]["control_tolerance"]: raise ValueError("Frozen-encoder drift exceeds tolerance")
    if meta["identity"]["mode"]=="head":
        import torch
        first=torch.load(run/"stages"/"stage_000"/"checkpoint.pt",map_location="cpu",weights_only=True)["parameters"]
        last=torch.load(run/"stages"/f"stage_{len(cfg['data']['tasks']):03d}"/"checkpoint.pt",map_location="cpu",weights_only=True)["parameters"]
        if not any(not torch.equal(first[k],last[k]) for k in first):
            raise ValueError("Head control did not actually update its readout")
    return {"pass":True,"max_feature_change_all_views":max_change}

def verify_frozen(plan):
    verify_plan(plan)
    p=plan_path(plan["name"]).parent/"FROZEN.json"
    if not p.exists(): raise ValueError("Freeze the protocol through CPU PBS first")
    receipt=read_json(p)
    if receipt["plan_identity"]!=plan["identity"]: raise ValueError("Freeze/plan mismatch")
    for registry in ("assets","tokens","data"):
        for rel,digest in receipt[registry].items():
            if digest_file(ROOT/rel)!=digest: raise ValueError("Frozen manifest changed")
    return receipt

def run_state(row):
    p=ROOT/row["run_path"]
    if (p/"COMPLETE.json").exists(): return "COMPLETE"
    if (p/"FAILURE.json").exists(): return read_json(p/"FAILURE.json")["type"]
    if (p/"run.json").exists(): return "INCOMPLETE"
    return "NOT_STARTED"

def submit(plan,limit=8,resume=False,execute=False,after=None,walltime="24:00:00"):
    from h2.checkpoints import exclusive_lock
    with exclusive_lock(plan_path(plan["name"]).parent/"submission.lock"):
        return _submit(plan,limit,resume,execute,after,walltime)

def _submit(plan,limit=8,resume=False,execute=False,after=None,walltime="24:00:00"):
    verify_frozen(plan)
    if limit<1: raise ValueError("Submission limit must be positive")
    if not re.fullmatch(r"\d{1,3}:[0-5]\d:[0-5]\d",walltime): raise ValueError("Invalid walltime")
    if after and not re.fullmatch(r"\d+(?:\.[\w.-]+)?",after): raise ValueError("Invalid predecessor job")
    ledger_path=plan_path(plan["name"]).parent/"submissions.json"
    ledger=read_json(ledger_path) if ledger_path.exists() else {"runs":{}}
    active_lines=[]
    if execute:
        if not (ROOT/"settings.env").exists(): raise ValueError("Create settings.env before submission")
        result=subprocess.run(["qstat","-u",getpass.getuser()],text=True,capture_output=True,check=True)
        active_lines=[line for line in result.stdout.splitlines()
                      if re.match(r"\s*\d",line) and not {"C","F"}.intersection(line.split())]
        unrelated=[line for line in active_lines if "workq" in line.split() and
                   not any(r["job_name"] in line for r in plan["runs"])]
        if unrelated and not after:
            raise ValueError("Other workq jobs are active. Use --after with the intended predecessor: "+str(unrelated))
    previous=after; count=0
    for row in plan["runs"]:
        if run_state(row)=="COMPLETE": continue
        state=ledger["runs"].get(row["id"],{})
        # Deterministic PBS job names reconcile a crash between qsub acceptance and ledger write.
        match=next((line for line in active_lines if row["job_name"] in line),None)
        if match:
            state["job_id"]=match.split()[0]; ledger["runs"][row["id"]]=state
            previous=state["job_id"]
            continue
        if state.get("job_id") and any(line.split() and line.split()[0]==state["job_id"] for line in active_lines):
            previous=state["job_id"]; continue
        if state and not resume: raise ValueError("Previous submission exists; inspect status then use --resume")
        if count>=limit: break
        cell=next(c for c in plan["cells"] if c["id"]==row["cell"])
        cfg=cell["config_path"]
        variables={"H2_CONFIG":cfg,"H2_RUN_REL":row["run_path"],"H2_MODE":row["mode"],
                   "H2_ORDER":row["order"],"H2_SEED":row["seed"],"H2_RESUME":int(resume),
                   "H2_PLAN":plan["name"],"H2_RUN_ID":row["id"]}
        command=["qsub","-N",row["job_name"],"-l","walltime="+walltime,"-v",
                 ",".join(f"{k}={v}" for k,v in variables.items())]
        if previous: command += ["-W","depend=afterany:"+previous]
        command += ["pbs/suite_gpu.pbs"]
        print(shlex.join(command),flush=True)
        if execute:
            state={"intent_at":now(),"command":command,"previous":state}
            ledger["runs"][row["id"]]=state; write_json(ledger_path,ledger)
            result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
            if result.returncode: raise RuntimeError(result.stderr)
            job=result.stdout.strip()
            if not re.fullmatch(r"\d+(?:\.[\w.-]+)?",job):
                raise RuntimeError("Unexpected qsub reply; reconcile qstat before retrying: "+job)
            state.update(job_id=job,accepted_at=now()); write_json(ledger_path,ledger)
            previous=job
        count+=1
    if execute: write_json(ledger_path,ledger)
    print("SUBMITTED" if execute else "DRY RUN; no jobs submitted",count)

def run_one(plan,run_id,resume=False):
    verify_frozen(plan)
    row=next(r for r in plan["runs"] if r["id"]==run_id)
    cell=next(c for c in plan["cells"] if c["id"]==row["cell"])
    # Engine accepts a full relative output path only from this wrapper's registered plan.
    os.environ["H2_REGISTERED_RUN_REL"]=row["run_path"]
    import sys
    from h2 import engine
    sys.argv=["h2.engine","--config",cell["config_path"],"--mode",row["mode"],
              "--order",row["order"],"--seed",str(row["seed"]),"--campaign",plan["name"]]
    if resume: sys.argv.append("--resume")
    engine.main()

def budget(plan,calibration_name=None):
    training=0; features=0; ckpt=0
    cal=read_plan(calibration_name) if calibration_name else None
    measured={}
    if cal:
        for row in cal["runs"]:
            p=ROOT/row["run_path"]
            cell=next(c for c in cal["cells"] if c["id"]==row["cell"])
            if row["mode"]==cell["mode"] and (p/"COMPLETE.json").exists():
                seconds=updates=0
                for stage in sorted((p/"stages").glob("stage_*")):
                    t=read_json(stage/"training.json")
                    if t: seconds+=t.get("seconds",0); updates+=t.get("updates",0)
                if updates: measured[cell["shape_signature"]]=seconds/updates
    hours=0; unmeasured=set()
    for row in plan["runs"]:
        c=next(c for c in plan["cells"] if c["id"]==row["cell"]); cfg=c["config"]
        m=MODELS[c["model_key"]]; k=len(cfg["data"]["tasks"])
        n=cfg["data"]["sizes"]["train"]; t=cfg["training"]
        per_task=math.ceil(math.ceil(n/t["microbatch"])/t["gradient_accumulation"])*t["epochs"]
        if t["max_steps_per_task"] is not None: per_task=min(per_task,t["max_steps_per_task"])
        if row["mode"] not in ("head","frozen"):
            training+=per_task*k
            sig=c["shape_signature"]
            if sig in measured: hours+=per_task*k*measured[sig]/3600
            else: unmeasured.add(c["id"])
        probes=sum(cfg["data"]["sizes"][s] for s in cfg["features"]["probe_splits"])
        endpoint=2*k*probes*m["hidden"]*4*len(cfg["features"]["layers"])*len(cfg["features"]["poolings"])
        middle=(k-1)*k*cfg["data"]["sizes"][cfg["analysis"]["probe"]]*m["hidden"]*4
        features+=endpoint+middle
        if row["mode"]=="full": ckpt+=(k+1)*m["billions"]*1e9*4
    result={"gpu_runs":len(plan["runs"]),"main_optimizer_updates":training,
        "uncompressed_feature_GiB":features/2**30,"full_tuning_boundary_checkpoint_GiB":ckpt/2**30,
        "measured_training_hours_excluding_evaluation_probing_analysis":hours if measured else None,
        "cells_without_matching_measured_timing":sorted(unmeasured),
        "warning":"Storage excludes model caches, token packs, LoRA/progress checkpoints, plots and analysis. "
                  "No walltime or memory fit guarantee. Probe bootstrap can dominate CPU cost."}
    print(json.dumps(result,indent=2)); return result

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="cmd",required=True)
    b=sub.add_parser("build"); b.add_argument("--name",required=True)
    b.add_argument("--profile",required=True,choices=["smoke","calibration","core","primary","stress","replication","scale14b","full_tuning"])
    b.add_argument("--like",default="primary",choices=["core","primary","stress","replication","scale14b","full_tuning"])
    b.add_argument("--models",nargs="+",choices=list(MODELS))
    for cmd in ("prepare","freeze","submit","status","budget","run-one","analyze","sensitivity"):
        q=sub.add_parser(cmd); q.add_argument("--name",required=True)
        if cmd in ("freeze","budget"): q.add_argument("--calibration-plan")
        if cmd=="run-one": q.add_argument("--run-id",required=True); q.add_argument("--resume",action="store_true")
        if cmd=="submit":
            q.add_argument("--limit",type=int,default=8); q.add_argument("--resume",action="store_true")
            q.add_argument("--submit",action="store_true"); q.add_argument("--after"); q.add_argument("--walltime",default="24:00:00")
        if cmd in ("analyze","sensitivity"): q.add_argument("--cell",help="Optional registered cell ID to process; aggregate still requires all")
    a=p.parse_args()
    if a.cmd=="build": return build(a.name,a.profile,a.models,a.like)
    plan=read_plan(a.name)
    if a.cmd=="prepare": return prepare(plan)
    if a.cmd=="freeze": return freeze(plan,a.calibration_plan)
    if a.cmd=="submit": return submit(plan,a.limit,a.resume,a.submit,a.after,a.walltime)
    if a.cmd=="run-one": return run_one(plan,a.run_id,a.resume)
    if a.cmd=="budget": return budget(plan,a.calibration_plan)
    if a.cmd=="status":
        for row in plan["runs"]: print(f"{run_state(row):18s} {row['id']}")
    if a.cmd in ("analyze","sensitivity"):
        from h2.suite_analysis import analyze,sensitivity
        return (analyze if a.cmd=="analyze" else sensitivity)(plan,a.cell)

if __name__=="__main__":
    try: main()
    except Exception as e:
        # Preserve traceback in scheduler logs; pauses use a distinguishable exit status.
        if type(e).__name__=="RunPaused": print("RUN PAUSED:",e,flush=True); raise SystemExit(75)
        raise
