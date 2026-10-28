"""Predeclared suite-wide multiplicity and descriptive replication checks."""
from collections import defaultdict
from contextlib import ExitStack
from itertools import combinations
import math
from pathlib import Path
import numpy as np
from .common import ROOT,digest_file,now,read_json,require_pbs,write_json
from .analyze import analyze_run,load_feature,write_csv
from .geometry import (simultaneous_intervals,inversion_flags,pair_vector,differences,
                       affinity_matrix,covariance_affinity_matrix)

def replication_counts(records,seed_count,order_names,fraction=.8):
    """A declared DESCRIPTIVE stability criterion, not an extra p-value."""
    by=defaultdict(set)
    for r in records:
        if r["eligible_inversion"]:
            by[r["comparison"],r["initial_direction"],r["order"]].add(r["seed"])
    threshold=math.ceil(fraction*seed_count)
    passing=defaultdict(list)
    for (comparison,direction,order),seeds in by.items():
        if len(seeds)>=threshold: passing[comparison,direction].append(order)
    return [{"comparison":c,"initial_direction":d,"orders":sorted(orders),
             "minimum_seeds_per_order":threshold,"seed_count":seed_count,
             "criterion":"descriptive >=80% seeds in >=min(3,planned orders) with same orientation"}
            for (c,d),orders in passing.items() if len(orders)>=min(3,len(order_names))]

def verify_run_artifacts(run):
    meta=read_json(run/"run.json")
    for i in range(len(meta["config"]["data"]["tasks"])+1):
        stage=run/"stages"/f"stage_{i:03d}"
        record=read_json(stage/"COMPLETE.json")
        if not record.get("sha256"): raise ValueError("Missing v3 stage integrity manifest")
        for rel,digest in record["sha256"].items():
            if digest_file(stage/rel)!=digest: raise ValueError("Artifact changed: "+str(stage/rel))
    return meta

def analyze(plan,selected_cell=None):
    from .checkpoints import exclusive_lock
    out=ROOT/"reports"/plan["name"]
    selected=[c["id"] for c in plan["cells"] if selected_cell is None or c["id"]==selected_cell]
    with ExitStack() as stack:
        for cell in sorted(selected):stack.enter_context(exclusive_lock(out/(cell+".analysis.lock")))
        return _analyze(plan,selected_cell)

def _analyze(plan,selected_cell=None):
    require_pbs("cpuq")
    import suite
    suite.verify_frozen(plan)
    out=ROOT/"reports"/plan["name"]; out.mkdir(parents=True,exist_ok=True)
    ledger=[{"run":r["id"],"state":suite.run_state(r)} for r in plan["runs"]]
    write_csv(out/"execution_status.csv",ledger)
    cells=[c for c in plan["cells"] if selected_cell is None or c["id"]==selected_cell]
    if not cells: raise ValueError("Unknown registered cell")
    alpha=plan["primary_alpha"]/plan["primary_families"]
    for cell in cells:
        rows=[r for r in plan["runs"] if r["cell"]==cell["id"]]
        missing=[r["id"] for r in rows if suite.run_state(r)!="COMPLETE"]
        if missing:
            write_json(out/(cell["id"]+"_incomplete.json"),{"missing":missing,"conclusion":"INCOMPLETE; no confirmatory conclusion"})
            raise ValueError("Cell incomplete, including controls: "+str(missing))
        analyses=[]; controls=[]; learning={}
        for row in rows:
            run=ROOT/row["run_path"]
            if (run/"COMPACT.json").exists():
                from .compact import load_capsule
                capsule=load_capsule(run,plan,row)
                if row["mode"] in ("frozen","head"):
                    controls.append(capsule["control"])
                else: learning[row["id"]]=capsule["learning"]
                analyses.append(run/capsule["analysis_rel"])
                continue
            verify_run_artifacts(run)
            if row["mode"] in ("frozen","head"):
                controls.append(suite.validate_control(run))
            else: learning[row["id"]]=suite.learning_audit(run)
            analyses.append(analyze_run(run,alpha=alpha))
        points=[]; draws=[]; reports=[]
        for path in analyses:
            report=read_json(path/"report.json"); reports.append(report)
            with np.load(path/"bootstrap.npz",allow_pickle=False) as z:
                points.append(z["point"]); draws.append(z["draws"])
        if len({r["resampling_signature"] for r in reports})!=1:
            raise ValueError("Paired bootstrap samples differ within the cell")
        point=np.stack(points); boot=np.stack(draws,axis=1)
        lo,hi,intervals=simultaneous_intervals(point,boot,alpha)
        cfg=cell["config"]; detail=[]; run_rows=[]
        tasks=reports[0]["tasks"]; pairs=list(combinations(tasks,2))
        comparisons=list(combinations(pairs,2))
        for i,(row,report) in enumerate(zip(rows,reports)):
            flags=inversion_flags(lo[i],hi[i],cfg["analysis"]["min_gap"])
            spectrum={}
            for endpoint in (0,-1):
                spectrum[endpoint]={t:report["subspace_diagnostics"][endpoint][t]["relative_cutoff_gap"]
                                    for t in tasks}
            eligible_count=0
            for k,(p,q) in enumerate(comparisons):
                involved=set(p+q)
                gaps=[spectrum[e][t] for e in (0,-1) for t in involved]
                stable=all(g is not None and g>=cfg["analysis"]["relative_cutoff_gap_floor"] for g in gaps)
                learned=learning.get(row["id"],{}).get("pass",True)
                eligible=bool(flags[k] and stable and learned and row["mode"] not in ("frozen","head"))
                eligible_count+=int(eligible)
                detail.append({"run":row["id"],"mode":row["mode"],"seed":row["seed"],"order":row["order"],
                    "comparison":k,"pair_p":"/".join(p),"pair_q":"/".join(q),
                    "initial_difference":float(point[i,0,k]),"final_difference":float(point[i,1,k]),
                    "initial_lower":float(lo[i,0,k]),"initial_upper":float(hi[i,0,k]),
                    "final_lower":float(lo[i,1,k]),"final_upper":float(hi[i,1,k]),
                    "initial_direction":int(np.sign(point[i,0,k])),"joint_interval_inversion":bool(flags[k]),
                    "cutoff_spectrum_pass":stable,"learning_pass":learned,"eligible_inversion":eligible})
            if row["mode"] in ("frozen","head") and flags.any():
                raise ValueError("A fixed-encoder control has an interval-supported inversion")
            run_rows.append({"run":row["id"],"mode":row["mode"],"order":row["order"],"seed":row["seed"],
                "raw_inversions":report["raw_inversions"],"joint_interval_inversions":int(flags.sum()),
                "eligible_inversions":eligible_count,"final_mean_accuracy":report["final_mean_accuracy"],
                "backward_transfer":report["backward_transfer"],
                "learning_pass":learning.get(row["id"],{}).get("pass")})
        trained=[r for r in detail if r["mode"] not in ("frozen","head")]
        repeated=replication_counts(trained,len(cfg["seeds"]),cfg["orders"])
        # Resample seed blocks across all FIXED orders, rather than treating pairs as replications.
        means=np.asarray([np.mean([r["eligible_inversions"] for r in run_rows
                         if r["seed"]==s and r["mode"] not in ("head","frozen")]) for s in cfg["seeds"]])
        rng=np.random.default_rng(5191)
        seed_means=rng.choice(means,size=(10000,len(means)),replace=True).mean(axis=1)
        directory=out/cell["id"]; directory.mkdir(exist_ok=True)
        write_csv(directory/"runs.csv",run_rows); write_csv(directory/"joint_pair_comparisons.csv",detail)
        np.savez_compressed(directory/"joint_intervals.npz",point=point,lower=lo,upper=hi)
        result={"cell":cell["id"],"created_at":now(),"complete":True,"plan_identity":plan["identity"],
            "evidential_role":cfg["purpose"],
            "confirmatory_eligible":cfg["purpose"]=="confirmation" and cfg["analysis"]["bootstrap"]>=1000,
            "alpha_per_cell":alpha,"multiplicity":"Bonferroni across ALL registered cells; joint max-standardized bootstrap across runs/endpoints/comparisons within each cell",
            "conditional_probe_intervals":intervals,"controls":controls,"learning":learning,
            "repeated_comparisons":repeated,"eligible_count":sum(r["eligible_inversions"] for r in run_rows),
            "seed_block_descriptive_mean_count_95_interval":np.quantile(seed_means,[.025,.975]).tolist(),
            "conclusion":("REPEATED MEASURED REVERSALS" if repeated else
                ("MEASURED REVERSALS WITHOUT DECLARED REPLICATION" if any(r["eligible_inversions"] for r in run_rows)
                 else "NO ELIGIBLE REVERSALS AT DECLARED RESOLUTION")),
            "limits":["Approximate intervals, not guaranteed finite-sample error control.",
                      "Probe uncertainty is conditional on fitted checkpoints; seed stability is descriptive.",
                      "No proof of universal H2, signed conflict, semantic similarity, H3/H4, or novelty."]}
        write_json(directory/"cell_report.json",result)
    summarize(plan,out)
    print("SUITE ANALYSIS COMPLETE",out,flush=True)

def summarize(plan,out):
    from .checkpoints import exclusive_lock
    with exclusive_lock(out/"summary.lock",blocking=True):
        return _summarize(plan,out)

def _summarize(plan,out):
    results=[]; pending=[]
    for c in plan["cells"]:
        p=out/c["id"]/"cell_report.json"
        if p.exists(): results.append(read_json(p))
        else: pending.append(c["id"])
    intersection={}
    for model in sorted({c["model_key"] for c in plan["cells"]}):
        selected=[r for r in results if next(c for c in plan["cells"] if c["id"]==r["cell"])["model_key"]==model]
        sets=[{(x["comparison"],x["initial_direction"]) for x in r["repeated_comparisons"]} for r in selected]
        intersection[model]=[list(x) for x in sorted(set.intersection(*sets))] if sets else []
    record={"complete":not pending,"pending_cells":pending,"cells":results,
        "same_oriented_repeated_comparisons_across_observed_cells_by_model":intersection,
        "interpretation":"Incomplete suites are descriptive. Completed suites retain null cells and all controls. "
                         "Cohort overlap and approximate interval assumptions limit generalization."}
    write_json(out/"suite_report.json",record)
    lines=["# H2 v3 suite evidence","",f"Complete registered suite: {not pending}",
           f"Analyzed cells: {len(results)}/{len(plan['cells'])}","",
           "| Cell | Eligible reversals summed across runs | Repeated comparisons | Result |",
           "|---|---:|---:|---|"]
    for r in results: lines.append(f"| {r['cell']} | {r['eligible_count']} | {len(r['repeated_comparisons'])} | {r['conclusion']} |")
    lines += ["","Counts are dependent observations, not independent experiments or p-values.",
              "Smoke/calibration cells remain functionality/development evidence regardless of their numerical outcomes.",
              "An absence of eligible reversals is a result at this protocol's resolution, not a universal disproof.",
              "Read joint_pair_comparisons.csv, the learning audits, cutoff spectra, controls and all failed-run records.",
              "Secondary sensitivity output is descriptive and does not replace the registered primary analysis."]
    (out/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

def sensitivity_run(run,row,cell):
    """Same descriptive views as v3, calculated before representations are released."""
    cfg=cell["config"]; primary=cfg["features"]["primary_layer"]
    views=[("rank8","probe_b","content",primary,8,"subspace"),
           ("rank32","probe_b","content",primary,32,"subspace"),
           ("early","probe_b","content",cfg["features"]["layers"][0],16,"subspace"),
           ("late","probe_b","content",cfg["features"]["layers"][-1],16,"subspace"),
           ("answer","probe_b","answer",primary,16,"subspace"),
           ("last_content","probe_b","last_content",primary,16,"subspace"),
           ("probe_a","probe_a","content",primary,16,"subspace"),
           ("covariance","probe_b","content",primary,16,"covariance")]
    result=[]
    for name,probe,pooling,layer,rank,estimator in views:
        points=[]
        for s in (0,len(cfg["data"]["tasks"])):
            f={t:load_feature(run/"stages"/f"stage_{s:03d}",probe,pooling,layer,t)["features"] for t in cfg["data"]["tasks"]}
            _,a,_=affinity_matrix(f,rank) if estimator=="subspace" else covariance_affinity_matrix(f)
            _,v=pair_vector(a); _,d=differences(v); points.append(d)
        for k in range(len(points[0])):
            result.append({"cell":cell["id"],"run":row["id"],"view":name,"comparison":k,
                "initial_difference":float(points[0][k]),"final_difference":float(points[1][k]),
                "raw_inversion":bool(points[0][k]*points[1][k]<0),
                "scope":"DESCRIPTIVE; no confidence or confirmatory selection claim"})
    return result


def sensitivity(plan,selected_cell=None):
    require_pbs("cpuq")
    import suite
    suite.verify_frozen(plan)
    out=ROOT/"reports"/plan["name"]; out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for cell in plan["cells"]:
        if selected_cell and cell["id"]!=selected_cell: continue
        for row in plan["runs"]:
            if row["cell"]!=cell["id"]: continue
            run=ROOT/row["run_path"]
            if suite.run_state(row)!="COMPLETE": raise ValueError("Missing run in sensitivity analysis")
            if (run/"COMPACT.json").exists():
                from .compact import load_capsule
                load_capsule(run,plan,row)
                rows.extend(read_json(run/"compact_sensitivity.json"))
            else:
                rows.extend(sensitivity_run(run,row,cell))
    write_csv(out/("sensitivity_"+(selected_cell or "all")+".csv"),rows)
