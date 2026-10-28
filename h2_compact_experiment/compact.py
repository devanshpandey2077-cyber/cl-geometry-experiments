"""CLI for prepare-analysis -> fetch/verify/ack -> explicit PBS release."""
import argparse
import json
import shutil
from pathlib import Path
from h2.common import ROOT,read_json,require_pbs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["status","ready","run-id","pack","shared","ack","release","verify","space"])
    parser.add_argument("--name")
    parser.add_argument("--run-id")
    parser.add_argument("--sha256")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--file")
    a=parser.parse_args()
    from h2 import compact
    if a.action=="verify":
        if not a.file: parser.error("--file required")
        m=compact.verify_archive(a.file)
        print(json.dumps({"verified":True,"kind":m["kind"],"files":len(m["files"])})); return
    if not a.name: parser.error("--name required")
    import suite
    plan=suite.read_plan(a.name); suite.verify_plan(plan)
    if plan.get("storage",{}).get("mode")!="compact":
        raise ValueError("Expected a plan built with --storage-mode compact")
    if a.action in ("pack","shared","release"):
        require_pbs("cpuq")
        if a.action!="shared" and not a.run_id: parser.error("--run-id required")
        result=(compact.shared(plan) if a.action=="shared" else
                (compact.pack if a.action=="pack" else compact.release)(plan,a.run_id))
        print(json.dumps(result,indent=2)); return
    if a.action=="ack":
        if not all((a.run_id,a.sha256,a.manifest_sha256)):
            parser.error("ack requires --run-id, --sha256 and --manifest-sha256")
        compact.acknowledge(plan,a.run_id,a.sha256,a.manifest_sha256)
        print("ACKNOWLEDGED",a.run_id,"(originals have NOT been removed)"); return
    if a.action=="ready":
        print(json.dumps(compact.ready(plan))); return
    if a.action=="run-id":
        ledger=compact.records(plan)
        for row in plan["runs"]:
            state=ledger["runs"].get(row["id"],{})
            if suite.run_state(row)=="COMPLETE" and not state.get("released"):
                print(row["id"]);return
        for row in plan["runs"]:
            if suite.run_state(row)!="COMPLETE":print(row["id"]);return
        raise ValueError("All registered runs are complete and released")
    if a.action=="space":
        ledger=compact.records(plan)
        total=sum(r.get("checkpoint_bytes",0) for r in ledger["runs"].values())
        print(json.dumps({"free_gib":shutil.disk_usage(ROOT).free/2**30,
            "measured_packed_runs":len(ledger["runs"]),"measured_checkpoint_gib":total/2**30,
            "archive_gib":sum(r["archive"]["bytes"] for r in ledger["runs"].values())/2**30,
            "warning":"Actual recorded sizes; not a projection for unmeasured runs."},indent=2)); return
    ledger=compact.records(plan)
    for row in plan["runs"]:
        value=ledger["runs"].get(row["id"],{})
        state=("RELEASED (archived on laptop)" if value.get("released") else
               "ACKNOWLEDGED; submit release CPU job" if value.get("acknowledged") else
               "READY TO FETCH" if value else suite.run_state(row))
        print(state,row["id"])
    print("Server free GiB:",round(shutil.disk_usage(ROOT).free/2**30,2))


if __name__=="__main__": main()
