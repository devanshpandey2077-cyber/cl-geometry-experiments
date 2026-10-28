"""Restore verified shared inputs or retained statistical capsules from laptop ZIPs.

Run on the server through PBS cpuq after copying an archive back. Never overwrites
different files. Raw features were intentionally not archived. Checkpoints remain
inside the ZIP by default; --checkpoints restores them too for later regeneration.
"""
import argparse
import os
from pathlib import Path
import zipfile
from h2.common import ROOT,canonical_hash,digest_file,read_json,write_json,require_pbs,now
from h2.compact import verify_archive,relative_name,records,records_path,locate
from h2.checkpoints import exclusive_lock
from h2.storage import guard


def restore(path,checkpoints=False):
    manifest=verify_archive(path)
    import suite
    if manifest["kind"]=="shared":
        name=manifest["plan_name"]; suite.safe_name(name)
        base=ROOT
        selected=set(manifest["files"])
    elif manifest["kind"]=="run":
        # Recover the plan name and retained inventory from the verified capsule.
        with zipfile.ZipFile(path) as z:
            import json
            capsule=json.loads(z.read("COMPACT.json"))
        name=capsule["plan_name"]; suite.safe_name(name)
        plan=suite.read_plan(name)
        if manifest["plan_identity"]!=plan["identity"]:raise ValueError("Different plan")
        _,base=locate(plan,manifest["run_id"])
        selected=set(capsule["retained"])|{"COMPACT.json"}
        if checkpoints:
            selected|={rel for rel in manifest["files"] if rel.endswith("/checkpoint.pt")}
        if not selected.issubset(manifest["files"]):raise ValueError("Incomplete capsule archive")
    else:
        raise ValueError("Unknown archive kind")
    base.mkdir(parents=True,exist_ok=True)
    guard(base,sum(manifest["files"][r]["bytes"] for r in selected),1.5)
    with zipfile.ZipFile(path) as z:
        for rel in sorted(selected):
            relative_name(rel)
            p=base/rel
            if p.is_symlink() or not p.resolve().is_relative_to(base.resolve()):
                raise ValueError("Restore path escapes target")
            expected=manifest["files"][rel]
            if p.exists():
                if digest_file(p)!=expected["sha256"]:raise ValueError("Different existing file: "+str(p))
                continue
            p.parent.mkdir(parents=True,exist_ok=True)
            temp=p.with_name(p.name+".restoring")
            with z.open(rel) as src,temp.open("wb") as dst:
                import shutil
                shutil.copyfileobj(src,dst)
                dst.flush();os.fsync(dst.fileno())
            if digest_file(temp)!=expected["sha256"]:raise ValueError("Restore verification failed")
            os.replace(temp,p)
    if manifest["kind"]=="run":
        plan=suite.read_plan(name);suite.verify_frozen(plan)
        with exclusive_lock(ROOT/"plans"/name/"compact.lock"):
            ledger=records(plan)
            record={"capsule_sha256":digest_file(base/"COMPACT.json"),
                "archive":{"path":str(Path(path).resolve()),"sha256":digest_file(path),
                           "manifest_sha256":canonical_hash(manifest),"bytes":Path(path).stat().st_size},
                "acknowledged":{"assertion":"restored from fully verified archive"},
                "released":now(),"restored":True}
            old=ledger["runs"].get(manifest["run_id"])
            if old and old["capsule_sha256"]!=record["capsule_sha256"]:
                raise ValueError("Different existing capsule ledger entry")
            if not old:
                ledger["runs"][manifest["run_id"]]=record;write_json(records_path(plan),ledger)
        from h2.compact import load_capsule
        load_capsule(base,plan)
    return {"kind":manifest["kind"],"restored_files":len(selected),"destination":str(base)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file",required=True);p.add_argument("--checkpoints",action="store_true")
    a=p.parse_args();require_pbs("cpuq")
    print(restore(a.file,a.checkpoints))


if __name__=="__main__":main()
