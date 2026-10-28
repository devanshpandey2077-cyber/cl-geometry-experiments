"""Laptop-only verified SSH/SCP downloader. Requires Python stdlib and OpenSSH.

No extraction, no credentials in files, no deletion. Acknowledgements are sent only
after SHA256 verification of the ZIP and every member on the local filesystem.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from h2.common import canonical_hash
from h2.compact import verify_archive


def file_hash(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def ssh(host,root,arguments,capture=True):
    command="cd "+shlex.quote(root)+" && "+shlex.join(arguments)
    result=subprocess.run(["ssh",host,command],text=True,capture_output=capture)
    if result.returncode:
        raise RuntimeError("SSH operation failed: "+(result.stderr or "inspect SSH output"))
    return result.stdout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host",required=True,help="Your existing SSH alias or user@host; no URL")
    p.add_argument("--remote-root",required=True)
    p.add_argument("--name",required=True)
    p.add_argument("--destination",required=True)
    p.add_argument("--remote-python",default="/Data4/YOUR_HPC_USER/.conda/envs/h2_h100/bin/python")
    a=p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*",a.host): p.error("Invalid SSH host/alias")
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+",a.remote_root): p.error("Use an absolute remote path without spaces")
    if not re.fullmatch(r"[A-Za-z0-9_-]+",a.name): p.error("Invalid plan name")
    if not shutil.which("ssh") or not shutil.which("scp"): p.error("Install/enable the Windows OpenSSH client")
    data=json.loads(ssh(a.host,a.remote_root,[a.remote_python,"compact.py","ready","--name",a.name]))
    destination=Path(a.destination).expanduser().resolve()/a.name
    destination.mkdir(parents=True,exist_ok=True)
    if not data["archives"]:
        print("No unacknowledged archives are ready. Check compact.py status on the server."); return
    for item in data["archives"]:
        rel=item["path"]
        if not re.fullmatch(r"archives/[A-Za-z0-9_./-]+\.zip",rel) or ".." in Path(rel).parts:
            raise ValueError("Unsafe remote archive path")
        path=destination/Path(rel).name
        temporary=path.with_suffix(".zip.partial")
        if path.exists() and file_hash(path)!=item["sha256"]:
            raise ValueError("An existing local archive has different contents; preserve it: "+str(path))
        if not path.exists():
            free=shutil.disk_usage(destination).free
            if free<item["bytes"]+2*2**30:
                raise RuntimeError("Insufficient laptop space; keep at least 2 GiB headroom")
            remote=a.host+":"+a.remote_root.rstrip("/")+"/"+rel
            subprocess.run(["scp",remote,str(temporary)],check=True)
            if file_hash(temporary)!=item["sha256"]:
                raise ValueError("Downloaded ZIP hash mismatch. Nothing acknowledged or deleted.")
            manifest=verify_archive(temporary)
            if canonical_hash(manifest)!=item["manifest_sha256"] or manifest["plan_identity"]!=data["plan_identity"]:
                raise ValueError("Downloaded archive identity mismatch")
            with temporary.open("r+b") as stream:
                stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary,path)
        # Revalidate existing files too, including a retry after SSH acknowledgement failed.
        manifest=verify_archive(path)
        if (canonical_hash(manifest)!=item["manifest_sha256"] or manifest["plan_identity"]!=data["plan_identity"]
                or manifest["kind"]!=item["kind"]
                or (item["kind"]=="run" and manifest["run_id"]!=item["id"])):
            raise ValueError("Archive binding mismatch; nothing acknowledged")
        receipt={"archive":path.name,"sha256":item["sha256"],"manifest_sha256":item["manifest_sha256"],
                 "plan_identity":data["plan_identity"],"bytes":path.stat().st_size}
        receipt_path=path.with_suffix(".verified.json")
        receipt_path.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
        output=ssh(a.host,a.remote_root,[a.remote_python,"compact.py","ack","--name",a.name,
                   "--run-id",item["id"],"--sha256",item["sha256"],
                   "--manifest-sha256",item["manifest_sha256"]])
        print("VERIFIED:",path,flush=True)
        print(output.strip(),flush=True)
    print("Done. No server originals were removed. Submit the separate release CPU job next.")


if __name__=="__main__": main()
