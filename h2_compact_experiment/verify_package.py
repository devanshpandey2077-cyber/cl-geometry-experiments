"""Verify the delivered source archive before building server-side plans. No heavy imports."""
import hashlib
import json
from pathlib import Path

def verify(root=None):
    root=Path(root) if root else Path(__file__).resolve().parent
    manifest=json.loads((root/"SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    bad=[]
    for item in manifest["files"]:
        path=(root/item["path"]).resolve()
        if not path.is_relative_to(root.resolve()): raise ValueError("Invalid manifest path")
        if not path.is_file():bad.append(item["path"]+" missing");continue
        h=hashlib.sha256()
        with path.open("rb") as stream:
            for part in iter(lambda:stream.read(1024*1024),b""):h.update(part)
        if h.hexdigest()!=item["sha256"]:bad.append(item["path"]+" changed")
    if bad:raise ValueError("Package integrity failed: "+str(bad))
    print("PACKAGE VERIFIED",len(manifest["files"]),"files")

if __name__=="__main__":verify()
