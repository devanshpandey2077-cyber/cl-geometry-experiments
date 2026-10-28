"""Reuse caches in a NEW experiment folder; never change the previous source or plan."""
import argparse
from pathlib import Path
import shutil
import json
from h2.common import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reuse",required=True,help="Existing server v3 experiment directory")
    a=p.parse_args()
    old=Path(a.reuse).expanduser().resolve()
    if old==ROOT or not (old/"h2"/"engine.py").is_file():
        p.error("Expected a different, existing experiment directory")
    if (ROOT/"settings.env").exists() or (ROOT/"assets").exists() or (ROOT/"data").exists():
        p.error("Run bootstrap once in a freshly extracted directory; existing paths were left unchanged")
    (ROOT/"assets").mkdir()
    if (old/"assets").exists():
        for path in (old/"assets").glob("*.json"):
            shutil.copy2(path,ROOT/"assets"/path.name)
    if (old/"data").is_dir():
        # Shares immutable prepared caches. New preparation writes NEW named cache directories.
        (ROOT/"data").symlink_to(old/"data",target_is_directory=True)
    else:
        (ROOT/"data").mkdir()
    settings=old/"settings.env"
    if settings.exists():
        shutil.copy2(settings,ROOT/"settings.env")
    else:
        shutil.copy2(ROOT/"settings.env.example",ROOT/"settings.env")
    print("Reused asset manifests and data cache without copying model weights.")
    print("Source, plans, runs and logs from the old experiment were not changed.")
    print("Inspect settings.env: use h2_h100, confirm the CURRENT authorized GPU selection,")
    print("and set H2_MAX_MINUTES for the actual PBS walltime. Then build a NEW plan.")


if __name__=="__main__": main()
