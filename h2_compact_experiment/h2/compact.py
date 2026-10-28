"""Verified analysis capsules and explicit archive/acknowledge/release lifecycle.

Raw representations are not sufficient-statistic compressed. All declared analyses
run BEFORE removal; their exact bootstrap draws and audits remain on the server.
"""
import json
import os
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from .common import ROOT, canonical_hash, digest_file, now, read_json, write_json
from .checkpoints import exclusive_lock
from .storage import GIB, guard

SCHEMA = 1


def relative_name(value):
    p = PurePosixPath(value)
    if not value or value=="." or p.is_absolute() or any(x in (".", "..") for x in p.parts) or "\\" in value or ":" in value:
        raise ValueError("Unsafe relative archive path: " + value)
    if p.as_posix() != value:
        raise ValueError("Noncanonical relative path")
    return p


def locate(plan, run_id):
    row = next((r for r in plan["runs"] if r["id"] == run_id), None)
    if row is None:
        raise ValueError("Run is not registered in this plan")
    relative_name(row["run_path"])
    run = ROOT / row["run_path"]
    if run.is_symlink() or not run.resolve().is_relative_to((ROOT/"runs").resolve()):
        raise ValueError("Run directory is outside this experiment's runs directory")
    return row, run


def records_path(plan):
    return ROOT/"plans"/plan["name"]/"compact_records.json"


def records(plan):
    p = records_path(plan)
    value = read_json(p) if p.exists() else {"plan_identity": plan["identity"], "runs": {}}
    if value["plan_identity"] != plan["identity"]:
        raise ValueError("Compact ledger belongs to a different plan")
    return value


def is_feature(rel):
    parts = PurePosixPath(rel).parts
    return len(parts) >= 4 and parts[0] == "stages" and parts[2] == "features"


def is_checkpoint(rel):
    parts = PurePosixPath(rel).parts
    return len(parts) == 3 and parts[0] == "stages" and parts[2] == "checkpoint.pt"


def inventory(run):
    result = {}
    for p in sorted(run.rglob("*")):
        if p.is_symlink():
            raise ValueError("Symlinks are not allowed inside run artifacts: " + str(p))
        if not p.is_file():
            continue
        rel = p.relative_to(run).as_posix()
        if rel in ("COMPACT.json", "RELEASED.json") or p.name.endswith(".lock"):
            continue
        result[rel] = {"sha256": digest_file(p), "bytes": p.stat().st_size}
    return result


def verify_files(base, entries):
    for rel, expected in entries.items():
        relative_name(rel)
        p = base/rel
        if p.is_symlink() or not p.resolve().is_relative_to(base.resolve()):
            raise ValueError("Artifact escapes its directory: " + rel)
        if not p.is_file() or p.stat().st_size != expected["bytes"] or digest_file(p) != expected["sha256"]:
            raise ValueError("Retained artifact missing or changed: " + str(p))


def load_capsule(run, plan=None, row=None):
    run = Path(run)
    capsule = read_json(run/"COMPACT.json")
    if plan is None:
        from suite import read_plan, verify_plan
        plan = read_plan(capsule["plan_name"])
        verify_plan(plan)
    row, expected_run = locate(plan, capsule["run_id"] if row is None else row["id"])
    if run.resolve() != expected_run.resolve():
        raise ValueError("Capsule run path mismatch")
    if capsule["schema"] != SCHEMA or capsule["plan_identity"] != plan["identity"] or capsule["run_id"] != row["id"]:
        raise ValueError("Capsule identity mismatch")
    receipt = records(plan)["runs"].get(row["id"], {})
    if digest_file(run/"COMPACT.json") != receipt.get("capsule_sha256"):
        raise ValueError("Capsule seal is missing or changed")
    verify_files(run, capsule["retained"])
    meta = read_json(run/"run.json")
    cell = next(c for c in plan["cells"] if c["id"] == row["cell"])
    if (meta["config"] != cell["config"] or meta["identity"]["mode"] != row["mode"]
            or meta["identity"]["seed"] != row["seed"] or meta["identity"]["order_name"] != row["order"]):
        raise ValueError("Run metadata does not match registered cell")
    if meta["source_sha256"] != plan["source_sha256"]:
        raise ValueError("Run source identity differs from the plan")
    analysis = capsule["analysis_rel"]
    relative_name(analysis)
    if analysis+"/bootstrap.npz" not in capsule["retained"] or analysis+"/report.json" not in capsule["retained"]:
        raise ValueError("Missing sealed primary analysis")
    report = read_json(run/analysis/"report.json")
    if report["plan"]["alpha"] != plan["primary_alpha"]/plan["primary_families"]:
        raise ValueError("Wrong registered per-cell alpha")
    if report["plan"]["resamples"] != cell["config"]["analysis"]["bootstrap"]:
        raise ValueError("Wrong bootstrap count")
    if capsule["policy"] != "all_declared_analyses_before_feature_removal":
        raise ValueError("Unsupported compaction policy")
    return capsule


def write_archive(destination, files, identity, reserve_gib):
    destination.parent.mkdir(parents=True, exist_ok=True)
    listing = {name: {"sha256": digest_file(path), "bytes": path.stat().st_size}
               for name, path in files.items()}
    manifest = {"schema": SCHEMA, **identity, "files": listing}
    guard(destination.parent, sum(x["bytes"] for x in listing.values()) + 64*2**20, reserve_gib)
    temporary = destination.with_suffix(".zip.partial")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as z:
        for name, path in sorted(files.items()):
            relative_name(name)
            z.write(path, name)
        z.writestr("ARCHIVE_MANIFEST.json", json.dumps(manifest, sort_keys=True))
    checked = verify_archive(temporary)
    if checked != manifest:
        raise ValueError("Archive verification returned a different manifest")
    os.replace(temporary, destination)
    return {"path": destination.relative_to(ROOT).as_posix(), "sha256": digest_file(destination),
            "bytes": destination.stat().st_size, "manifest_sha256": canonical_hash(manifest)}


def verify_archive(path):
    """Stream and hash every member, without extracting paths or requiring torch/numpy."""
    import hashlib
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive member")
        manifest = json.loads(z.read("ARCHIVE_MANIFEST.json"))
        if manifest["schema"] != SCHEMA or set(names) != set(manifest["files"]) | {"ARCHIVE_MANIFEST.json"}:
            raise ValueError("Archive inventory mismatch")
        for name, item in manifest["files"].items():
            relative_name(name)
            info = z.getinfo(name)
            if info.file_size != item["bytes"]:
                raise ValueError("Archive size mismatch: " + name)
            h = hashlib.sha256()
            count = 0
            with z.open(name) as stream:
                for block in iter(lambda: stream.read(1024*1024), b""):
                    h.update(block)
                    count += len(block)
            if count != item["bytes"] or h.hexdigest() != item["sha256"]:
                raise ValueError("Archive hash mismatch: " + name)
    return manifest


def shared(plan):
    from suite import verify_frozen
    from .common import model_asset_path, data_dir, token_dir
    verify_frozen(plan)
    with exclusive_lock(ROOT/"plans"/plan["name"]/"compact_shared.lock"):
        receipt_path = ROOT/"plans"/plan["name"]/"compact_shared.json"
        if receipt_path.exists():
            old = read_json(receipt_path)
            if old["plan_identity"] != plan["identity"]:
                raise ValueError("Shared snapshot identity mismatch")
            if old.get("acknowledged"):
                return old
            if digest_file(ROOT/old["archive"]["path"]) != old["archive"]["sha256"]:
                raise ValueError("Shared archive changed")
            return old
        files = {}
        def add(p):
            files[p.relative_to(ROOT).as_posix()] = p
        for rel in plan["source_sha256"]:
            add(ROOT/rel)
        directory = ROOT/"plans"/plan["name"]
        for name in ("plan.json", "FROZEN.json"):
            add(directory/name)
        for cell in plan["cells"]:
            cfg = cell["config"]
            add(ROOT/cell["config_path"])
            add(model_asset_path(cfg))
            for directory in (data_dir(cfg), token_dir(cfg)):
                add(directory/"manifest.json")
                for item in read_json(directory/"manifest.json")["files"]:
                    p=directory/item["path"]
                    if digest_file(p)!=item["sha256"]:
                        raise ValueError("Prepared input changed before shared archival: "+str(p))
                    add(p)
        lock = ROOT/"assets"/"dataset_revisions.json"
        if lock.exists():
            add(lock)
        archive = write_archive(ROOT/"archives"/plan["name"]/"shared.zip", files,
                                {"kind": "shared", "plan_identity": plan["identity"], "plan_name": plan["name"]},
                                plan["storage"]["reserve_gib"])
        result = {"plan_identity": plan["identity"], "archive": archive, "created_at": now()}
        write_json(receipt_path, result)
        return result


def pack(plan, run_id):
    from suite import verify_frozen, validate_control, learning_audit
    from .analyze import analyze_run
    from .suite_analysis import verify_run_artifacts, sensitivity_run
    verify_frozen(plan)
    if plan.get("storage", {}).get("mode") != "compact":
        raise ValueError("Build a new plan with --storage-mode compact before using this workflow")
    row, run = locate(plan, run_id)
    if row["mode"] not in ("lora", "head", "frozen"):
        raise ValueError("Compact workflow currently supports LoRA and its controls only")
    with exclusive_lock(ROOT/"plans"/plan["name"]/"compact.lock"):
        shared(plan)
        ledger = records(plan)
        existing = ledger["runs"].get(run_id)
        if existing:
            load_capsule(run, plan, row)
            if not existing.get("released"):
                a = existing["archive"]
                if digest_file(ROOT/a["path"]) != a["sha256"]:
                    raise ValueError("Ready archive is missing or changed")
            return existing
        if not (run/"COMPLETE.json").exists():
            raise ValueError("Training is incomplete; resume the same run before packing")
        verify_run_artifacts(run)
        # A crash during packing may leave an uncommitted generated capsule. The
        # unchanged original stage inventory above is the authority for retrying.
        (run/"COMPACT.json").unlink(missing_ok=True)
        cell = next(c for c in plan["cells"] if c["id"] == row["cell"])
        meta = read_json(run/"run.json")
        if meta["config"] != cell["config"] or meta["source_sha256"] != plan["source_sha256"]:
            raise ValueError("Run configuration/source differs from registered plan")
        control = validate_control(run) if row["mode"] in ("frozen", "head") else None
        learning = learning_audit(run) if row["mode"] == "lora" else None
        analysis = analyze_run(run, alpha=plan["primary_alpha"]/plan["primary_families"])
        write_json(run/"compact_sensitivity.json", sensitivity_run(run, row, cell))
        before = inventory(run)
        retained = {r: v for r, v in before.items() if not is_feature(r) and not is_checkpoint(r)}
        capsule = {"schema": SCHEMA, "plan_name": plan["name"], "plan_identity": plan["identity"],
                   "run_id": run_id, "created_at": now(),
                   "policy": "all_declared_analyses_before_feature_removal",
                   "analysis_rel": analysis.relative_to(run).as_posix(), "control": control,
                   "learning": learning, "retained": retained,
                   "removed_after_ack": {r: v for r, v in before.items() if is_feature(r) or is_checkpoint(r)},
                   "limitations": ["Raw features are not in this archive. Reanalysis may require regeneration.",
                                   "Boundary checkpoints omit optimizer state; they support inference and boundary resume.",
                                   "Checksums detect changes; they are not signatures or backups."]}
        write_json(run/"COMPACT.json", capsule)
        files = {r: run/r for r in before if not is_feature(r)}
        files["COMPACT.json"] = run/"COMPACT.json"
        archive = write_archive(ROOT/"archives"/plan["name"]/(run_id+".zip"), files,
                                {"kind": "run", "plan_identity": plan["identity"], "run_id": run_id},
                                plan["storage"]["reserve_gib"])
        record = {"capsule_sha256": digest_file(run/"COMPACT.json"), "archive": archive,
                  "created_at": now(), "feature_bytes": sum(v["bytes"] for r,v in before.items() if is_feature(r)),
                  "checkpoint_bytes": sum(v["bytes"] for r,v in before.items() if is_checkpoint(r))}
        ledger["runs"][run_id] = record
        write_json(records_path(plan), ledger)
        load_capsule(run, plan, row)
        return record


def ready(plan):
    result = []
    path = ROOT/"plans"/plan["name"]/"compact_shared.json"
    if path.exists():
        value = read_json(path)
        if not value.get("acknowledged"):
            result.append({"kind": "shared", "id": "shared", **value["archive"]})
    ledger = records(plan)
    for row in plan["runs"]:
        value = ledger["runs"].get(row["id"])
        if value and not value.get("acknowledged"):
            result.append({"kind": "run", "id": row["id"], **value["archive"]})
    return {"schema": SCHEMA, "plan_identity": plan["identity"], "archives": result}


def acknowledge(plan, identifier, sha256, manifest_sha256):
    # Lightweight receipt operation for the SSH login shell. The client MUST have fully
    # verified the archive. Hashing all originals and deleting them happens in a PBS CPU job.
    with exclusive_lock(ROOT/"plans"/plan["name"]/"compact.lock"):
        if identifier == "shared":
            path = ROOT/"plans"/plan["name"]/"compact_shared.json"
            value = read_json(path)
        else:
            locate(plan, identifier)
            path = records_path(plan)
            ledger = records(plan)
            value = ledger["runs"][identifier]
        a = value["archive"]
        if a["sha256"] != sha256 or a["manifest_sha256"] != manifest_sha256:
            raise ValueError("Acknowledgement does not match this archive")
        value["acknowledged"] = {"at": now(), "sha256": sha256, "manifest_sha256": manifest_sha256,
                                 "assertion": "client verified every archive member on persistent local storage"}
        write_json(path, value if identifier == "shared" else ledger)


def release(plan, run_id):
    from suite import verify_frozen
    from .suite_analysis import verify_run_artifacts
    verify_frozen(plan)
    row, run = locate(plan, run_id)
    with exclusive_lock(ROOT/"plans"/plan["name"]/"compact.lock"):
        ledger = records(plan)
        record = ledger["runs"].get(run_id, {})
        capsule = load_capsule(run, plan, row)
        shared_path = ROOT/"plans"/plan["name"]/"compact_shared.json"
        shared_receipt = read_json(shared_path)
        if not record.get("acknowledged") or not shared_receipt.get("acknowledged"):
            raise ValueError("Verified laptop acknowledgement is required for BOTH shared and run archives")
        if record.get("released"):
            return record
        archive = ROOT/record["archive"]["path"]
        if archive.exists() and digest_file(archive) != record["archive"]["sha256"]:
            raise ValueError("Server archive changed before release")
        if not archive.exists() and not record.get("release_started"):
            raise ValueError("Server archive is missing before release")
        # Recheck originals before first removal. After an interrupted removal, verify every
        # remaining original against the sealed inventory, then safely continue the same list.
        for rel, item in capsule["removed_after_ack"].items():
            relative_name(rel)
            p = run/rel
            if not (is_feature(rel) or is_checkpoint(rel)):
                raise ValueError("Release inventory contains a nonreleasable artifact")
            if p.is_symlink() or not p.resolve().is_relative_to(run.resolve()):
                raise ValueError("Release path escapes the run directory")
            if p.exists() and (p.stat().st_size != item["bytes"] or digest_file(p) != item["sha256"]):
                raise ValueError("Original artifact changed before release: " + rel)
        if not record.get("release_started"):
            verify_run_artifacts(run)
            record["release_started"] = now()
            write_json(records_path(plan), ledger)
        for rel in capsule["removed_after_ack"]:
            (run/rel).unlink(missing_ok=True)
        archive.unlink(missing_ok=True)
        shared_archive = ROOT/shared_receipt["archive"]["path"]
        if shared_archive.exists():
            if digest_file(shared_archive) != shared_receipt["archive"]["sha256"]:
                raise ValueError("Shared archive changed")
            shared_archive.unlink()
        record["released"] = now()
        write_json(records_path(plan), ledger)
        write_json(run/"RELEASED.json", {"at": record["released"], "plan_identity": plan["identity"],
                   "archive_sha256": record["archive"]["sha256"], "capsule_sha256": record["capsule_sha256"]})
        load_capsule(run, plan, row)
        return record


def can_submit(plan):
    if plan.get("storage", {}).get("mode") != "compact":
        return
    ledger = records(plan)
    for row in plan["runs"]:
        _, run = locate(plan, row["id"])
        if (run/"COMPLETE.json").exists() and not ledger["runs"].get(row["id"], {}).get("released"):
            raise ValueError("Pack, fetch, acknowledge and release completed run before another GPU run: " + row["id"])
