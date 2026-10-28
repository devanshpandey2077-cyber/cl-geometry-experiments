"""Storage lifecycle tests on CPU fixtures; these are not H100 fit/learning evidence."""
import copy
from contextlib import ExitStack,redirect_stdout
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import zipfile
import sys
import numpy as np
import suite
import test_core
from h2 import compact,suite_analysis
from h2.common import read_json,write_json,digest_file,canonical_hash
from h2.storage import guard


class CompactTests(unittest.TestCase):
    def setUp(self):
        self.f=test_core.PipelineTests(); self.f.setUp()
        self.root=self.f.root
        self.stack=ExitStack()
        for module in (suite,compact,suite_analysis):
            self.stack.enter_context(patch.object(module,"ROOT",self.root))
        self.stack.enter_context(patch.object(suite,"verify_frozen",return_value={}))
        self.stack.enter_context(patch.object(suite,"verify_plan",return_value=None))
        self.stack.enter_context(patch.object(suite_analysis,"require_pbs"))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        cfg=self.f.cfg
        cfg["analysis"].update(relative_cutoff_gap_floor=.001,minimum_learning_tasks=2,
                               learning_nll_improvement=.01)
        cfg["seeds"]=[101]
        self.cell={"id":"tiny","model_key":"qwen1.7b","config":cfg}
        rows=[]
        for mode in ("frozen","head","lora"):
            path=self.root/"runs"/mode
            self.f.run_engine(path,mode,self.f.make_model(mode))
            rows.append({"id":mode,"cell":"tiny","mode":mode,"order":"O1","seed":101,"run_path":"runs/"+mode})
        for row in rows:
            self.expand_features(self.root/row["run_path"],row["mode"])
        source=read_json(self.root/"runs/lora/run.json")["source_sha256"]
        self.plan={"name":"fixture","identity":"fixture","cells":[self.cell],"runs":rows,
                   "primary_alpha":.05,"primary_families":1,"source_sha256":source,
                   "storage":{"mode":"compact","reserve_gib":0}}
        directory=self.root/"plans/fixture"; directory.mkdir(parents=True)
        write_json(directory/"plan.json",self.plan)
        # Shared-archive fixture uses real ZIP verification, independent of network and data download.
        payload=self.root/"shared_data.json"; write_json(payload,{"probe_identity":"fixture"})
        a=compact.write_archive(self.root/"archives/fixture/shared.zip",{"shared_data.json":payload},
                               {"kind":"shared","plan_identity":"fixture","plan_name":"fixture"},0)
        write_json(directory/"compact_shared.json",{"plan_identity":"fixture","archive":a})

    def expand_features(self,path,mode):
        # Give the synthetic feature fixture enough rows and dimensions for every original
        # descriptive sensitivity (including rank32). Control stages are exactly identical.
        cfg=self.f.cfg
        tasks=cfg["data"]["tasks"]
        cfg["features"].update(layers=[1,2],primary_layer=1,
            poolings=["content","answer","last_content"],probe_splits=["probe_a","probe_b"])
        cfg["data"]["sizes"].update(probe_a=40,probe_b=40)
        meta=read_json(path/"run.json"); meta["config"]=copy.deepcopy(cfg); write_json(path/"run.json",meta)
        for stage_index in range(len(tasks)+1):
            stage=path/"stages"/f"stage_{stage_index:03d}"
            for split_i,split in enumerate(cfg["features"]["probe_splits"]):
                for layer in cfg["features"]["layers"]:
                    for pool_i,pooling in enumerate(cfg["features"]["poolings"]):
                        folder=stage/"features"/split/pooling/f"layer_{layer:03d}"
                        folder.mkdir(parents=True,exist_ok=True)
                        for task_i,task in enumerate(tasks):
                            rng=np.random.default_rng(500+task_i+split_i*10+pool_i*100+layer*1000)
                            x=rng.normal(size=(40,40))*np.linspace(2,.1,40)
                            if mode=="lora" and stage_index:
                                x=x @ np.linalg.qr(np.random.default_rng(stage_index*100+task_i).normal(size=(40,40)))[0]
                            np.savez_compressed(folder/(task+".npz"),features=x.astype(np.float32),
                                ids=np.asarray([f"{task}-{i}" for i in range(40)]),
                                groups=np.asarray([f"{task}-g{i}" for i in range(40)]),labels=np.arange(40)%2)
            record=read_json(stage/"COMPLETE.json")
            record["sha256"]={p.relative_to(stage).as_posix():digest_file(p)
                              for p in stage.rglob("*") if p.is_file() and p.name!="COMPLETE.json"}
            write_json(stage/"COMPLETE.json",record)

    def tearDown(self):
        self.stack.close(); self.f.tearDown()

    def ack(self,identifier):
        item=next(a for a in compact.ready(self.plan)["archives"] if a["id"]==identifier)
        manifest=compact.verify_archive(self.root/item["path"])
        self.assertEqual(canonical_hash(manifest),item["manifest_sha256"])
        compact.acknowledge(self.plan,identifier,item["sha256"],item["manifest_sha256"])

    def test_full_joint_and_sensitivity_outputs_identical_after_release(self):
        suite_analysis.analyze(self.plan)
        suite_analysis.sensitivity(self.plan)
        out=self.root/"reports/fixture/tiny"
        comparison=(out/"joint_pair_comparisons.csv").read_bytes()
        runs=(out/"runs.csv").read_bytes()
        sensitivity=(self.root/"reports/fixture/sensitivity_all.csv").read_bytes()
        report=read_json(out/"cell_report.json")
        with np.load(out/"joint_intervals.npz") as z:
            intervals={k:z[k].copy() for k in z.files}
        expected_draws={}
        original={}
        for row in self.plan["runs"]:
            run=self.root/row["run_path"]
            original[row["id"]]=digest_file(run/"stages/stage_003/checkpoint.pt")
            packed=compact.pack(self.plan,row["id"])
            capsule=compact.load_capsule(run,self.plan,row)
            with np.load(run/capsule["analysis_rel"]/"bootstrap.npz") as z:
                expected_draws[row["id"]]=z["draws"].copy()
            # The laptop archive fixture persists the real verified ZIP before acknowledgement.
            local=self.root/"laptop"/(row["id"]+".zip"); local.parent.mkdir(exist_ok=True)
            shutil.copy2(self.root/packed["archive"]["path"],local)
            m=compact.verify_archive(local)
            self.assertEqual(m["files"]["stages/stage_003/checkpoint.pt"]["sha256"],original[row["id"]])
        self.ack("shared")
        for row in self.plan["runs"]:
            self.ack(row["id"])
            compact.release(self.plan,row["id"])
            compact.release(self.plan,row["id"])  # retry is idempotent
            run=self.root/row["run_path"]
            self.assertFalse(list((run/"stages").rglob("*.npz")))
            self.assertFalse(list((run/"stages").rglob("checkpoint.pt")))
            capsule=compact.load_capsule(run,self.plan,row)
            with np.load(run/capsule["analysis_rel"]/"bootstrap.npz") as z:
                self.assertTrue(np.array_equal(z["draws"],expected_draws[row["id"]]))
        suite_analysis.analyze(self.plan)
        suite_analysis.sensitivity(self.plan)
        self.assertEqual(comparison,(out/"joint_pair_comparisons.csv").read_bytes())
        self.assertEqual(runs,(out/"runs.csv").read_bytes())
        self.assertEqual(sensitivity,(self.root/"reports/fixture/sensitivity_all.csv").read_bytes())
        actual=read_json(out/"cell_report.json")
        report.pop("created_at"); actual.pop("created_at")
        self.assertEqual(report,actual)
        with np.load(out/"joint_intervals.npz") as z:
            for key in intervals: self.assertTrue(np.array_equal(intervals[key],z[key]))
        self.assertTrue(read_json(self.root/"reports/fixture/suite_report.json")["complete"])

    def test_release_requires_both_verified_acknowledgements(self):
        compact.pack(self.plan,"lora")
        self.ack("lora")
        with self.assertRaisesRegex(ValueError,"BOTH"):
            compact.release(self.plan,"lora")
        self.assertTrue((self.root/"runs/lora/stages/stage_003/checkpoint.pt").exists())
        with self.assertRaisesRegex(ValueError,"does not match"):
            compact.acknowledge(self.plan,"shared","0"*64,"0"*64)

    def test_corrupt_retained_statistics_and_capsule_rejected(self):
        compact.pack(self.plan,"lora")
        run=self.root/"runs/lora"
        cap=compact.load_capsule(run,self.plan,self.plan["runs"][-1])
        with (run/cap["analysis_rel"]/"bootstrap.npz").open("ab") as f:f.write(b"corrupt")
        with self.assertRaisesRegex(ValueError,"artifact missing or changed"):
            compact.load_capsule(run,self.plan,self.plan["runs"][-1])

    def test_missing_archive_and_corrupt_original_prevent_release(self):
        compact.pack(self.plan,"lora"); self.ack("shared"); self.ack("lora")
        feature=next((self.root/"runs/lora/stages").rglob("*.npz"))
        with feature.open("ab") as f:f.write(b"changed")
        with self.assertRaisesRegex(ValueError,"Original artifact changed"):
            compact.release(self.plan,"lora")
        self.assertTrue(feature.exists())

    def test_pack_retry_and_submission_gate(self):
        a=compact.pack(self.plan,"lora")
        self.assertEqual(a,compact.pack(self.plan,"lora"))
        with self.assertRaisesRegex(ValueError,"Pack, fetch"):
            compact.can_submit(self.plan)

    def test_laptop_client_verifies_archives_and_acknowledges_without_deletion(self):
        import archive_client
        compact.pack(self.plan,"lora")
        acknowledgements=[]
        def fake_ssh(host,root,args,capture=True):
            if args[2]=="ready":return json.dumps(compact.ready(self.plan))
            self.assertEqual(args[2],"ack")
            identifier=args[args.index("--run-id")+1]
            sha=args[args.index("--sha256")+1]
            seal=args[args.index("--manifest-sha256")+1]
            compact.acknowledge(self.plan,identifier,sha,seal)
            acknowledgements.append(identifier)
            return "ACKNOWLEDGED"
        def fake_scp(args,check):
            source=args[1].split(":/experiment/",1)[1]
            shutil.copyfile(self.root/source,args[2])
        args=["archive_client.py","--host","user@hpc","--remote-root","/experiment",
              "--name","fixture","--destination",str(self.root/"laptop")]
        with patch.object(sys,"argv",args),patch.object(archive_client,"ssh",side_effect=fake_ssh),\
             patch.object(archive_client.shutil,"which",return_value="available"),\
             patch.object(archive_client.subprocess,"run",side_effect=fake_scp):
            archive_client.main()
            archive_client.main()  # no duplicate transfer after acknowledgement
        self.assertEqual(acknowledgements,["shared","lora"])
        self.assertTrue((self.root/"laptop/fixture/lora.zip").exists())
        self.assertTrue((self.root/"runs/lora/stages/stage_003/checkpoint.pt").exists())

    def test_restore_retained_statistics_from_verified_laptop_archive(self):
        import restore_compact
        packed=compact.pack(self.plan,"lora")
        local=self.root/"laptop/lora.zip";local.parent.mkdir()
        shutil.copy2(self.root/packed["archive"]["path"],local)
        self.ack("shared");self.ack("lora");compact.release(self.plan,"lora")
        run=self.root/"runs/lora"
        # Simulate loss of one retained analysis artifact; recovery must use its archive.
        cap=compact.load_capsule(run,self.plan,self.plan["runs"][-1])
        (run/cap["analysis_rel"]/"bootstrap.npz").unlink()
        with patch.object(restore_compact,"ROOT",self.root):
            restore_compact.restore(local)
        compact.load_capsule(run,self.plan,self.plan["runs"][-1])

    def test_shared_snapshot_verifies_selected_raw_and_token_inputs(self):
        import h2.common as common
        source_root=common.ROOT
        for rel in self.plan["source_sha256"]:
            target=self.root/rel;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source_root/rel,target)
        receipt=self.root/"plans/fixture/compact_shared.json";receipt.unlink()
        self.cell["config_path"]="plans/fixture/cells/tiny.json"
        write_json(self.root/self.cell["config_path"],self.f.cfg)
        write_json(self.root/"plans/fixture/FROZEN.json",{})
        with patch.object(common,"ROOT",self.root):
            write_json(common.model_asset_path(self.f.cfg),{"repo":"fixture","revision":"pinned"})
            for directory in (common.data_dir(self.f.cfg),common.token_dir(self.f.cfg)):
                write_json(directory/"selected.json",{"ids":["a","b"]})
                write_json(directory/"manifest.json",{"files":[{"path":"selected.json",
                    "sha256":digest_file(directory/"selected.json")} ]})
            result=compact.shared(self.plan)
            manifest=compact.verify_archive(self.root/result["archive"]["path"])
            self.assertIn(common.data_dir(self.f.cfg).relative_to(self.root).as_posix()+"/selected.json",manifest["files"])
            self.assertIn(common.token_dir(self.f.cfg).relative_to(self.root).as_posix()+"/selected.json",manifest["files"])
            receipt.unlink()
            (common.data_dir(self.f.cfg)/"selected.json").write_text("changed")
            with self.assertRaisesRegex(ValueError,"Prepared input changed"):
                compact.shared(self.plan)

    def test_guard_and_zip_path_validation(self):
        with self.assertRaisesRegex(RuntimeError,"headroom"):
            guard(self.root,10**18,1)
        for name in ("../bad","/bad","a/../bad","C:/bad","a\\bad","."):
            with self.assertRaises(ValueError): compact.relative_name(name)
        path=self.root/"malformed.zip"
        with zipfile.ZipFile(path,"w") as z:
            z.writestr("ARCHIVE_MANIFEST.json",json.dumps({"schema":1,"files":{}}))
            z.writestr("unexpected","bad")
        with self.assertRaisesRegex(ValueError,"inventory mismatch"):compact.verify_archive(path)


if __name__=="__main__":unittest.main()
