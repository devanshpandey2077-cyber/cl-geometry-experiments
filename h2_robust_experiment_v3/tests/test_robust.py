import copy
import io
import os
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import numpy as np
import torch
import test_core
import suite
from h2.common import ROOT,load_config,read_json
from h2 import engine
from h2.checkpoints import RunPaused
from h2.lora import trainable_state
from h2.robust_data import stable_assign,group_role
from h2.data import audit_splits
from h2.suite_analysis import replication_counts,verify_run_artifacts
from h2.geometry import covariance_affinity_matrix,subspace

class RobustEngineTests(unittest.TestCase):
    def setUp(self):
        self.f=test_core.PipelineTests(); self.f.setUp()
    def tearDown(self): self.f.tearDown()
    def test_optimizer_step_resume_matches_uninterrupted_dropout_training(self):
        cfg=self.f.cfg; task="yelp"; codes=self.f.tokens["label_token_ids"]
        first=self.f.make_model("lora")
        state=self.f.root/"progress.pt"
        with redirect_stdout(io.StringIO()),patch("h2.engine.pause_requested",return_value=True):
            with self.assertRaises(RunPaused):
                engine.train_task(first,self.f.packs[task]["train"],cfg,"lora",codes,0,"cpu",44,task,
                                  self.f.root/"paused.jsonl",state)
        self.assertTrue(state.exists())
        self.assertEqual(torch.load(state,weights_only=True)["totals"][0],1)
        resumed=self.f.make_model("lora")
        with redirect_stdout(io.StringIO()):
            result=engine.train_task(resumed,self.f.packs[task]["train"],cfg,"lora",codes,0,"cpu",44,task,
                                    self.f.root/"resumed.jsonl",state)
        direct=self.f.make_model("lora")
        with redirect_stdout(io.StringIO()):
            expected=engine.train_task(direct,self.f.packs[task]["train"],cfg,"lora",codes,0,"cpu",44,task,
                                      self.f.root/"direct.jsonl")
        self.assertEqual(result["updates"],expected["updates"])
        self.assertEqual(result["examples"],expected["examples"])
        for name,value in trainable_state(resumed).items():
            self.assertTrue(torch.equal(value,trainable_state(direct)[name]),name)
    def test_changed_completed_feature_is_rejected(self):
        path=self.f.root/"corrupt"
        self.f.run_engine(path,"lora",self.f.make_model("lora"))
        verify_run_artifacts(path)
        feature=next((path/"stages"/"stage_000"/"features").rglob("*.npz"))
        with feature.open("ab") as f:f.write(b"changed")
        with self.assertRaisesRegex(ValueError,"Artifact changed"): verify_run_artifacts(path)
        with self.assertRaisesRegex(ValueError,"Corrupted"):
            self.f.run_engine(path,"lora",self.f.make_model("lora"),True)
    def test_auxiliary_views_only_saved_at_declared_endpoints(self):
        self.f.cfg["features"].update(auxiliary_endpoints_only=True)
        path=self.f.root/"retained"
        self.f.run_engine(path,"frozen",self.f.make_model("frozen"))
        self.assertTrue((path/"stages/stage_000/features/probe_b/answer/layer_002/yelp.npz").exists())
        self.assertFalse((path/"stages/stage_001/features/probe_b/answer/layer_002/yelp.npz").exists())
        self.assertTrue((path/"stages/stage_001/features/probe_b/content/layer_001/yelp.npz").exists())
        self.assertTrue((path/"stages/stage_003/features/probe_b/answer/layer_002/yelp.npz").exists())
    def test_complete_suite_analysis_including_both_controls(self):
        from h2 import suite_analysis
        self.f.cfg["analysis"].update(relative_cutoff_gap_floor=.001,minimum_learning_tasks=2,
                                     learning_nll_improvement=.01)
        self.f.cfg["seeds"]=[101]
        cell={"id":"tiny","model_key":"qwen1.7b","config":self.f.cfg}
        rows=[]
        for mode in ("frozen","head","lora"):
            path=self.f.root/"runs"/mode
            self.f.run_engine(path,mode,self.f.make_model(mode))
            rows.append({"id":mode,"cell":"tiny","mode":mode,"order":"O1","seed":101,
                         "run_path":"runs/"+mode})
        plan={"name":"fixture","identity":"fixture","cells":[cell],"runs":rows,
              "primary_alpha":.05,"primary_families":1}
        with patch.object(suite_analysis,"ROOT",self.f.root),patch.object(suite,"ROOT",self.f.root),\
             patch.object(suite,"verify_frozen",return_value={}),patch.object(suite_analysis,"require_pbs"),\
             redirect_stdout(io.StringIO()):
            suite_analysis.analyze(plan)
        report=read_json(self.f.root/"reports/fixture/suite_report.json")
        self.assertTrue(report["complete"])
        self.assertEqual(len(report["cells"][0]["controls"]),2)
        self.assertFalse(report["cells"][0]["confirmatory_eligible"])

class StableDataTests(unittest.TestCase):
    def make_groups(self):
        groups=[]
        for task in ("yelp","qqp","ag_news"):
            classes=len(test_core.TASKS[task].labels)
            for i in range(1200):
                r=test_core.fixture_record(task,i,i%classes)
                r["texts"]=[task+" "+text for text in r["texts"]]
                groups.append([r])
        return groups
    def test_sampling_seed_changes_rows_without_changing_roles(self):
        groups=self.make_groups(); sizes=dict(train=80,validation=20,probe_a=20,probe_b=20,test=20)
        a=stable_assign(groups,["yelp","qqp","ag_news"],sizes,1,91)
        b=stable_assign(groups,["yelp","qqp","ag_news"],sizes,2,91)
        owner={r["id"]:s for s,tasks in a.items() for rows in tasks.values() for r in rows}
        overlap=0
        for s,tasks in b.items():
            for rows in tasks.values():
                for r in rows:
                    if r["id"] in owner:self.assertEqual(owner[r["id"]],s); overlap+=1
        self.assertGreater(overlap,0)
        self.assertEqual(audit_splits(a)["cross_split_group_overlap"],0)
    def test_deficits_fail_instead_of_duplicating(self):
        with self.assertRaises(ValueError):
            stable_assign(self.make_groups()[:3],["yelp","qqp","ag_news"],
                          dict(train=80,validation=20,probe_a=20,probe_b=20,test=20),1,91)
    def test_role_does_not_depend_on_cohort_or_requested_counts(self):
        roles=[group_role(f"g{i}",23) for i in range(1000)]
        self.assertEqual(roles,[group_role(f"g{i}",23) for i in range(1000)])
        self.assertEqual(set(roles),{"train","validation","probe_a","probe_b","test"})

class SuiteDesignTests(unittest.TestCase):
    def test_profile_counts_unique_seeds_and_run_paths(self):
        base=load_config(ROOT/"configs/full.json")
        expected={"smoke":(1,3),"core":(1,17),"primary":(4,208),"stress":(11,121),
                  "replication":(1,52),"calibration":(2,6),"scale14b":(1,17),"full_tuning":(1,17)}
        with tempfile.TemporaryDirectory(dir=os.environ.get("H2_TEST_TMPDIR")) as d:
          for profile,(cells,runs) in expected.items():
            with patch.object(suite,"ROOT",Path(d)),patch.object(suite,"load_config",return_value=copy.deepcopy(base)),\
                 patch.object(suite,"source_inventory",return_value={}),redirect_stdout(io.StringIO()):
                plan=suite.build(profile,profile)
            self.assertEqual(len(plan["cells"]),cells)
            self.assertEqual(len(plan["runs"]),runs)
            self.assertEqual(len({r["run_path"] for r in plan["runs"]}),runs)
            self.assertEqual(len({r["job_name"] for r in plan["runs"]}),runs)
            for cell in plan["cells"]:
                self.assertEqual(len(cell["config"]["seeds"]),len(set(cell["config"]["seeds"])))
                if profile!="calibration": self.assertEqual(len(cell["config"]["orders"]),len({tuple(o) for o in cell["config"]["orders"].values()}))
    def test_calibration_matches_reference_training_shape(self):
        base=load_config(ROOT/"configs/full.json")
        with tempfile.TemporaryDirectory(dir=os.environ.get("H2_TEST_TMPDIR")) as d:
            with patch.object(suite,"ROOT",Path(d)),patch.object(suite,"load_config",return_value=base),\
                 patch.object(suite,"source_inventory",return_value={}),redirect_stdout(io.StringIO()):
                primary=suite.build("p","primary")
                cal=suite.build("c","calibration")
        signatures={c["shape_signature"] for c in cal["cells"]}
        self.assertTrue(all(c["shape_signature"] in signatures for c in primary["cells"]))
    def test_replication_rule_does_not_count_duplicate_rows_as_seeds(self):
        r={"eligible_inversion":True,"comparison":0,"initial_direction":1,"order":"O1","seed":1}
        self.assertEqual(replication_counts([r]*100,5,["O1","O2","O3"]),[])
        rows=[dict(r,seed=s,order=o) for s in (1,2,3,4) for o in ("O1","O2","O3")]
        self.assertEqual(len(replication_counts(rows,5,["O1","O2","O3"])),1)
        rows[-1]["initial_direction"]=-1
        self.assertEqual(replication_counts(rows,5,["O1","O2","O3"]),[])
    def test_covariance_overlap_in_common_feature_coordinates(self):
        rng=np.random.default_rng(912)
        a=rng.normal(size=(30,4))*[9,2,1,.1]; b=rng.normal(size=(37,4))*[1,8,1,.1]
        q=np.linalg.qr(rng.normal(size=(4,4)))[0]
        _,x,_=covariance_affinity_matrix({"a":a,"b":b})
        _,y,_=covariance_affinity_matrix({"a":a@q,"b":b@q})
        np.testing.assert_allclose(x,y,atol=1e-12)
        self.assertTrue(0<=x[0,1]<=1)
    def test_large_matrix_iterative_svd_matches_dense_feature_subspace(self):
        rng=np.random.default_rng(37)
        x=rng.normal(size=(120,96))*np.linspace(8,.2,96)
        x-=x.mean(0)
        basis,info=subspace(x,5)
        _,singular,vt=np.linalg.svd(x,full_matrices=False)
        np.testing.assert_allclose(basis@basis.T,vt[:5].T@vt[:5],atol=1e-7)
        np.testing.assert_allclose(info["singular_values"],singular[:6],rtol=1e-7)
        repeated,_=subspace(x,5)
        np.testing.assert_allclose(basis@basis.T,repeated@repeated.T,atol=1e-10)

class SchedulerTests(unittest.TestCase):
    def plan(self):
        return {"name":"p","cells":[{"id":"c","config_path":"plans/p/cells/c.json"}],
                "runs":[{"id":f"r{i}","cell":"c","mode":"lora","order":"O1","seed":i,
                         "run_path":f"runs/p/r{i}","job_name":f"h2test{i}"} for i in range(3)]}
    def test_bounded_serial_submissions_and_accepted_job_ledger(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("H2_TEST_TMPDIR")) as d:
            root=Path(d); (root/"settings.env").write_text("",encoding="utf-8")
            calls=[]
            def run(command,**kwargs):
                calls.append(command)
                return subprocess.CompletedProcess(command,0,"" if command[0]=="qstat" else f"{100+len(calls)}.master\n","")
            with patch.object(suite,"ROOT",root),patch.object(suite,"verify_frozen"),\
                 patch.object(suite.subprocess,"run",side_effect=run),redirect_stdout(io.StringIO()):
                suite.submit(self.plan(),limit=2,execute=True)
            qs=[c for c in calls if c[0]=="qsub"]
            self.assertEqual(len(qs),2)
            self.assertTrue(any(x.startswith("depend=afterany:") for x in qs[1]))
            ledger=read_json(root/"plans/p/submissions.json")
            self.assertEqual(len(ledger["runs"]),2)
            self.assertTrue(all(v["job_id"].endswith(".master") for v in ledger["runs"].values()))
    def test_active_deterministic_job_name_prevents_duplicate_launch(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("H2_TEST_TMPDIR")) as d:
            root=Path(d); (root/"settings.env").write_text("",encoding="utf-8")
            calls=[]
            def run(command,**kwargs):
                calls.append(command)
                return subprocess.CompletedProcess(command,0,
                    "77.master me workq h2test0 Q\n" if command[0]=="qstat" else "88.master\n","")
            with patch.object(suite,"ROOT",root),patch.object(suite,"verify_frozen"),\
                 patch.object(suite.subprocess,"run",side_effect=run),redirect_stdout(io.StringIO()):
                suite.submit(self.plan(),limit=1,execute=True,resume=True)
            qs=[c for c in calls if c[0]=="qsub"]
            self.assertEqual(len(qs),1)
            self.assertIn("h2test1",qs[0]); self.assertIn("depend=afterany:77.master",qs[0])
    def test_invalid_plan_names_cannot_traverse_paths(self):
        for name in ("../other","a/b","a,b","a\nb"):
            with self.assertRaises(ValueError):suite.safe_name(name)

if __name__=="__main__":unittest.main()
