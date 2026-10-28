from __future__ import annotations
import copy
import io
import os
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from h2.common import ROOT, load_config, read_json, seed_everything, to_numpy
from h2.data import assign_splits, audit_splits, grouped_records, in_source_partition, unique_records
from h2.geometry import (affinity, affinity_matrix, differences, inversion_flags,
                         joint_group_bootstrap_indices, pair_vector, simultaneous_intervals, subspace)
from h2.lora import inject_lora, restore_trainable, trainable_state
from h2.prompts import Pack, collate, encode_record, label_token_ids, pool_hidden, render, save_pack
from h2.tasks import TASKS
from h2 import engine
from h2.analyze import analyze_run, summarize_campaign

torch.set_num_threads(1)


class CharacterTokenizer:
    pad_token_id = 0
    is_fast = True
    def encode(self, text, add_special_tokens=True):
        return [ord(c) + 1 for c in text]
    def decode(self, ids, **kwargs):
        return "".join(chr(int(i)-1) for i in ids if i)
    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text),
                "offset_mapping": [(i, i+1) for i in range(len(text))]}


class TinyCausalLM(nn.Module):
    """A causal CPU fixture, never used as a substitute for an H100 model test."""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(512, 16)
        self.q_proj = nn.Linear(16, 16)
        self.o_proj = nn.Linear(16, 16)
        self.lm_head = nn.Linear(16, 512, bias=False)
        self.config = SimpleNamespace(hidden_size=16, use_cache=False)
    def forward(self, input_ids, attention_mask, position_ids, output_hidden_states=False,
                use_cache=False, return_dict=True, logits_to_keep=0):
        e = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).to(e.dtype)
        cumulative = (e * mask).cumsum(1) / mask.cumsum(1).clamp_min(1)
        h1 = torch.tanh(self.q_proj(cumulative))
        h2 = h1 + torch.tanh(self.o_proj(h1))
        z = h2[:, -logits_to_keep:, :] if logits_to_keep else h2
        return SimpleNamespace(logits=self.lm_head(z), hidden_states=(e, h1, h2))


def fixture_record(task, i, label=0):
    rng = np.random.default_rng(i + 196)
    alphabet = np.asarray(list("abcdefghijklmnopqrstuvwxyz"))
    texts = ["".join(rng.choice(alphabet, size=20+j*3)) for j in range(len(TASKS[task].fields))]
    return {"task": task, "texts": texts, "label": label, "id": f"{task}-{i}",
            "group_id": f"g-{task}-{i}"}


class GeometryTests(unittest.TestCase):
    def test_affinity_feature_coordinates_and_common_rotation(self):
        rng = np.random.default_rng(9)
        u = np.eye(8)[:, :2]
        v = np.eye(8)[:, 2:4]
        self.assertAlmostEqual(affinity(u, u), 1)
        self.assertAlmostEqual(affinity(u, v), 0)
        x = rng.normal(size=(100, 8)) @ np.diag([10, 8, 4, 3, 2, 1, .5, .2])
        y = rng.normal(size=(80, 8)) @ np.diag([9, 1, 5, 3, 2, 1, .5, .2])
        rotation = np.linalg.qr(rng.normal(size=(8,8)))[0]
        a = affinity(subspace(x, 2)[0], subspace(y, 2)[0])
        b = affinity(subspace(x @ rotation, 2)[0], subspace(y @ rotation, 2)[0])
        self.assertAlmostEqual(a, b, places=10)
        self.assertEqual(subspace(x, 2)[0].shape, (8,2))

    def test_constructed_reversal_and_near_tie_rule(self):
        rng = np.random.default_rng(73)
        def features(angle):
            z = rng.normal(size=(60,1))
            direction = np.array([np.cos(angle), np.sin(angle), 0, 0])
            return z * direction + rng.normal(size=(60,4))*1e-5
        initial = {"a": features(0), "b": features(.1), "c": features(1.3)}
        final = {"a": features(0), "b": features(1.3), "c": features(.1)}
        values = []
        for stage in (initial, final):
            _, a, _ = affinity_matrix(stage, 1)
            _, v = pair_vector(a)
            _, d = differences(v)
            values.append(d)
        point = np.asarray(values)
        boots = point[None,:,:] + rng.normal(0,.001,size=(100,2,3))
        lo, hi, info = simultaneous_intervals(point, boots)
        self.assertTrue(inversion_flags(lo,hi,.01)[0])
        self.assertEqual(info["family_size"], 6)
        near = np.array([[.0001],[-.0001]])
        lo, hi, _ = simultaneous_intervals(near, np.repeat(near[None], 30, axis=0))
        self.assertFalse(inversion_flags(lo,hi,.01)[0])

    def test_global_groups_share_multiplicity(self):
        groups = {"a": np.array(["shared", "a1", "a2"]),
                  "b": np.array(["shared", "b1", "b2"])}
        for seed in range(20):
            indices = joint_group_bootstrap_indices(groups, np.random.default_rng(seed))
            self.assertEqual(int((indices["a"] == 0).sum()), int((indices["b"] == 0).sum()))

    def test_reject_degenerate_rank(self):
        with self.assertRaises(ValueError):
            subspace(np.ones((20,8)), 2)


class DataAndPromptTests(unittest.TestCase):
    def test_exploratory_confirmation_exact_fields_are_disjoint(self):
        rows = [fixture_record(t,i)["texts"] for t in ("yelp","qqp") for i in range(200)]
        pools = {p: {text.lower() for texts in rows if in_source_partition(texts,p) for text in texts}
                 for p in ("development","confirmation")}
        self.assertTrue(pools["development"])
        self.assertTrue(pools["confirmation"])
        self.assertFalse(pools["development"] & pools["confirmation"])
        for texts in rows:
            self.assertFalse(in_source_partition(texts,"development") and
                             in_source_partition(texts,"confirmation"))
    def test_exact_conflicts_and_cross_task_question_group(self):
        a = {"id":"a", "task":"qqp", "texts":["Repeated QUESTION", "Other text"], "label":1}
        b = {"id":"b", "task":"qnli", "texts":["repeated question", "Different sentence"], "label":0}
        groups, _ = grouped_records([a,b])
        self.assertEqual(len(groups), 1)
        conflict = copy.deepcopy(a)
        conflict.update(id="c",label=0)
        rows, info = unique_records([a,conflict,b])
        self.assertEqual([r["id"] for r in rows], ["b"])
        self.assertEqual(info["conflicting_content_keys"], 1)

    def test_grouped_split_balance_and_leakage_detection(self):
        tasks = ["yelp","ag_news"]
        rows = [fixture_record(t, i, i % len(TASKS[t].labels)) for t in tasks for i in range(160)]
        groups, _ = grouped_records(rows)
        sizes = dict(train=24, validation=8, probe_a=8, probe_b=8, test=8)
        splits = assign_splits(groups,tasks,sizes,17)
        self.assertEqual(audit_splits(splits)["cross_split_group_overlap"], 0)
        for split in sizes:
            for task in tasks:
                self.assertEqual(len(splits[split][task]), sizes[split])
        splits["test"]["yelp"].append(splits["train"]["yelp"][0])
        with self.assertRaises(ValueError):
            audit_splits(splits)
        with self.assertRaises(ValueError):
            assign_splits(groups, tasks, {s:1000 for s in sizes},17)

    def test_content_spans_and_truncation_keep_answer(self):
        tok = CharacterTokenizer()
        rec = fixture_record("qqp", 1)
        row = encode_record(tok,rec,512)
        text, spans = render("qqp", rec["texts"])
        selected = "".join(c for c,m in zip(text,row["content_mask"]) if m)
        self.assertEqual(selected, "".join(rec["texts"]))
        rec["texts"] = ["long text " * 200, "other text " * 200]
        short = encode_record(tok, rec,256)
        self.assertTrue(short["truncated"])
        self.assertLessEqual(len(short["input_ids"]),256)
        self.assertTrue(tok.decode(short["input_ids"]).endswith("Answer:"))
        self.assertEqual(len(label_token_ids(tok)),4)

    def test_padding_and_pooling_exclude_instruction_values(self):
        rows = [dict(input_ids=[4,5],content_mask=[0,1],label=0),
                dict(input_ids=[7,8,9],content_mask=[0,1,1],label=1)]
        batch = collate(rows,0,"cpu")
        self.assertEqual(batch["position_ids"].tolist(), [[0,0,1],[0,1,2]])
        h = torch.tensor([[[999.],[100.],[2.]], [[100.],[2.],[4.]]])
        self.assertEqual(pool_hidden(h,batch,"content").flatten().tolist(),[2.,3.])
        self.assertEqual(pool_hidden(h,batch,"last_content").flatten().tolist(),[2.,4.])


class LoRATests(unittest.TestCase):
    def test_zero_init_freezes_base_and_checkpoint_roundtrip(self):
        seed_everything(13)
        model = TinyCausalLM()
        x = torch.randn(3,16)
        expected = model.q_proj(x).detach().clone()
        inject_lora(model,dict(targets=["q_proj","o_proj"],rank=2,alpha=4,dropout=0))
        self.assertTrue(torch.equal(expected,model.q_proj(x)))
        base = model.q_proj.base.weight.detach().clone()
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.1)
        model.q_proj(x).square().mean().backward()
        opt.step()
        self.assertTrue(torch.equal(base,model.q_proj.base.weight))
        self.assertGreater(float(model.q_proj.lora_B.abs().sum()),0)
        state = trainable_state(model)
        with torch.no_grad():
            model.q_proj.lora_B.zero_()
        restore_trainable(model,state)
        self.assertTrue(torch.equal(model.q_proj.lora_B,state["q_proj.lora_B"]))
        self.assertTrue(all("lora_" in k for k in state))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="h2-test-", dir=os.environ.get("H2_TEST_TMPDIR"))
        self.root = Path(self.tmp.name)
        self.cfg = load_config(ROOT / "configs/smoke.json")
        self.cfg["data"]["tasks"] = ["yelp","qqp","ag_news"]
        self.cfg["orders"] = {"O1":self.cfg["data"]["tasks"]}
        self.cfg["features"].update(layers=[1,2],primary_layer=1,probe_splits=["probe_b"])
        self.cfg["training"].update(epochs=1,microbatch=3,gradient_accumulation=2,
                                      max_steps_per_task=2,eval_batch=8,learning_rate=.01)
        self.cfg["training"]["lora"] = dict(targets=["q_proj","o_proj"],rank=2,alpha=4,dropout=.05)
        self.cfg["analysis"].update(rank=2,bootstrap=20)
        tok = CharacterTokenizer()
        self.tokens = {"label_token_ids":label_token_ids(tok),"pad_token_id":0}
        self.packs = {}
        for task in self.cfg["data"]["tasks"]:
            self.packs[task] = {}
            for split in ("train","validation","test","probe_b"):
                n = 32 if split == "probe_b" else 12
                rows = [encode_record(tok,fixture_record(task,i,i % len(TASKS[task].labels)),512)
                        for i in range(n)]
                path = self.root / "packs" / task / (split + ".npz")
                save_pack(path,rows)
                self.packs[task][split] = Pack(path)
    def tearDown(self):
        self.tmp.cleanup()
    def make_model(self, mode):
        seed_everything(101)
        model = TinyCausalLM().requires_grad_(False)
        if mode == "lora":
            inject_lora(model,self.cfg["training"]["lora"])
        elif mode == "head":
            model.h2_readout = nn.Linear(16,4)
        return model
    def run_engine(self, path, mode, model, resume=False):
        with redirect_stdout(io.StringIO()):
            engine.execute_run(self.cfg,model,self.packs,self.tokens,path,mode,"O1",101,"cpu",
                               resume, {"data_manifest":{"recipe_hash":"fixture"}})
    def test_all_stages_analysis_and_resume_after_interruption(self):
        model = self.make_model("lora")
        baseline = model.embedding.weight.detach().clone()
        run = self.root / "lora_O1_s101"
        original = engine.extract
        def interrupt_at_second_stage(*args, **kwargs):
            import inspect
            destination = inspect.signature(original).bind(*args,**kwargs).arguments["destination"]
            if "stage_001.tmp" in str(destination):
                raise RuntimeError("simulated interruption")
            return original(*args,**kwargs)
        with patch.object(engine,"extract",side_effect=interrupt_at_second_stage):
            with self.assertRaises(RuntimeError):
                self.run_engine(run,"lora",model)
        self.assertTrue((run / "stages/stage_000/COMPLETE.json").exists())
        resumed = self.make_model("lora")
        self.run_engine(run,"lora",resumed,True)
        self.assertTrue((run / "COMPLETE.json").exists())
        self.assertTrue(torch.equal(baseline,resumed.embedding.weight))
        self.assertTrue(any(float(p.abs().sum()) > 0 for n,p in resumed.named_parameters() if "lora_B" in n))
        self.assertEqual(read_json(run / "stages/stage_001/training.json")["updates"],2)
        direct = self.root / "direct"
        self.run_engine(direct,"lora",self.make_model("lora"))
        for name, value in trainable_state(resumed).items():
            other = torch.load(direct / "stages/stage_003/checkpoint.pt",weights_only=True)["parameters"][name]
            self.assertTrue(torch.equal(value,other),name)
        with redirect_stdout(io.StringIO()):
            analysis = analyze_run(run)
        report = read_json(analysis / "report.json")
        self.assertEqual(report["comparisons"],3)
        self.assertEqual(report["uncertainty"]["family_size"],6)
        self.assertTrue((analysis / "accuracy.csv").exists())
        with self.assertRaises(FileExistsError):
            self.run_engine(run,"lora",resumed)
    def test_frozen_and_head_controls_hold_encoder_fixed(self):
        analyses = []
        for mode in ("frozen","head"):
            model = self.make_model(mode)
            before = model.embedding.weight.detach().clone()
            head_before = model.h2_readout.weight.detach().clone() if mode == "head" else None
            run = self.root / mode
            self.run_engine(run,mode,model)
            self.assertTrue(torch.equal(before,model.embedding.weight))
            with redirect_stdout(io.StringIO()):
                analysis = analyze_run(run)
            analyses.append(analysis)
            report = read_json(analysis / "report.json")
            self.assertTrue(report["fixed_representation_control_pass"])
            self.assertEqual(report["within_run_interval_inversions"],0)
            if mode == "head":
                self.assertFalse(torch.equal(head_before,model.h2_readout.weight))
        with patch("h2.analyze.ROOT",self.root):
            report_path = summarize_campaign("test_controls",analyses,plan={"runs":["frozen","head"]})
        campaign = read_json(report_path / "campaign_report.json")
        self.assertTrue(campaign["all_present_controls_pass"])
        self.assertEqual(campaign["joint_uncertainty"]["family_size"],12)
        self.assertEqual(campaign["controls_present"],2)
        self.assertTrue(all(r["joint_interval_inversions"] == 0 for r in campaign["runs"]))


if __name__ == "__main__":
    unittest.main()
