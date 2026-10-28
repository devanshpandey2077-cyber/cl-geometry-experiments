"""No real qsub calls: verify serial allocation and the accepted-job ledger."""
import io
import os
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from h2.common import ROOT, load_config, read_json, write_json
import submit_campaign as submit


class SubmissionTests(unittest.TestCase):
    def test_full_matrix_is_fifteen_trainings_and_two_controls(self):
        cfg = load_config(ROOT / "configs/full.json")
        rows = submit.campaign_rows(cfg,"test",list(cfg["orders"]),cfg["seeds"])
        self.assertEqual(len(rows),17)
        self.assertEqual(sum(r["mode"] == "lora" for r in rows),15)
        self.assertEqual(len({r["run_path"] for r in rows}),17)

    def test_submissions_are_serial_and_recorded_incrementally(self):
        with tempfile.TemporaryDirectory(prefix="h2-submit-",dir=os.environ.get("H2_TEST_TMPDIR")) as tmp:
            root = Path(tmp)
            cfg = load_config(ROOT / "configs/full.json")
            write_json(root / "configs/full.json",cfg)
            (root / "settings.env").write_text("MIG_UUID=fixture\n")
            command = ["submit_campaign.py","--config",str(root / "configs/full.json"),
                       "--campaign","fixture","--with-analysis","--submit"]
            calls = []
            def fake_submit(cmd):
                calls.append(cmd)
                return str(9000 + len(calls)-1) + ".master"
            with patch.object(submit,"ROOT",root), patch.object(submit,"checked_submit",side_effect=fake_submit), \
                 patch.object(submit.subprocess,"run",return_value=SimpleNamespace(returncode=0,stdout="")), \
                 patch("sys.argv",command), redirect_stdout(io.StringIO()):
                submit.main()
            self.assertEqual(len(calls),18)
            self.assertNotIn("-W",calls[0])
            for i in range(1,len(calls)):
                self.assertIn("depend=afterok:" + str(9000+i-1) + ".master",calls[i])
            plan = read_json(root / "campaigns/fixture.json")
            self.assertEqual(plan["runs"][-1]["job_id"],"9016.master")
            self.assertEqual(plan["analysis_job_id"],"9017.master")
            # A new ordinary submit refuses to overwrite the registered campaign.
            with patch.object(submit,"ROOT",root), patch("sys.argv",command):
                with self.assertRaises(ValueError):
                    submit.main()

    def test_reject_invalid_pbs_variable_separators(self):
        with self.assertRaises(ValueError):
            submit.pbs_variables({"H2_CONFIG":"config,INJECT=1"})


if __name__ == "__main__":
    unittest.main()
