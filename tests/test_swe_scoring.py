import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rsi import ROOT
from rsi.backends import SWEBackend, harness_command
from rsi.cli import load_config, main
from rsi.common import write_json
from rsi.swe import evaluation_dataset_name, validate_swe_tasks


class SWEScoringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = load_config(ROOT / "configs/deepseek.yaml")
        self.task = {
            "id": "astropy__astropy-12907", "benchmark": "swe",
            "instance": {"repo": "astropy/astropy", "base_commit": "abc", "version": "4.3"},
        }
        self.row = {
            **self.task["instance"], "instance_id": self.task["id"],
            "image": "swebench/example:latest", "eval_script": "pytest -rA",
            "log_parser": "parse_log_pytest", "eval_type": "pass_and_fail",
            "FAIL_TO_PASS": ["test_fixed"], "PASS_TO_PASS": ["test_existing"],
            "patch": "SECRET_GOLD_PATCH", "test_patch": "SECRET_TESTS",
        }

    def backend(self):
        # Exercise the real score method without starting an extra solver container.
        backend = SWEBackend.__new__(SWEBackend)
        backend.task, backend.config, backend.folder = self.task, self.config, self.root
        return backend

    def report(self, run_id, **fields):
        path = self.root / "logs/evaluation" / run_id / "rsi" / self.task["id"] / "report.json"
        write_json(path, {self.task["id"]: {"resolved": True, "infra_failure": False, **fields}})
        return path

    def test_legacy_config_and_harness_use_current_dataset(self):
        legacy = {**self.config, "swe_dataset": "princeton-nlp/SWE-bench_Verified"}
        path = self.root / "config.yaml"
        write_json(path, legacy)  # JSON is also valid YAML.
        self.assertEqual(load_config(path)["swe_dataset"], "SWE-bench/SWE-bench_Verified")
        command = harness_command(legacy, self.root / "predictions.json", self.task["id"], "test")
        self.assertEqual(command[command.index("--dataset_name") + 1], "SWE-bench/SWE-bench_Verified")
        for name in ("local.json", "other-owner/SWE-bench_Verified"):
            self.assertEqual(evaluation_dataset_name(name), name)

    def test_image_pull_budget_defaults_for_old_configs_and_rejects_nonpositive(self):
        legacy = dict(self.config)
        legacy.pop("image_pull_timeout")
        path = self.root / "legacy.yaml"
        write_json(path, legacy)
        loaded = load_config(path)
        self.assertEqual(loaded["image_pull_timeout"], 7200)
        self.assertEqual(loaded["rollout_timeout"], 1800)
        self.assertEqual(loaded["evaluation_timeout"], 1800)
        for timeout in (0, -1):
            write_json(path, {**legacy, "image_pull_timeout": timeout})
            with self.assertRaisesRegex(ValueError, "image_pull_timeout must be positive"):
                load_config(path)

    def test_solver_container_receives_configured_image_pull_budget(self):
        task = copy.deepcopy(self.task)
        task["instance"]["instance_id"] = task["id"]
        config = {**self.config, "image_pull_timeout": 10800}
        with patch("rsi.docker_environment.CompatibleDockerEnvironment") as environment:
            environment.return_value.execute.return_value = {"output": "abc\n"}
            backend = SWEBackend(task, self.root, config)
        self.assertEqual(environment.call_args.kwargs["pull_timeout"], 10800)
        self.assertEqual(environment.call_args.kwargs["timeout"], config["command_timeout"])
        self.assertEqual(backend.base, "abc")

    def test_incompatible_dataset_stops_cli_before_solver_work(self):
        legacy_row = {k: v for k, v in self.row.items()
                      if k not in ("image", "eval_script", "log_parser", "eval_type")}
        with patch("rsi.cli.load_tasks", return_value=[self.task]), \
             patch("rsi.cli.shutil.which", return_value="docker"), \
             patch("rsi.cli.os.getenv", return_value="configured"), \
             patch("rsi.swe.load_evaluation_rows", return_value=[legacy_row]), \
             patch("rsi.runner.run_experiment") as run:
            with self.assertRaisesRegex(ValueError, "missing evaluation fields.*image"):
                main(["run", "--tasks", "unused.json", "--output", str(self.root / "run")])
        run.assert_not_called()
        self.assertFalse((self.root / "run").exists())

    def test_preflight_keeps_hidden_fields_out_of_solver_tasks(self):
        tasks = [self.task, {"id": "terminal-example", "benchmark": "terminal"}]
        original = copy.deepcopy(tasks)
        with patch("rsi.swe.load_evaluation_rows", return_value=[self.row]) as load:
            validate_swe_tasks(tasks, self.config)
        load.assert_called_once_with(self.config, [self.task["id"]])
        self.assertEqual(tasks, original)
        self.assertNotIn("SECRET_", json.dumps(tasks))

    def test_preflight_rejects_a_different_checkout(self):
        with patch("rsi.swe.load_evaluation_rows", return_value=[{**self.row, "base_commit": "wrong"}]):
            with self.assertRaisesRegex(ValueError, "base_commit differs"):
                validate_swe_tasks([self.task], self.config)

    def test_terminal_only_run_does_not_load_swe_data(self):
        with patch("rsi.swe.load_evaluation_rows") as load:
            validate_swe_tasks([{"benchmark": "terminal"}], self.config)
        load.assert_not_called()

    def test_missing_report_cannot_reuse_an_old_success(self):
        self.report("old-run")
        result = subprocess.CompletedProcess([], 1, "", "KeyError: 'image'")
        with patch("rsi.backends.subprocess.run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "no instance report.*evaluation.log"):
                self.backend().score("candidate patch")
        self.assertIn("KeyError: 'image'", (self.root / "evaluation.log").read_text())

    def test_score_reads_only_the_current_official_report(self):
        self.report("old-run", resolved=False)
        path = self.report("rsi-" + "a" * 12)
        result = subprocess.CompletedProcess([], 0, "evaluation complete", "")
        with patch("rsi.backends.uuid.uuid4", return_value=SimpleNamespace(hex="a" * 32)), \
             patch("rsi.backends.subprocess.run", return_value=result):
            score = self.backend().score("candidate patch")
        self.assertEqual(score, {"success": True, "report": str(path)})

    def test_infrastructure_failure_does_not_become_a_failed_solution(self):
        self.report("rsi-" + "a" * 12, resolved=False, infra_failure=True,
                    infra_failure_reason="dependency installation failed")
        result = subprocess.CompletedProcess([], 0, "", "")
        with patch("rsi.backends.uuid.uuid4", return_value=SimpleNamespace(hex="a" * 32)), \
             patch("rsi.backends.subprocess.run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "infrastructure failure.*dependency"):
                self.backend().score("candidate patch")

    def test_empty_patch_is_unresolved_without_a_harness_report(self):
        with patch("rsi.backends.subprocess.run") as run:
            score = self.backend().score("\n")
        self.assertFalse(score["success"])
        self.assertEqual(score["reason"], "empty_patch")
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
