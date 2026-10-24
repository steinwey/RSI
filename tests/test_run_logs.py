import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from rsi import ROOT
from rsi.config import load_config
from rsi.cli import main
from rsi.common import read_json, write_json
from rsi.run_logs import create_run_directory, update_run, write_run_index
from rsi.runner import run_experiment


class RunLogTests(unittest.TestCase):
    def setUp(self):
        gate = patch("rsi.runner.functional_gate", return_value={"passed": True})
        gate.start()
        self.addCleanup(gate.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = load_config(ROOT / "configs/deepseek.yaml")
        self.config["wandb"] = {"enabled": False}
        self.task = {"id": "example-1", "benchmark": "swe", "prompt": "Fix parsing"}
        self.time = datetime(2026, 10, 19, 20, 0, 0, tzinfo=timezone(timedelta(hours=8)))

    def test_cli_automatically_names_repeated_runs_without_overwriting(self):
        with patch("rsi.cli.load_config", side_effect=lambda path: dict(self.config)), \
             patch("rsi.cli.load_tasks", return_value=[self.task]), \
             patch("rsi.swe.validate_swe_tasks"), \
             patch("rsi.cli.shutil.which", return_value="docker"), \
             patch("rsi.cli.os.getenv", return_value="configured"), \
             patch("rsi.run_logs._now", return_value=self.time), \
             patch("rsi.runner.run_group", return_value=({"success": True, "tokens": 10, "seconds": 1}, [])), \
             patch("rsi.runner.induce", return_value={"changes": {}, "memory_items": []}):
            for _ in range(2):
                main(["run", "--tasks", "unused.json", "--runs-dir", str(self.root),
                      "--fixed-skill", "--label", "第一轮 / 基线"])
        directories = sorted(path for path in self.root.iterdir() if path.is_dir())
        self.assertEqual(len(directories), 2)
        for directory in directories:
            self.assertIn("20261019-200000__第一轮-基线__swe-fixed__deepseek-flash__1tasks__v-", directory.name)
            info = read_json(directory / "run-info.json")
            self.assertEqual(info["status"], "completed")
            self.assertEqual(info["completed_tasks"], 1)
            self.assertEqual(info["timezone"], "Asia/Shanghai")
            self.assertTrue((directory / "summary.json").exists())
            self.assertTrue((directory / "metrics.json").exists())
            self.assertFalse(read_json(directory / "config.json")["wandb"]["enabled"])
        self.assertEqual(read_json(directories[0] / "run-info.json")["experiment_version"],
                         read_json(directories[1] / "run-info.json")["experiment_version"])
        self.assertTrue(directories[1].name.endswith("__02"))
        latest = read_json(self.root / "latest.json")["benchmark"]
        self.assertEqual(Path(latest["path"]), directories[1])

    def test_failure_is_visible_in_index_and_keeps_the_output(self):
        directory = self.root / "custom-name"
        with patch("rsi.runner.run_group", side_effect=RuntimeError("all rollouts failed")):
            with self.assertRaisesRegex(RuntimeError, "all rollouts failed"):
                run_experiment([self.task], self.config, directory)
        info = read_json(directory / "run-info.json")
        self.assertEqual(info["status"], "failed")
        self.assertIn("all rollouts failed", info["error"])
        self.assertIn("报错中止", (self.root / "INDEX.md").read_text())
        self.assertTrue((directory / "config.json").exists())
        with self.assertRaises(FileExistsError):
            run_experiment([self.task], self.config, directory)

    def test_legacy_runs_are_indexed_without_changing_existing_logs(self):
        legacy = self.root / "swe-12907"
        write_json(legacy / "config.json", self.config)
        write_json(legacy / "tasks.json", [self.task])
        write_json(legacy / "summary.json", [{"success": None}])
        score = self.root / "old-score-check" / "score.json"
        write_json(score, {"success": True, "report": "original-absolute-report-path"})
        before = score.read_bytes()
        entries = write_run_index(self.root)
        self.assertEqual({entry["kind"] for entry in entries}, {"benchmark", "score-check"})
        self.assertEqual(score.read_bytes(), before)
        self.assertFalse((legacy / "run-info.json").exists())
        index = (self.root / "INDEX.md").read_text()
        self.assertIn("评分未完成", index)
        self.assertIn("old-score-check/score.json", index)

    def test_latest_scoring_check_does_not_hide_the_latest_benchmark(self):
        benchmark = create_run_directory([self.task], self.config, runs_dir=self.root)
        score = create_run_directory([self.task], self.config, runs_dir=self.root, kind="score-check")
        update_run(score, status="completed")
        latest = read_json(self.root / "latest.json")
        self.assertEqual(Path(latest["benchmark"]["path"]), benchmark)
        self.assertEqual(Path(latest["score-check"]["path"]), score)
        self.assertEqual(latest["benchmark"]["mode"], "evolve")
        self.assertEqual(latest["score-check"]["status"], "completed")
        self.assertIsNone(read_json(score / "run-info.json")["trajectory_pattern"])

    def test_single_run_metadata_matches_trajectory_count(self):
        self.config.update(execution_mode="single", rollouts=1)
        directory = create_run_directory([self.task], self.config, runs_dir=self.root)
        info = read_json(directory / "run-info.json")
        self.assertEqual(info["execution_mode"], "single")
        self.assertEqual(info["rollouts"], 1)
        self.assertEqual(info["trajectory_pattern"], "<task_id>/rollout-<0..0>/trajectory.json")

    def test_finished_experiment_with_unknown_score_is_labelled_unscored(self):
        directory = self.root / "unscored-run"
        with patch("rsi.runner.run_group", return_value=({"success": None, "tokens": 10, "seconds": 1}, [])), \
             patch("rsi.runner.induce", return_value={"changes": {}, "memory_items": []}):
            run_experiment([self.task], self.config, directory)
        info = read_json(directory / "run-info.json")
        self.assertEqual(info["status"], "unscored")
        self.assertEqual(info["scored_tasks"], 0)
        self.assertEqual(info["completed_tasks"], 1)


if __name__ == "__main__":
    unittest.main()
