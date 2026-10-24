import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsi import ROOT
from rsi.baselines import MODES, set_execution_mode
from rsi.config import load_config
from rsi.cli import main
from rsi.common import file_map, read_json, write_json
from rsi.dataset_split import prior_exposure
from rsi.skills import load_initial_skill
from rsi.workflow import run_train_test, validate_train_test


def tasks(repo, cid):
    return [{"id": f"{cid}-{i}", "benchmark": "swe-evo", "prompt": f"Fix {repo} behavior {i}",
             "instance": {"instance_id": f"{cid}-{i}", "repo": repo,
                          "problem_statement": f"Fix {repo} behavior {i}"},
             "chain": {"id": cid, "index": i, "length": 2, "source_instance_id": f"source-{cid}-{i}"}}
            for i in range(2)]


class TrainTestWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = load_config(ROOT / "configs/qwen.yaml")
        self.config.update(evolve=True, wandb={"enabled": False})
        self.train, self.test = tasks("owner/train", "train-chain"), tasks("owner/test", "test-chain")
        self.guide = file_map(ROOT / "skills/test-and-diagnose")["SKILL.md"]
        self.learned = self.guide + "\nLearned during training: reproduce before editing.\n"
        self.seen = []
        self.proposals = 0
        self.scores = {}
        gate = patch("rsi.runner.functional_gate", return_value={"passed": True})
        gate.start()
        self.addCleanup(gate.stop)

    def group(self, task, folder, skill, config):
        self.seen.append({"task": task["id"], "guide": file_map(skill)["SKILL.md"], "version": skill.name,
                          "config": copy.deepcopy(config), "skill_path": skill})
        return {"task_id": task["id"], "success": self.scores.get(task["id"], True), "tokens": 2,
                "input_tokens": 1, "output_tokens": 1, "seconds": 1,
                "skill_version": skill.name, "chain": task["chain"]}, []

    def propose(self, *args, **kwargs):
        self.proposals += 1
        return {"changes": {"SKILL.md": self.learned} if self.proposals == 1 else {}}

    def test_training_automatically_runs_a_frozen_test_with_the_accepted_artifact(self):
        self.scores[self.train[0]["id"]] = False  # A wrong solution does not abort training.
        original_config = copy.deepcopy(self.config)
        output = self.root / "experiment"
        with patch("rsi.runner.run_group", side_effect=self.group), \
             patch("rsi.runner.induce", side_effect=self.propose) as updater:
            result = run_train_test(self.train, self.test, self.config, output)
        self.assertEqual(updater.call_count, len(self.train))
        self.assertEqual([x["task"] for x in self.seen], [t["id"] for t in self.train + self.test])
        self.assertEqual(self.seen[0]["guide"], self.guide)
        self.assertEqual([x["guide"] for x in self.seen[1:]], [self.learned] * 3)
        self.assertEqual([x["config"]["evolve"] for x in self.seen], [True, True, False, False])
        self.assertTrue(all(x["config"]["reflection_enabled"] for x in self.seen))
        trained = read_json(output / "train/skills/current.json")["version"]
        self.assertEqual({x["version"] for x in self.seen[2:]}, {trained})
        self.assertTrue(all(x["skill_path"].is_relative_to(output / "test") for x in self.seen[2:]))
        self.assertEqual(read_json(output / "test/run-info.json")["initial_skill_source"]["kind"], "version")
        self.assertEqual(self.config, original_config)
        self.assertEqual((result["status"], result["phase"]), ("completed", "finished"))
        self.assertEqual(result["train"]["metrics"]["resolved"], 1)
        self.assertEqual(result["test"]["metrics"]["resolved"], 2)
        self.assertFalse((output / "test/skills/history.jsonl").exists())
        info = read_json(output / "run-info.json")
        self.assertEqual((info["completed_tasks"], info["resolved"]), (4, 3))
        self.assertEqual(info["kind"], "train-test")
        self.assertEqual(info["current_skill_version"], trained)
        self.assertEqual(result, read_json(output / "workflow.json"))
        files, source = load_initial_skill(output)
        self.assertEqual(files["SKILL.md"], self.learned)
        self.assertEqual((source["kind"], source["version"]), ("workflow", trained))
        listed, executed = prior_exposure(self.root)
        self.assertEqual(listed, {t["chain"]["source_instance_id"] for t in self.train + self.test})
        self.assertEqual(executed, listed)
        self.assertIn("训练后冻结测试", (self.root / "INDEX.md").read_text(encoding="utf-8"))
        self.assertIn("训练／测试汇总", (self.root / "INDEX.md").read_text(encoding="utf-8"))

    def test_every_execution_mode_and_budget_is_preserved_between_phases(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                self.seen, self.proposals = [], 0
                config = copy.deepcopy(self.config)
                set_execution_mode(config, mode)
                with patch("rsi.runner.run_group", side_effect=self.group), \
                     patch("rsi.runner.induce", side_effect=self.propose):
                    run_train_test(self.train, self.test, config, self.root / mode)
                for row in self.seen:
                    for key in ("execution_mode", "rollouts", "model", "judge_model", "reflection_model",
                                "reflection_max_rounds", "step_limit", "reflection_step_limit"):
                        self.assertEqual(row["config"][key], config[key])
                self.assertEqual(self.proposals, 2)

    def test_training_failure_or_interrupt_skips_test_and_keeps_logs(self):
        for error, status in ((RuntimeError("training failed"), "failed"), (KeyboardInterrupt(), "interrupted")):
            with self.subTest(status=status):
                output = self.root / status
                with patch("rsi.runner.run_group", side_effect=error) as group, self.assertRaises(type(error)):
                    run_train_test(self.train, self.test, self.config, output)
                self.assertEqual(group.call_count, 1)
                state = read_json(output / "workflow.json")
                self.assertEqual(state["status"], status)
                self.assertEqual(state["train"]["status"], status)
                self.assertEqual(state["test"]["status"], "skipped")
                self.assertFalse((output / "test").exists())
                self.assertEqual(read_json(output / "train/run-info.json")["status"], status)
                self.assertEqual(read_json(output / "run-info.json")["status"], status)

    def test_partial_training_failure_preserves_completed_task_metrics(self):
        def group(task, folder, skill, config):
            if task["id"] == self.train[1]["id"]:
                raise RuntimeError("later training task failed")
            return self.group(task, folder, skill, config)

        output = self.root / "partial"
        with patch("rsi.runner.run_group", side_effect=group), \
             patch("rsi.runner.induce", side_effect=self.propose), self.assertRaises(RuntimeError):
            run_train_test(self.train, self.test, self.config, output)
        self.assertEqual(read_json(output / "workflow.json")["train"]["metrics"]["tasks"], 1)
        self.assertEqual(read_json(output / "run-info.json")["completed_tasks"], 1)
        self.assertFalse((output / "test").exists())

    def test_test_failure_keeps_the_finished_training_artifact(self):
        def group(task, folder, skill, config):
            if not config["evolve"]:
                raise RuntimeError("test failed")
            return self.group(task, folder, skill, config)

        output = self.root / "test-failure"
        with patch("rsi.runner.run_group", side_effect=group), \
             patch("rsi.runner.induce", side_effect=self.propose), self.assertRaisesRegex(RuntimeError, "test failed"):
            run_train_test(self.train, self.test, self.config, output)
        state = read_json(output / "workflow.json")
        self.assertEqual((state["train"]["status"], state["test"]["status"]), ("completed", "failed"))
        trained = read_json(output / "train/skills/current.json")["version"]
        self.assertEqual(file_map(output / "train/skills/versions" / trained)["SKILL.md"], self.learned)
        self.assertEqual(read_json(output / "train/run-info.json")["status"], "completed")

    def test_unknown_training_scores_are_reported_and_do_not_block_finished_trajectories(self):
        self.scores = {t["id"]: None for t in self.train}
        with patch("rsi.runner.run_group", side_effect=self.group), \
             patch("rsi.runner.induce", side_effect=self.propose):
            state = run_train_test(self.train, self.test, self.config, self.root / "unscored")
        self.assertEqual(state["status"], "unscored")
        self.assertEqual((state["train"]["status"], state["test"]["status"]), ("unscored", "completed"))
        self.assertEqual(state["train"]["metrics"]["scoring_coverage"], 0)

    def test_incomplete_training_return_does_not_start_test(self):
        output = self.root / "incomplete"
        with patch("rsi.workflow.run_experiment", return_value=[]) as run, \
             self.assertRaisesRegex(RuntimeError, "Training did not finish"):
            run_train_test(self.train, self.test, self.config, output)
        run.assert_called_once()
        self.assertEqual(read_json(output / "workflow.json")["test"]["status"], "skipped")

    def test_invalid_or_overlapping_splits_fail_before_creating_outputs(self):
        shared_repo, shared_prompt, shared_id, shared_source = (copy.deepcopy(self.test) for _ in range(4))
        shared_repo[0]["instance"]["repo"] = self.train[0]["instance"]["repo"]
        shared_prompt[0]["prompt"] = self.train[0]["prompt"]
        shared_id[0]["id"] = self.train[0]["id"]
        shared_source[0]["chain"]["source_instance_id"] = self.train[0]["chain"]["source_instance_id"]
        variants = (([], self.test), (self.train, []), (self.train[:1], self.test),
                    *((self.train, rows) for rows in (shared_repo, shared_prompt, shared_id, shared_source)))
        with patch("rsi.workflow.run_experiment") as run:
            for index, (train, test) in enumerate(variants):
                with self.subTest(index=index), self.assertRaises(ValueError):
                    run_train_test(train, test, self.config, self.root / f"bad-{index}")
                self.assertFalse((self.root / f"bad-{index}").exists())
        run.assert_not_called()

    def test_fixed_training_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "evolve=true"):
            run_train_test(self.train, self.test, {**self.config, "evolve": False}, self.root / "fixed")
        self.assertFalse((self.root / "fixed").exists())

    def test_current_dataset_passes_train_test_checks(self):
        path = ROOT / "tasks/splits/evoarena-frozen-v3"
        if not path.exists():
            self.skipTest("Local dataset not available")
        validate_train_test(read_json(path / "train.json"), read_json(path / "test.json"))

    def test_cli_connects_training_and_test_with_one_command(self):
        for name, rows in (("train", self.train), ("test", self.test)):
            write_json(self.root / f"{name}.json", rows)
        output = self.root / "cli"
        with patch("rsi.cli.require_credentials"), patch("rsi.cli.shutil.which", return_value="docker"), \
             patch("rsi.swe.validate_swe_tasks"), patch("rsi.evo_swe.validate_tasks") as validate, \
             patch("rsi.runner.run_group", side_effect=self.group), \
             patch("rsi.runner.induce", side_effect=self.propose):
            main(["run", "--tasks", str(self.root / "train.json"), "--test-tasks", str(self.root / "test.json"),
                  "--execution-mode", "single", "--output", str(output)])
        self.assertEqual(len(validate.call_args.args[0]), 4)
        self.assertEqual(read_json(output / "workflow.json")["status"], "completed")
        self.assertTrue(read_json(output / "train/config.json")["evolve"])
        self.assertFalse(read_json(output / "test/config.json")["evolve"])
        self.assertEqual(read_json(output / "test/config.json")["execution_mode"], "single")

    def test_cli_validates_test_environments_before_training(self):
        with patch("rsi.cli.require_credentials"), patch("rsi.cli.shutil.which", return_value="docker"), \
             patch("rsi.cli.load_tasks", side_effect=[self.train, self.test]), \
             patch("rsi.evo_swe.validate_tasks", side_effect=RuntimeError("Missing test image")), \
             patch("rsi.workflow.run_train_test") as run, \
             self.assertRaisesRegex(RuntimeError, "Missing test image"):
            main(["run", "--tasks", "train.json", "--test-tasks", "test.json"])
        run.assert_not_called()

    def test_cli_rejects_fixed_training_with_automatic_test(self):
        with patch("rsi.cli.require_credentials") as credentials, self.assertRaises(SystemExit):
            main(["run", "--tasks", "train.json", "--test-tasks", "test.json", "--fixed-skill"])
        credentials.assert_not_called()


if __name__ == "__main__":
    unittest.main()
