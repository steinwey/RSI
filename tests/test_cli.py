import contextlib
import io
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rsi import ROOT
from rsi.config import load_config
from rsi.cli import main
from rsi.common import read_json, write_json
from rsi.config import require_credentials
from rsi.tasks import load_tasks, validate_environments


class CLIUploadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        config = load_config(ROOT / "configs/deepseek.yaml")
        # A legacy config must not opt a new CLI invocation into uploads.
        config["wandb"] = {"enabled": True, "project": "custom-project", "entity": "custom-entity"}
        self.path = self.root / "config.yaml"
        write_json(self.path, config)

    def invoke(self, flags, *, wandb_key="configured"):
        env = {"DEEPSEEK_API_KEY": "configured", "WANDB_API_KEY": wandb_key}
        with patch("rsi.cli.os.getenv", side_effect=lambda key: env.get(key)), \
             patch("rsi.cli.shutil.which", return_value="docker"), \
             patch("rsi.cli.load_tasks", return_value=[]) as load_tasks, \
             patch("rsi.swe.validate_swe_tasks"), \
             patch("rsi.runner.run_experiment", return_value=[]) as run, \
             contextlib.redirect_stdout(io.StringIO()):
            main(["run", "--config", str(self.path), "--tasks", "unused.json", *flags])
        return run.call_args.args[1], load_tasks

    def test_upload_is_disabled_by_default_despite_legacy_yaml_and_existing_key(self):
        config, _ = self.invoke([])
        self.assertFalse(config["wandb"]["enabled"])
        self.assertEqual(config["wandb"]["project"], "custom-project")

    def test_wandb_flag_enables_upload_and_preserves_the_destination(self):
        config, _ = self.invoke(["--wandb"])
        self.assertTrue(config["wandb"]["enabled"])
        self.assertEqual(config["wandb"]["entity"], "custom-entity")

    def test_no_wandb_flag_runs_without_a_wandb_key(self):
        config, _ = self.invoke(["--no-wandb"], wandb_key=None)
        self.assertFalse(config["wandb"]["enabled"])

    def test_explicit_upload_without_a_key_fails_before_loading_tasks_or_running(self):
        with patch("rsi.cli.os.getenv", return_value=None), \
             patch("rsi.cli.load_tasks") as load_tasks, \
             patch("rsi.runner.run_experiment") as run, \
             contextlib.redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit) as raised:
                main(["run", "--config", str(self.path), "--tasks", "unused.json", "--wandb"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--wandb requires WANDB_API_KEY", errors.getvalue())
        load_tasks.assert_not_called()
        run.assert_not_called()


class CLICleanupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = load_config(ROOT / "configs/qwen.yaml")
        self.path = self.root / "config.yaml"
        write_json(self.path, self.config)
        env = patch("rsi.cli.load_env")
        env.start()
        self.addCleanup(env.stop)

    def test_removed_and_abbreviated_options_are_rejected_before_loading_settings(self):
        variants = (["--validation", "unused.json"], ["--init", "skill"],
                    ["--fixed-skill", "--test-tasks", "test.json"],
                    ["--output", "one", "--runs-dir", "two"])
        with patch("rsi.cli.load_config") as config:
            for flags in variants:
                with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(["run", "--tasks", "train.json", *flags])
        config.assert_not_called()

    def test_invalid_dataset_selection_is_rejected_before_any_dataset_download(self):
        variants = (["--limit", "0"], ["--offset", "-1"], ["--ids", "a", "--offset", "1"])
        with patch("datasets.load_dataset") as download:
            for flags in variants:
                with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(["prepare-swe", "--output", str(self.root / "tasks.json"), *flags])
        download.assert_not_called()

    def test_reference_check_validates_metadata_and_images_once(self):
        rows = [{"id": f"evo-{i}", "benchmark": "swe-evo",
                 "chain": {"id": "chain-a", "index": i, "length": 2, "source_instance_id": f"source-{i}"}}
                for i in range(2)]
        with patch("rsi.cli.load_tasks", return_value=rows), \
             patch("rsi.evo_swe.validate_tasks") as validate, \
             patch("rsi.evo_swe.verify_references") as verify, \
             patch("rsi.cli.require_credentials", side_effect=AssertionError("Offline command needs no model key")):
            main(["verify-evo-swe", "--tasks", "unused.json", "--limit", "1"])
        validate.assert_called_once()
        self.assertEqual(validate.call_args.args[0], rows[:1])
        self.assertTrue(validate.call_args.kwargs["require_images"])
        verify.assert_called_once()

    def test_prepare_swe_omits_oracles_and_does_not_require_model_credentials(self):
        row = {"instance_id": "owner__repo-1", "repo": "owner/repo", "version": "1",
               "base_commit": "a" * 40, "problem_statement": "Fix behavior", "patch": "SECRET_ORACLE"}

        class Dataset(list):
            _fingerprint = "source-fingerprint"

        output = self.root / "tasks.json"
        with patch("datasets.load_dataset", return_value=Dataset([row])), \
             patch("rsi.cli.require_credentials", side_effect=AssertionError("Offline command needs no model key")):
            main(["prepare-swe", "--ids", row["instance_id"], "--output", str(output)])
        result = read_json(output)
        self.assertEqual(result[0]["id"], row["instance_id"])
        self.assertNotIn("SECRET_ORACLE", json.dumps(result))

    def test_doctor_reports_only_configured_active_provider_credentials(self):
        with patch("rsi.cli.shutil.which", return_value=None), \
             patch("rsi.cli.subprocess.run", return_value=SimpleNamespace(returncode=1, stdout="")), \
             patch("rsi.cli.os.getenv", return_value="configured"), \
             contextlib.redirect_stdout(io.StringIO()) as stream:
            main(["doctor"])
        result = json.loads(stream.getvalue())
        self.assertEqual(result["credentials"], {"QWEN_TOKEN_PLAN_API_KEY": True})
        self.assertNotIn("deepseek_key_set", result)
        self.assertNotIn("opencode_key_set", result)
        self.assertNotIn("planner_model", result["models"])

    def test_probe_requires_only_the_solver_key_and_no_docker(self):
        config = copy.deepcopy(self.config)
        config["judge_model"]["api_key_env"] = "UNUSED_JUDGE_KEY"
        config["reflection_model"]["api_key_env"] = "UNUSED_REVIEW_KEY"
        write_json(self.path, config)
        with patch("rsi.config.os.getenv", side_effect=lambda name: "configured" if name == "QWEN_TOKEN_PLAN_API_KEY" else None), \
             patch("rsi.cli.shutil.which", side_effect=AssertionError("Probe does not need Docker")), \
             patch("rsi.model.probe_model", return_value={"ok": True}) as probe, \
             contextlib.redirect_stdout(io.StringIO()):
            main(["probe-model", "--config", str(self.path)])
        probe.assert_called_once()

    def test_fixed_single_mode_requires_only_roles_it_uses(self):
        config = copy.deepcopy(self.config)
        config.update(execution_mode="single", rollouts=1, evolve=False, reflection_enabled=False)
        config["judge_model"]["api_key_env"] = "UNUSED_JUDGE_KEY"
        with patch("rsi.config.os.getenv", side_effect=lambda name: "configured" if name == "QWEN_TOKEN_PLAN_API_KEY" else None):
            require_credentials(config)
            config["evolve"] = True
            with self.assertRaisesRegex(RuntimeError, "UNUSED_JUDGE_KEY"):
                require_credentials(config)

    def test_empty_or_malformed_task_files_are_rejected(self):
        path = self.root / "tasks.json"
        for rows in ([], {}, None, [True], [{"id": 1}]):
            write_json(path, rows)
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                load_tasks(path, self.config)

    def test_environment_checks_dispatch_only_to_the_relevant_benchmark(self):
        with patch("rsi.evo_swe.validate_tasks") as evo, patch("rsi.swe.validate_swe_tasks") as swe:
            validate_environments([{"benchmark": "terminal"}], self.config)
            evo.assert_not_called()
            swe.assert_not_called()
            validate_environments([{"benchmark": "swe-evo"}], self.config)
            evo.assert_called_once()
            swe.assert_not_called()
            validate_environments([{"benchmark": "swe"}], self.config)
            swe.assert_called_once()
            self.assertEqual(evo.call_count, 1)


if __name__ == "__main__":
    unittest.main()
