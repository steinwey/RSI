import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsi import ROOT
from rsi.cli import load_config, main
from rsi.common import write_json


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
             patch("rsi.cli.validate_swe_tasks"), \
             patch("rsi.runner.run_experiment", return_value=[]) as run, \
             contextlib.redirect_stdout(io.StringIO()):
            main(["run", "--config", str(self.path), "--tasks", "unused.json", *flags])
        return run.call_args.args[2], load_tasks

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


if __name__ == "__main__":
    unittest.main()
