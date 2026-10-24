import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rsi import ROOT
from rsi.config import load_config, select_model
from rsi.cli import main
from rsi.common import write_json
from rsi.model import Model, prepare_request, request_options
from rsi.tracking import public_config


def _response(content="```bash\necho ok\n```", prompt=3, completion=2, cost=None):
    usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, cost=cost)
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=message)])


class OpenCodeConfigTests(unittest.TestCase):
    def test_opencode_config_selects_deepseek_with_thinking_disabled(self):
        config = load_config(ROOT / "configs/opencode.yaml")
        self.assertEqual(config["allowed_models"], ["deepseek-v4.1-flash", "glm-5.3-flash"])
        self.assertEqual(config["model"]["name"], "deepseek-v4.1-flash")
        self.assertEqual(config["model"]["reasoning_effort"], "none")
        self.assertEqual(config["model"]["extra_body"]["thinking"]["type"], "disabled")
        self.assertEqual(config["judge_model"]["reasoning_effort"], "none")
        self.assertEqual(config["model"]["custom_llm_provider"], "openai")
        self.assertEqual(config["model"]["api_base"], "https://opencode.ai/inference/openai/v1")
        self.assertEqual(config["model"]["api_key_env"], "OPENCODE_API_KEY")
        self.assertNotIn("api_key", config["model"])
        self.assertNotIn("api_key", config["judge_model"])
        select_model(config, "glm-5.3-flash")
        self.assertEqual(config["model"]["name"], "glm-5.3-flash")
        self.assertEqual(config["judge_model"]["name"], "glm-5.3-flash")
        with self.assertRaisesRegex(ValueError, "not one of"):
            select_model(config, "glm-5.3-flash-free")

    def test_saved_config_rejects_secrets_and_model_fallbacks(self):
        source = load_config(ROOT / "configs/opencode.yaml")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.yaml"
            write_json(path, {**source, "model": {**source["model"], "fallbacks": ["deepseek-v4.1-flash"]}})
            with self.assertRaisesRegex(ValueError, "fallback"):
                load_config(path)
            write_json(path, {**source, "model": {**source["model"], "api_key": "secret-key"}})
            with self.assertRaisesRegex(ValueError, "environment variables"):
                load_config(path)
            write_json(path, {**source, "judge_model": {**source["judge_model"], "name": "other-model"}})
            with self.assertRaisesRegex(ValueError, "allowed_models"):
                load_config(path)

    def test_deepseek_config_is_unchanged(self):
        config = load_config(ROOT / "configs/deepseek.yaml")
        self.assertEqual(config["model"]["name"], "deepseek/deepseek-flash")
        self.assertNotIn("allowed_models", config)
        self.assertNotIn("api_base", config["model"])

    def test_cli_model_flag_switches_solver_and_judge_before_the_run(self):
        with patch("rsi.cli.os.getenv", side_effect=lambda key: "configured" if key == "OPENCODE_API_KEY" else None), \
             patch("rsi.cli.shutil.which", return_value="docker"), \
             patch("rsi.cli.load_tasks", return_value=[]), \
             patch("rsi.swe.validate_swe_tasks"), \
             patch("rsi.runner.run_experiment", return_value=[]) as run, \
             contextlib.redirect_stdout(io.StringIO()):
            main(["run", "--config", str(ROOT / "configs/opencode.yaml"), "--tasks", "unused.json",
                  "--model", "glm-5.3-flash"])
        config = run.call_args.args[1]
        self.assertEqual(config["model"]["name"], "glm-5.3-flash")
        self.assertEqual(config["judge_model"]["name"], "glm-5.3-flash")
        self.assertEqual(config["model"]["api_base"], "https://opencode.ai/inference/openai/v1")

    def test_cli_rejects_a_model_outside_the_configured_list(self):
        with patch("rsi.cli.load_tasks") as load_tasks, \
             patch("rsi.runner.run_experiment") as run, \
             contextlib.redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit) as raised:
                main(["run", "--config", str(ROOT / "configs/opencode.yaml"), "--tasks", "unused.json",
                      "--model", "glm-5.3-flash-free"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("not one of", errors.getvalue())
        load_tasks.assert_not_called()
        run.assert_not_called()


class OpenCodeCallTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/opencode.yaml")["model"]

    def test_completion_uses_the_openai_compatible_endpoint_and_records_usage(self):
        model = Model(dict(self.config))
        with patch.dict("os.environ", {"OPENCODE_API_KEY": "test-key"}, clear=False), \
             patch("litellm.completion", return_value=_response()) as completion, \
             patch("litellm.completion_cost", return_value=0.02):
            content = model.query([{"role": "user", "content": "Print ok."}])
        kwargs = completion.call_args.kwargs
        self.assertEqual(kwargs["model"], "deepseek-v4.1-flash")
        self.assertEqual(kwargs["custom_llm_provider"], "openai")
        self.assertEqual(kwargs["api_base"], "https://opencode.ai/inference/openai/v1")
        self.assertEqual(kwargs["api_key"], "test-key")
        self.assertEqual(kwargs["reasoning_effort"], "none")
        self.assertEqual(kwargs["extra_body"]["thinking"], {"type": "disabled"})
        self.assertEqual(kwargs["extra_body"]["reasoning_effort"], "none")
        self.assertNotIn("fallbacks", kwargs)
        self.assertNotIn("tools", kwargs)
        self.assertNotIn("api_key", model.options)
        self.assertEqual(content, {"content": "```bash\necho ok\n```"})
        stats = model.stats()
        self.assertEqual(stats["provider"], "opencode-zen")
        self.assertEqual(stats["model"], "deepseek-v4.1-flash")
        self.assertEqual(stats["reasoning_tokens"], 0)
        self.assertEqual((stats["input_tokens"], stats["output_tokens"], stats["cost_usd"]), (3, 2, 0.02))
        self.assertNotIn("test-key", json.dumps(stats))
        from minisweagent.agents.default import DefaultAgent

        class Env:
            def get_template_vars(self):
                return {}

        self.assertEqual(DefaultAgent(model, Env()).parse_action(content)["action"], "echo ok")

    def test_missing_pricing_keeps_a_provider_reported_cost(self):
        model = Model(dict(self.config))
        with patch.dict("os.environ", {"OPENCODE_API_KEY": "test-key"}, clear=False), \
             patch("litellm.completion", return_value=_response(cost=0.004)), \
             patch("litellm.completion_cost", side_effect=Exception("unpriced")):
            model.query([{"role": "user", "content": "hi"}])
        self.assertEqual(model.stats()["cost_usd"], 0.004)

    def test_unknown_cost_is_recorded_as_unknown(self):
        model = Model(dict(self.config))
        with patch.dict("os.environ", {"OPENCODE_API_KEY": "test-key"}, clear=False), \
             patch("litellm.completion", return_value=_response()), \
             patch("litellm.completion_cost", side_effect=Exception("unpriced")):
            model.query([{"role": "user", "content": "hi"}])
        self.assertIsNone(model.stats()["cost_usd"])

    def test_glm_on_opencode_does_not_keep_the_deepseek_thinking_switch(self):
        options = dict(self.config)
        options["name"] = "glm-5.3-flash"
        model = Model(options)
        with patch.dict("os.environ", {"OPENCODE_API_KEY": "test-key"}, clear=False), \
             patch("litellm.completion", return_value=_response(content="ok")) as completion, \
             patch("litellm.completion_cost", return_value=0):
            model.query([{"role": "user", "content": "hi"}])
        kwargs = completion.call_args.kwargs
        self.assertEqual(kwargs["model"], "glm-5.3-flash")
        self.assertNotIn("reasoning_effort", kwargs)
        self.assertNotIn("thinking", kwargs.get("extra_body") or {})

    def test_evoarena_uses_qwen_token_plan_without_thinking(self):
        config = load_config(ROOT / "configs/evoarena.yaml")
        self.assertEqual(config["model"]["name"], "qwen3.8-max")
        self.assertEqual(config["judge_model"]["name"], "qwen3.8-max")
        with patch.dict("os.environ", {"QWEN_TOKEN_PLAN_API_KEY": "test-key"}, clear=False):
            prepared = prepare_request(config["judge_model"]["name"], config["judge_model"])
        self.assertEqual(prepared["api_base"], "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
        self.assertFalse(prepared["extra_body"]["enable_thinking"])

    def test_deepseek_requests_do_not_gain_opencode_settings(self):
        model = Model(load_config(ROOT / "configs/deepseek.yaml")["model"])
        with patch("litellm.completion", return_value=_response(content="ok")) as completion, \
             patch("litellm.completion_cost", return_value=0.01):
            model.query([{"role": "user", "content": "hi"}])
        kwargs = completion.call_args.kwargs
        self.assertEqual(kwargs["model"], "deepseek/deepseek-flash")
        self.assertNotIn("api_key", kwargs)
        self.assertNotIn("api_base", kwargs)
        self.assertNotIn("custom_llm_provider", kwargs)
        self.assertNotIn("reasoning_effort", kwargs)
        self.assertEqual(kwargs["extra_body"]["thinking"]["type"], "disabled")
        self.assertEqual(model.stats()["provider"], "deepseek")

    def test_request_options_refuse_a_fallback_and_a_missing_key(self):
        with self.assertRaisesRegex(ValueError, "fallback"):
            request_options({"fallbacks": ["glm-5.3-flash"]})
        env = {key: value for key, value in os.environ.items() if key != "OPENCODE_API_KEY"}
        with patch.dict("os.environ", env, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENCODE_API_KEY"):
                request_options({"api_key_env": "OPENCODE_API_KEY"})

    def test_published_config_names_the_provider_without_the_secret(self):
        config = load_config(ROOT / "configs/opencode.yaml")
        config["model"]["api_key"] = "secret-key"
        published = json.dumps(public_config(config))
        self.assertIn("opencode-zen", published)
        self.assertIn("deepseek-v4.1-flash", published)
        self.assertIn("none", published)
        self.assertIn("disabled", published)
        self.assertIn("https://opencode.ai/inference/openai/v1", published)
        self.assertNotIn("secret-key", published)


if __name__ == "__main__":
    unittest.main()
