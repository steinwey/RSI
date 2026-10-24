"""Verify the real LiteLLM/OpenAI serialization against an in-memory HTTP transport."""
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import OpenAI

from rsi import ROOT
from rsi.config import load_config
from rsi.model import Model, object_messages, probe_model


BASE = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"


class QwenInterfaceTests(unittest.TestCase):
    def test_all_active_configs_route_every_role_to_qwen(self):
        for filename in ("qwen.yaml", "evoarena.yaml", "baseline-single.yaml", "baseline-serial_shared.yaml"):
            config = load_config(ROOT / "configs" / filename)
            self.assertFalse(config["planner_enabled"])
            for role in ("model", "judge_model", "reflection_model"):
                with self.subTest(config=filename, role=role):
                    block = config[role]
                    self.assertEqual(block["name"], "qwen3.8-max")
                    self.assertEqual(block["custom_llm_provider"], "openai")
                    self.assertEqual(block["api_base"], BASE)
                    self.assertEqual(block["api_key_env"], "QWEN_TOKEN_PLAN_API_KEY")
                    self.assertIs(block["extra_body"]["enable_thinking"], False)

    def test_real_sdk_sends_correct_endpoint_auth_and_body_for_each_role(self):
        config = load_config(ROOT / "configs/qwen.yaml")
        for role in ("model", "judge_model", "reflection_model"):
            with self.subTest(role=role):
                received = []
                content = "```bash\necho ok\n```" if role == "model" else '{"stop": true, "reason": "done", "feedback": ""}'

                def respond(request):
                    received.append(request)
                    return httpx.Response(200, json={
                        "id": "chatcmpl-mocked", "object": "chat.completion", "created": 1,
                        "model": "qwen3.8-max",
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": content}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    })

                with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
                    client = OpenAI(api_key="wire-test-key", base_url=BASE, http_client=http_client)
                    with patch.dict(os.environ, {"QWEN_TOKEN_PLAN_API_KEY": "wire-test-key",
                                                  "DEEPSEEK_API_KEY": "unused-deepseek-key",
                                                  "OPENAI_API_KEY": "unused-openai-key"}), \
                         patch("rsi.model.recorded_cost", return_value=None):
                        model = Model(config[role])
                        messages = [{"role": "user", "content": "Print ok."}] if role == "model" else object_messages("Review", {})
                        response = model.query(messages, client=client, timeout=2, num_retries=0)
                self.assertEqual(len(received), 1)
                request = received[0]
                self.assertEqual(str(request.url), BASE + "/chat/completions")
                self.assertEqual(request.method, "POST")
                self.assertEqual(request.headers["authorization"], "Bearer wire-test-key")
                body = json.loads(request.content)
                self.assertEqual(body["model"], "qwen3.8-max")
                self.assertIs(body["enable_thinking"], False)
                self.assertNotIn("extra_body", body)
                self.assertNotIn("api_key_env", body)
                self.assertNotIn("provider", body)
                self.assertNotIn("tools", body)
                self.assertEqual(body["max_tokens"], config[role]["max_tokens"])
                self.assertEqual(response["content"], content)
                self.assertEqual((model.input_tokens, model.output_tokens), (10, 5))
                self.assertEqual(model.last_finish_reason, "stop")

    def test_probe_does_not_claim_thinking_disabled_from_missing_fields(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
                content="```bash\necho ok\n```", tool_calls=None))])
        config = load_config(ROOT / "configs/qwen.yaml")
        with patch.dict(os.environ, {"QWEN_TOKEN_PLAN_API_KEY": "probe-test-key"}), \
             patch("litellm.completion", return_value=response), \
             patch("rsi.model.recorded_cost", return_value=None):
            result = probe_model(config)
        self.assertTrue(result["thinking_disable_requested"])
        self.assertFalse(result["reasoning_observed"])
        self.assertIsNone(result["thinking_disabled"])
        self.assertTrue(result["bash_parse_ok"])

    def test_probe_exposes_reasoning_and_truncated_empty_output(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=64,
                                  completion_tokens_details={"reasoning_tokens": 64}),
            choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(
                content=None, reasoning_content="thinking", tool_calls=None))])
        config = load_config(ROOT / "configs/qwen.yaml")
        with patch.dict(os.environ, {"QWEN_TOKEN_PLAN_API_KEY": "probe-test-key"}), \
             patch("litellm.completion", return_value=response), \
             patch("rsi.model.recorded_cost", return_value=None):
            result = probe_model(config)
        self.assertTrue(result["reasoning_observed"])
        self.assertFalse(result["thinking_disabled"])
        self.assertEqual(result["finish_reason"], "length")
        self.assertFalse(result["bash_parse_ok"])


if __name__ == "__main__":
    unittest.main()
