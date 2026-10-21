import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsi.command_logs import RecordingEnvironment
from rsi.common import read_json
from rsi.evidence import archive_request, build_evidence, serialized, sha
from rsi.model import Model, object_messages
from rsi.runner import induce, solver_step_limit_note
from rsi import ROOT


def rollout(messages, artifact="", index=0):
    return {"rollout": index, "status": "Submitted", "error": None,
            "messages": messages, "artifact": artifact, "score": "SECRET_VERIFIER"}


class EvidenceTests(unittest.TestCase):
    def test_steps_spans_budgets_and_unmodified_sources(self):
        messages = [{"role": "system", "content": "system"*2000},
                    {"role": "user", "content": "issue"*3000}]
        for i in range(15):
            messages += [{"role": "assistant", "content": f"step {i}\n```bash\necho {i}\n```"},
                         {"role": "user", "content": "0123456789\n"*800}]
        messages[15]["content"] = "<returncode>1</returncode>\nTraceback\nValueError: bad value"
        messages[-1]["content"] = "<returncode>0</returncode>\n3 passed"
        source = rollout(messages)
        before = copy.deepcopy(source)
        for purpose in ("selector", "updater"):
            bundle = build_evidence([source, {**source, "rollout": 1}], 30, purpose)
            self.assertEqual(len(bundle["shared_context"]), 1)
            self.assertNotIn("SECRET_VERIFIER", serialized(bundle))
            evidence = bundle["trajectories"][0]["messages"]
            self.assertEqual(bundle["trajectories"][0]["steps_omitted"], 0)
            self.assertEqual([m["message"] for m in evidence], list(range(2, len(messages))))
            self.assertEqual([m["step"] for m in evidence], [i//2 for i in range(30)])
            for m in evidence:
                raw = messages[m["message"]]["content"]
                previous = 0
                for a, b, text in m["spans"]:
                    self.assertGreaterEqual(a, previous)
                    self.assertEqual(text, raw[a:b])
                    previous = b
                self.assertEqual(m["omitted"], len(raw)-sum(b-a for a,b,_ in m["spans"]))
            self.assertEqual(next(m for m in evidence if m["message"] == 15)["omitted"], 0)
            self.assertEqual(evidence[-1]["omitted"], 0)
        self.assertEqual(source, before)

    def test_patch_all_files_and_hunks_keep_exact_offsets(self):
        artifact = "".join(f"diff --git a/f{i} b/f{i}\n--- a/f{i}\n+++ b/f{i}\n"
                           + "@@ -1 +1 @@\n" + "+added line\n"*1000 for i in range(3))
        units = build_evidence([rollout([], artifact)], 30)["trajectories"][0]["artifact"]
        self.assertEqual(len(units), 6)
        self.assertEqual("".join(text for unit in units for _, _, text in unit["spans"]), artifact)
        for unit in units:
            self.assertTrue(unit["spans"])
            for a, b, text in unit["spans"]:
                self.assertEqual(artifact[unit["offset"]+a:unit["offset"]+b], text)

    def test_step_cap_keeps_the_latest_steps_verbatim(self):
        messages = []
        for i in range(31):
            messages += [{"role": "assistant", "content": f"step-{i}-" + "x"*50},
                         {"role": "user", "content": f"obs-{i}"}]
        route = build_evidence([rollout(messages, "patch")], 30)["trajectories"][0]
        self.assertEqual(route["steps_total"], 31)
        self.assertEqual(route["steps_omitted"], 1)
        self.assertEqual(route["messages"][0]["step"], 1)
        self.assertEqual(route["messages"][-1]["step"], 30)
        self.assertNotIn("step-0-", serialized(route["messages"]))
        self.assertTrue(all(m["omitted"] == 0 for m in route["messages"]))
        self.assertEqual(route["artifact"][0]["spans"][0][2], "patch")
        with self.assertRaisesRegex(ValueError, "step_limit"):
            build_evidence([rollout(messages)], 0)

    def test_archive_is_actual_model_request_and_allowlist(self):
        source = rollout([{"role": "assistant", "content": "act"}])
        data = {"task": "fix", **build_evidence([source], 30)}
        with tempfile.TemporaryDirectory() as temp:
            archive_request(temp, "selector", "choose", data, [source], 30)
            model = Model({"name": "mock"})
            with patch.object(model, "query", return_value={"content": "{}"}) as query:
                model.object("choose", data)
            saved = read_json(Path(temp)/"evidence/selector.json")
            self.assertEqual(query.call_args.args[0], saved)
            manifest = read_json(Path(temp)/"evidence/manifest.json")["requests"]["selector"]
            self.assertEqual(manifest["request_sha256"], sha(serialized(saved)))
            snapshot = read_json(Path(temp)/"evidence/sources/rollout-0.json")
            self.assertNotIn("SECRET_VERIFIER", serialized(snapshot))
            self.assertEqual(manifest["sources"][0]["message_sha256"], [sha("act")])

    def test_judge_retries_until_the_response_is_json(self):
        model = Model({"name": "mock"})
        replies = iter([{"content": '{"rollout": 1, "reason": "unterminated'}, {"content": "not-json"},
                         {"content": '{"rollout": 2, "reason": "visible tests"}'}])
        with patch.object(model, "query", side_effect=lambda messages: next(replies)) as query:
            self.assertEqual(model.object("choose", {"task": "fix"}), {"rollout": 2, "reason": "visible tests"})
        self.assertEqual(query.call_count, 3)
        with patch.object(model, "query", return_value={"content": "still bad"}):
            with self.assertRaisesRegex(ValueError, "invalid JSON after 3 attempts"):
                model.object("choose", {"task": "fix"})

    def test_updater_archive_exists_before_failed_provider_call(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            class FailureModel:
                def __init__(self, config):
                    pass
                def object(self, instruction, data):
                    self_request = read_json(folder/"evidence/updater.json")
                    if self_request != object_messages(instruction, data):
                        raise AssertionError("request differs")
                    raise RuntimeError("provider unavailable")
            with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                induce({"prompt": "fix"}, [rollout([])], ROOT/"skills/test-and-diagnose",
                       {"judge_model": {}, "step_limit": 30}, FailureModel, folder=folder)
            saved = serialized(read_json(folder/"evidence/updater.json"))
            self.assertIn("hard limit of 30 steps", solver_step_limit_note(30))
            self.assertIn("at most 30 steps", saved)
            self.assertIn("At most 30 steps are included", saved)
            self.assertTrue((folder/"evidence/manifest.json").exists())

    def test_raw_output_and_timeout_bytes_saved_without_changing_result(self):
        class Env:
            def execute(self, command, cwd=""):
                if command == "timeout":
                    raise subprocess.TimeoutExpired(command, 1, output=b"\xffraw\r\n", stderr=b"err")
                return result
        result = {"output": "A"*20000 + "\r\nEND", "returncode": 2}
        with tempfile.TemporaryDirectory() as temp:
            env = RecordingEnvironment(Env(), temp, lambda: 8)
            self.assertIs(env.execute("long", cwd="/repo"), result)
            raw = (Path(temp)/"commands/000001.output.txt").read_bytes()
            self.assertEqual(raw, result["output"].encode())
            metadata = read_json(Path(temp)/"commands/000001.json")
            self.assertEqual(metadata["assistant_message"], 8)
            self.assertEqual(metadata["output"]["sha256"], hashlib.sha256(raw).hexdigest())
            with self.assertRaises(subprocess.TimeoutExpired):
                env.execute("timeout")
            self.assertEqual((Path(temp)/"commands/000002.output.bin").read_bytes(), b"\xffraw\r\n")

    def test_command_capture_precedes_upstream_visible_truncation(self):
        import yaml
        from minisweagent.agents.default import DefaultAgent
        from rsi import UPSTREAM
        raw = "HEAD\n" + "x"*10000 + "MIDDLE_EVIDENCE" + "y"*10000 + "\nTAIL"
        class Env:
            def execute(self, command, cwd=""):
                return {"output": raw, "returncode": 0}
            def get_template_vars(self):
                return {}
        class ModelStub:
            def get_template_vars(self):
                return {}
        with tempfile.TemporaryDirectory() as temp:
            settings = yaml.safe_load((UPSTREAM/"third_party/src/minisweagent/config/extra/swebench.yaml").read_text(encoding="utf-8"))["agent"]
            env = RecordingEnvironment(Env(), temp, lambda: 0)
            agent = DefaultAgent(ModelStub(), env, **settings)
            agent.get_observation({"content": "```bash\ncat file\n```"})
            self.assertNotIn("MIDDLE_EVIDENCE", agent.messages[-1]["content"])
            self.assertEqual((Path(temp)/"commands/000001.output.txt").read_bytes(), raw.encode())


if __name__ == "__main__":
    unittest.main()
