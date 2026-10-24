import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from rsi import ROOT
from rsi.config import load_config
from rsi.common import file_map, write_json
from rsi.runner import run_group, run_experiment
from rsi.evidence import public_rollout
from rsi.skills import SkillStore, functional_gate
from rsi.tracking import public_config, task_metrics


class FakeModel:
    selection = 3
    def __init__(self, config):
        self.cost = self.n_calls = 0
        self.messages = []
    def query(self, messages):
        self.n_calls += 1
        self.messages = messages
        return {"content": "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```"}
    def get_template_vars(self):
        return {}
    def stats(self):
        return {"calls": self.n_calls, "input_tokens": 10, "output_tokens": 5, "cost_usd": 0}
    def object(self, instruction, data):
        assert "SECRET_VERIFIER" not in json.dumps(data)
        assert all("score" not in r and "success" not in r for r in data["trajectories"])
        return {"rollout": self.selection, "reason": "visible evidence"}


class FakeEnvironment:
    def execute(self, command, cwd=""):
        return {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n" if command.startswith("echo ") else "", "returncode": 0}
    def get_template_vars(self):
        return {}


class FakeBackend:
    instances = []
    barrier = None
    fail_index = None
    def __init__(self, task, folder, config):
        self.env = FakeEnvironment()
        self.folder = folder
        self.index = int(folder.name.split("-")[-1])
        self.closed = self.scored = False
        self.instances.append(self)
        if self.barrier:
            self.barrier.wait(timeout=10)
    def artifact(self):
        if self.index == self.fail_index:
            raise RuntimeError("injected rollout failure")
        return f"patch-{self.index}"
    def score(self, artifact):
        assert (self.folder.parent / "selection.json").exists()
        self.scored = True
        return {"success": True, "hidden": "SECRET_VERIFIER"}
    def close(self):
        self.closed = True


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = load_config(ROOT / "configs/deepseek.yaml")
        self.config["wandb"] = {**self.config.get("wandb", {}), "enabled": False}
        self.store = SkillStore(self.root / "skills", ROOT / "skills/test-and-diagnose")
        FakeBackend.instances = []
        FakeBackend.barrier = threading.Barrier(5)
        FakeBackend.fail_index = None
        FakeModel.selection = 3
        self.task = {"id": "example-1", "benchmark": "swe", "prompt": "Fix parsing"}
    def tearDown(self):
        self.tmp.cleanup()
    def group(self):
        return run_group(self.task, self.root / "group", self.store.current(), self.config,
                         backend_factory=FakeBackend, model_factory=FakeModel)
    def test_serial_shared_uses_fresh_environments_and_only_prior_evidence(self):
        FakeBackend.barrier = None
        config = {**self.config, "execution_mode": "serial_shared", "planner_enabled": False}
        report, results = run_group(self.task, self.root / "serial", self.store.current(), config,
                                   backend_factory=FakeBackend, model_factory=FakeModel)
        self.assertEqual(report["execution_mode"], "serial_shared")
        self.assertEqual(len(FakeBackend.instances), 5)
        for i, result in enumerate(results):
            self.assertEqual(result["shared_from_rollouts"], list(range(i)))
            if i:
                evidence = json.loads((self.root / "serial" / f"rollout-{i}" / "shared-evidence.json").read_text())
                self.assertEqual([r["rollout"] for r in evidence], list(range(i)))
                self.assertNotIn("SECRET_VERIFIER", json.dumps(evidence))
                self.assertTrue(all(r["messages"][0]["role"] == "assistant" for r in evidence))
                self.assertNotIn("Previous candidates below", json.dumps(evidence))
        self.assertEqual(report["planner_usage"]["calls"], 0)
        self.assertEqual(sum(b.scored for b in FakeBackend.instances), 1)

    def test_single_skips_planner_and_selector(self):
        FakeBackend.barrier = None
        config = {**self.config, "execution_mode": "single", "rollouts": 1, "planner_enabled": False}
        with patch.object(FakeModel, "object", side_effect=AssertionError("selector called")):
            report, results = run_group(self.task, self.root / "single", self.store.current(), config,
                                       backend_factory=FakeBackend, model_factory=FakeModel)
        self.assertEqual(len(results), 1)
        self.assertEqual(report["selected"], 0)
        self.assertEqual(report["selector_usage"]["calls"], 0)
        self.assertEqual(report["planner_usage"]["calls"], 0)
        self.assertEqual(report["tokens"], 15)
        self.assertTrue(FakeBackend.instances[0].closed)

    def test_five_reflection_loops_run_in_parallel_before_selection(self):
        reviewers = []
        barrier = threading.Barrier(5)
        class FeedbackModel(FakeModel):
            def query(self, messages, **kwargs):
                if self.n_calls == 0:
                    barrier.wait(timeout=10)
                self.n_calls += 1
                return {"content": json.dumps({"stop": self.n_calls > 1,
                        "reason": "review", "feedback": "Verify the fix and submit."})}
            def stats(self):
                return {"calls": self.n_calls, "input_tokens": self.n_calls * 10,
                        "output_tokens": self.n_calls * 2, "cost_usd": 0}
        def factory(config):
            if config.get("name") == "qwen3.8-max":
                reviewer = FeedbackModel(config)
                reviewers.append(reviewer)
                return reviewer
            return FakeModel(config)
        config = {**self.config, "reflection_enabled": True, "reflection_max_rounds": 5,
                  "reflection_step_limit": 2, "reflection_model": {"name": "qwen3.8-max"}}
        report, results = run_group(self.task, self.root / "reflect", self.store.current(), config,
                                   backend_factory=FakeBackend, model_factory=factory)
        self.assertEqual(len(reviewers), 5)
        self.assertTrue(all(r.n_calls == 2 for r in reviewers))
        self.assertEqual(report["reflection_usage"]["input_tokens"], 100)
        self.assertEqual(report["input_tokens"], 160)
        self.assertEqual(report["reflection_revisions"], 5)
        self.assertTrue(all(r["usage"]["calls"] == 2 for r in results))
        self.assertTrue(all(r["reflection"]["stop_reason"] == "reviewer_stop" for r in results))
        self.assertTrue(all(b.closed for b in FakeBackend.instances))
        self.assertEqual(sum(b.scored for b in FakeBackend.instances), 1)

    def test_selector_request_saved_before_provider_failure_and_cleanup(self):
        from rsi.common import read_json
        from rsi.model import object_messages
        def fail(instruction, data):
            request = read_json(self.root / "group/evidence/selector.json")
            self.assertEqual(request, object_messages(instruction, data))
            raise RuntimeError("provider unavailable")
        with patch.object(FakeModel, "object", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                self.group()
        self.assertTrue(all(x.closed for x in FakeBackend.instances))
        self.assertFalse(any(x.scored for x in FakeBackend.instances))
    def test_five_parallel_upstream_agents_select_only_one(self):
        report, results = self.group()
        self.assertEqual(len(results), 5)
        self.assertEqual(report["selected"], 3)
        self.assertEqual(report["tokens"], 90)
        self.assertEqual(report["input_tokens"], 60)
        self.assertEqual(report["output_tokens"], 30)
        saved = json.loads((self.root / "group/summary.json").read_text())
        self.assertEqual((saved["input_tokens"], saved["output_tokens"]), (60, 30))
        self.assertEqual(sum(x.scored for x in FakeBackend.instances), 1)
        self.assertTrue(all(x.closed for x in FakeBackend.instances))
        self.assertEqual({r["skill_version"] for r in results}, {self.store.current().name})
        self.assertNotIn("SECRET_VERIFIER", json.dumps(results))
        pred = json.loads((self.root / "group/predictions.json").read_text())
        self.assertEqual(pred[0]["model_patch"], "patch-3")
    def test_invalid_selection_cleans_all_environments(self):
        FakeModel.selection = 99
        with self.assertRaises(ValueError):
            self.group()
        self.assertTrue(all(x.closed for x in FakeBackend.instances))
        self.assertFalse(any(x.scored for x in FakeBackend.instances))
    def test_planner_routes_are_isolated_and_cost_is_included(self):
        plan = {"status": "planned", "strategies": [
            {"rollout": i, "hypothesis": f"unique-hypothesis-{i}"} for i in range(1, 5)],
            "usage": {"calls": 1, "input_tokens": 7, "output_tokens": 3}}
        with patch("rsi.runner.make_plan", return_value=plan):
            report, results = self.group()
        self.assertEqual(report["tokens"], 100)
        self.assertIsNone(results[0]["strategy"])
        self.assertNotIn("Task-local exploration plan", results[0]["messages"][0]["content"])
        for i in range(1, 5):
            prompt = results[i]["messages"][0]["content"]
            self.assertIn(f"unique-hypothesis-{i}", prompt)
            for j in set(range(1, 5)) - {i}:
                self.assertNotIn(f"unique-hypothesis-{j}", prompt)
        self.assertEqual(results[0]["planner"], plan)
    def test_one_failed_rollout_preserves_others(self):
        FakeBackend.fail_index = 1
        report, results = self.group()
        self.assertTrue(report["success"])
        self.assertIn("injected", results[1]["error"])
    def test_skill_rejection_and_acceptance_are_versioned(self):
        old = self.store.current()
        candidate, diff = self.store.propose({"SKILL.md": "Revised workflow"})
        self.assertIn("Revised workflow", diff)
        self.assertEqual(self.store.current(), old)
        self.store.record(candidate, False, {"reason": "no improvement"})
        self.assertEqual(self.store.current(), old)
        self.store.record(candidate, True, {"reason": "validated"})
        self.assertEqual(self.store.current(), candidate)
        self.assertNotEqual(file_map(old)["SKILL.md"], "Revised workflow")
    def test_updater_cannot_modify_tests_or_escape_directory(self):
        for name in ("../escape.py", "/absolute.py", "tests/check_skill.py", "rsi/runner.py"):
            with self.assertRaises(ValueError):
                self.store.propose({name: "pass"})
    def test_script_edits_are_rejected(self):
        with self.assertRaises(ValueError):
            self.store.propose({"scripts/inspect_project.py": "def :"})
    def test_hidden_feedback_excluded(self):
        result = {"rollout": 0, "status": "Submitted", "messages": [], "artifact": "", "error": None,
                  "score": {"success": True}, "hidden_tests": "secret"}
        self.assertNotIn("score", public_rollout(result))
        self.assertNotIn("hidden_tests", public_rollout(result))
    def test_run_directory_cannot_be_reused(self):
        directory = self.root / "existing"
        directory.mkdir()
        with self.assertRaises(FileExistsError):
            run_experiment([], self.config, directory)

    def test_complete_update_cycle_changes_next_task_only(self):
        calls = []
        def group(task, folder, skill, config):
            calls.append((task["id"], skill.name))
            tokens = 20
            return {"task_id": task["id"], "success": True,
                    "tokens": tokens, "input_tokens": tokens * 4 // 5,
                    "output_tokens": tokens // 5, "seconds": 1}, []
        proposals = [
            {"changes": {"SKILL.md": "A validated new strategy"}, "reason": "observed repeated omission",
             "memory_items": [{"title": "reproduce", "description": "before editing", "content": "Run a targeted reproduction"}]},
            {"changes": {}, "memory_items": []},
        ]
        tasks = [self.task, {**self.task, "id": "example-2"}]
        output = self.root / "experiment"
        with patch("rsi.runner.run_group", side_effect=group), patch("rsi.runner.induce", side_effect=proposals), patch("rsi.runner.functional_gate", return_value={"passed": True}) as initial_check:
            run_experiment(tasks, self.config, output)
        initial_check.assert_called_once()
        self.assertEqual([c[0] for c in calls], ["example-1", "example-2"])
        self.assertNotEqual(calls[0][1], calls[1][1])
        self.assertFalse((output / "example-1/validation").exists())
        self.assertFalse((output / "memory.json").exists())
        self.assertFalse((output / "validation-tasks.json").exists())
        self.assertNotIn("validation_count", json.loads((output / "run-info.json").read_text()))
        totals = json.loads((output / "metrics.json").read_text())
        self.assertEqual((totals["input_tokens"], totals["output_tokens"], totals["tokens"]), (32, 8, 40))

    def test_fixed_mode_does_not_call_updater(self):
        config = {**self.config, "evolve": False}
        output = self.root / "fixed"
        report = {"success": True, "tokens": 15, "seconds": 1}
        with patch("rsi.runner.run_group", return_value=(report, [])), \
             patch("rsi.runner.induce") as updater, \
             patch("rsi.runner.functional_gate", return_value={"passed": True}):
            run_experiment([self.task], config, output)
        updater.assert_not_called()
        update = json.loads((output / self.task["id"] / "update.json").read_text())
        self.assertEqual(update["decision"], "fixed_skill")
        self.assertIn("tools_hash", update)
        self.assertTrue((output / "initial-skill/current.json").exists())

    def test_no_validation_activates_candidate_after_checks(self):
        proposal = {"changes": {"SKILL.md": "candidate"}, "memory_items": []}
        output = self.root / "experiment"
        with patch("rsi.runner.run_group", return_value=({"success": True, "tokens": 15, "input_tokens": 10, "output_tokens": 5, "seconds": 1}, [])), patch("rsi.runner.induce", return_value=proposal), patch("rsi.runner.functional_gate", return_value={"passed": True}):
            run_experiment([self.task], self.config, output)
        history = json.loads((output / "skills/history.jsonl").read_text())
        self.assertTrue(history["accepted"])
        pointer = json.loads((output / "skills/current.json").read_text())
        self.assertNotEqual(pointer["version"], history["before"])

    def test_failed_checks_keep_previous_skill(self):
        output = self.root / "experiment"
        with patch("rsi.runner.run_group", return_value=({"success": False, "tokens": 15,
                "input_tokens": 10, "output_tokens": 5, "seconds": 1}, [])), \
             patch("rsi.runner.induce", return_value={"changes": {"SKILL.md": "bad candidate"}}), \
             patch("rsi.runner.functional_gate", return_value={"passed": True}), \
             patch("rsi.runner.document_gate", side_effect=[{"passed": True}, {"passed": False, "stderr": "failed"}]):
            run_experiment([self.task], self.config, output)
        history = json.loads((output / "skills/history.jsonl").read_text())
        self.assertFalse(history["accepted"])
        self.assertEqual(json.loads((output / "skills/current.json").read_text())["version"], history["before"])
        self.assertEqual(json.loads((output / self.task["id"] / "update.json").read_text())["decision"], "checks_failed")

    def test_task_files_are_read_as_utf8(self):
        path = self.root / "instruction.md"
        path.write_bytes("概率分布 café\n".encode())
        self.assertEqual(path.read_text(), "概率分布 café\n")

    def test_wandb_payload_omits_secrets_and_task_text(self):
        config = {**self.config, "model": {**self.config["model"], "api_key": "secret-key"},
                  "wandb": {"enabled": True, "project": "rsi", "api_key": "secret-key"}}
        published = public_config(config)
        encoded = json.dumps(published)
        self.assertNotIn("secret-key", encoded)
        self.assertNotIn("problem", encoded)
        self.assertEqual(published["model/name"], self.config["model"]["name"])
        self.assertEqual(published["rollouts"], 5)
        self.assertEqual(published["wandb/project"], "rsi")
        report = {"task_id": "example-1", "selected": 1, "success": True, "tokens": 15, "seconds": 1.5,
                  "skill_version": "abc", "selector_usage": {"calls": 1, "input_tokens": 3, "output_tokens": 4, "cost_usd": 0.1}}
        row = task_metrics({"benchmark": "swe", "prompt": "hidden task text"}, report,
                           {"accepted": False, "memory_items": 2, "diff": "secret patch"})
        self.assertEqual(row["success"], 1)
        self.assertEqual(row["selector/input_tokens"], 3)
        self.assertEqual(row["memory_items"], 2)
        self.assertNotIn("hidden task text", json.dumps(row))
        self.assertNotIn("diff", row)

    def test_document_gate_and_tool_hash_are_published(self):
        report = {"task_id": "example-1", "selected": 0, "success": True, "tokens": 1,
                  "seconds": 1, "skill_version": "v", "skill_injected_rollouts": 5,
                  "skill_script_called_rollouts": 1, "skill_script_command_attempts": 2}
        update = {"accepted": True, "document_gate": {"passed": True}, "tools_hash": "fixed",
                  "acceptance_policy": "document_only_fixed_tools", "decision": "checks_passed"}
        row = task_metrics(self.task, report, update)
        self.assertEqual(row["document_gate_passed"], 1)
        self.assertEqual(row["tools_hash"], "fixed")
        self.assertEqual(row["skill_script_command_attempts"], 2)

    def test_harness_writes_eval_scripts_with_lf(self):
        import subprocess
        import sys
        from rsi.backends import _POSIX_TEXT
        target = self.root / "eval.sh"
        script = _POSIX_TEXT.split("sys.argv", 1)[0] + f"pathlib.Path({str(target)!r}).write_text('set -e\\n')"
        subprocess.run([sys.executable, "-c", script], check=True)
        self.assertNotIn(b"\r", target.read_bytes())
        raw = self.root / "utf8.json"
        raw.write_bytes('{"x":"“"}'.encode())
        shown = subprocess.run([sys.executable, "-c", _POSIX_TEXT.split("sys.argv", 1)[0]
                                + f"print(pathlib.Path({str(raw)!r}).read_text())"],
                               check=True, capture_output=True, text=True)
        self.assertIn("“", shown.stdout)

    def test_gate_timeout_removes_container(self):
        import subprocess
        with patch("rsi.skills.subprocess.run", side_effect=[subprocess.TimeoutExpired("docker", 90), None]) as run:
            result = functional_gate(self.store.current(), ROOT / "skills/test-and-diagnose/tests")
        self.assertFalse(result["passed"])
        self.assertEqual(run.call_args_list[1].args[0][:3], ["docker", "rm", "-f"])


if __name__ == "__main__":
    unittest.main()
