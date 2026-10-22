import json
import tempfile
import unittest
from pathlib import Path

from rsi.planner import make_plan


class PlannerTests(unittest.TestCase):
    def test_plans_exclude_free_route_and_use_current_skill(self):
        seen = []
        class Fake:
            def __init__(self, config):
                pass
            def query(self, messages):
                seen.append(messages)
                return {"content": json.dumps({"strategies": [
                    {"rollout": i, "hypothesis": f"hypothesis {i}", "first_experiment": f"probe {i}",
                     "stop_condition": "contradicting evidence", "exploration_steps": 5}
                    for i in range(1, 5)]})}
            def stats(self):
                return {"calls": 1, "input_tokens": 12, "output_tokens": 8}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            result = make_plan({"prompt": "task"}, "current guide", {
                "planner_enabled": True, "rollouts": 5, "step_limit": 30, "judge_model": {}}, folder, Fake)
            self.assertEqual(result["status"], "planned")
            self.assertEqual([p["rollout"] for p in result["strategies"]], [1, 2, 3, 4])
            self.assertEqual(json.loads(seen[0][1]["content"])["skill"], "current guide")
            self.assertEqual(json.loads((folder / "plan.json").read_text())["usage"]["calls"], 1)

    def test_malformed_plan_falls_back_without_retry_and_preserves_cost(self):
        class Fake:
            def __init__(self, config):
                self.calls = 0
            def query(self, messages):
                self.calls += 1
                return {"content": '{"strategies": []}'}
            def stats(self):
                return {"calls": self.calls, "input_tokens": 7, "output_tokens": 3}
        with tempfile.TemporaryDirectory() as tmp:
            result = make_plan({"prompt": "task"}, "guide", {
                "planner_enabled": True, "rollouts": 5, "step_limit": 30, "judge_model": {}}, Path(tmp), Fake)
            self.assertEqual(result["status"], "fallback")
            self.assertEqual(result["strategies"], [])
            self.assertEqual(result["usage"]["calls"], 1)

    def test_disabled_does_not_call_model(self):
        def forbidden(config):
            raise AssertionError("Unexpected model call")
        result = make_plan({}, "guide", {"rollouts": 5}, Path("unused"), forbidden)
        self.assertEqual(result["status"], "disabled")


if __name__ == "__main__":
    unittest.main()
