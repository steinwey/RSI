"""Task-local exploration plans; only the versioned skill stores learned rules."""
import time

from .common import write_json
from .model import Model, object_messages


INSTRUCTION = """Allocate distinct exploration strategies using the supplied task and skill's planning rules.
Rollout 0 remains independent and must not receive a plan. Generate one strategy for each other rollout.
These are hypotheses, not known facts or prescribed patches. Each solver must implement, test and submit
a complete solution and may abandon its strategy when evidence contradicts it, explaining why.
Differentiate hypotheses or experiments, not just role names. Do not force artificial differences when
the task is straightforward. No repository has been inspected for you; do not invent observations.
Do not fetch upstream solutions. Reserve most steps for implementation, verification and submission.
Return {"strategies": [{"rollout": int, "hypothesis": str, "first_experiment": str,
"stop_condition": str, "exploration_steps": int}]}.
Use rollout IDs 1 through planned_rollouts, exactly once each. exploration_steps must be between
1 and max_exploration_steps. Follow the skill's reusable allocation rules within this fixed contract."""


def make_plan(task, skill_text, config, folder, model_factory=Model):
    record = {"status": "disabled", "strategies": [], "usage": {
        "calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0}}
    if not config.get("planner_enabled", False) or config["rollouts"] < 2:
        return record
    start = time.monotonic()
    model = None
    budget = max(1, config["step_limit"] // 3)
    data = {"task": task["prompt"], "skill": skill_text,
            "planned_rollouts": config["rollouts"] - 1, "max_exploration_steps": budget}
    write_json(folder / "planner-request.json", object_messages(INSTRUCTION, data))
    try:
        model = model_factory(config.get("planner_model", config["judge_model"]))
        # One logical request; do not repeat a malformed plan using the judge JSON retry loop.
        from .common import parse_object
        raw = model.query(object_messages(INSTRUCTION, data))["content"]
        write_json(folder / "planner-response.json", {"content": raw})
        plans = parse_object(raw).get("strategies")
        if not isinstance(plans, list) or len(plans) != config["rollouts"] - 1:
            raise ValueError("Wrong number of strategies")
        ids, signatures = set(), set()
        for plan in plans:
            if not isinstance(plan, dict):
                raise ValueError("Strategy must be an object")
            rid, steps = plan.get("rollout"), plan.get("exploration_steps")
            if type(rid) is not int or rid not in range(1, config["rollouts"]) or rid in ids:
                raise ValueError("Invalid or duplicate rollout")
            if type(steps) is not int or not 1 <= steps <= budget:
                raise ValueError("Invalid exploration budget")
            fields = [plan.get(k) for k in ("hypothesis", "first_experiment", "stop_condition")]
            if any(not isinstance(v, str) or not v.strip() or len(v) > 2000 for v in fields):
                raise ValueError("Missing or oversized strategy fields")
            signature = tuple(" ".join(v.lower().split()) for v in fields[:2])
            if signature in signatures:
                raise ValueError("Duplicate exploration strategy")
            ids.add(rid)
            signatures.add(signature)
        record.update(status="planned", strategies=plans)
    except Exception as exc:
        record.update(status="fallback", error=f"{type(exc).__name__}: {exc}")
    finally:
        if model is not None:
            record["usage"] = model.stats()
        record["seconds"] = time.monotonic() - start
        write_json(folder / "plan.json", record)
    return record
