"""Load solver-facing tasks and dispatch environment checks by benchmark."""
from pathlib import Path

from .common import read_json, safe_id


def load_tasks(path, config):
    tasks = read_json(path)
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Task manifest must contain a nonempty list")
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("id"), str):
            raise ValueError("Every task must have a string id")
        safe_id(task["id"])
        if task["benchmark"] in ("swe", "swe-evo"):
            if task["instance"]["instance_id"] != task["id"]:
                raise ValueError("SWE task ID does not match instance")
            task["prompt"] = task["instance"]["problem_statement"]
        elif task["benchmark"] == "terminal":
            from harbor.models.task.task import Task
            location = Path(task["path"])
            if not location.is_absolute():
                location = Path(path).resolve().parent / location
            task["path"] = str(location.resolve())
            parsed = Task(location)
            task["prompt"] = parsed.instruction
            task["timeout"] = parsed.config.agent.timeout_sec or config["rollout_timeout"]
        else:
            raise ValueError("benchmark must be swe, swe-evo or terminal")
    return tasks


def validate_environments(tasks, config, *, require_images=True):
    from .evo_swe import DATASET, validate_tasks as validate_evo_tasks
    from .swe import validate_swe_tasks
    evo = [t for t in tasks if t["benchmark"] == "swe-evo"]
    swe = [t for t in tasks if t["benchmark"] == "swe"]
    if evo:
        if len(evo) == len(tasks):
            config["swe_dataset"] = DATASET
        validate_evo_tasks(evo, config, require_images=require_images)
    if swe:
        validate_swe_tasks(swe, config)
