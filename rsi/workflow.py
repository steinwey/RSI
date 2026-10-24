"""Train a persistent skill, then evaluate an independent frozen snapshot."""
import copy
from pathlib import Path

from .common import read_json, write_json
from .evo_swe import validate_order
from .run_logs import create_run_directory, update_run
from .runner import progress, run_experiment
from .skills import SkillStore, load_initial_skill
from .tracking import summary_metrics


def validate_train_test(train_tasks, test_tasks):
    if not train_tasks or not test_tasks:
        raise ValueError("Automatic train/test requires nonempty training and test sets")
    validate_order(train_tasks, test_tasks)
    for rows in (train_tasks, test_tasks):
        if len({t["id"] for t in rows}) != len(rows):
            raise ValueError("Duplicate task IDs in train/test")
    identities = {
        "task ID": lambda t: t["id"],
        "repository": lambda t: t.get("instance", {}).get("repo"),
        "problem statement": lambda t: " ".join(
            (t.get("prompt") or t.get("instance", {}).get("problem_statement") or "").split()),
    }
    for name, identity in identities.items():
        train = {identity(t) for t in train_tasks} - {None, ""}
        test = {identity(t) for t in test_tasks} - {None, ""}
        if train & test:
            raise ValueError(f"Train/test {name} overlap")


def run_train_test(train_tasks, test_tasks, config, output=None, *, runs_dir=None, label=None, resume=False):
    """Return per-phase metrics; exceptions stop the workflow and retain its logs."""
    validate_train_test(train_tasks, test_tasks)
    if not config["evolve"]:
        raise ValueError("Automatic train/test requires evolve=true for training")
    if resume:
        from .checkpoints import activate_checkpoint
        from .common import digest
        output = Path(output).resolve()
        if (read_json(output / "config.json") != config or
                read_json(output / "train-tasks.json") != train_tasks or
                read_json(output / "test-tasks.json") != test_tasks):
            raise ValueError("Cannot change workflow configuration or tasks when resuming")
        seed_files, _ = load_initial_skill(output / "initial-skill")
        seed_source = read_json(output / "initial-skill-source.json")
        state = read_json(output / "workflow.json")
        reports = {phase: read_json(output / phase / "summary.json")
                   if (output / phase / "summary.json").exists() else [] for phase in ("train", "test")}
        _, trained = load_initial_skill(output / "train")
        activate_checkpoint(output, train_tasks + test_tasks,
                            {"reports": reports["train"] + reports["test"], "skill_version": trained["version"],
                             "config_version": digest(config)})
        state.update(status="running", phase="train")
        state.pop("error", None)
        state.pop("stopped_at", None)
        if state["test"]["status"] == "skipped":
            state["test"]["status"] = "pending"
    else:
        seed_files, seed_source = load_initial_skill(config.get("initial_skill"))
        output = create_run_directory(train_tasks + test_tasks, config, output,
                                      runs_dir=runs_dir, label=label, kind="train-test",
                                      initial_skill=seed_source)
        state = {"protocol": "train_then_frozen_test", "status": "running", "phase": "train",
                 "output": str(output), "initial_skill_source": seed_source,
                 "train": {"directory": str(output / "train"), "tasks": len(train_tasks), "status": "pending"},
                 "test": {"directory": str(output / "test"), "tasks": len(test_tasks), "status": "pending"}}
        reports = {"train": [], "test": []}

    def save(**fields):
        write_json(output / "workflow.json", state)
        rows = reports["train"] + reports["test"]
        update_run(output, phase=state["phase"], completed_tasks=len(rows),
                   scored_tasks=sum(r.get("success") is not None for r in rows),
                   resolved=sum(r.get("success") is True for r in rows), **fields)

    print(f"Train/test output: {output}", flush=True)
    try:
        write_json(output / "config.json", config)
        write_json(output / "train-tasks.json", train_tasks)
        write_json(output / "test-tasks.json", test_tasks)
        write_json(output / "initial-skill-source.json", seed_source)
        # Pin the training seed too, even when its external source pointer later moves.
        seed = SkillStore(output / "initial-skill", seed_files).current()
        train_config = copy.deepcopy(config)
        train_config.update(initial_skill=str(seed), evolve=True)
        state["train"]["status"] = "running"
        save()
        progress("Training phase started")
        reports["train"] = run_experiment(train_tasks, train_config, output / "train", label="train",
                                         resume=resume and (output / "train").exists())
        if [r.get("task_id") for r in reports["train"]] != [t["id"] for t in train_tasks]:
            raise RuntimeError("Training did not finish every task in manifest order")
        state["train"].update(status="unscored" if any(r.get("success") is None for r in reports["train"])
                              else "completed", metrics=summary_metrics(reports["train"]))
        _, trained = load_initial_skill(output / "train")
        state["trained_skill_source"] = trained
        state["phase"] = "test"
        state["test"]["status"] = "running"
        save(current_skill_version=trained["version"])
        test_config = copy.deepcopy(config)
        test_config.update(initial_skill=trained["resolved_path"], evolve=False)
        progress(f"Training finished; frozen test started with skill {trained['version']}")
        reports["test"] = run_experiment(test_tasks, test_config, output / "test", label="test-trained",
                                        resume=resume and (output / "test").exists())
        if [r.get("task_id") for r in reports["test"]] != [t["id"] for t in test_tasks]:
            raise RuntimeError("Testing did not finish every task in manifest order")
        state["test"].update(status="unscored" if any(r.get("success") is None for r in reports["test"])
                             else "completed", metrics=summary_metrics(reports["test"]))
        state["status"] = ("unscored" if any(state[p]["status"] == "unscored" for p in reports)
                           else "completed")
        state["phase"] = "finished"
        save(status=state["status"], current_skill_version=trained["version"])
    except BaseException as exc:
        state["status"] = "failed" if isinstance(exc, Exception) else "interrupted"
        phase = state["phase"]
        if phase in reports:
            state[phase]["status"] = state["status"]
            summary = output / phase / "summary.json"
            if summary.is_file():
                reports[phase] = read_json(summary)
                state[phase]["metrics"] = summary_metrics(reports[phase])
        state["error"] = f"{type(exc).__name__}: {exc}"
        if state["test"]["status"] == "pending":
            state["test"]["status"] = "skipped"
        save(status=state["status"], error=state["error"])
        raise
    return state
