"""Validate task-boundary checkpoints and preserve incomplete attempts on resume."""
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from . import ROOT
from .common import digest, file_map, read_json, safe_id, write_json
from .config import load_config
from .skills import load_initial_skill
from .tasks import load_tasks


def task_metadata(tasks):
    fields = ("id", "benchmark", "dataset_fingerprint", "path", "chain")
    return [{k: t[k] for k in fields if k in t} for t in tasks]


def resume_request(output):
    """Load saved inputs, never substitute today's model or task configuration."""
    output = Path(output).expanduser().resolve()
    info = read_json(output / "run-info.json")
    if info.get("status") == "running":
        raise ValueError("Run is recorded as running; stop and verify its processes before resuming")
    config = load_config(output / "config.json")
    if digest(config) != info["config_version"]:
        raise ValueError("Saved configuration fingerprint changed")
    workflow = info["kind"] == "train-test"
    names = ("train-tasks.json", "test-tasks.json") if workflow else ("task-manifest.json",)
    if any(not (output / name).is_file() for name in names):
        raise ValueError("Checkpoint has no full task snapshot; recover and verify its original manifest first")
    groups = [load_tasks(output / name, config) for name in names]
    if [t["id"] for group in groups for t in group] != info["task_ids"]:
        raise ValueError("Checkpoint task IDs or order differ from the original run")
    if info.get("task_manifest_version") and digest([t for group in groups for t in group]) != info["task_manifest_version"]:
        raise ValueError("Saved task manifest fingerprint changed")
    for phase, tasks in zip(("train", "test") if workflow else ("",), groups):
        directory = output / phase
        if (directory / "tasks.json").exists() and read_json(directory / "tasks.json") != task_metadata(tasks):
            raise ValueError("Checkpoint task metadata differs from the saved manifest")
        if (directory / "config.json").exists():
            inspect_checkpoint(directory, tasks, read_json(directory / "config.json"))
    return output, config, groups


def inspect_checkpoint(output, tasks, config):
    """Require a consistent scored prefix, skill history, fixed tools and config."""
    output = Path(output).resolve()
    if read_json(output / "config.json") != config:
        raise ValueError("Cannot change experiment configuration when resuming")
    info = read_json(output / "run-info.json")
    if digest(config) != info["config_version"]:
        raise ValueError("Saved configuration fingerprint changed")
    if read_json(output / "tasks.json") != task_metadata(tasks):
        raise ValueError("Cannot change task order or metadata when resuming")
    manifest = output / "task-manifest.json"
    if manifest.exists() and read_json(manifest) != tasks:
        raise ValueError("Checkpoint task contents changed")
    if info.get("task_manifest_version") and digest(tasks) != info["task_manifest_version"]:
        raise ValueError("Saved task manifest fingerprint changed")
    if read_json(output / "upstreams.lock.json") != read_json(ROOT / "upstreams.lock.json"):
        raise ValueError("Checkpoint upstream versions changed")
    _, seed = load_initial_skill(output / "initial-skill")
    _, current = load_initial_skill(output / "skills")
    gate = read_json(output / "fixed-tools-check.json")
    if not gate.get("passed") or gate.get("tools_hash") != seed["tools_hash"]:
        raise ValueError("Checkpoint fixed-tool check is missing or invalid")
    reports = read_json(output / "summary.json") if (output / "summary.json").exists() else []
    if not isinstance(reports, list) or len(reports) > len(tasks):
        raise ValueError("Invalid checkpoint summary")
    history_file = output / "skills/history.jsonl"
    history = [json.loads(line) for line in history_file.read_text(encoding="utf-8").splitlines() if line.strip()] if history_file.exists() else []
    prefix = [t["id"] for t in tasks[:len(reports)]]
    if any(event.get("task_id") not in prefix for event in history):
        raise ValueError("Skill history includes an uncommitted task; cannot safely replay its update")
    expected = seed["version"]
    for task, report in zip(tasks, reports):
        directory = output / safe_id(task["id"])
        if report.get("task_id") != task["id"] or read_json(directory / "summary.json") != report:
            raise ValueError("Completed task summary is inconsistent")
        score, update = read_json(directory / "score.json"), read_json(directory / "update.json")
        if type(score.get("success")) is not bool or score["success"] != report.get("success"):
            raise ValueError("Cannot resume through a missing or inconsistent official score")
        if update.get("decision") == "update_error" or type(update.get("accepted")) is not bool:
            raise ValueError("Resolve the saved skill update error before resuming")
        if report.get("skill_version") != expected:
            raise ValueError("Completed task used an unexpected skill version")
        transitions = [e for e in history if e.get("task_id") == task["id"]]
        accepted = [e for e in transitions if e.get("accepted")]
        if len(accepted) != int(update["accepted"]) or any(e["before"] != expected for e in transitions):
            raise ValueError("Checkpoint skill history and decisions disagree")
        if accepted:
            expected = accepted[0]["after"]
            load_initial_skill(output / "skills/versions" / expected)
    if current["version"] != expected:
        raise ValueError("Current skill pointer differs from the committed task prefix")
    for task in tasks[len(reports):]:
        if (output / safe_id(task["id"]) / "update.json").exists():
            raise ValueError("Task state advanced beyond the committed summary; inspect before replay")
    return {"reports": reports, "skill_version": expected, "config_version": digest(config)}


def activate_checkpoint(output, tasks, checkpoint):
    """Keep completed tasks intact; move incomplete work into this run's archive."""
    from .run_logs import update_run
    output = Path(output).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    archive = output / "resume-attempts" / stamp
    moves = [(output / safe_id(t["id"]), archive / safe_id(t["id"]))
             for t in tasks[len(checkpoint["reports"]):] if (output / safe_id(t["id"])).exists()]
    for source, destination in moves:
        if source.resolve().parent != output or not destination.resolve().is_relative_to(output):
            raise ValueError("Resume archive path escaped the run directory")
    if not archive.resolve().is_relative_to(output):
        raise ValueError("Resume archive escaped the run directory")
    archive.mkdir(parents=True)
    info = read_json(output / "run-info.json")
    event = {"resumed_at": datetime.now(timezone.utc).isoformat(), "completed_tasks": len(checkpoint["reports"]),
             "skill_version": checkpoint["skill_version"], "config_version": checkpoint["config_version"],
             "code_version": digest(file_map(ROOT / "rsi")), "previous_run_info": info,
             "archived_tasks": [source.name for source, _ in moves]}
    write_json(archive / "resume.json", event)
    for source, destination in moves:
        shutil.move(str(source), str(destination))
    with (output / "resume-history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({k: v for k, v in event.items() if k != "previous_run_info"}) + "\n")
    update_run(output, status="running", error=None, stopped_at=None, resumed_at=event["resumed_at"],
               resume_count=info.get("resume_count", 0) + 1, resume_code_version=event["code_version"],
               current_skill_version=checkpoint["skill_version"])


@contextmanager
def run_lock(output):
    """An OS lock releases automatically even if the resumed process is killed."""
    with (Path(output) / "resume.lock").open("a+b") as stream:
        if not stream.tell():
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("Another process already holds this run's resume lock") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
