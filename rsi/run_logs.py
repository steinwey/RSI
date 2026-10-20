"""Name experiment outputs and maintain a readable index without moving old logs."""
import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

from . import ROOT
from .common import digest, file_map, read_json, write_json


def _now():
    return datetime.now(timezone(timedelta(hours=8)))


def _slug(value, limit=24):
    return re.sub(r"[^\w.-]+", "-", value).strip(".-_")[:limit] or "unnamed"


def create_run_directory(tasks, validation, config, output=None, *, runs_dir=None,
                         label=None, kind="benchmark"):
    started = _now()
    benchmarks = {task["benchmark"] for task in tasks}
    benchmark = next(iter(benchmarks)) if len(benchmarks) == 1 else "mixed"
    mode = ("evolve" if validation else "propose") if config["evolve"] else "fixed"
    if kind != "benchmark":
        mode = kind
    config_version = digest(config)
    code_version = digest(file_map(ROOT / "rsi"))
    skill_version = digest(file_map(ROOT / "skills/test-and-diagnose"))
    version = digest({"kind": kind, "config": config_version, "code": code_version,
                      "skill": skill_version, "tasks": tasks, "validation": validation})[:8]
    model = config["model"]["name"]
    if output is not None:
        directory = Path(output).resolve()
        directory.mkdir(parents=True, exist_ok=False)
    else:
        root = Path(runs_dir or ROOT / "runs").resolve()
        root.mkdir(parents=True, exist_ok=True)
        parts = [started.strftime("%Y%m%d-%H%M%S"), f"{benchmark}-{mode}",
                 _slug(model.rsplit("/", 1)[-1]), f"{len(tasks)}tasks", f"v-{version}"]
        if label:
            parts.insert(1, _slug(label))
        stem = "__".join(parts)
        suffix = 1
        while True:
            directory = root / (stem if suffix == 1 else f"{stem}__{suffix:02d}")
            try:
                directory.mkdir()
                break
            except FileExistsError:
                suffix += 1
    write_json(directory / "run-info.json", {
        "started_at": started.isoformat(timespec="seconds"), "timezone": "Asia/Shanghai",
        "kind": kind, "benchmark": benchmark, "mode": mode, "label": label,
        "model": model, "task_count": len(tasks), "validation_count": len(validation),
        "task_ids": [task["id"] for task in tasks],
        "validation_ids": [task["id"] for task in validation],
        "chain_ids": list(dict.fromkeys(t["chain"]["id"] for t in tasks if t.get("chain"))),
        "experiment_version": version, "config_version": config_version,
        "code_version": code_version,
        "initial_skill_version": skill_version if kind == "benchmark" else None,
        "current_skill_version": skill_version if kind == "benchmark" else None,
        "status": "running", "completed_tasks": 0,
        "trajectory_pattern": "<task_id>/rollout-<0..4>/trajectory.json" if kind == "benchmark" else None,
    })
    write_run_index(directory.parent)
    return directory


def update_run(directory, **fields):
    directory = Path(directory)
    info = read_json(directory / "run-info.json")
    info.update(fields)
    info["updated_at"] = _now().isoformat(timespec="seconds")
    write_json(directory / "run-info.json", info)
    write_run_index(directory.parent)


def _read(path, default):
    try:
        return read_json(path)
    except (OSError, ValueError):
        return default


def _legacy_info(directory):
    config = _read(directory / "config.json", {})
    tasks = _read(directory / "tasks.json", [])
    reports = list(directory.glob("logs/evaluation/**/report.json")) if not config else []
    if not config and not reports and not (directory / "score.json").exists():
        return None
    benchmarks = {task.get("benchmark") for task in tasks}
    mode = ("evolve" if _read(directory / "validation-tasks.json", []) else "propose")
    mode = mode if config.get("evolve") else "fixed"
    summary = _read(directory / "summary.json", [])
    status = "incomplete"
    if config:
        if summary and len(summary) == len(tasks):
            status = "completed" if all(r.get("success") is not None for r in summary) else "unscored"
        elif any(directory.glob("*/group-error.json")):
            status = "failed"
    elif reports or _read(directory / "score.json", {}).get("success") is not None:
        status = "completed"
    return {
        "started_at": datetime.fromtimestamp(directory.stat().st_ctime, _now().tzinfo).isoformat(timespec="seconds"),
        "kind": "benchmark" if config else "score-check", "legacy": True,
        "benchmark": next(iter(benchmarks)) if len(benchmarks) == 1 else "swe" if reports else "unknown",
        "mode": mode if config else "score-check", "model": config.get("model", {}).get("name", "未记录"),
        "task_count": len(tasks) if config else len(reports) or 1,
        "validation_count": len(_read(directory / "validation-tasks.json", [])),
        "status": status,
        "completed_tasks": len(summary) if config else len(reports) or int(status == "completed"),
        "current_skill_version": _read(directory / "skills/current.json", {}).get("version"),
    }


def _atomic_text(path, text):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_run_index(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for directory in root.iterdir():
        if directory.is_dir():
            info = _read(directory / "run-info.json", None) or _legacy_info(directory)
            if info:
                entries.append({**info, "directory": directory.name})
    entries.sort(key=lambda item: (item["started_at"], item["directory"]), reverse=True)
    latest = {}
    for entry in entries:
        if entry["kind"] not in latest:
            latest[entry["kind"]] = {**entry, "path": str(root / entry["directory"])}
    _atomic_text(root / "latest.json", json.dumps(latest, ensure_ascii=False, indent=2))

    def link(directory, filename, title):
        return f"[{title}]({quote(directory + '/' + filename)})"

    modes = {"fixed": "固定技能", "propose": "仅提议修改", "evolve": "验证后更新", "score-check": "单独评分"}
    statuses = {"running": "运行中", "completed": "已完成", "failed": "报错中止",
                "interrupted": "已中断", "incomplete": "未完成", "unscored": "评分未完成"}
    lines = ["# 实验日志索引", "", "自动更新；时间为北京时间（UTC+08:00）。最新指最近启动的运行。", ""]
    for kind, title in (("benchmark", "最近的完整实验流程"), ("score-check", "最近的单独评分")):
        if kind in latest:
            entry = latest[kind]
            target = next(filename for filename in
                          ("run-info.json", "summary.json", "score.json", "tasks.json", "logs/")
                          if (root / entry["directory"] / filename).exists())
            lines += [f"- {title}：{link(entry['directory'], target, entry['directory'])}（{statuses.get(entry['status'], entry['status'])}）"]
    lines += ["", "| 开始时间 | 目录 | 模式 | 模型 | 执行题 / 验证题 | 状态 | 实验版本 / 当前技能 | 查看 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for entry in entries:
        name = entry["directory"]
        directory = root / name
        links = [link(name, filename, title) for filename, title in
                 (("run-info.json", "运行信息"), ("summary.json", "汇总"), ("score.json", "评分"),
                  ("skills/current.json", "当前技能"), ("config.json", "配置")) if (directory / filename).exists()]
        if (directory / "logs").exists():
            links.append(link(name, "logs/", "评分日志"))
        version = entry.get("experiment_version", "历史记录")
        skill = entry.get("current_skill_version") or "—"
        values = [entry["started_at"].replace("T", " ").replace("+08:00", ""), name,
                  modes.get(entry["mode"], entry["mode"]), entry["model"],
                  f"{entry['task_count']} / {entry['validation_count']}",
                  statuses.get(entry["status"], entry["status"]), f"{version} / {skill}", " · ".join(links)]
        lines.append("| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in values) + " |")
    lines += ["", "目录名中的 v- 是实验版本（代码、配置、初始技能和任务清单的指纹），技能版本单独记录。",
              "模式：fixed 固定技能；propose 仅保存候选；evolve 使用独立验证题决定是否启用候选。",
              "五路轨迹位于各实验的 <task_id>/rollout-0 到 rollout-4/trajectory.json；技能历史位于 skills/history.jsonl。",
              "历史目录未移动；没有运行元数据的历史记录，其时间和状态按现有文件推断。", ""]
    _atomic_text(root / "INDEX.md", "\n".join(lines))
    return entries
