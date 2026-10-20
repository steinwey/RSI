import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import yaml
from . import ROOT
from .common import read_json, safe_id, write_json
from .swe import evaluation_dataset_name, validate_swe_tasks
from .run_logs import write_run_index
from .tracking import summary_metrics


def doctor():
    checks = {"python": sys.version.split()[0], "harbor_python_ok": sys.version_info >= (3, 12),
              "docker_cli": shutil.which("docker"), "deepseek_key_set": bool(os.getenv("DEEPSEEK_API_KEY")),
              "harbor_installed": importlib.util.find_spec("harbor") is not None,
              "swebench_installed": importlib.util.find_spec("swebench") is not None}
    if checks["docker_cli"]:
        try:
            run = subprocess.run(["docker", "info"], capture_output=True, timeout=15)
            checks["docker_daemon"] = run.returncode == 0
            run = subprocess.run(["docker", "compose", "version"], capture_output=True, timeout=15)
            checks["docker_compose"] = run.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            checks["docker_daemon"] = False
    pins = read_json(ROOT / "upstreams.lock.json")
    checks["upstreams"] = {}
    for name, entry in pins.items():
        path = ROOT / "vendor" / name
        run = subprocess.run(["git", "-c", f"safe.directory={path.as_posix()}", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
        checks["upstreams"][name] = run.returncode == 0 and run.stdout.strip() == entry["commit"]
    print(json.dumps(checks, indent=2))
    return checks


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    config.setdefault("image_pull_timeout", 7200)
    config["swe_dataset"] = evaluation_dataset_name(config["swe_dataset"])
    if config["rollouts"] != 5:
        raise ValueError("This experiment requires exactly five rollouts per task")
    for key in ("step_limit", "command_timeout", "image_pull_timeout", "rollout_timeout", "evaluation_timeout", "evidence_chars"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for cfg in (config["model"], config["judge_model"], config.get("wandb") or {}):
        if isinstance(cfg, dict) and any(key in cfg for key in ("api_key", "key")):
            raise ValueError("Put credentials in environment variables, not saved configuration")
    return config


def load_tasks(path, config):
    tasks = read_json(path)
    for task in tasks:
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="ReasoningBank five-rollout skill evolution")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    listing = sub.add_parser("runs", help="List experiment logs and refresh INDEX.md")
    listing.add_argument("--runs-dir", default=str(ROOT / "runs"))
    prepare = sub.add_parser("prepare-swe")
    choice = prepare.add_mutually_exclusive_group()
    choice.add_argument("--ids", nargs="+")
    choice.add_argument("--limit", type=int, default=1)
    prepare.add_argument("--offset", type=int, default=0)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--config", default=str(ROOT / "configs/deepseek.yaml"))
    terminal = sub.add_parser("prepare-terminal")
    terminal.add_argument("--paths", nargs="+", required=True)
    terminal.add_argument("--output", required=True)
    evo = sub.add_parser("prepare-evo-swe", help="Prepare complete SWE-Chain-Evo chains")
    evo.add_argument("--data-dir", default=str(ROOT / "datasets/evoarena-swe"))
    evo.add_argument("--chains", nargs="+", required=True)
    evo.add_argument("--output", required=True)
    evo.add_argument("--download", action="store_true", help="Download selected chain metadata")
    evo.add_argument("--revision", default="f98ba6aed4c1a40c527f6d70c186a6887888a332")
    for name, help_text in (("prepare-evo-images", "Download, import and prepare milestone images"),
                            ("verify-evo-swe", "Grade reference patches without calling a model")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--config", default=str(ROOT / "configs/evoarena.yaml"))
        command.add_argument("--tasks", required=True)
        command.add_argument("--limit", type=int, help="Prepare/check only the first N milestones")
        if name == "prepare-evo-images":
            command.add_argument("--cache", default=str(ROOT / "datasets/evoarena-swe/image-cache"))
        else:
            command.add_argument("--output")
    run = sub.add_parser("run")
    run.add_argument("--config", default=str(ROOT / "configs/deepseek.yaml"))
    run.add_argument("--tasks", required=True)
    run.add_argument("--validation")
    destination = run.add_mutually_exclusive_group()
    destination.add_argument("--output", help="Exact output directory; existing directories are never overwritten")
    destination.add_argument("--runs-dir", help="Parent directory for automatically named runs (default: runs/)")
    run.add_argument("--label", help="Optional experiment label included in automatic directory names")
    run.add_argument("--fixed-skill", action="store_true")
    run.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False,
                     help="Upload metrics to W&B; requires WANDB_API_KEY (default: disabled)")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        doctor()
        return
    if args.command == "runs":
        for entry in write_run_index(args.runs_dir):
            print(f"{entry['started_at']}  {entry['kind']}  {entry['mode']}  {entry['status']}  {entry['directory']}")
        print(f"Log index: {Path(args.runs_dir).resolve() / 'INDEX.md'}")
        return
    if args.command == "prepare-terminal":
        from harbor.models.task.task import Task
        tasks = [{"id": safe_id(Path(p).name), "benchmark": "terminal", "path": str(Path(p).resolve())} for p in args.paths]
        for task in tasks:
            Task(task["path"])
        write_json(args.output, tasks)
        return
    if args.command == "prepare-evo-swe":
        from .evo_swe import download_chains, prepare_tasks
        if args.download:
            download_chains(args.data_dir, args.chains, args.revision)
        tasks = prepare_tasks(args.data_dir, args.chains, args.output)
        print(f"Prepared {len(args.chains)} chains, {len(tasks)} milestones: {args.output}")
        return
    config = load_config(args.config)
    if args.command in ("prepare-evo-images", "verify-evo-swe"):
        from .evo_swe import prepare_images, validate_order, validate_tasks, verify_references
        tasks = load_tasks(args.tasks, config)
        validate_order(tasks)
        if not tasks or any(t["benchmark"] != "swe-evo" for t in tasks):
            parser.error("This command requires a SWE-Chain-Evo manifest")
        if args.limit is not None and args.limit <= 0:
            parser.error("--limit must be positive")
        selected = tasks[:args.limit] if args.limit else tasks
        validate_tasks(selected, config, require_images=False)
        if args.command == "prepare-evo-images":
            prepare_images(selected, args.cache, config["image_pull_timeout"])
        else:
            validate_tasks(selected, config)
            verify_references(selected, config, args.output)
        return
    if args.command == "prepare-swe":
        from .evo_swe import DATASET
        if config["swe_dataset"] == DATASET:
            parser.error("Use prepare-evo-swe --chains ... for SWE-Chain-Evo")
        from datasets import load_dataset
        data = load_dataset(config["swe_dataset"], split=config["swe_split"])
        if args.limit <= 0 or args.offset < 0:
            raise ValueError("limit must be positive and offset must be nonnegative")
        ids = args.ids or list(data["instance_id"])[args.offset:args.offset + args.limit]
        if not ids:
            raise ValueError("No tasks selected")
        selected = {row["instance_id"]: row for row in data if row["instance_id"] in ids}
        if set(ids) - selected.keys():
            raise ValueError(f"Unknown instances: {set(ids)-selected.keys()}")
        # Do not persist gold patch or hidden test expectations into solver manifests.
        fields = ("instance_id", "problem_statement", "repo", "base_commit", "version", "image_name", "image")
        tasks = [{"id": iid, "benchmark": "swe", "dataset_fingerprint": data._fingerprint,
                  "instance": {k: v for k, v in selected[iid].items() if k in fields}} for iid in ids]
        write_json(args.output, tasks)
        return
    # Uploads are an explicit per-run CLI choice, overriding legacy YAML flags.
    config["wandb"] = {**(config.get("wandb") or {}), "enabled": args.wandb}
    if args.wandb and not os.getenv("WANDB_API_KEY"):
        parser.error("--wandb requires WANDB_API_KEY in the environment")
    if not shutil.which("docker"):
        raise RuntimeError("Docker is required. Run inside the configured WSL environment; see README.md")
    if any(c["name"].startswith("deepseek/") for c in (config["model"], config["judge_model"])) and not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("Set DEEPSEEK_API_KEY in the environment")
    from .runner import run_experiment
    config["evolve"] = not args.fixed_skill and config["evolve"]
    tasks = load_tasks(args.tasks, config)
    validation = load_tasks(args.validation, config) if args.validation else []
    from .evo_swe import DATASET, validate_order, validate_tasks
    validate_order(tasks, validation)
    validate_tasks(tasks + validation, config)
    if tasks and all(t["benchmark"] == "swe-evo" for t in tasks):
        config["swe_dataset"] = DATASET
    validate_swe_tasks(tasks + validation, config)
    if config["evolve"] and not validation:
        print("No validation tasks: candidates will be recorded but cannot become active.")
    reports = run_experiment(tasks, validation, config, args.output, runs_dir=args.runs_dir, label=args.label)
    print(json.dumps(summary_metrics(reports), indent=2))


if __name__ == "__main__":
    main()
