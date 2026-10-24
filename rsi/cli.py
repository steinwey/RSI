"""CLI argument definitions and command dispatch."""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import ROOT
from .baselines import MODES, set_execution_mode
from .common import read_json, safe_id, write_json
from .config import DEFAULT_CONFIG, credential_envs, load_config, load_env, model_configs, require_credentials, select_model
from .evo_swe import DATASET, REVISION, validate_order
from .run_logs import write_run_index
from .tasks import load_tasks, validate_environments
from .tracking import summary_metrics


def _doctor(args):
    config = args.settings
    checks = {"python": sys.version.split()[0], "harbor_python_ok": sys.version_info >= (3, 12),
              "docker_cli": shutil.which("docker"),
              "models": {role: block["name"] for role, block in model_configs(config, active_only=True).items()},
              "credentials": {name: bool(os.getenv(name)) for name in credential_envs(config)},
              "harbor_installed": importlib.util.find_spec("harbor") is not None,
              "swebench_installed": importlib.util.find_spec("swebench") is not None}
    if checks["docker_cli"]:
        for name, command in (("docker_daemon", ["docker", "info"]),
                              ("docker_compose", ["docker", "compose", "version"])):
            try:
                checks[name] = subprocess.run(command, capture_output=True, timeout=15).returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                checks[name] = False
    checks["upstreams"] = {}
    for name, entry in read_json(ROOT / "upstreams.lock.json").items():
        path = ROOT / "vendor" / name
        try:
            result = subprocess.run(["git", "-c", f"safe.directory={path.as_posix()}", "-C", str(path),
                                     "rev-parse", "HEAD"], capture_output=True, text=True)
            checks["upstreams"][name] = result.returncode == 0 and result.stdout.strip() == entry["commit"]
        except OSError:
            checks["upstreams"][name] = False
    print(json.dumps(checks, indent=2))


def _runs(args):
    for entry in write_run_index(args.runs_dir):
        print(f"{entry['started_at']}  {entry['kind']}  {entry['mode']}  {entry['status']}  {entry['directory']}")
    print(f"Log index: {Path(args.runs_dir).resolve() / 'INDEX.md'}")


def _prepare_swe(args):
    from datasets import load_dataset
    config = args.settings
    if config["swe_dataset"] == DATASET:
        raise ValueError("Use prepare-evo-swe --chains ... for SWE-Chain-Evo")
    data = load_dataset(config["swe_dataset"], split=config["swe_split"])
    ids = args.ids or list(data["instance_id"])[args.offset:args.offset + args.limit]
    if not ids:
        raise ValueError("No tasks selected")
    selected = {row["instance_id"]: row for row in data if row["instance_id"] in ids}
    if set(ids) - selected.keys():
        raise ValueError(f"Unknown instances: {set(ids) - selected.keys()}")
    fields = ("instance_id", "problem_statement", "repo", "base_commit", "version", "image_name", "image")
    tasks = [{"id": iid, "benchmark": "swe", "dataset_fingerprint": data._fingerprint,
              "instance": {k: v for k, v in selected[iid].items() if k in fields}} for iid in ids]
    write_json(args.output, tasks)


def _prepare_terminal(args):
    from harbor.models.task.task import Task
    tasks = [{"id": safe_id(Path(p).name), "benchmark": "terminal", "path": str(Path(p).resolve())} for p in args.paths]
    for task in tasks:
        Task(task["path"])
    write_json(args.output, tasks)


def _prepare_evo(args):
    from .evo_swe import download_chains, prepare_tasks
    if args.download:
        download_chains(args.data_dir, args.chains, args.revision)
    tasks = prepare_tasks(args.data_dir, args.chains, args.output)
    print(f"Prepared {len(args.chains)} chains, {len(tasks)} milestones: {args.output}")


def _evo_tasks(args):
    tasks = load_tasks(args.tasks, args.settings)
    validate_order(tasks)
    if any(t["benchmark"] != "swe-evo" for t in tasks):
        raise ValueError("This command requires a SWE-Chain-Evo manifest")
    return tasks[:args.limit] if args.limit else tasks


def _prepare_images(args):
    from .evo_swe import prepare_images
    tasks = _evo_tasks(args)
    validate_environments(tasks, args.settings, require_images=False)
    prepare_images(tasks, args.cache, args.settings["image_pull_timeout"])


def _verify_evo(args):
    from .evo_swe import verify_references
    tasks = _evo_tasks(args)
    validate_environments(tasks, args.settings)
    verify_references(tasks, args.settings, args.output)


def _run(args):
    config = args.settings
    if args.resume:
        from .checkpoints import resume_request, run_lock
        output, config, groups = resume_request(args.resume)
        if not shutil.which("docker"):
            raise RuntimeError("Docker CLI is required")
        require_credentials(config)
        if len(groups) == 2:
            from .workflow import run_train_test, validate_train_test
            validate_train_test(*groups)
        else:
            validate_order(groups[0])
        validate_environments([t for group in groups for t in group], config)
        with run_lock(output):
            if len(groups) == 2:
                result = run_train_test(*groups, config, output, resume=True)
            else:
                from .runner import run_experiment
                result = summary_metrics(run_experiment(groups[0], config, output, resume=True))
        print(json.dumps(result, indent=2))
        return
    config["evolve"] = config["evolve"] and not args.fixed_skill
    if args.initial_skill is not None:
        config["initial_skill"] = str(Path(args.initial_skill).expanduser().resolve())
    config["wandb"] = {**(config.get("wandb") or {}), "enabled": bool(args.wandb)}
    if not shutil.which("docker"):
        raise RuntimeError("Docker CLI is required; install Docker Desktop or configure Docker on Linux")
    require_credentials(config)
    tasks = load_tasks(args.tasks, config)
    test_tasks = load_tasks(args.test_tasks, config) if args.test_tasks else []
    if args.test_tasks:
        from .workflow import run_train_test, validate_train_test
        validate_train_test(tasks, test_tasks)
    else:
        validate_order(tasks)
    validate_environments(tasks + test_tasks, config)
    if args.test_tasks:
        result = run_train_test(tasks, test_tasks, config, args.output, runs_dir=args.runs_dir, label=args.label)
    else:
        from .runner import run_experiment
        reports = run_experiment(tasks, config, args.output, runs_dir=args.runs_dir, label=args.label)
        result = summary_metrics(reports)
    print(json.dumps(result, indent=2))


def _probe(args):
    from .model import probe_model
    require_credentials(args.settings, solver_only=True)
    print(json.dumps(probe_model(args.settings), indent=2))


def _positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description="Coding-agent skill evolution and frozen evaluation", allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Check dependencies and the configured model credentials", allow_abbrev=False)
    doctor.add_argument("--config", default=DEFAULT_CONFIG)
    doctor.set_defaults(handler=_doctor)
    listing = sub.add_parser("runs", help="Refresh the experiment log index", allow_abbrev=False)
    listing.add_argument("--runs-dir", default=ROOT / "runs")
    listing.set_defaults(handler=_runs)
    prepare = sub.add_parser("prepare-swe", help="Prepare SWE-bench tasks", allow_abbrev=False)
    choice = prepare.add_mutually_exclusive_group()
    choice.add_argument("--ids", nargs="+")
    choice.add_argument("--limit", type=_positive_int, default=1)
    prepare.add_argument("--offset", type=int, default=0)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--config", default=DEFAULT_CONFIG)
    prepare.set_defaults(handler=_prepare_swe)
    terminal = sub.add_parser("prepare-terminal", help="Prepare local Terminal-Bench tasks", allow_abbrev=False)
    terminal.add_argument("--paths", nargs="+", required=True)
    terminal.add_argument("--output", required=True)
    terminal.set_defaults(handler=_prepare_terminal)
    evo = sub.add_parser("prepare-evo-swe", help="Prepare complete SWE-Chain-Evo chains", allow_abbrev=False)
    evo.add_argument("--data-dir", default=ROOT / "datasets/evoarena-swe")
    evo.add_argument("--chains", nargs="+", required=True)
    evo.add_argument("--output", required=True)
    evo.add_argument("--download", action="store_true")
    evo.add_argument("--revision", default=REVISION)
    evo.set_defaults(handler=_prepare_evo)
    for name, handler, help_text in (
        ("prepare-evo-images", _prepare_images, "Download and prepare milestone images"),
        ("verify-evo-swe", _verify_evo, "Grade reference patches without model calls"),
    ):
        command = sub.add_parser(name, help=help_text, allow_abbrev=False)
        command.add_argument("--config", default=DEFAULT_CONFIG)
        command.add_argument("--tasks", required=True)
        command.add_argument("--limit", type=_positive_int, help="Prepare/check only the first N milestones")
        if name == "prepare-evo-images":
            command.add_argument("--cache", default=ROOT / "datasets/evoarena-swe/image-cache")
        else:
            command.add_argument("--output")
        command.set_defaults(handler=handler)
    run = sub.add_parser("run", help="Evolve a skill, evaluate a frozen skill, or train then test", allow_abbrev=False)
    run.add_argument("--config")
    run.add_argument("--model", help="Override solver/judge/planner using config.allowed_models")
    run.add_argument("--execution-mode", choices=list(MODES), help="Execution protocol; disables planner")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--tasks", help="Full task manifest for a new run")
    source.add_argument("--resume", help="Continue an interrupted run or train/test workflow in place")
    behavior = run.add_mutually_exclusive_group()
    behavior.add_argument("--test-tasks", help="Automatically evaluate these tasks with the trained skill frozen")
    behavior.add_argument("--fixed-skill", action="store_true", help="Freeze the skill; within-task reflection stays enabled")
    destination = run.add_mutually_exclusive_group()
    destination.add_argument("--output", help="Exact output directory; existing directories are never overwritten")
    destination.add_argument("--runs-dir", help="Parent for automatically named runs (default: runs/)")
    run.add_argument("--label", help="Label included in automatic directory names")
    run.add_argument("--initial-skill", help="Load a workflow, training run, skill store, version directory or SKILL.md")
    run.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=None, help="Upload metrics (default: disabled)")
    run.set_defaults(handler=_run)
    probe = sub.add_parser("probe-model", help="Send one solver-model request and check its bash action", allow_abbrev=False)
    probe.add_argument("--config", default=DEFAULT_CONFIG)
    probe.add_argument("--model", help="Model id from config.allowed_models")
    probe.set_defaults(handler=_probe)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "resume", None) and (
            any(getattr(args, key, None) for key in ("config", "model", "execution_mode", "test_tasks", "fixed_skill",
                                                    "initial_skill", "output", "runs_dir", "label")) or args.wandb is not None):
        parser.error("--resume uses saved settings; do not combine it with run overrides")
    load_env()
    if hasattr(args, "config"):
        try:
            args.settings = load_config(Path(args.resume).resolve() / "config.json" if getattr(args, "resume", None)
                                        else args.config or DEFAULT_CONFIG)
            select_model(args.settings, getattr(args, "model", None))
            if getattr(args, "execution_mode", None):
                set_execution_mode(args.settings, args.execution_mode)
            if args.command == "run":
                if args.test_tasks and not args.settings["evolve"]:
                    raise ValueError("--test-tasks requires evolve=true for training")
                if args.wandb and not os.getenv("WANDB_API_KEY"):
                    raise ValueError("--wandb requires WANDB_API_KEY in the environment")
            if args.command == "prepare-swe" and (args.offset < 0 or (args.ids and args.offset)):
                raise ValueError("--offset must be nonnegative and cannot be combined with --ids")
        except ValueError as exc:
            parser.error(str(exc))
    return args.handler(args)


if __name__ == "__main__":
    main()
