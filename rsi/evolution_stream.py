"""Construct chronological chain blocks for an online skill-evolution benchmark."""
import argparse
import hashlib
import random
from collections import Counter
from pathlib import Path

from . import ROOT
from .common import read_json, write_json
from .dataset_split import prior_exposure, select_longest_chains, source_id
from .evo_swe import validate_order


def build_streams(tasks, spec):
    validate_order(tasks)
    if not tasks or any(t["benchmark"] != "swe-evo" for t in tasks):
        raise ValueError("Expected a nonempty SWE-Chain-Evo manifest")
    development = spec["development_repositories"]
    benchmark = spec["benchmark_repositories"]
    repos = development + benchmark
    if not development or not benchmark or len(repos) != len(set(repos)):
        raise ValueError("Development and benchmark need nonempty disjoint repository groups")
    if set(repos) != {t["instance"]["repo"] for t in tasks}:
        raise ValueError("Repository groups must cover exactly the input repositories")
    seeds = spec["order_seeds"]
    if (not seeds or any(type(s) is not int for s in seeds)
            or len(seeds) != len(set(seeds)) or spec["primary_seed"] not in seeds):
        raise ValueError("Supply unique integer order seeds including primary_seed")
    development_tasks = select_longest_chains(tasks, development)
    streams = {}
    for seed in seeds:
        order = list(benchmark)
        random.Random(seed).shuffle(order)
        rows = select_longest_chains(tasks, order)
        validate_order(development_tasks, rows)
        dev_prompts = {" ".join(t["instance"]["problem_statement"].split()) for t in development_tasks}
        if dev_prompts & {" ".join(t["instance"]["problem_statement"].split()) for t in rows}:
            raise ValueError("Identical problem statement overlaps development and benchmark")
        for group in (development_tasks, rows):
            ids = [source_id(t) for t in group]
            if len(ids) != len(set(ids)):
                raise ValueError("Primary stream contains a repeated original task")
        streams[seed] = rows
    return development_tasks, streams


def episode_plan(rows, *, warmup_count=0):
    """Host-side schedule only; never add future task descriptions to solver prompts."""
    return [{"episode": i + 1, "phase": "warmup" if i < warmup_count else "benchmark",
             "task_id": task["id"], "source_instance_id": source_id(task),
             "repository": task["instance"]["repo"], "chain_id": task["chain"]["id"],
             "chain_index": task["chain"]["index"], "image": task["instance"]["image"]}
            for i, task in enumerate(rows)]


def write_streams(source, spec_path, output, data_dir, runs_dir):
    tasks, spec = read_json(source), read_json(spec_path)
    dataset = read_json(Path(data_dir) / "dataset-source.json")
    if dataset["revision"] != spec["dataset_revision"]:
        raise ValueError("Dataset revision differs from stream specification")
    development, streams = build_streams(tasks, spec)
    output = Path(output)
    artifacts = {"development.json": development}
    for seed, rows in streams.items():
        artifacts[f"benchmark-seed-{seed}.json"] = rows
    primary = streams[spec["primary_seed"]]
    artifacts["benchmark.json"] = primary
    artifacts["warmup-and-benchmark.json"] = development + primary
    # Keep the unmodified official chains available, outside the primary no-replay protocol.
    for name, repos in (("development", spec["development_repositories"]),
                        ("benchmark", spec["benchmark_repositories"])):
        artifacts[f"full-chains/{name}.json"] = [t for t in tasks if t["instance"]["repo"] in repos]
    for filename, rows in artifacts.items():
        validate_order(rows)
        write_json(output / filename, rows)
    listed, executed = prior_exposure(runs_dir)
    languages = {}
    for task in development + primary:
        chain = task["chain"]
        path = Path(data_dir) / "chains" / chain["id"] / "instances" / f"{chain['index']:03d}-{source_id(task)}.json"
        languages[source_id(task)] = read_json(path)["raw"]["language"].lower()
    report = {
        "name": spec["name"], "dataset": dataset,
        "input_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "spec_sha256": hashlib.sha256(Path(spec_path).read_bytes()).hexdigest(),
        "semantics": {"skill_reset": "new_run_only", "skill_updates": "after_each_task_and_scoring_attempt",
                      "ordering": "seeded_repository_blocks_with_original_chain_order",
                      "task_environment": "official_reference_base_for_each_milestone",
                      "solver_patch_carried_forward": False, "official_score_sent_to_updater": False,
                      "primary_seed": spec["primary_seed"]},
        "manifests": {name: {
            "tasks": len(rows), "unique_tasks": len({source_id(t) for t in rows}),
            "repositories": len({t["instance"]["repo"] for t in rows}),
            "chains": len({t["chain"]["id"] for t in rows}),
            "languages": dict(Counter(languages[source_id(t)] for t in rows)),
            "previously_listed_tasks": len({source_id(t) for t in rows} & listed),
            "previously_executed_tasks": len({source_id(t) for t in rows} & executed),
            "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(),
        } for name, rows in artifacts.items()},
        "schedules": {f"benchmark-seed-{seed}.json": episode_plan(rows) for seed, rows in streams.items()},
        "warmup_schedule": episode_plan(development + primary, warmup_count=len(development)),
        "limitations": [
            "Online adaptation is evaluated; this is not frozen-skill unseen-repository evaluation.",
            "Benchmark comprises seven Go repositories and one Python repository; no claim about other languages.",
            "Seed variants reuse tasks and are repeated order experiments, not independent new task samples.",
            "Only local runs are checked for prior exposure; no near-duplicate-code or pretraining-contamination audit.",
            "full-chains manifests contain replay and may revisit earlier versions; not the primary dynamic stream.",
            "Solver errors can abort a run and scoring infrastructure failures remain unscored; report coverage."
        ],
    }
    write_json(output / "protocol.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "tasks/evoarena-all.json")
    parser.add_argument("--spec", type=Path, default=ROOT / "configs/evoarena-dynamic.json")
    parser.add_argument("--output", type=Path, default=ROOT / "tasks/streams/evoarena-dynamic-v1")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets/evoarena-swe")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    report = write_streams(args.source, args.spec, args.output, args.data_dir, args.runs_dir)
    for name in ("development.json", "benchmark.json", "warmup-and-benchmark.json"):
        info = report["manifests"][name]
        print(f"{name}: {info['tasks']} tasks, {info['repositories']} repositories, {info['chains']} chains")
    print(f"Protocol: {args.output / 'protocol.json'}")


if __name__ == "__main__":
    main()
