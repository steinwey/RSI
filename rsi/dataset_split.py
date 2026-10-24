"""Build repository-held-out manifests without changing official chain metadata."""
import argparse
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

from . import ROOT
from .common import read_json, write_json
from .evo_swe import validate_order


PARTITIONS = ("train", "dev", "test", "reserve")


def source_id(task):
    return task["chain"]["source_instance_id"]


def select_longest_chains(tasks, repositories):
    """Select original complete chains, refusing to silently omit unique tasks."""
    if len(repositories) != len(set(repositories)):
        raise ValueError("Repository selection contains duplicates")
    by_repo = defaultdict(lambda: defaultdict(list))
    for task in tasks:
        by_repo[task["instance"]["repo"]][task["chain"]["id"]].append(task)
    selected = []
    for repo in repositories:
        if repo not in by_repo:
            raise ValueError(f"Unknown repository: {repo}")
        chains = by_repo[repo]
        longest = min(chains.values(), key=lambda rows: (-len(rows), rows[0]["chain"]["id"]))
        all_ids = {source_id(t) for rows in chains.values() for t in rows}
        if {source_id(t) for t in longest} != all_ids:
            raise ValueError(f"Longest chain does not cover all unique tasks: {repo}")
        selected.extend(longest)
    validate_order(selected)
    return selected


def order_repositories(assignments, seeds):
    """Shuffle whole repositories independently without changing membership."""
    if not isinstance(seeds, dict) or not seeds.keys() <= assignments.keys():
        raise ValueError("Repository order seeds must name existing partitions")
    ordered = {p: list(repos) for p, repos in assignments.items()}
    for partition, seed in seeds.items():
        if type(seed) is not int:
            raise ValueError("Repository order seeds must be integers")
        # Canonical input makes a seed independent of JSON list order and other partitions.
        ordered[partition].sort()
        random.Random(seed).shuffle(ordered[partition])
    return ordered


def build_split(tasks, assignments, *, repository_order_seeds=None):
    """Return complete-chain views; reject overlap and omitted unique tasks."""
    validate_order(tasks)
    if not tasks or any(t["benchmark"] != "swe-evo" for t in tasks):
        raise ValueError("Expected a nonempty SWE-Chain-Evo manifest")
    if (not {"train", "test"} <= assignments.keys() or not assignments.keys() <= set(PARTITIONS)
            or any(not rows for rows in assignments.values())):
        raise ValueError("Supply nonempty train/test lists and optional dev/reserve lists")
    if repository_order_seeds is not None:
        assignments = order_repositories(assignments, repository_order_seeds)
    partitions = [p for p in PARTITIONS if p in assignments]
    owners = {}
    for partition in partitions:
        for repo in assignments[partition]:
            if repo in owners:
                raise ValueError(f"Repository assigned more than once: {repo}")
            owners[repo] = partition
    by_repo = defaultdict(lambda: defaultdict(list))
    for task in tasks:
        by_repo[task["instance"]["repo"]][task["chain"]["id"]].append(task)
    if set(owners) != set(by_repo):
        raise ValueError("Assignments must cover exactly the input repositories")
    full, primary = {}, {}
    for partition in partitions:
        full[partition], primary[partition] = [], []
        for repo in assignments[partition]:
            chains = by_repo[repo]
            for cid in sorted(chains):
                full[partition].extend(chains[cid])
            primary[partition].extend(select_longest_chains(tasks, [repo]))
        validate_order(full[partition])
        validate_order(primary[partition])
        ids = [source_id(t) for t in primary[partition]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Repeated original task in primary {partition}")
    # ID checks catch source aliases; content checks catch identical questions under new IDs.
    for identity in (
        source_id,
        lambda t: (t["instance"]["repo"], t["instance"]["base_commit"]),
        lambda t: " ".join(t["instance"]["problem_statement"].split()),
    ):
        seen = {}
        for partition in partitions:
            for task in full[partition]:
                key = identity(task)
                if key in seen and seen[key] != partition:
                    raise ValueError(f"Task identity or identical prompt overlaps {seen[key]} and {partition}")
                seen[key] = partition
    return primary, full


def prior_exposure(runs_dir):
    """Read identifiers only, never scores, patches or model responses."""
    listed, executed = set(), set()
    root = Path(runs_dir)
    paths = [path for pattern in ("*/tasks.json", "*/train/tasks.json", "*/test/tasks.json")
             for path in root.glob(pattern)]
    for path in sorted(paths):
        for task in read_json(path):
            if not task.get("chain"):
                continue
            iid = source_id(task)
            listed.add(iid)
            folder = path.parent / task["id"]
            if (folder / "summary.json").exists() or any(folder.glob("rollout-*/trajectory.json")):
                executed.add(iid)
    return listed, executed


def write_split(source, spec_path, output, data_dir, runs_dir, image_tags=None):
    tasks, spec = read_json(source), read_json(spec_path)
    source_info = read_json(Path(data_dir) / "dataset-source.json")
    if source_info["revision"] != spec["dataset_revision"]:
        raise ValueError("Dataset revision differs from split specification")
    order_seeds = spec.get("repository_order_seeds", {})
    primary, full = build_split(tasks, spec["repositories"], repository_order_seeds=order_seeds)
    repository_order = {p: list(dict.fromkeys(t["instance"]["repo"] for t in rows))
                        for p, rows in primary.items()}
    languages = {}
    for rows in primary.values():
        for task in rows:
            chain = task["chain"]
            path = Path(data_dir) / "chains" / chain["id"] / "instances" / f"{chain['index']:03d}-{source_id(task)}.json"
            languages[source_id(task)] = read_json(path)["raw"]["language"].lower()
    language_counts = {p: dict(Counter(languages[source_id(t)] for t in rows)) for p, rows in primary.items()}
    for partition, expected in spec.get("expected_language_tasks", {}).items():
        if partition not in primary or language_counts[partition] != expected:
            raise ValueError(f"Language task counts differ from fixed specification: {partition}")
    tags = None if image_tags is None else set(Path(image_tags).read_text(encoding="utf-8-sig").splitlines())
    for partition in spec.get("require_cached_partitions", []):
        if partition not in primary:
            raise ValueError(f"Unknown required cached partition: {partition}")
        if tags is None:
            raise ValueError("This split requires --image-tags to verify active environments")
        missing = {t["instance"]["image"] for t in primary[partition]} - tags
        if missing:
            raise ValueError(f"Missing {len(missing)} images in {partition}; keep original split or prepare environments")
    prefix_lengths = list(spec.get("training_prefix_lengths", []))
    for repo_count in spec.get("training_prefix_repository_counts", []):
        if type(repo_count) is not int or not 1 <= repo_count <= len(repository_order["train"]):
            raise ValueError("Training prefix repository count must fit the training set")
        repos = set(repository_order["train"][:repo_count])
        prefix_lengths.append(sum(t["instance"]["repo"] in repos for t in primary["train"]))
    if any(type(count) is not int or not 1 <= count <= len(primary["train"]) for count in prefix_lengths):
        raise ValueError("Training prefix length must fit the training set")
    prefixes = {}
    for count in sorted(set(prefix_lengths)):
        prefix = primary["train"][:count]
        validate_order(prefix)
        prefixes[f"training-prefixes/train-{count}.json"] = prefix
    listed, executed = prior_exposure(runs_dir)
    output = Path(output)
    repo_rows = []
    for filename, rows in prefixes.items():
        write_json(output / filename, rows)
    for partition in primary:
        write_json(output / f"{partition}.json", primary[partition])
        write_json(output / "full-chains" / f"{partition}.json", full[partition])
        for repo in repository_order[partition]:
            rows = [t for t in full[partition] if t["instance"]["repo"] == repo]
            selected = [t for t in primary[partition] if t["instance"]["repo"] == repo]
            language_tasks = Counter(languages[source_id(t)] for t in selected)
            ids = {source_id(t) for t in rows}
            images = {t["instance"]["image"] for t in selected}
            repo_rows.append({
                "repository": repo, "partition": partition, "languages": sorted(language_tasks),
                "language_tasks": dict(language_tasks),
                "original_chains": len({t["chain"]["id"] for t in rows}),
                "original_slots": len(rows), "unique_tasks": len(ids),
                "selected_chain": selected[0]["chain"]["id"],
                "cached_image_tags": None if tags is None else len(images & tags),
                "missing_images": None if tags is None else sorted(images - tags),
                "previously_listed_tasks": len(ids & listed), "previously_executed_tasks": len(ids & executed),
            })
    report = {
        "name": spec["name"], "dataset": source_info,
        "skill_policy": spec.get("skill_policy"),
        "ordering": {"unit": "repository", "within_chain": "original_chronological_order",
                     "repository_order_seeds": order_seeds, "repository_order": repository_order},
        "input_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "spec_sha256": hashlib.sha256(Path(spec_path).read_bytes()).hexdigest(),
        "original": {"repositories": len(repo_rows), "chains": len({t['chain']['id'] for t in tasks}),
                     "slots": len(tasks), "unique_tasks": len({source_id(t) for t in tasks})},
        "partitions": {p: {
            "repositories": len(spec['repositories'][p]), "primary_tasks": len(primary[p]),
            "primary_chains": len({t['chain']['id'] for t in primary[p]}), "original_slots": len(full[p]),
            "language_tasks": language_counts[p],
            "manifest_sha256": hashlib.sha256((output / f"{p}.json").read_bytes()).hexdigest(),
            "cached_image_tags": None if tags is None else sum(t['instance']['image'] in tags for t in primary[p]),
        } for p in primary},
        "training_prefixes": {name: {"tasks": len(rows), "repositories": len({t['instance']['repo'] for t in rows}),
                                      "last_task": rows[-1]["id"],
                                      "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest()}
                              for name, rows in prefixes.items()},
        "repositories": repo_rows,
        "checks": {"repository_overlap": 0, "source_id_overlap": 0, "repo_base_overlap": 0,
                   "identical_prompt_overlap": 0, "primary_unique_coverage": True, "complete_original_chains": True},
        "limits": [
            "No near-duplicate code or fork genealogy audit was performed; exact identity/prompt checks do not prove their absence.",
            "Prior exposure covers identifiers in this local runs directory only, not other machines or pretrained model data.",
            "Cache counts check tag presence only, not repository isolation or reference patch scoring.",
            "New runs default to the built-in skill; use --initial-skill with --fixed-skill to freeze a separate copy of a training artifact during evaluation."
        ],
    }
    write_json(output / "audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "tasks/evoarena-all.json")
    parser.add_argument("--spec", type=Path, default=ROOT / "configs/evoarena-frozen-shuffled.json")
    parser.add_argument("--output", type=Path, default=ROOT / "tasks/splits/evoarena-frozen-v3")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets/evoarena-swe")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--image-tags", type=Path, help="Optional output of docker image ls --format '{{.Repository}}:{{.Tag}}'")
    args = parser.parse_args()
    report = write_split(args.source, args.spec, args.output, args.data_dir, args.runs_dir, args.image_tags)
    for partition, stats in report["partitions"].items():
        print(f"{partition}: {stats['repositories']} repositories, {stats['primary_tasks']} tasks")
    print(f"Audit: {args.output / 'audit.json'}")


if __name__ == "__main__":
    main()
