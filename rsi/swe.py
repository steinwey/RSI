"""Dataset compatibility checks for the pinned SWE-bench evaluation harness."""


EVALUATION_FIELDS = {
    "instance_id", "image", "repo", "version", "FAIL_TO_PASS", "PASS_TO_PASS",
    "log_parser", "eval_type", "eval_script",
}


def evaluation_dataset_name(name):
    # The legacy namespace still serves rows without the v5 evaluation metadata.
    legacy = {
        f"princeton-nlp/{dataset}": f"SWE-bench/{dataset}"
        for dataset in ("SWE-bench", "SWE-bench_Lite", "SWE-bench_Verified")
    }
    return legacy.get(name, name)


def load_evaluation_rows(config, instance_ids):
    from swebench.harness.utils import load_swebench_dataset
    return load_swebench_dataset(
        evaluation_dataset_name(config["swe_dataset"]), config["swe_split"], instance_ids)


def validate_swe_tasks(tasks, config):
    """Fail before solver/API work; keep grading-only fields out of task manifests."""
    selected = [task for task in tasks if task["benchmark"] == "swe"]
    if not selected:
        return
    rows = load_evaluation_rows(config, sorted({task["id"] for task in selected}))
    by_id = {}
    for row in rows:
        missing = EVALUATION_FIELDS - row.keys()
        if missing:
            raise ValueError(
                f"SWE-bench dataset {config['swe_dataset']!r} is incompatible with the pinned "
                f"harness: missing evaluation fields {', '.join(sorted(missing))}. "
                "Use the current SWE-bench/SWE-bench_Verified dataset or a local dataset "
                "with complete evaluation metadata.")
        by_id[row["instance_id"]] = row
    for task in selected:
        row = by_id.get(task["id"])
        if row is None:
            raise ValueError(f"SWE-bench evaluation dataset has no instance {task['id']!r}")
        for field in ("repo", "base_commit", "version"):
            expected = task["instance"].get(field)
            if expected is not None and row.get(field) != expected:
                raise ValueError(
                    f"SWE-bench instance {task['id']!r}: {field} differs between the task "
                    "manifest and evaluation dataset; regenerate the manifest with prepare-swe.")
