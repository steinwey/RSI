"""Adapt official SWE-Chain-Evo milestones to the pinned SWE-bench harness."""
import base64
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
import urllib.request
from pathlib import Path, PurePosixPath

from .common import digest, read_json, safe_id, write_json

DATASET = "Aiden0526/SWE-Chain-Evo"
REVISION = "f98ba6aed4c1a40c527f6d70c186a6887888a332"
IMAGE_DATASET = "zhiyuanhucs/SWE-EVO-50chain-images"
IMAGE_REVISION = "b73f94440a191aa2f7c47cdd83747e6f8b8cfa0b"


def hub_endpoint():
    """Domestic mirror by default. Set HF_ENDPOINT=https://huggingface.co to use the origin."""
    return os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")


def _fetch(path, repo=DATASET, revision=REVISION):
    url = f"{hub_endpoint()}/datasets/{repo}/resolve/{revision}/{path}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def download_chains(root, chain_ids, revision=REVISION):
    root = Path(root)
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use a pinned 40-character dataset revision")
    for chain_id in chain_ids:
        safe_id(chain_id)
        relative = f"chains/{chain_id}/chain.json"
        content = _fetch(relative, revision=revision)
        manifest = json.loads(content)
        directory = root / "chains" / chain_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "chain.json").write_bytes(content)
        for entry in manifest["instances"]:
            iid = safe_id(entry["instance_id"])
            index = entry["version_chain_index"]
            filename = f"instances/{index:03d}-{iid}.json"
            destination = directory / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(_fetch(f"chains/{chain_id}/{filename}", revision=revision))
    write_json(root / "dataset-source.json", {"dataset": DATASET, "revision": revision})


def _relative_file(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe repository path: {name!r}")
    return path.as_posix()


def eval_script(row):
    """Restore official test files after applying a model patch, then run tests."""
    raw = row["raw"]
    base = raw["base_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", base):
        raise ValueError("Invalid base commit")
    language = raw.get("language", "").lower()
    if language not in ("python", "go"):
        raise ValueError(f"Unsupported EvoArena test language: {language!r}")
    commands = raw.get("test_cmds")
    if not isinstance(commands, list) or not commands or any(not isinstance(c, str) or not c.strip() for c in commands):
        raise ValueError("A milestone must supply executable test_cmds")
    for command in commands:
        if language == "python" and "pytest" not in command:
            raise ValueError("Python milestones must use pytest logs")
        if language == "go" and not re.search(r"\bgo test\b.*(?:^|\s)-v(?:\s|$)", command):
            raise ValueError("Go milestones must use verbose go test logs")
    files = set(raw.get("files", {}).get("test", []))
    for line in raw["test_patch"].splitlines():
        if line.startswith("diff --git "):
            parts = shlex.split(line)
            files.update(part[2:] for part in parts[2:] if part.startswith(("a/", "b/")))
    lines = ["#!/bin/bash", "set -e", "cd /testbed",
             "if [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then",
             "  source /opt/miniconda3/etc/profile.d/conda.sh; conda activate testbed", "fi"]
    if language == "python":
        lines.append('export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} -rA --color=no"')
    for name in sorted(files):
        name = _relative_file(name)
        path = shlex.quote(name)
        lines.append(f"if git cat-file -e {shlex.quote(base + ':' + name)} 2>/dev/null; then "
                     f"git checkout {base} -- {path}; else rm -f -- {path}; fi")
    payload = base64.b64encode(raw["test_patch"].encode()).decode()
    if payload:
        lines.extend([f"printf %s {shlex.quote(payload)} | base64 -d > /tmp/rsi-evo-tests.patch",
                      "git apply --whitespace=nowarn /tmp/rsi-evo-tests.patch"])
    # One compound command makes the harness capture the tests' exit code.
    lines.extend(["set +e", "echo '>>>>> Start Test Output'",
                  " && ".join(f"( {command} )" for command in commands),
                  "echo '>>>>> End Test Output'"])
    return "\n".join(lines) + "\n"


def prepare_tasks(root, chain_ids, output):
    """Write a solver-safe manifest and a separate grading-only local dataset."""
    root = Path(root).resolve()
    tasks, evaluation = [], []
    for chain_id in chain_ids:
        safe_id(chain_id)
        directory = root / "chains" / chain_id
        manifest = read_json(directory / "chain.json")
        entries = manifest["instances"]
        count = manifest["instance_count"]
        if manifest["chain_id"] != chain_id or count != len(entries) or not count:
            raise ValueError("Invalid or incomplete chain manifest")
        if [e["version_chain_index"] for e in entries] != list(range(count)):
            raise ValueError("Chain instances must be in complete chronological order")
        for index, entry in enumerate(entries):
            iid = safe_id(entry["instance_id"])
            row = read_json(directory / "instances" / f"{index:03d}-{iid}.json")
            raw = row["raw"]
            if (row["instance_id"] != iid or row["version_chain_id"] != chain_id
                    or row["version_chain_index"] != index or row["repo"] != manifest["repo"]
                    or row["image"] != entry["image"] or row["version"] != entry["version"]
                    or raw["instance_id"] != iid or raw["repo"] != row["repo"]):
                raise ValueError(f"Milestone differs from chain manifest: {iid}")
            if row["image"] != f"sweb.eval.x86_64.{iid.lower()}:latest":
                raise ValueError("Unexpected official image tag")
            f2p, p2p = row["fail_to_pass"], row["pass_to_pass"]
            if not f2p or not all(isinstance(t, str) and t for t in f2p + p2p):
                raise ValueError("Every milestone needs explicit Fail-to-Pass tests")
            task_id = f"evo-{digest(chain_id)[:8]}-{index:03d}"
            image = f"rsi-evo/{iid.lower()}:{raw['base_commit'][:12]}"
            instance = {"instance_id": task_id, "repo": row["repo"], "version": row["version"],
                        "base_commit": raw["base_commit"], "image": image,
                        "problem_statement": row["problem_statement"]}
            chain = {"id": chain_id, "index": index, "length": count, "source_instance_id": iid}
            tasks.append({"id": task_id, "benchmark": "swe-evo", "chain": chain,
                          "dataset_fingerprint": digest(row), "instance": instance})
            evaluation.append({**instance, "source_image": row["image"],
                               "image_archive": f"images/{row['image'].removeprefix('sweb.eval.x86_64.').removesuffix(':latest')}.tar",
                               "FAIL_TO_PASS": f2p, "PASS_TO_PASS": p2p,
                               "eval_type": "pass_and_fail", "eval_script": eval_script(row),
                               "log_parser": "parse_log_gotest" if raw["language"].lower() == "go" else "parse_log_pytest",
                               "reference_patch": raw["patch"], "chain": chain})
    if not tasks or len({t["id"] for t in tasks}) != len(tasks):
        raise ValueError("Select distinct nonempty chains")
    evaluation_path = root / f"evaluation-{digest(evaluation)}.json"
    write_json(evaluation_path, evaluation)
    fingerprint = hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
    for task in tasks:
        task["evaluation_dataset"] = str(evaluation_path)
        task["evaluation_sha256"] = fingerprint
    write_json(output, tasks)
    return tasks


def validate_order(tasks, validation=()):
    """Require whole chains and prevent held-out milestones leaking into training."""
    if any(t["benchmark"] == "swe-evo" for t in list(tasks) + list(validation)):
        if any(t["benchmark"] != "swe-evo" for t in list(tasks) + list(validation)):
            raise ValueError("Use separate complete EvoArena chains for both training and validation")
    for group in (tasks, validation):
        chains, finished = {}, set()
        previous = None
        for task in group:
            if task["benchmark"] != "swe-evo":
                if any(t["benchmark"] == "swe-evo" for t in group):
                    raise ValueError("Run EvoArena chains separately from other benchmarks")
                continue
            chain = task["chain"]
            cid = safe_id(chain["id"])
            if cid != previous and cid in finished:
                raise ValueError("Chain steps must be contiguous")
            if previous is not None and cid != previous:
                finished.add(previous)
            chains.setdefault(cid, []).append(chain)
            previous = cid
        for steps in chains.values():
            length = steps[0]["length"]
            if length != len(steps) or any(s["length"] != length for s in steps) or [s["index"] for s in steps] != list(range(length)):
                raise ValueError("Supply complete chains in chronological step order")
    training = {t["chain"]["source_instance_id"] for t in tasks if t.get("chain")}
    heldout = {t["chain"]["source_instance_id"] for t in validation if t.get("chain")}
    train_chains = {t["chain"]["id"] for t in tasks if t.get("chain")}
    val_chains = {t["chain"]["id"] for t in validation if t.get("chain")}
    if training & heldout or train_chains & val_chains:
        raise ValueError("Training and validation chains must not share milestones or chain IDs")


def evaluation_rows(tasks):
    rows, loaded = {}, {}
    for task in tasks:
        if task["benchmark"] != "swe-evo":
            continue
        path = Path(task["evaluation_dataset"])
        if path not in loaded:
            content = path.read_bytes()
            loaded[path] = (hashlib.sha256(content).hexdigest(),
                            {row["instance_id"]: row for row in json.loads(content)})
        fingerprint, by_id = loaded[path]
        if fingerprint != task["evaluation_sha256"]:
            raise ValueError("EvoArena evaluation metadata changed; prepare the manifest again")
        row = by_id[task["id"]]
        if row["chain"] != task["chain"] or any(row.get(k) != v for k, v in task["instance"].items()):
            raise ValueError("Task manifest differs from grading metadata")
        rows[task["id"]] = row
    return rows


def validate_tasks(tasks, config, *, require_images=True):
    from .swe import validate_swe_tasks
    selected = [t for t in tasks if t["benchmark"] == "swe-evo"]
    evaluation_rows(selected)
    for path in {t["evaluation_dataset"] for t in selected}:
        group = [{**t, "benchmark": "swe"} for t in selected if t["evaluation_dataset"] == path]
        validate_swe_tasks(group, {**config, "swe_dataset": path})
    if require_images:
        missing = [image for image in sorted({t["instance"]["image"] for t in selected})
                   if subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=30).returncode]
        if missing:
            raise RuntimeError("Missing EvoArena images; run prepare-evo-images first: " + ", ".join(missing))


def _download_image(path, metadata, timeout):
    """Resume interrupted archives and verify the publisher's LFS SHA256."""
    expected = metadata["lfs"]["oid"]
    size = metadata["size"]
    def valid(file):
        if not file.exists() or file.stat().st_size != size:
            return False
        with file.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest() == expected
    if valid(path):
        return
    part = path.with_suffix(path.suffix + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    for attempt in range(3):
        if valid(part):
            part.replace(path)
            return
        offset = part.stat().st_size if part.exists() else 0
        if offset >= size:
            offset = 0
            part.write_bytes(b"")
        request = urllib.request.Request(
            f"{hub_endpoint()}/datasets/{IMAGE_DATASET}/resolve/{IMAGE_REVISION}/{metadata['path']}",
            headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                append = offset and response.status == 206
                with part.open("ab" if append else "wb") as stream:
                    while block := response.read(1024 * 1024):
                        if time.monotonic() > deadline:
                            raise TimeoutError("EvoArena image download budget exhausted")
                        stream.write(block)
            if not valid(part):
                raise ValueError("Incomplete image archive or SHA256 mismatch")
            part.replace(path)
            return
        except (OSError, ValueError):
            if attempt == 2 or time.monotonic() > deadline:
                raise


def isolate_repository_script(base):
    """Drop commits that are not ancestors of the milestone, matching SWE-bench's future-history removal.

    The published archives keep later tags and branches. Resetting HEAD leaves those objects reachable.
    Grading only checks out this base commit and applies its hidden test patch, so those objects can go.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", base):
        raise ValueError("Invalid base commit")
    return "\n".join([
        "#!/bin/bash",
        "set -euo pipefail",
        "cd /testbed",
        "git config --global --add safe.directory /testbed",
        f"git reset --hard {base}",
        "git clean -fd",
        "while read -r remote; do",
        '  [ -n "$remote" ] && git remote remove "$remote"',
        "done < <(git remote)",
        "head_ref=$(git symbolic-ref -q HEAD || true)",
        "while read -r ref; do",
        '  [ -n "$ref" ] || continue',
        '  if [ "$ref" = "$head_ref" ]; then continue; fi',
        '  if git rev-parse -q --verify "${ref}^{commit}" >/dev/null && git merge-base --is-ancestor "${ref}^{commit}" HEAD; then',
        "    continue",
        "  fi",
        '  git update-ref -d "$ref"',
        "done < <(git for-each-ref --format='%(refname)')",
        "git reflog expire --expire=now --all",
        "git gc --prune=now",
        "count=$(git rev-list --all --not HEAD --count)",
        '[ "$count" -eq 0 ]',
        "",
    ])


def _git_in_image(image, *args, timeout=60):
    return subprocess.run(["docker", "run", "--rm", "--entrypoint", "git", image, *args],
                          capture_output=True, text=True, timeout=timeout)


def prepare_images(tasks, cache, timeout=7200):
    cache = Path(cache).resolve()
    rows = evaluation_rows(tasks)
    archives = None
    prepared = set()
    for row in rows.values():
        image = row["image"]
        if image in prepared:
            continue
        prepared.add(image)
        if subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=30).returncode == 0:
            check = _git_in_image(image, "rev-parse", "HEAD")
            future = _git_in_image(image, "rev-list", "--all", "--not", "HEAD", "--count")
            if (check.returncode == 0 and check.stdout.strip() == row["base_commit"]
                    and future.returncode == 0 and future.stdout.strip() == "0"):
                print(f"Image cached: {image}", flush=True)
                continue
        source = row["source_image"]
        if subprocess.run(["docker", "image", "inspect", source], capture_output=True, timeout=30).returncode:
            if archives is None:
                index = cache / f"archive-index-{IMAGE_REVISION}.json"
                if not index.exists():
                    url = f"{hub_endpoint()}/api/datasets/{IMAGE_DATASET}/tree/{IMAGE_REVISION}/images"
                    with urllib.request.urlopen(url, timeout=30) as response:
                        write_json(index, json.load(response))
                archives = {item["path"]: item for item in read_json(index)}
            archive = cache / row["image_archive"]
            print(f"Preparing archive: {row['image_archive']}", flush=True)
            _download_image(archive, archives[row["image_archive"]], timeout)
            subprocess.run(["docker", "load", "--input", str(archive)], check=True, timeout=timeout)
        directory = cache / "build" / row["instance_id"]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "isolate-history.sh").write_text(isolate_repository_script(row["base_commit"]),
                                                      encoding="utf-8", newline="\n")
        (directory / "Dockerfile").write_text(
            f"FROM {source}\nUSER root\nWORKDIR /testbed\nCOPY isolate-history.sh /tmp/isolate-history.sh\n"
            "RUN bash /tmp/isolate-history.sh && rm /tmp/isolate-history.sh\n",
            encoding="utf-8", newline="\n")
        subprocess.run(["docker", "build", "--network=none", "--pull=false", "--tag", image, str(directory)],
                       check=True, timeout=timeout)
        actual = _git_in_image(image, "rev-parse", "HEAD", timeout=60)
        future = _git_in_image(image, "rev-list", "--all", "--not", "HEAD", "--count", timeout=60)
        if actual.returncode != 0 or actual.stdout.strip() != row["base_commit"]:
            raise RuntimeError("Prepared image does not contain the expected oracle base snapshot")
        if future.returncode != 0 or future.stdout.strip() != "0":
            raise RuntimeError("Prepared image still contains commits after the milestone base")
        print(f"Image ready: {image}", flush=True)


def verify_references(tasks, config, output=None):
    """Exercise the real grading path without solver or updater model calls."""
    from .backends import SWEBackend
    from .run_logs import create_run_directory, update_run
    rows = evaluation_rows(tasks)
    config = {**config, "swe_dataset": DATASET}
    output = create_run_directory(tasks, [], config, output, label="evo-reference", kind="score-check")
    reports = []
    try:
        for task in tasks:
            folder = output / task["id"]
            (folder / "reference").mkdir(parents=True)
            (folder / "baseline").mkdir()
            # score() starts the harness's isolated verifier; no solver is needed.
            backend = SWEBackend.__new__(SWEBackend)
            backend.task, backend.folder = task, folder / "reference"
            backend.config = {**config, "swe_dataset": task["evaluation_dataset"]}
            result = backend.score(rows[task["id"]]["reference_patch"])
            write_json(folder / "reference/score.json", result)
            # A harmless new file exercises the identical patch/evaluation path
            # without fixing anything; it must leave at least one F2P test failing.
            control = ("diff --git a/.rsi-evo-control b/.rsi-evo-control\n"
                       "new file mode 100644\n--- /dev/null\n+++ b/.rsi-evo-control\n"
                       "@@ -0,0 +1 @@\n+RSI baseline calibration\n")
            backend.folder = folder / "baseline"
            baseline = backend.score(control)
            write_json(folder / "baseline/score.json", baseline)
            baseline_report = read_json(baseline["report"])[task["id"]]
            failed_tests = baseline_report.get("tests_status", {}).get("FAIL_TO_PASS", {}).get("failure", [])
            reports.append({"task_id": task["id"], "report": result["report"],
                            "reference_success": result["success"], "baseline_success": baseline["success"],
                            "baseline_f2p_failures": len(failed_tests),
                            "success": result["success"] is True and baseline["success"] is False and bool(failed_tests)})
            write_json(output / "score.json", {"success": all(r["success"] is True for r in reports), "reports": reports})
            if result["success"] is not True:
                raise RuntimeError(f"Reference patch failed for {task['id']}; inspect {folder / 'reference/score.json'}")
            if baseline["success"] is not False or not failed_tests:
                raise RuntimeError(f"Baseline calibration failed for {task['id']}; tests do not distinguish the fix")
        update_run(output, status="completed", completed_tasks=len(reports), scored_tasks=len(reports),
                   resolved=sum(r["success"] is True for r in reports))
    except BaseException as exc:
        update_run(output, status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    print(json.dumps({"output": str(output), "reports": reports}, indent=2))
    return reports
