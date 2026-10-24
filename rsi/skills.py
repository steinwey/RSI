import difflib
import hashlib
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from . import ROOT
from .common import digest, file_map, read_json, write_json

SCRIPT_PATH = re.compile(r"scripts/[A-Za-z][A-Za-z0-9_]{0,40}\.py")
MAX_FILE_CHARS = 12000


class SkillProposalRejected(ValueError):
    """An invalid candidate leaves the current skill unchanged."""


def editable(path):
    return path == "SKILL.md"


def skill_files(skill):
    """Guide plus read-only tools available as updater context; exclude fixed checks."""
    return {k: v for k, v in file_map(skill).items() if editable(k) or SCRIPT_PATH.fullmatch(k)}


def load_initial_skill(source=None):
    """Read and validate a seed once; never change its source store or pointer."""
    builtin = ROOT / "skills/test-and-diagnose"
    fixed_files = {k: v for k, v in file_map(builtin).items() if k != "SKILL.md"}
    if source is not None and (not isinstance(source, (str, Path)) or not str(source).strip()):
        raise ValueError("initial_skill must be a nonempty path")
    path = Path(source).expanduser().resolve() if source is not None else builtin.resolve()
    requested = path
    pointer = None
    kind = "builtin" if source is None else "directory"
    if path.is_dir():
        if (path / "workflow.json").is_file() and (path / "train/skills/current.json").is_file():
            pointer, kind = path / "train/skills/current.json", "workflow"
        elif (path / "skills/current.json").is_file():
            pointer, kind = path / "skills/current.json", "run"
        elif (path / "current.json").is_file():
            pointer, kind = path / "current.json", "store"
    elif path.is_file() and path.name == "current.json":
        pointer, kind = path, "pointer"
    elif path.is_file() and path.name == "SKILL.md":
        kind = "document"
    else:
        raise ValueError(f"Initial skill not found or unsupported: {path}")
    if pointer is not None:
        try:
            version = read_json(pointer)["version"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid skill pointer: {pointer}") from exc
        if not isinstance(version, str) or not re.fullmatch(r"[0-9a-f]{16}", version):
            raise ValueError(f"Invalid skill version in pointer: {pointer}")
        path = pointer.parent / "versions" / version
    elif kind == "directory" and re.fullmatch(r"[0-9a-f]{16}", path.name):
        kind = "version"
    if kind != "document" and not (path / "SKILL.md").is_file():
        raise ValueError(f"Initial skill directory must contain SKILL.md: {path}")
    files = ({**fixed_files, "SKILL.md": path.read_text(encoding="utf-8")}
             if kind == "document" else file_map(path))
    validate_document(files, fixed_files)
    version = digest(files)
    if (pointer is not None or kind == "version") and version != path.name:
        raise ValueError(f"Initial skill content differs from its version: {path}")
    metadata = {"kind": kind, "requested_path": str(requested), "resolved_path": str(path.resolve()),
                "version": version, "tools_hash": digest(fixed_files),
                "document_sha256": hashlib.sha256(files["SKILL.md"].encode("utf-8")).hexdigest()}
    return files, metadata


class SkillStore:
    def __init__(self, root, seed):
        self.root = Path(root)
        self.pointer = self.root / "current.json"
        files = dict(seed) if isinstance(seed, dict) else file_map(seed)
        self.fixed_files = {k: v for k, v in files.items() if k != "SKILL.md"}
        self.tools_hash = digest(self.fixed_files)
        if not self.pointer.exists():
            version = self.materialize(files)
            write_json(self.pointer, {"version": version})

    def materialize(self, files):
        version = digest(files)
        folder = self.root / "versions" / version
        for name, text in files.items():
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
        return version

    def current(self):
        version = read_json(self.pointer)["version"]
        return self.root / "versions" / version

    def propose(self, changes):
        if not isinstance(changes, dict) or any(p != "SKILL.md" for p in changes):
            raise SkillProposalRejected("Only SKILL.md may be changed; tools and checks are fixed")
        if "SKILL.md" in changes and changes["SKILL.md"] is None:
            raise SkillProposalRejected("SKILL.md cannot be deleted")
        old = file_map(self.current())
        new = old | changes
        try:
            validate_document(new, self.fixed_files)
        except ValueError as exc:
            raise SkillProposalRejected(str(exc)) from exc
        version = self.materialize(new)
        diff = "\n".join("\n".join(difflib.unified_diff(old.get(p, "").splitlines(), new.get(p, "").splitlines(),
                         fromfile=f"old/{p}", tofile=f"new/{p}", lineterm="")) for p in sorted(changes))
        return self.root / "versions" / version, diff

    def record(self, candidate, accepted, evidence):
        if accepted:
            validate_document(file_map(candidate), self.fixed_files)
        before = self.current().name
        event = {"timestamp": datetime.now(timezone.utc).isoformat(), "before": before,
                 "after": candidate.name, "accepted": accepted, **evidence, "tools_hash": self.tools_hash}
        # EvoMem-inspired append-only transition evidence, including rejected proposals.
        import json
        with (self.root / "history.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        if accepted:
            write_json(self.pointer, {"version": candidate.name})


def validate_document(files, fixed_files):
    if {k: v for k, v in files.items() if k != "SKILL.md"} != fixed_files:
        raise ValueError("Fixed tools or checks were modified")
    guide = files.get("SKILL.md")
    if not isinstance(guide, str) or not guide.strip():
        raise ValueError("SKILL.md must be non-empty text")
    if len(guide) > MAX_FILE_CHARS:
        raise ValueError(f"SKILL.md exceeds {MAX_FILE_CHARS} characters; merge existing rules")
    for ref in re.findall(r"scripts/[A-Za-z0-9_./-]+\.py", guide):
        if ref not in fixed_files:
            raise ValueError(f"Unknown fixed tool reference: {ref}")


def document_gate(candidate, fixed_files):
    try:
        validate_document(file_map(candidate), fixed_files)
        return {"passed": True, "tools_hash": digest(fixed_files)}
    except ValueError as exc:
        return {"passed": False, "error": str(exc)}


def functional_gate(candidate, fixed_tests, image="python:3.12-slim"):
    """Check the initial fixed tools in a restricted disposable container."""
    name = "rsi-gate-" + uuid.uuid4().hex[:12]
    command = ["docker", "run", "--rm", "--name", name, "--network", "none", "--read-only",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--pids-limit", "64", "--memory", "256m", "--cpus", "1",
               "--tmpfs", "/tmp:rw,nosuid,size=64m", "-e", "PYTHONDONTWRITEBYTECODE=1",
               "-v", f"{candidate.resolve()}:/skill:ro",
               "-v", f"{fixed_tests.resolve()}:/checks:ro", image,
               "python", "/checks/check_skill.py", "/skill"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=900)
        return {"passed": result.returncode == 0, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"passed": False, "error": str(exc)}
    finally:
        # Killing a timed-out docker client does not necessarily stop its container.
        try:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            pass
