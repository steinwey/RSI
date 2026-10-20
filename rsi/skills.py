import difflib
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from .common import digest, file_map, read_json, write_json

ALLOWED = {"SKILL.md", "scripts/inspect_project.py"}


class SkillStore:
    def __init__(self, root, seed):
        self.root = Path(root)
        self.pointer = self.root / "current.json"
        if not self.pointer.exists():
            files = file_map(seed)
            version = self.materialize(files)
            write_json(self.pointer, {"version": version})

    def materialize(self, files):
        version = digest(files)
        folder = self.root / "versions" / version
        for name, text in files.items():
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return version

    def current(self):
        version = read_json(self.pointer)["version"]
        return self.root / "versions" / version

    def propose(self, changes):
        if not isinstance(changes, dict) or set(changes) - ALLOWED:
            raise ValueError("Only SKILL.md and scripts/inspect_project.py may be changed")
        if any(not isinstance(text, str) or len(text) > 60000 for text in changes.values()):
            raise ValueError("Skill files must be text shorter than 60000 characters")
        old = file_map(self.current())
        new = old | changes
        if not new.get("SKILL.md", "").strip():
            raise ValueError("Skill description cannot be empty")
        compile(new["scripts/inspect_project.py"], "inspect_project.py", "exec")
        version = self.materialize(new)
        diff = "\n".join("\n".join(difflib.unified_diff(old.get(p, "").splitlines(), new[p].splitlines(),
                         fromfile=f"old/{p}", tofile=f"new/{p}", lineterm="")) for p in sorted(changes))
        return self.root / "versions" / version, diff

    def record(self, candidate, accepted, evidence):
        before = self.current().name
        event = {"timestamp": datetime.now(timezone.utc).isoformat(), "before": before,
                 "after": candidate.name, "accepted": accepted, **evidence}
        # EvoMem-inspired append-only transition evidence, including rejected proposals.
        import json
        with (self.root / "history.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        if accepted:
            write_json(self.pointer, {"version": candidate.name})


def functional_gate(candidate, fixed_tests, image="python:3.12-slim"):
    """Execute proposed code only inside a restricted disposable container."""
    name = "rsi-gate-" + uuid.uuid4().hex[:12]
    command = ["docker", "run", "--rm", "--name", name, "--network", "none", "--read-only",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--pids-limit", "64", "--memory", "256m", "--cpus", "1",
               "--tmpfs", "/tmp:rw,nosuid,size=64m", "-e", "PYTHONDONTWRITEBYTECODE=1",
               "-v", f"{candidate.resolve()}:/skill:ro",
               "-v", f"{fixed_tests.resolve()}:/checks:ro", image,
               "python", "/checks/check_skill.py", "/skill"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        return {"passed": result.returncode == 0, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"passed": False, "error": str(exc)}
    finally:
        # Killing a timed-out docker client does not necessarily stop its container.
        try:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            pass


def accept_metrics(baseline, candidate):
    """Frozen rule: no success regression and better success or fewer tokens."""
    if not baseline or len(baseline) != len(candidate):
        return False
    if any(r.get("success") is None for r in baseline + candidate):
        return False
    old = sum(r["success"] for r in baseline)
    new = sum(r["success"] for r in candidate)
    old_tokens = sum(r["tokens"] for r in baseline)
    new_tokens = sum(r["tokens"] for r in candidate)
    return new >= old and (new > old or new_tokens < old_tokens)
