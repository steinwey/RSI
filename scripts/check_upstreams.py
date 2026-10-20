"""Verify exact source revisions; no global git configuration changes."""
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent.parent
for name, info in json.loads((root / "upstreams.lock.json").read_text()).items():
    path = root / "vendor" / name
    sha = subprocess.check_output(["git", "-c", f"safe.directory={path.as_posix()}", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    assert sha == info["commit"], (name, sha)
    print(f"{name}: {sha}")
