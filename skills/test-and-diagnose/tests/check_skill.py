"""Fixed acceptance checks. The updater cannot edit this file."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def check(skill):
    script = skill / "scripts" / "inspect_project.py"
    cases = [({}, set()), ({"pytest.ini": "[pytest]\n"}, {"pytest"}),
             ({"pyproject.toml": "[tool.pytest.ini_options]\n"}, {"pytest"}),
             ({"setup.cfg": "[tool:pytest]\n"}, {"pytest"}),
             ({"package.json": '{"scripts":{"test":"echo test"}}'}, {"npm"}),
             ({"Cargo.toml": "[package]\n"}, {"cargo"}),
             ({"go.mod": "module example\n"}, {"go"}),
             ({"package.json": "invalid json"}, set())]
    for files, expected in cases:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, text in files.items():
                (root / name).write_text(text, encoding="utf-8")
            run = subprocess.run([sys.executable, str(script), "--path", tmp],
                                 capture_output=True, text=True, timeout=10, check=True)
            result = json.loads(run.stdout)
            actual = {item["framework"] for item in result["detected"]}
            assert actual == expected, (files, actual, expected)
            assert {p.name for p in root.iterdir()} == set(files), "Inspector modified project"
    print(f"Passed {len(cases)} fixed skill checks")


if __name__ == "__main__":
    check(Path(sys.argv[1]))
