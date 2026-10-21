"""Fixed acceptance checks. The updater cannot edit this file."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

MAX_SCRIPTS = 8
SCRIPT_TIMEOUT = 20
CACHE_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
FIXTURES = {
    "empty": {},
    "python": {
        "pytest.ini": "[pytest]\n",
        "pkg/__init__.py": "",
        "pkg/calc.py": "def add(a, b):\n    return a + b\n",
        "tests/test_calc.py": "from pkg.calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
    },
    "node": {"package.json": '{"scripts":{"test":"echo test"}}', "index.js": "module.exports = 1;\n"},
}


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file() and not CACHE_PARTS & set(p.relative_to(root).parts) and p.suffix != ".pyc"}


def write_tree(root, files):
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def run(script, *args, cwd):
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True,
                          timeout=SCRIPT_TIMEOUT, cwd=cwd)


def check_inspect_project(skill):
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
            write_tree(root, files)
            result = run(script, "--path", tmp, cwd=tmp)
            assert result.returncode == 0, (files, result.stderr)
            actual = {item["framework"] for item in json.loads(result.stdout)["detected"]}
            assert actual == expected, (files, actual, expected)
            assert {p.name for p in root.iterdir()} == set(files), "inspect_project modified project"
    return len(cases)


def check_script(script):
    compile(script.read_text(encoding="utf-8"), script.name, "exec")
    with tempfile.TemporaryDirectory() as tmp:
        shown = run(script, "--help", cwd=tmp)
    assert shown.returncode == 0, f"{script.name} --help failed: {shown.stderr[-500:]}"
    for name, files in FIXTURES.items():
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            write_tree(root, files)
            before = snapshot(root)
            result = run(script, "--path", str(root), cwd=root)
            assert result.returncode in (0, 1, 2), f"{script.name} on {name}: exit {result.returncode}"
            assert "Traceback (most recent call last)" not in result.stderr, \
                f"{script.name} on {name} crashed: {result.stderr[-500:]}"
            if result.returncode == 0:
                json.loads(result.stdout)
            assert snapshot(root) == before, f"{script.name} modified the {name} repository"


def check(skill):
    skill = skill.resolve()
    scripts = sorted((skill / "scripts").glob("*.py"))
    assert (skill / "scripts" / "inspect_project.py") in scripts, "scripts/inspect_project.py is required"
    assert len(scripts) <= MAX_SCRIPTS, f"At most {MAX_SCRIPTS} scripts are allowed"
    guide = (skill / "SKILL.md").read_text(encoding="utf-8")
    undocumented = [s.name for s in scripts if s.name not in guide]
    assert not undocumented, f"SKILL.md does not describe: {undocumented}"
    cases = check_inspect_project(skill)
    for script in scripts:
        check_script(script)
    print(f"Passed {cases} inspect_project checks and contract checks for {len(scripts)} scripts")


if __name__ == "__main__":
    check(Path(sys.argv[1]))
