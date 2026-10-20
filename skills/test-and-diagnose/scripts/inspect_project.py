"""Read project metadata and suggest test commands; never execute project code."""
import argparse
import configparser
import json
from pathlib import Path


def inspect(path):
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError("Project path must be a directory")
    found, warnings = [], []
    def add(framework, evidence, command):
        found.append({"framework": framework, "evidence": evidence, "command": command})
    if (root / "pytest.ini").is_file():
        add("pytest", "pytest.ini", "python -m pytest")
    if (root / "setup.cfg").is_file():
        try:
            cfg = configparser.ConfigParser(interpolation=None)
            cfg.read(root / "setup.cfg", encoding="utf-8")
            if cfg.has_section("tool:pytest"):
                add("pytest", "setup.cfg", "python -m pytest")
        except configparser.Error as exc:
            warnings.append(str(exc))
    if (root / "pyproject.toml").is_file():
        try:
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            cfg = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
            if "pytest" in cfg.get("tool", {}):
                add("pytest", "pyproject.toml", "python -m pytest")
        except (ValueError, ImportError) as exc:
            warnings.append(str(exc))
    if (root / "package.json").is_file():
        try:
            package = json.loads((root / "package.json").read_text(encoding="utf-8"))
            if isinstance(package, dict) and package.get("scripts", {}).get("test"):
                add("npm", "package.json", "npm test")
        except (ValueError, AttributeError) as exc:
            warnings.append(str(exc))
    if (root / "Cargo.toml").is_file():
        add("cargo", "Cargo.toml", "cargo test")
    if (root / "go.mod").is_file():
        add("go", "go.mod", "go test ./...")
    return {"project": str(root), "detected": found, "warnings": warnings}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    print(json.dumps(inspect(parser.parse_args().path), ensure_ascii=False))
