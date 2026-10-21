import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rsi import ROOT
from rsi.common import file_map
from rsi.runner import induce, skill_prompt, skill_usage
from rsi.skills import SkillStore

CHECKS = ROOT / "skills/test-and-diagnose/tests/check_skill.py"
GOOD_SCRIPT = '''"""List test files."""
import argparse, json, pathlib
parser = argparse.ArgumentParser()
parser.add_argument("--path", required=True)
root = pathlib.Path(parser.parse_args().path)
print(json.dumps({"tests": sorted(p.relative_to(root).as_posix() for p in root.rglob("test_*.py"))}))
'''


class SkillUpdateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = SkillStore(self.root / "skills", ROOT / "skills/test-and-diagnose")
        self.guide = file_map(self.store.current())["SKILL.md"]

    def tearDown(self):
        self.tmp.cleanup()

    def gate(self, changes):
        candidate, _ = self.store.propose(changes)
        return subprocess.run([sys.executable, str(CHECKS), str(candidate)], capture_output=True,
                              text=True, timeout=300)

    def test_new_scripts_can_be_added_and_deleted(self):
        candidate, diff = self.store.propose({"scripts/find_tests.py": GOOD_SCRIPT,
                                              "SKILL.md": self.guide + "\nUse find_tests.py to list tests.\n"})
        self.assertIn("scripts/find_tests.py", file_map(candidate))
        self.assertIn("+print(json.dumps", diff)
        self.store.record(candidate, True, {})
        removed, _ = self.store.propose({"scripts/find_tests.py": None})
        self.assertNotIn("scripts/find_tests.py", file_map(removed))
        self.assertIn("tests/check_skill.py", file_map(removed))

    def test_required_files_paths_and_script_count_are_enforced(self):
        for path in ("SKILL.md", "scripts/inspect_project.py"):
            with self.assertRaisesRegex(ValueError, "cannot be deleted"):
                self.store.propose({path: None})
        for path in ("scripts/sub/tool.py", "scripts/1tool.py", "scripts/tool.sh", "tests/check_skill.py"):
            with self.assertRaisesRegex(ValueError, "may be changed"):
                self.store.propose({path: "pass"})
        with self.assertRaisesRegex(ValueError, "At most 8"):
            self.store.propose({f"scripts/tool_{i}.py": "pass" for i in range(8)})
        with self.assertRaises(SyntaxError):
            self.store.propose({"scripts/broken.py": "def :"})

    def test_fixed_checks_accept_documented_read_only_script(self):
        result = self.gate({"scripts/find_tests.py": GOOD_SCRIPT,
                            "SKILL.md": self.guide + "\nUse find_tests.py to list tests.\n"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("2 scripts", result.stdout)

    def test_fixed_checks_reject_unsafe_or_undocumented_scripts(self):
        writer = GOOD_SCRIPT.replace("print(", "(root / 'created.txt').write_text('x'); print(")
        crash = GOOD_SCRIPT.replace("print(", "raise RuntimeError('boom'); print(")
        cases = {"undocumented": ({"scripts/find_tests.py": GOOD_SCRIPT}, "does not describe"),
                 "writes": ({"scripts/find_tests.py": writer}, "modified"),
                 "crashes": ({"scripts/find_tests.py": crash}, "crashed")}
        for name, (changes, message) in cases.items():
            with self.subTest(name):
                if name != "undocumented":
                    changes = {**changes, "SKILL.md": self.guide + "\nUse find_tests.py to list tests.\n"}
                result = self.gate(changes)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_solver_prompt_lists_tools_and_does_not_interpret_guide_as_template(self):
        candidate, _ = self.store.propose({"SKILL.md": "Literal {{ not_a_variable }} guide with inspect_project.py"})
        text = skill_prompt(candidate)
        self.assertIn("python /opt/rsi-skills/scripts/inspect_project.py --path <repository>", text)
        self.assertIn("Use these tools when the guide below says they apply", text)
        from jinja2 import Template
        self.assertIn("{{ not_a_variable }}", Template("{{ rsi_skill }}").render(rsi_skill=text))

    def test_updater_receives_full_skill_and_script_usage_but_not_fixed_checks(self):
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"},
                    {"role": "assistant", "content": "```bash\npython /opt/rsi-skills/scripts/inspect_project.py --path .\n```"},
                    {"role": "user", "content": "<returncode>0</returncode>\n<output>{}</output>"},
                    {"role": "assistant", "content": "```bash\nls\n```"}]
        results = [{"rollout": 0, "status": "Submitted", "error": None, "messages": messages, "artifact": ""},
                   {"rollout": 1, "status": "Submitted", "error": None, "messages": messages[:2], "artifact": ""}]
        self.assertEqual(skill_usage(results), [
            {"rollout": 0, "calls": [{"message": 2, "scripts": ["scripts/inspect_project.py"], "returncode": 0}]},
            {"rollout": 1, "calls": []}])
        seen = {}
        class Updater:
            def __init__(self, config):
                pass
            def object(self, instruction, data):
                seen.update(instruction=instruction, data=data)
                return {"memory_items": [], "changes": {}}
            def stats(self):
                return {}
        induce({"prompt": "fix"}, results, self.store.current(), {"judge_model": {}, "step_limit": 30}, Updater)
        self.assertEqual(set(seen["data"]["current_skill"]), {"SKILL.md", "scripts/inspect_project.py"})
        self.assertEqual(seen["data"]["skill_usage"][0]["calls"][0]["returncode"], 0)
        self.assertIn("add new scripts", seen["instruction"])
        self.assertIn("null to delete", seen["instruction"])
        self.assertIn("Put a repeated procedure into SKILL.md", seen["instruction"])
        self.assertNotIn("memory_items", seen["instruction"])
        self.assertNotIn("websites", seen["instruction"])


if __name__ == "__main__":
    unittest.main()
