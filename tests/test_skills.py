import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rsi import ROOT
from rsi.common import file_map
from rsi.runner import skill_prompt
from rsi.evolution import induce, skill_usage
from rsi.skills import SkillStore, document_gate, MAX_FILE_CHARS

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

    def test_tool_add_edit_delete_and_paths_are_rejected(self):
        for path in ("scripts/find_tests.py", "scripts/inspect_project.py", "../SKILL.md",
                     "tests/check_skill.py", "scripts/sub/tool.py"):
            for value in (GOOD_SCRIPT, None):
                with self.subTest(path=path, value=value), self.assertRaises(ValueError):
                    self.store.propose({path: value})

    def test_document_validation_and_fixed_tool_hash(self):
        for guide in (None, "", "   ", "x" * (MAX_FILE_CHARS + 1), "Use scripts/missing.py"):
            with self.subTest(guide=str(guide)[:20]), self.assertRaises(ValueError):
                self.store.propose({"SKILL.md": guide})
        candidate, _ = self.store.propose({"SKILL.md": self.guide + "\nReproduce first."})
        gate = document_gate(candidate, self.store.fixed_files)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["tools_hash"], self.store.tools_hash)
        self.assertEqual({k: v for k, v in file_map(candidate).items() if k != "SKILL.md"}, self.store.fixed_files)
        (candidate / "scripts/inspect_project.py").write_text("pass")
        self.assertFalse(document_gate(candidate, self.store.fixed_files)["passed"])
        with self.assertRaises(ValueError):
            self.store.record(candidate, True, {})

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
        call = {"message": 2, "sequence": 1, "scripts": ["scripts/inspect_project.py"],
                "returncode": 0, "error": None}
        results[0]["skill_script_calls"] = [call]
        self.assertEqual(skill_usage(results), [
            {"rollout": 0, "source": "command_log", "calls": [call]},
            {"rollout": 1, "source": "unavailable", "calls": []}])
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
        budget = seen["data"]["skill_document_budget"]
        self.assertEqual(budget["current_characters"], len(self.guide))
        self.assertEqual(budget["remaining_characters"], MAX_FILE_CHARS - len(self.guide))
        self.assertEqual(seen["data"]["skill_usage"][0]["calls"][0]["returncode"], 0)
        self.assertIn("Only SKILL.md may be changed", seen["instruction"])
        self.assertIn("Do not add, rewrite or delete scripts", seen["instruction"])
        self.assertIn("Put a repeated procedure into SKILL.md", seen["instruction"])
        self.assertNotIn("memory_items", seen["instruction"])
        self.assertNotIn("websites", seen["instruction"])


if __name__ == "__main__":
    unittest.main()
