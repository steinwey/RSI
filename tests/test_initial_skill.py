import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsi import ROOT
from rsi.config import load_config
from rsi.cli import main
from rsi.common import digest, file_map, read_json, write_json
from rsi.runner import run_experiment
from rsi.run_logs import create_run_directory
from rsi.skills import MAX_FILE_CHARS, SkillStore, load_initial_skill


class InitialSkillTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.training = self.root / "training"
        self.store = SkillStore(self.training / "skills", ROOT / "skills/test-and-diagnose")
        self.guide = file_map(self.store.current())["SKILL.md"] + "\nTrained rule: reproduce the failure first.\n"
        self.accepted, _ = self.store.propose({"SKILL.md": self.guide})
        self.store.record(self.accepted, True, {"task_id": "train-1"})
        rejected, _ = self.store.propose({"SKILL.md": "Rejected proposal must not become the default seed."})
        self.store.record(rejected, False, {"task_id": "train-2"})
        self.config = load_config(ROOT / "configs/qwen.yaml")
        self.config.update(initial_skill=str(self.training), evolve=False, wandb={"enabled": False})
        self.tasks = [{"id": f"test-{cid}-{i}", "benchmark": "swe-evo", "prompt": "Fix behavior",
                       "chain": {"id": cid, "index": i, "length": length,
                                 "source_instance_id": f"source-{cid}-{i}"}}
                      for cid, length in (("chain-a", 2), ("chain-b", 1)) for i in range(length)]

    def group(self, task, folder, skill, config):
        return {"task_id": task["id"], "success": True, "tokens": 1, "seconds": 1,
                "skill_version": skill.name, "chain": task["chain"]}, []

    def test_supported_sources_load_the_accepted_version_without_modifying_the_source(self):
        before = {str(p.relative_to(self.training)): p.read_bytes()
                  for p in self.training.rglob("*") if p.is_file()}
        for source, kind in ((self.training, "run"), (self.store.root, "store"),
                             (self.store.pointer, "pointer"), (self.accepted, "version"),
                             (self.accepted / "SKILL.md", "document")):
            with self.subTest(source=source):
                files, metadata = load_initial_skill(source)
                self.assertEqual(files, file_map(self.accepted))
                self.assertEqual(metadata["kind"], kind)
                self.assertEqual(metadata["version"], self.accepted.name)
                self.assertEqual(metadata["tools_hash"], self.store.tools_hash)
        self.assertEqual(before, {str(p.relative_to(self.training)): p.read_bytes()
                                  for p in self.training.rglob("*") if p.is_file()})

    def test_builtin_seed_is_unchanged_when_no_source_is_supplied(self):
        files, metadata = load_initial_skill()
        self.assertEqual(files, file_map(ROOT / "skills/test-and-diagnose"))
        self.assertEqual(metadata["kind"], "builtin")
        self.assertEqual(metadata["version"], digest(files))

    def test_document_only_source_uses_current_fixed_tools(self):
        source = self.root / "document/SKILL.md"
        source.parent.mkdir()
        source.write_text(self.guide, encoding="utf-8")
        files, metadata = load_initial_skill(source)
        self.assertEqual(files, file_map(self.accepted))
        self.assertEqual(metadata["tools_hash"], self.store.tools_hash)

    def test_invalid_sources_and_pointers_fail_without_falling_back(self):
        for source in ("", True, self.root / "missing", self.root):
            with self.subTest(source=source), self.assertRaises(ValueError):
                load_initial_skill(source)
        for i, pointer in enumerate(({"version": "../escape"}, {"version": True},
                                     {"version": "f" * 16}, {}, [], None)):
            source = self.root / f"bad-{i}"
            write_json(source / "current.json", pointer)
            with self.subTest(pointer=pointer), self.assertRaises(ValueError):
                load_initial_skill(source)

    def test_tool_mismatch_and_tampered_version_are_rejected(self):
        files = file_map(self.accepted)
        changed = SkillStore(self.root / "changed-tools", {**files, "scripts/inspect_project.py": "pass\n"})
        with self.assertRaisesRegex(ValueError, "Fixed tools"):
            load_initial_skill(changed.root)
        (self.accepted / "SKILL.md").write_text(self.guide + "tampered", encoding="utf-8")
        for source in (self.store.root, self.accepted):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "differs from its version"):
                load_initial_skill(source)

    def test_invalid_loaded_documents_are_rejected(self):
        source = self.root / "SKILL.md"
        for guide in ("", " ", "x" * (MAX_FILE_CHARS + 1), "Use scripts/missing.py"):
            source.write_text(guide, encoding="utf-8")
            with self.subTest(guide=guide[:20]), self.assertRaises(ValueError):
                load_initial_skill(source)

    def test_frozen_run_uses_an_independent_trained_snapshot_across_chains(self):
        seen = []
        source_before = {str(p.relative_to(self.training)): p.read_bytes()
                         for p in self.training.rglob("*") if p.is_file()}
        output = self.root / "test"

        def group(task, folder, skill, config):
            seen.append((skill.name, file_map(skill), config["reflection_enabled"]))
            self.assertFalse(config["evolve"])
            self.assertTrue(skill.is_relative_to(output))
            return self.group(task, folder, skill, config)

        with patch("rsi.runner.run_group", side_effect=group), \
             patch("rsi.runner.induce") as updater, \
             patch("rsi.runner.functional_gate", return_value={"passed": True}) as gate:
            run_experiment(self.tasks, self.config, output)
        updater.assert_not_called()
        gate.assert_called_once()
        self.assertEqual(len(seen), 3)
        self.assertTrue(all(v == self.accepted.name and files["SKILL.md"] == self.guide and reflection
                            for v, files, reflection in seen))
        info = read_json(output / "run-info.json")
        self.assertEqual(info["initial_skill_version"], self.accepted.name)
        self.assertEqual(info["current_skill_version"], self.accepted.name)
        self.assertEqual(info["initial_skill_source"], read_json(output / "initial-skill-source.json"))
        self.assertEqual(info["initial_skill_source"]["kind"], "run")
        self.assertEqual(info["initial_skill_source"]["tools_hash"], self.store.tools_hash)
        for store in ("skills", "initial-skill"):
            version = read_json(output / store / "current.json")["version"]
            self.assertEqual(file_map(output / store / "versions" / version), file_map(self.accepted))
            content = (output / store / "versions" / version / "SKILL.md").read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), info["initial_skill_source"]["document_sha256"])
            self.assertFalse((output / store / "history.jsonl").exists())
        for task in self.tasks:
            self.assertEqual(read_json(output / task["id"] / "update.json")["decision"], "fixed_skill")
        self.assertEqual(source_before, {str(p.relative_to(self.training)): p.read_bytes()
                                        for p in self.training.rglob("*") if p.is_file()})

    def test_source_pointer_moving_during_evaluation_does_not_change_the_snapshot(self):
        seen = []

        def group(task, folder, skill, config):
            seen.append(file_map(skill)["SKILL.md"])
            if len(seen) == 1:
                candidate, _ = self.store.propose({"SKILL.md": "Later training version"})
                self.store.record(candidate, True, {})
            return self.group(task, folder, skill, config)

        with patch("rsi.runner.run_group", side_effect=group), \
             patch("rsi.runner.functional_gate", return_value={"passed": True}), \
             patch("rsi.runner.load_initial_skill", wraps=load_initial_skill) as loader:
            run_experiment(self.tasks, self.config, self.root / "test")
        loader.assert_called_once()
        self.assertEqual(seen, [self.guide] * 3)

    def test_loaded_seed_can_also_start_a_new_evolution_run(self):
        seen = []

        def group(task, folder, skill, config):
            seen.append(file_map(skill)["SKILL.md"])
            return self.group(task, folder, skill, config)

        revised = self.guide + "\nNew training-stage rule."
        with patch("rsi.runner.run_group", side_effect=group), \
             patch("rsi.runner.induce", side_effect=[{"changes": {"SKILL.md": revised}}, {"changes": {}}]), \
             patch("rsi.runner.functional_gate", return_value={"passed": True}):
            run_experiment(self.tasks[:2], {**self.config, "evolve": True}, self.root / "warm-start")
        self.assertEqual(seen, [self.guide, revised])
        self.assertEqual(self.store.current(), self.accepted)

    def test_bad_seed_fails_before_creating_a_run_or_starting_the_backend(self):
        output = self.root / "test"
        with patch("rsi.runner.run_group") as group, patch("rsi.runner.functional_gate") as gate:
            with self.assertRaises(ValueError):
                run_experiment(self.tasks, {**self.config, "initial_skill": str(self.root / "missing")}, output)
        self.assertFalse(output.exists())
        group.assert_not_called()
        gate.assert_not_called()

    def test_experiment_fingerprint_includes_the_loaded_content(self):
        with patch("rsi.runner.run_group", side_effect=self.group), \
             patch("rsi.runner.functional_gate", return_value={"passed": True}):
            run_experiment(self.tasks, self.config, self.root / "test-a")
            candidate, _ = self.store.propose({"SKILL.md": self.guide + "\nAnother accepted rule."})
            self.store.record(candidate, True, {})
            run_experiment(self.tasks, self.config, self.root / "test-b")
        a, b = (read_json(self.root / name / "run-info.json") for name in ("test-a", "test-b"))
        self.assertEqual(a["config_version"], b["config_version"])
        self.assertNotEqual(a["initial_skill_version"], b["initial_skill_version"])
        self.assertNotEqual(a["experiment_version"], b["experiment_version"])

    def test_run_directory_metadata_respects_the_configured_seed(self):
        output = create_run_directory(self.tasks, self.config, self.root / "metadata-only")
        info = read_json(output / "run-info.json")
        self.assertEqual(info["initial_skill_version"], self.accepted.name)
        self.assertEqual(info["initial_skill_source"]["kind"], "run")

    def test_cli_loads_the_training_seed_and_freezes_it(self):
        output = self.root / "cli-test"
        config = copy.deepcopy(self.config)
        config.pop("initial_skill")
        config["evolve"] = True
        with patch("rsi.cli.load_config", return_value=config), \
             patch("rsi.cli.load_tasks", return_value=self.tasks), \
             patch("rsi.cli.require_credentials"), \
             patch("rsi.cli.shutil.which", return_value="docker"), \
             patch("rsi.swe.validate_swe_tasks"), \
             patch("rsi.evo_swe.validate_tasks"), \
             patch("rsi.runner.run_group", side_effect=self.group), \
             patch("rsi.runner.functional_gate", return_value={"passed": True}), \
             patch("rsi.runner.induce") as updater:
            main(["run", "--tasks", "unused.json", "--initial-skill", str(self.training),
                  "--fixed-skill", "--execution-mode", "single", "--output", str(output)])
        updater.assert_not_called()
        config = read_json(output / "config.json")
        self.assertFalse(config["evolve"])
        self.assertEqual(config["initial_skill"], str(self.training.resolve()))
        self.assertEqual((config["execution_mode"], config["rollouts"]), ("single", 1))
        self.assertTrue(config["reflection_enabled"])
        self.assertEqual(read_json(output / "run-info.json")["initial_skill_version"], self.accepted.name)


if __name__ == "__main__":
    unittest.main()
