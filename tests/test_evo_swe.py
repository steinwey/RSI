import copy
import hashlib
import io
import json
import shutil
import tempfile
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from rsi import ROOT
from rsi.cli import load_config, load_tasks, main
from rsi.backends import SWEChainBackend
from rsi.common import read_json, write_json
from rsi.evo_swe import (_download_image, evaluation_rows, hub_endpoint, isolate_repository_script,
                         prepare_tasks, prepare_images, validate_order, validate_tasks, verify_references)
from rsi.runner import run_experiment
from rsi.tracking import chain_metrics


class EvoSWETests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = load_config(ROOT / "configs/evoarena.yaml")
        self.config["wandb"] = {"enabled": False}

    def chain(self, cid="repo-chain-a", count=2, source_prefix=None, language="python"):
        directory = self.root / "data/chains" / cid
        entries = []
        for index in range(count):
            iid = f"{source_prefix or cid}_lc_{index:04d}"
            image = f"sweb.eval.x86_64.{iid}:latest"
            entries.append({"instance_id": iid, "version_chain_index": index,
                            "image": image, "version": str(index)})
            row = {"instance_id": iid, "version_chain_id": cid, "version_chain_index": index,
                   "image": image, "version": str(index), "repo": "owner/project",
                   "problem_statement": "Fix a localized behavior",
                   "fail_to_pass": ["test_new"], "pass_to_pass": ["test_existing"],
                   "raw": {"instance_id": iid, "repo": "owner/project", "language": language,
                           "base_commit": f"{index + 1:040x}", "patch": "SECRET_REFERENCE_PATCH",
                           "test_patch": "SECRET_HIDDEN_TEST_PATCH", "files": {"test": ["tests/test_new.py"]},
                           "test_cmds": ["go test -v ./..." if language == "go" else "python -m pytest -rA tests"]}}
            write_json(directory / "instances" / f"{index:03d}-{iid}.json", row)
        write_json(directory / "chain.json", {"chain_id": cid, "repo": "owner/project",
                                              "instance_count": count, "instances": entries})
        return cid

    def prepare(self, chain_ids, name="tasks.json"):
        output = self.root / name
        prepare_tasks(self.root / "data", chain_ids, output)
        return load_tasks(output, self.config)

    def test_real_local_click_chain_is_prepared_without_oracle_exposure(self):
        source = ROOT / "datasets/evoarena-swe"
        if not (source / "chains/pallets__click_multistep_v25_L05_s0/chain.json").exists():
            self.skipTest("Official metadata sample not downloaded")
        cid = "pallets__click_multistep_v25_L05_s0"
        shutil.copytree(source / "chains" / cid, self.root / "official/chains" / cid)
        tasks = prepare_tasks(self.root / "official", [cid], self.root / "official.json")
        self.assertEqual(len(tasks), 5)
        self.assertEqual([t["chain"]["index"] for t in tasks], list(range(5)))
        self.assertNotIn('"FAIL_TO_PASS"', json.dumps(tasks))
        self.assertNotIn('"reference_patch"', json.dumps(tasks))

    def test_manifest_separates_reference_patches_and_tests(self):
        tasks = self.prepare([self.chain()])
        encoded = json.dumps(tasks)
        self.assertNotIn("SECRET_REFERENCE_PATCH", encoded)
        self.assertNotIn("SECRET_HIDDEN_TEST_PATCH", encoded)
        self.assertNotIn("test_existing", encoded)
        rows = evaluation_rows(tasks)
        row = rows[tasks[0]["id"]]
        self.assertEqual(row["reference_patch"], "SECRET_REFERENCE_PATCH")
        self.assertEqual(row["log_parser"], "parse_log_pytest")
        self.assertIn("git checkout", row["eval_script"])
        self.assertIn("Start Test Output", row["eval_script"])
        with patch("rsi.swe.load_evaluation_rows", return_value=list(rows.values())):
            validate_tasks(tasks, self.config, require_images=False)

    def test_changed_evaluation_metadata_is_rejected(self):
        tasks = self.prepare([self.chain()])
        path = Path(tasks[0]["evaluation_dataset"])
        path.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "metadata changed"):
            evaluation_rows(tasks)

    def test_go_metadata_selects_the_existing_go_parser(self):
        tasks = self.prepare([self.chain(language="go")])
        row = evaluation_rows(tasks)[tasks[0]["id"]]
        self.assertEqual(row["log_parser"], "parse_log_gotest")
        self.assertIn("go test -v", row["eval_script"])

    def test_incomplete_reordered_and_interleaved_chains_are_rejected(self):
        tasks = self.prepare([self.chain(), self.chain("repo-chain-b")])
        validate_order(tasks)
        for invalid in (tasks[:1], [tasks[1], tasks[0]], [tasks[0], tasks[2], tasks[1], tasks[3]]):
            with self.subTest(invalid=[t["id"] for t in invalid]):
                with self.assertRaises(ValueError):
                    validate_order(invalid)

    def test_overlapping_source_milestones_cannot_be_validation_tasks(self):
        main = self.prepare([self.chain(source_prefix="same")], "main.json")
        heldout = self.prepare([self.chain("repo-chain-b", source_prefix="same")], "val.json")
        with self.assertRaisesRegex(ValueError, "share milestones"):
            validate_order(main, heldout)

    def test_partially_solved_or_ungraded_chains_do_not_count_as_resolved(self):
        tasks = self.prepare([self.chain(), self.chain("repo-chain-b")])
        reports = [{"chain": t["chain"], "success": result}
                   for t, result in zip(tasks, [True, True, True, False])]
        self.assertEqual(chain_metrics(reports)["chain_accuracy"], 0.5)
        self.assertEqual(chain_metrics(reports)["step_accuracy"], 0.75)
        partial = chain_metrics(reports[:1])
        self.assertEqual(partial["completed_chains"], 0)
        self.assertEqual(partial["resolved_chains"], 0)
        reports[-1]["success"] = None
        self.assertEqual(chain_metrics(reports)["scored_chains"], 1)
        self.assertEqual(chain_metrics(reports)["chain_accuracy"], 0.5)

    def test_memory_and_accepted_skill_versions_are_isolated_between_chains(self):
        main = self.prepare([self.chain(), self.chain("repo-chain-b")], "main.json")
        heldout = self.prepare([self.chain("repo-chain-v")], "val.json")
        calls = []
        def group(task, folder, skill, memory, config):
            calls.append((task["id"], folder, skill.name, copy.deepcopy(memory)))
            tokens = 10 if folder.name == "candidate" else 20
            return {"task_id": task["id"], "chain": task["chain"], "success": True,
                    "tokens": tokens, "input_tokens": tokens - 2, "output_tokens": 2,
                    "seconds": 1}, []
        proposals = [{"changes": {"SKILL.md": "A general tested workflow"}, "memory_items": [{"content": "first"}]}]
        proposals += [{"changes": {}, "memory_items": [{"content": "next"}]} for _ in range(3)]
        output = self.root / "run"
        with patch("rsi.runner.run_group", side_effect=group), \
             patch("rsi.runner.induce", side_effect=proposals), \
             patch("rsi.runner.functional_gate", return_value={"passed": True}):
            run_experiment(main, heldout, self.config, output)
        main_calls = [call for call in calls if call[1].parent == output]
        self.assertNotEqual(main_calls[0][2], main_calls[1][2])
        self.assertEqual(main_calls[0][2], main_calls[2][2])
        self.assertTrue(all(call[3] == [] for call in calls))
        self.assertEqual(read_json(output / "metrics.json")["resolved_chains"], 2)
        self.assertTrue((output / "chains/repo-chain-a/skills/history.jsonl").exists())
        self.assertFalse((output / "chains/repo-chain-b/skills/history.jsonl").exists())

    def test_archive_download_resumes_and_checks_the_published_hash(self):
        content = b"previous bytes and remaining bytes"
        path = self.root / "image.tar"
        path.with_suffix(".tar.part").write_bytes(content[:10])
        response = io.BytesIO(content[10:])
        response.status = 206
        metadata = {"path": "images/image.tar", "size": len(content),
                    "lfs": {"oid": hashlib.sha256(content).hexdigest()}}
        with patch("rsi.evo_swe.urllib.request.urlopen", return_value=response) as urlopen:
            _download_image(path, metadata, 60)
        self.assertEqual(path.read_bytes(), content)
        self.assertEqual(urlopen.call_args.args[0].get_header("Range"), "bytes=10-")
        self.assertTrue(urlopen.call_args.args[0].full_url.startswith(f"{hub_endpoint()}/datasets/"))
        self.assertFalse(path.with_suffix(".tar.part").exists())

    def test_cached_images_need_no_huggingface_network_connection(self):
        tasks = self.prepare([self.chain()])
        bases = {t["instance"]["image"]: t["instance"]["base_commit"] for t in tasks}
        def execute(command, **kwargs):
            stdout = ""
            if command[1] == "run" and "rev-list" in command:
                stdout = "0\n"
            elif command[1] == "run":
                stdout = bases[command[5]] + "\n"
            return subprocess.CompletedProcess(command, 0, stdout)
        with patch("rsi.evo_swe.subprocess.run", side_effect=execute), \
             patch("rsi.evo_swe.urllib.request.urlopen") as request:
            prepare_images(tasks, self.root / "images")
        request.assert_not_called()

    def test_future_commits_are_removed_before_an_image_is_reused(self):
        tasks = self.prepare([self.chain()])
        bases = {t["instance"]["image"]: t["instance"]["base_commit"] for t in tasks}
        base = tasks[0]["instance"]["base_commit"]
        script = isolate_repository_script(base)
        self.assertIn(f"git reset --hard {base}", script)
        self.assertIn("git remote remove", script)
        self.assertIn("git reflog expire --expire=now --all", script)
        self.assertIn("git gc --prune=now", script)
        self.assertIn('git rev-list --all --not HEAD --count', script)
        self.assertIn('git merge-base --is-ancestor', script)
        leaked = set()
        def execute(command, **kwargs):
            stdout = ""
            if command[1] == "run" and "rev-list" in command:
                image = command[5]
                stdout = "0\n" if image in leaked else "4\n"
                leaked.add(image)
            elif command[1] == "run":
                stdout = bases[command[5]] + "\n"
            return subprocess.CompletedProcess(command, 0, stdout)
        with patch("rsi.evo_swe.subprocess.run", side_effect=execute) as run, \
             patch("rsi.evo_swe.urllib.request.urlopen") as request:
            prepare_images(tasks, self.root / "images")
        request.assert_not_called()
        self.assertTrue(any(call.args[0][1] == "build" for call in run.call_args_list))
        written = (self.root / "images" / "build" / tasks[0]["id"] / "isolate-history.sh").read_text(encoding="utf-8")
        self.assertEqual(written, script)

    def test_backend_checks_the_base_snapshot_and_uses_the_private_dataset(self):
        task = self.prepare([self.chain()])[0]
        base = task["instance"]["base_commit"]
        with patch("rsi.docker_environment.CompatibleDockerEnvironment") as environment:
            environment.return_value.execute.return_value = {"output": base + "\n"}
            backend = SWEChainBackend(task, self.root, self.config)
            self.assertEqual(backend.config["swe_dataset"], task["evaluation_dataset"])
            self.assertEqual(environment.call_args.kwargs["run_args"], ["--rm", "--network", "none"])
            environment.return_value.execute.return_value = {"output": "different snapshot\n"}
            with patch.object(SWEChainBackend, "close") as close:
                with self.assertRaisesRegex(RuntimeError, "image base differs"):
                    SWEChainBackend(task, self.root, self.config)
            close.assert_called_once()

    def test_missing_images_stop_the_cli_before_any_solver_work(self):
        tasks = self.prepare([self.chain()])
        rows = list(evaluation_rows(tasks).values())
        with patch("rsi.cli.shutil.which", return_value="docker"), \
             patch("rsi.cli.os.getenv", return_value="configured"), \
             patch("rsi.swe.load_evaluation_rows", return_value=rows), \
             patch("rsi.evo_swe.subprocess.run", return_value=subprocess.CompletedProcess([], 1)), \
             patch("rsi.runner.run_experiment") as run:
            with self.assertRaisesRegex(RuntimeError, "prepare-evo-images"):
                main(["run", "--config", str(ROOT / "configs/evoarena.yaml"), "--tasks", str(self.root / "tasks.json")])
        run.assert_not_called()

    def test_reference_verification_requires_a_failing_unfixed_control(self):
        tasks = self.prepare([self.chain()])[:1]
        for baseline_passes in (False, True):
            with self.subTest(baseline_passes=baseline_passes):
                output = self.root / f"verification-{baseline_passes}"
                def score(backend, artifact):
                    reference = artifact == "SECRET_REFERENCE_PATCH"
                    resolved = reference or baseline_passes
                    report = backend.folder / "report.json"
                    write_json(report, {backend.task["id"]: {"resolved": resolved,
                        "tests_status": {"FAIL_TO_PASS": {"failure": [] if resolved else ["test_new"]}}}})
                    return {"success": resolved, "report": str(report)}
                with patch("rsi.backends.SWEBackend.score", new=score):
                    if baseline_passes:
                        with self.assertRaisesRegex(RuntimeError, "Baseline calibration failed"):
                            verify_references(tasks, self.config, output)
                        self.assertEqual(read_json(output / "run-info.json")["status"], "failed")
                        self.assertFalse(read_json(output / "score.json")["success"])
                    else:
                        reports = verify_references(tasks, self.config, output)
                        self.assertTrue(reports[0]["success"])
                        self.assertEqual(reports[0]["baseline_f2p_failures"], 1)


if __name__ == "__main__":
    unittest.main()
