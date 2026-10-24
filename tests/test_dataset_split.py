import copy
import tempfile
import unittest
from pathlib import Path

from rsi import ROOT
from rsi.common import read_json, write_json
from rsi.dataset_split import build_split, order_repositories, source_id, write_split
from rsi.evo_swe import evaluation_rows, validate_order


def chain(repo, cid, originals):
    return [{"id": f"{cid}-{i}", "benchmark": "swe-evo",
             "chain": {"id": cid, "index": i, "length": len(originals), "source_instance_id": iid},
             "instance": {"repo": repo, "base_commit": f"{repo}-{iid}",
                          "image": f"rsi/{repo}:{iid}",
                          "problem_statement": f"Fix {repo} behavior {iid}"}}
            for i, iid in enumerate(originals)]


class DatasetSplitTests(unittest.TestCase):
    def setUp(self):
        self.assignments = {"train": ["owner/a"], "dev": ["owner/b"], "test": ["owner/c"]}
        self.tasks = (chain("owner/a", "a-short", ["a1"])
                      + chain("owner/a", "a-long", ["a1", "a2"])
                      + chain("owner/b", "b", ["b1"])
                      + chain("owner/c", "c", ["c1"]))

    def test_preserves_original_rows_and_full_chains_without_repeated_training_tasks(self):
        original = copy.deepcopy(self.tasks)
        primary, full = build_split(self.tasks, self.assignments)
        self.assertEqual([source_id(t) for t in primary["train"]], ["a1", "a2"])
        self.assertEqual(len(full["train"]), 3)
        self.assertEqual(self.tasks, original)
        for rows in primary.values():
            validate_order(rows)

    def test_rejects_duplicate_or_omitted_repository_assignments(self):
        for assignments in (
            {**self.assignments, "test": ["owner/a"]},
            {**self.assignments, "test": ["owner/unknown"]},
        ):
            with self.assertRaises(ValueError):
                build_split(self.tasks, assignments)

    def test_rejects_longest_chain_that_would_silently_drop_other_tasks(self):
        tasks = chain("owner/a", "a-other", ["a3"]) + self.tasks
        with self.assertRaisesRegex(ValueError, "cover all unique"):
            build_split(tasks, self.assignments)

    def test_rejects_original_task_alias_across_partitions(self):
        self.tasks[-1]["chain"]["source_instance_id"] = "a1"
        with self.assertRaisesRegex(ValueError, "overlaps"):
            build_split(self.tasks, self.assignments)

    def test_rejects_identical_prompt_with_different_task_ids(self):
        self.tasks[-1]["instance"]["problem_statement"] = self.tasks[0]["instance"]["problem_statement"]
        with self.assertRaisesRegex(ValueError, "overlaps"):
            build_split(self.tasks, self.assignments)

    def test_seeded_order_preserves_membership_chronology_and_source_metadata(self):
        assignments = {"train": ["owner/a", "owner/b", "owner/c"],
                       "test": ["owner/d", "owner/e"], "reserve": ["owner/f"]}
        tasks = [t for letter in "abcdef"
                 for t in chain(f"owner/{letter}", letter, [f"{letter}1", f"{letter}2"])]
        original_tasks, original_assignments = copy.deepcopy(tasks), copy.deepcopy(assignments)
        seeds = {"train": 0, "test": 0}
        primary, full = build_split(tasks, assignments, repository_order_seeds=seeds)
        expected = order_repositories(assignments, seeds)
        self.assertNotEqual(expected["train"], assignments["train"])
        for partition in assignments:
            self.assertEqual(set(expected[partition]), set(assignments[partition]))
            self.assertEqual(list(dict.fromkeys(t["instance"]["repo"] for t in primary[partition])),
                             expected[partition])
            self.assertEqual({t["id"] for t in full[partition]}, {t["id"] for t in primary[partition]})
            validate_order(primary[partition])
            for t in primary[partition]:
                self.assertEqual(t, next(row for row in original_tasks if row["id"] == t["id"]))
        self.assertEqual(tasks, original_tasks)
        self.assertEqual(assignments, original_assignments)
        reversed_lists = {p: list(reversed(rows)) for p, rows in assignments.items()}
        self.assertEqual(expected, order_repositories(reversed_lists, seeds))
        self.assertEqual(expected["test"], order_repositories(assignments, {**seeds, "train": 1})["test"])

    def test_rejects_invalid_repository_order_seeds(self):
        for seeds in ([], {"missing": 0}, {"train": True}, {"test": "0"}):
            with self.assertRaisesRegex(ValueError, "[Rr]epository order seeds"):
                build_split(self.tasks, self.assignments, repository_order_seeds=seeds)

    def fixture(self, root):
        source, spec_path = root / "tasks.json", root / "spec.json"
        assignments = {"train": ["owner/a"], "test": ["owner/b"], "reserve": ["owner/c"]}
        write_json(source, self.tasks)
        spec = {"name": "fixture", "dataset_revision": "1" * 40, "repositories": assignments,
                "require_cached_partitions": ["train", "test"], "training_prefix_lengths": [2]}
        write_json(spec_path, spec)
        data = root / "data"
        write_json(data / "dataset-source.json", {"revision": "1" * 40})
        for t in self.tasks:
            c = t["chain"]
            write_json(data / "chains" / c["id"] / "instances" / f"{c['index']:03d}-{source_id(t)}.json",
                       {"raw": {"language": "go"}})
        tags = root / "tags.txt"
        tags.write_text("\n".join(t["instance"]["image"] for t in self.tasks
                                  if t["instance"]["repo"] != "owner/c"), encoding="utf-8")
        return source, spec_path, data, tags

    def test_cached_train_test_split_can_reserve_unavailable_repositories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, spec, data, tags = self.fixture(root)
            out = root / "split"
            audit = write_split(source, spec, out, data, root / "runs", tags)
            self.assertEqual(set(audit["partitions"]), {"train", "test", "reserve"})
            self.assertEqual(audit["partitions"]["train"]["cached_image_tags"], 2)
            self.assertEqual(audit["partitions"]["test"]["cached_image_tags"], 1)
            self.assertEqual(audit["partitions"]["reserve"]["cached_image_tags"], 0)
            self.assertEqual(read_json(out / "training-prefixes/train-2.json"), read_json(out / "train.json"))

    def test_cache_check_fails_before_writing_incomplete_active_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, spec, data, tags = self.fixture(root)
            tags.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing"):
                write_split(source, spec, root / "split", data, root / "runs", tags)
            self.assertFalse((root / "split").exists())
            with self.assertRaisesRegex(ValueError, "image-tags"):
                write_split(source, spec, root / "split", data, root / "runs")

    def test_training_prefix_cannot_truncate_a_chain(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, spec, data, tags = self.fixture(root)
            changed = read_json(spec)
            changed["training_prefix_lengths"] = [1]
            write_json(spec, changed)
            with self.assertRaisesRegex(ValueError, "complete chains"):
                write_split(source, spec, root / "split", data, root / "runs", tags)
            self.assertFalse((root / "split").exists())

    def test_language_counts_are_fixed_before_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, spec, data, tags = self.fixture(root)
            changed = read_json(spec)
            changed["expected_language_tasks"] = {"train": {"go": 1, "python": 1}}
            write_json(spec, changed)
            with self.assertRaisesRegex(ValueError, "Language task counts"):
                write_split(source, spec, root / "split", data, root / "runs", tags)
            self.assertFalse((root / "split").exists())

    def test_repository_prefixes_end_on_whole_chain_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, spec, data, tags = self.fixture(root)
            changed = read_json(spec)
            changed.pop("training_prefix_lengths")
            changed["training_prefix_repository_counts"] = [1]
            changed["expected_language_tasks"] = {"train": {"go": 2}, "test": {"go": 1}}
            changed["repository_order_seeds"] = {"train": 0, "test": 0}
            write_json(spec, changed)
            out = root / "split"
            audit = write_split(source, spec, out, data, root / "runs", tags)
            prefix = read_json(out / "training-prefixes/train-2.json")
            self.assertEqual(prefix, read_json(out / "train.json"))
            self.assertEqual(audit["training_prefixes"]["training-prefixes/train-2.json"]["repositories"], 1)
            self.assertEqual(audit["ordering"]["repository_order_seeds"], {"train": 0, "test": 0})
            self.assertEqual(audit["partitions"]["train"]["language_tasks"], {"go": 2})
            changed["training_prefix_repository_counts"] = [2]
            write_json(spec, changed)
            with self.assertRaisesRegex(ValueError, "repository count"):
                write_split(source, spec, root / "invalid-split", data, root / "runs", tags)
            self.assertFalse((root / "invalid-split").exists())

    def test_shuffled_manifests_keep_frozen_membership_and_languages(self):
        source = ROOT / "tasks/evoarena-all.json"
        if not source.exists():
            self.skipTest("Local EvoArena metadata not available")
        spec = read_json(ROOT / "configs/evoarena-frozen-shuffled.json")
        primary, full = build_split(read_json(source), spec["repositories"],
                                    repository_order_seeds=spec["repository_order_seeds"])
        out = ROOT / "tasks/splits/evoarena-frozen-v3"
        audit = read_json(out / "audit.json")
        self.assertEqual(sum(map(len, full.values())), 493)
        for partition, rows in primary.items():
            self.assertEqual(rows, read_json(out / f"{partition}.json"))
            previous = read_json(ROOT / f"tasks/splits/evoarena-frozen-v2/{partition}.json")
            self.assertEqual({t["id"]: t for t in rows}, {t["id"]: t for t in previous})
            self.assertEqual(audit["partitions"][partition]["language_tasks"],
                             spec["expected_language_tasks"][partition])
            self.assertEqual(len(rows), {"train": 70, "test": 45, "reserve": 30}[partition])
        for name in audit["training_prefixes"]:
            validate_order(read_json(out / name))
        selected = primary["train"] + primary["test"]
        if all(Path(t["evaluation_dataset"]).exists() for t in selected):
            self.assertEqual(len(evaluation_rows(selected)), 115)

    def test_current_frozen_split_is_disjoint_and_preserves_all_tasks(self):
        source = ROOT / "tasks/evoarena-all.json"
        if not source.exists():
            self.skipTest("Local EvoArena metadata not available")
        tasks = read_json(source)
        spec = read_json(ROOT / "configs/evoarena-frozen-split.json")
        primary, full = build_split(tasks, spec["repositories"])
        self.assertEqual({p: len(rows) for p, rows in primary.items()}, {"train": 70, "test": 45, "reserve": 30})
        self.assertEqual(sum(map(len, full.values())), len(tasks))
        self.assertEqual({source_id(t) for rows in primary.values() for t in rows}, {source_id(t) for t in tasks})
        out = ROOT / "tasks/splits/evoarena-frozen-v2"
        for partition, rows in primary.items():
            self.assertEqual(rows, read_json(out / f"{partition}.json"))
        selected = primary["train"] + primary["test"]
        if all(Path(t["evaluation_dataset"]).exists() for t in selected):
            self.assertEqual(len(evaluation_rows(selected)), 115)

    def test_generated_manifests_preserve_coverage_and_grading_metadata(self):
        # Generated artifacts are checked when local source data is available.
        source = ROOT / "tasks/evoarena-all.json"
        if not source.exists():
            self.skipTest("Local EvoArena metadata not available")
        source_tasks = read_json(source)
        spec = read_json(ROOT / "configs/evoarena-repo-split.json")
        primary, full = build_split(source_tasks, spec["repositories"])
        selected = [t for rows in primary.values() for t in rows]
        self.assertEqual({source_id(t) for t in selected}, {source_id(t) for t in source_tasks})
        self.assertEqual(len(selected), 145)
        self.assertEqual(sum(map(len, full.values())), 493)
        for partition, rows in primary.items():
            self.assertEqual(rows, read_json(ROOT / f"tasks/splits/evoarena-repo-v1/{partition}.json"))
        if all(Path(t["evaluation_dataset"]).exists() for t in selected):
            self.assertEqual(len(evaluation_rows(selected)), len(selected))


if __name__ == "__main__":
    unittest.main()
