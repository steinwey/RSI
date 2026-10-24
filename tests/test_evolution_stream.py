import copy
import unittest
from pathlib import Path

from rsi import ROOT
from rsi.common import read_json
from rsi.dataset_split import source_id
from rsi.evolution_stream import build_streams, episode_plan
from rsi.evo_swe import evaluation_rows, validate_order


def chain(repo, cid, count):
    return [{"id": f"{cid}-{i}", "benchmark": "swe-evo",
             "chain": {"id": cid, "index": i, "length": count, "source_instance_id": f"{repo}-{i}"},
             "instance": {"repo": repo, "base_commit": f"{repo}-{i}", "image": f"image/{repo}:{i}",
                          "problem_statement": f"Fix {repo} milestone {i}"}}
            for i in range(count)]


class EvolutionStreamTests(unittest.TestCase):
    def setUp(self):
        self.tasks = (chain("dev", "dev-short", 1) + chain("dev", "dev-long", 2)
                      + chain("b", "b", 3) + chain("c", "c", 2) + chain("d", "d", 1))
        self.spec = {"development_repositories": ["dev"], "benchmark_repositories": ["b", "c", "d"],
                     "order_seeds": [0, 1, 2], "primary_seed": 0}

    def test_order_variants_keep_complete_chains_and_same_unique_tasks(self):
        before = copy.deepcopy(self.tasks)
        development, streams = build_streams(self.tasks, self.spec)
        self.assertEqual(len(development), 2)
        expected = {source_id(t) for t in self.tasks if t["instance"]["repo"] != "dev"}
        for rows in streams.values():
            validate_order(rows)
            self.assertEqual(len(rows), len(expected))
            self.assertEqual({source_id(t) for t in rows}, expected)
            for repo in ("b", "c", "d"):
                actual = [t for t in rows if t["instance"]["repo"] == repo]
                self.assertEqual(actual, [t for t in self.tasks if t["instance"]["repo"] == repo])
        self.assertEqual(build_streams(self.tasks, self.spec), (development, streams))
        self.assertNotEqual(streams[0], streams[1])
        self.assertEqual(self.tasks, before)

    def test_rejects_repository_and_original_task_leakage(self):
        spec = {**self.spec, "benchmark_repositories": ["dev", "b", "c", "d"]}
        with self.assertRaisesRegex(ValueError, "disjoint"):
            build_streams(self.tasks, spec)
        self.tasks[-1]["chain"]["source_instance_id"] = source_id(self.tasks[0])
        with self.assertRaisesRegex(ValueError, "share milestones"):
            build_streams(self.tasks, self.spec)

    def test_rejects_broken_chronology(self):
        self.tasks[3], self.tasks[4] = self.tasks[4], self.tasks[3]
        with self.assertRaisesRegex(ValueError, "chronological"):
            build_streams(self.tasks, self.spec)

    def test_warmup_plan_marks_phase_boundary_without_changing_tasks(self):
        development, streams = build_streams(self.tasks, self.spec)
        rows = development + streams[0]
        plan = episode_plan(rows, warmup_count=len(development))
        self.assertEqual([p["phase"] for p in plan], ["warmup"] * 2 + ["benchmark"] * 6)
        self.assertEqual([p["episode"] for p in plan], list(range(1, 9)))
        self.assertNotIn("problem_statement", plan[0])
        self.assertNotIn("phase", rows[0])

    def test_generated_local_streams_preserve_grading_and_exclude_development(self):
        source = ROOT / "tasks/evoarena-all.json"
        if not source.exists():
            self.skipTest("Local EvoArena metadata not available")
        tasks = read_json(source)
        spec = read_json(ROOT / "configs/evoarena-dynamic.json")
        development, streams = build_streams(tasks, spec)
        self.assertEqual(len(development), 40)
        primary = streams[spec["primary_seed"]]
        self.assertEqual(len(primary), 105)
        self.assertEqual(len({t["chain"]["id"] for t in primary}), 8)
        self.assertEqual({source_id(t) for t in development + primary}, {source_id(t) for t in tasks})
        out = ROOT / "tasks/streams/evoarena-dynamic-v1"
        self.assertEqual(primary, read_json(out / "benchmark.json"))
        for seed, rows in streams.items():
            self.assertEqual(rows, read_json(out / f"benchmark-seed-{seed}.json"))
        if all(Path(t["evaluation_dataset"]).exists() for t in primary):
            self.assertEqual(len(evaluation_rows(primary)), len(primary))


if __name__ == "__main__":
    unittest.main()
