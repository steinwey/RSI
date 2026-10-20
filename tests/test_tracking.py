import unittest
from unittest.mock import Mock, patch

from rsi.tracking import Tracker, scoring_metrics, summary_metrics, task_metrics, start_wandb


class TrackingTests(unittest.TestCase):
    def report(self, task_id, success):
        return {"task_id": task_id, "success": success, "selected": 0,
                "tokens": 10, "input_tokens": 7, "output_tokens": 3,
                "seconds": 2, "skill_version": "initial"}

    def test_summary_reports_the_pilot_accuracy(self):
        reports = [self.report(str(i), success)
                   for i, success in enumerate([False, True, True, False, True])]
        metrics = summary_metrics(reports)
        self.assertEqual(metrics["resolved"], 3)
        self.assertEqual(metrics["scored"], 5)
        self.assertEqual(metrics["accuracy"], 0.6)
        self.assertEqual(metrics["accuracy_pct"], 60)
        self.assertEqual(metrics["resolution_rate"], 0.6)
        self.assertEqual(metrics["scoring_coverage"], 1)
        self.assertEqual(metrics["tokens"], 50)
        self.assertEqual(metrics["input_tokens"], 35)
        self.assertEqual(metrics["output_tokens"], 15)

    def test_unknown_scores_do_not_become_failed_solutions(self):
        metrics = scoring_metrics([{"success": True}, {"success": None}, {"success": False}])
        self.assertEqual(metrics["tasks"], 3)
        self.assertEqual(metrics["scored"], 2)
        self.assertEqual(metrics["unscored"], 1)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["resolution_rate"], 1 / 3)
        self.assertEqual(metrics["scoring_coverage"], 2 / 3)

    def test_empty_or_unscored_runs_have_no_accuracy(self):
        for reports in ([], [{"success": None}]):
            with self.subTest(reports=reports):
                metrics = scoring_metrics(reports)
                self.assertIsNone(metrics["accuracy"])
                self.assertIsNone(metrics["accuracy_pct"])
        self.assertIsNone(scoring_metrics([])["scoring_coverage"])
        self.assertEqual(scoring_metrics([{"success": None}])["scoring_coverage"], 0)

    def test_tracker_logs_cumulative_rates_and_persists_final_summary(self):
        run = Mock(summary={})
        tracker = Tracker(run)
        reports = [self.report(str(i), success) for i, success in enumerate([True, None, False])]
        for report in reports:
            tracker.log_task({"benchmark": "swe"}, report, {})
        rows = [call.args[0] for call in run.log.call_args_list]
        self.assertEqual([row["tasks"] for row in rows], [1, 2, 3])
        self.assertEqual([row["accuracy"] for row in rows], [1, 1, 0.5])
        self.assertNotIn("success", rows[1])
        self.assertEqual(rows[-1]["success"], 0)
        self.assertEqual(rows[-1]["tokens"], 10)
        self.assertEqual((rows[-1]["input_tokens"], rows[-1]["output_tokens"]), (7, 3))
        tracker.log_summary(reports)
        self.assertEqual(run.summary["accuracy"], 0.5)
        self.assertEqual(run.summary["unscored"], 1)
        self.assertEqual(run.summary["tokens"], 30)
        self.assertEqual((run.summary["input_tokens"], run.summary["output_tokens"]), (21, 9))

    def test_validation_uses_the_same_scoring_denominators(self):
        update = {"validation": {
            "baseline": [self.report("v1", False), self.report("v2", None)],
            "candidate": [self.report("v1", True), self.report("v2", False)],
        }}
        metrics = task_metrics({"benchmark": "swe"}, self.report("main", True), update)
        self.assertEqual(metrics["validation/baseline/accuracy"], 0)
        self.assertEqual(metrics["validation/baseline/unscored"], 1)
        self.assertEqual(metrics["validation/candidate/accuracy_pct"], 50)
        self.assertEqual(metrics["validation/candidate/scoring_coverage"], 1)
        self.assertEqual(metrics["validation/candidate/input_tokens"], 14)
        self.assertEqual(metrics["validation/candidate/output_tokens"], 6)

    def test_legacy_token_totals_are_not_guessed_as_input_or_output(self):
        report = self.report("legacy", True)
        report.pop("input_tokens")
        report.pop("output_tokens")
        metrics = summary_metrics([self.report("new", True), report])
        self.assertEqual(metrics["tokens"], 20)
        self.assertIsNone(metrics["input_tokens"])
        self.assertIsNone(metrics["output_tokens"])

    def test_updater_usage_is_uploaded_separately(self):
        update = {"updater_usage": {"calls": 1, "input_tokens": 20,
                                    "output_tokens": 4, "cost_usd": 0.01}}
        metrics = task_metrics({"benchmark": "swe"}, self.report("main", True), update)
        self.assertEqual(metrics["updater/input_tokens"], 20)
        self.assertEqual(metrics["updater/output_tokens"], 4)
        self.assertEqual(metrics["tokens"], 10)

    def test_disabled_upload_does_not_initialize_wandb_even_with_a_key(self):
        sdk = Mock()
        with patch.dict("sys.modules", {"wandb": sdk}), \
             patch("rsi.tracking.os.getenv", return_value="configured"):
            tracker = start_wandb({"wandb": {"enabled": False}}, None)
        self.assertIsNone(tracker.run)
        sdk.init.assert_not_called()


if __name__ == "__main__":
    unittest.main()
