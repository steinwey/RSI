"""Verify that the watchdog stops failures and only touches its own containers."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import monitor_experiment as monitor


class MonitorTests(unittest.TestCase):
    def test_wrong_answer_and_step_limit_are_normal_results(self):
        self.assertIsNone(monitor.problem_in(Path('score.json'), {'success': False}))
        self.assertIsNone(monitor.problem_in(Path('trajectory.json'), {'status': 'LimitsExceeded', 'error': None}))
        self.assertIsNone(monitor.problem_in(Path('update.json'), {'decision': 'checks_failed'}))

    def test_missing_score_and_api_errors_stop_the_run(self):
        self.assertIn('Scoring failed', monitor.problem_in(Path('score.json'), {'success': None, 'error': 'bad decoding'}))
        self.assertIn('Rollout failed', monitor.problem_in(Path('trajectory.json'), {'error': 'AuthenticationError'}))
        self.assertIn('Skill update failed', monitor.problem_in(Path('update.json'), {'decision': 'update_error', 'error': 'timeout'}))

    def test_reflection_failure_stops_but_reviewer_stop_is_normal(self):
        path = Path('reflection/summary.json')
        self.assertIn('Reflection failed', monitor.problem_in(path, {'stop_reason': 'reflection_error', 'error': 'empty reply'}))
        self.assertIsNone(monitor.problem_in(path, {'stop_reason': 'reviewer_stop'}))

    def test_only_recorded_container_id_is_stopped(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            name, cid = 'minisweagent-aabbccdd', 'a' * 64
            stdout = f'Started container {name} with ID\n{cid}\n'
            inventory = [{'Names': name, 'ID': cid}, {'Names': 'unrelated', 'ID': 'b' * 64}]
            calls = []

            def run(command, **kwargs):
                calls.append(command)
                if command[1] == 'ps':
                    return subprocess.CompletedProcess(command, 0, '\n'.join(map(json.dumps, inventory)), '')
                return subprocess.CompletedProcess(command, 0, b'patch', b'')

            with patch.object(monitor.subprocess, 'run', side_effect=run):
                records = monitor.cleanup_containers(root, stdout)
            self.assertEqual([r['id'] for r in records], [cid])
            self.assertEqual((root / 'interrupted-containers' / name / 'patch.diff').read_bytes(), b'patch')
            self.assertFalse(any('b' * 64 in command for command in calls))

    def test_reused_container_name_cannot_match_a_different_id(self):
        with tempfile.TemporaryDirectory() as temp:
            stdout = 'Started container minisweagent-aabbccdd with ID\n' + 'a' * 64
            inventory = json.dumps({'Names': 'minisweagent-aabbccdd', 'ID': 'b' * 64})
            with patch.object(monitor.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, inventory, '')) as run:
                self.assertEqual(monitor.cleanup_containers(Path(temp), stdout), [])
            self.assertEqual(run.call_count, 1)

    def test_snapshot_failure_still_stops_the_owned_container(self):
        with tempfile.TemporaryDirectory() as temp:
            name, cid = 'minisweagent-aabbccdd', 'a' * 64
            stdout = f'Started container {name} with ID\n{cid}'
            inventory = json.dumps({'Names': name, 'ID': cid})

            def run(command, **kwargs):
                if command[1] == 'ps':
                    return subprocess.CompletedProcess(command, 0, inventory, '')
                if command[1] == 'exec':
                    raise subprocess.TimeoutExpired(command, 20)
                return subprocess.CompletedProcess(command, 0)

            with patch.object(monitor.subprocess, 'run', side_effect=run) as execute:
                records = monitor.cleanup_containers(Path(temp), stdout)
            self.assertIn('snapshot_error', records[0])
            self.assertEqual(records[0]['stop_exit'], 0)
            self.assertTrue(any(call.args[0][1] == 'stop' for call in execute.call_args_list))

    def test_grading_container_requires_this_runs_evaluation_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / 'train/evo-example-000/rollout-0/logs/evaluation/rsi-123/rsi/evo-example-000'
            directory.mkdir(parents=True)
            _, grading = monitor.owned_containers(root, '')
            self.assertEqual(grading, {'sweb.eval.evo-example-000.rsi-123'})

    def test_watchdog_exception_kills_its_child_process_tree(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            process = SimpleNamespace(pid=12345, poll=Mock(return_value=None), wait=Mock(return_value=1))
            with patch.object(monitor, 'ROOT', root), \
                 patch.object(monitor.subprocess, 'Popen', return_value=process), \
                 patch.object(monitor.shutil, 'disk_usage', side_effect=OSError('probe failed')), \
                 patch.object(monitor.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as kill, \
                 patch.object(monitor, 'cleanup_containers', return_value=[]), \
                 patch.object(monitor, 'mark_stopped'):
                self.assertEqual(monitor.monitor(), 1)
            self.assertEqual(kill.call_args.args[0], ['taskkill.exe', '/PID', '12345', '/T', '/F'])
            launch = monitor.read_json(root / '_work/active-parallel-launch.json')
            self.assertEqual(launch['status'], 'stopped')

    def test_normal_completion_does_not_kill_processes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            def start(command, **kwargs):
                output = Path(command[command.index('--output') + 1])
                output.mkdir(parents=True)
                monitor.write_json(output / 'workflow.json', {'status': 'completed', 'phase': 'finished'})
                return SimpleNamespace(pid=12345, poll=lambda: 0)

            with patch.object(monitor, 'ROOT', root), \
                 patch.object(monitor.subprocess, 'Popen', side_effect=start), \
                 patch.object(monitor.shutil, 'disk_usage', return_value=SimpleNamespace(free=20 * 2**30)), \
                 patch.object(monitor, 'cleanup_containers') as cleanup, \
                 patch.object(monitor.subprocess, 'run') as kill:
                self.assertEqual(monitor.monitor(), 0)
            cleanup.assert_not_called()
            kill.assert_not_called()

    def test_resume_uses_saved_inputs_and_ignores_archived_attempt_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / 'runs/previous'
            old = output / 'train/evo-example-000/rollout-0/reflection/summary.json'
            old.parent.mkdir(parents=True)
            monitor.write_json(old, {'stop_reason': 'reflection_error', 'error': 'previous attempt'})
            monitor.write_json(output / 'workflow.json', {'status': 'completed', 'phase': 'finished'})
            with patch.object(monitor, 'ROOT', root), \
                 patch('rsi.checkpoints.resume_request', return_value=(output, {'execution_mode': 'parallel'}, [[], []])), \
                 patch.object(monitor.subprocess, 'Popen', return_value=SimpleNamespace(pid=12345, poll=lambda: 0)) as start, \
                 patch.object(monitor.shutil, 'disk_usage', return_value=SimpleNamespace(free=20 * 2**30)), \
                 patch.object(monitor, 'cleanup_containers') as cleanup, \
                 patch.object(monitor.subprocess, 'run') as kill:
                self.assertEqual(monitor.monitor(output), 0)
            command = start.call_args.args[0]
            self.assertEqual(command[-2:], ['--resume', str(output)])
            self.assertNotIn('--config', command)
            cleanup.assert_not_called()
            kill.assert_not_called()


if __name__ == '__main__':
    unittest.main()
