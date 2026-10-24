import tempfile
import unittest
from types import SimpleNamespace

from rsi.command_logs import RecordingEnvironment, script_invocations
from rsi.evolution import skill_usage


class ScriptUsageTests(unittest.TestCase):
    def test_mentions_and_python_code_are_not_script_invocations(self):
        path = '/opt/rsi-skills/scripts/inspect_project.py'
        for command in (f'echo {path}', f'cat {path}', f'python -c "print(\'{path}\')"'):
            self.assertEqual(script_invocations(command), [])
        self.assertEqual(script_invocations(f'cd /testbed && python3 -B "{path}" --path .'),
                         ['scripts/inspect_project.py'])

    def test_failed_command_attempt_is_recorded_without_claiming_success(self):
        env = SimpleNamespace(execute=lambda command, cwd='': {'output': 'failure', 'returncode': 1})
        with tempfile.TemporaryDirectory() as tmp:
            recorder = RecordingEnvironment(env, tmp, lambda: 4)
            recorder.execute('python /opt/rsi-skills/scripts/inspect_project.py --path /missing')
            call = recorder.script_calls[0]
            self.assertEqual(call['returncode'], 1)
            self.assertEqual(call['message'], 4)

    def test_legacy_mentions_do_not_become_calls(self):
        result = {'rollout': 0, 'messages': [{'role': 'assistant', 'content':
                  'python /opt/rsi-skills/scripts/inspect_project.py --path .'}]}
        self.assertEqual(skill_usage([result])[0],
                         {'rollout': 0, 'source': 'unavailable', 'calls': []})
