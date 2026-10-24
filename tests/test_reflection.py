import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from rsi.reflection import refine_rollout, evidence_step_limit


class Reviewer:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.n_calls = self.cost = 0
        self.requests = []

    def query(self, messages, **kwargs):
        self.requests.append(messages)
        self.n_calls += 1
        return {'content': json.dumps(next(self.replies))}

    def stats(self):
        return {'calls': self.n_calls, 'input_tokens': self.n_calls * 10,
                'output_tokens': self.n_calls * 2, 'cost_usd': 0}


STOP = {'stop': True, 'reason': 'No concrete issue remains', 'feedback': ''}
FIX = {'stop': False, 'reason': 'Missing edge case', 'feedback': 'Check None handling and test it.'}


class ReflectionTests(unittest.TestCase):
    def run_loop(self, replies, *, expired=False):
        reviewer = Reviewer(replies)
        self.result = {'rollout': 0, 'status': 'Submitted', 'artifact': 'initial', 'error': None,
                       'messages': [], 'score': {'success': True}, 'hidden_tests': 'SECRET'}
        agent = SimpleNamespace(messages=[{'role': 'assistant', 'content': 'initial work'}],
                                model=SimpleNamespace(cost=0))
        agent.add_message = lambda role, content: agent.messages.append({'role': role, 'content': content})
        revisions = []
        def resume(agent, feedback, steps):
            revisions.append(feedback)
            agent.add_message('assistant', f'revision {len(revisions)}')
            return 'Submitted', 'done'
        backend = SimpleNamespace(artifact=lambda: f'patch-{len(revisions)}')
        config = {'reflection_max_rounds': 5, 'reflection_model': {'timeout': 10},
                  'step_limit': 30, 'reflection_step_limit': 30, 'cost_limit': 3}
        with tempfile.TemporaryDirectory() as temp:
            log = refine_rollout({'prompt': 'Fix task'}, self.result, agent, backend, reviewer,
                                 config, Path(temp), time.monotonic() + (-1 if expired else 30), resume)
            self.assertTrue((Path(temp) / 'summary.json').exists())
        self.assertNotIn('SECRET', json.dumps(reviewer.requests))
        self.assertNotIn('"score"', json.dumps(reviewer.requests))
        return log, reviewer, revisions

    def test_stop_without_revision(self):
        log, reviewer, revisions = self.run_loop([STOP])
        self.assertEqual(log['stop_reason'], 'reviewer_stop')
        self.assertEqual(reviewer.n_calls, 1)
        self.assertEqual(revisions, [])
        self.assertEqual(self.result['artifact'], 'initial')

    def test_refine_then_review_updated_artifact(self):
        log, reviewer, revisions = self.run_loop([FIX, STOP])
        self.assertEqual(log['revisions'], 1)
        second = json.loads(reviewer.requests[1][1]['content'])
        self.assertEqual(second['trajectory']['artifact'], 'patch-1')
        self.assertIn('revision 1', json.dumps(second))

    def test_maximum_five_revisions(self):
        log, reviewer, revisions = self.run_loop([FIX] * 6)
        self.assertEqual(log['stop_reason'], 'max_rounds')
        self.assertEqual((reviewer.n_calls, len(revisions)), (5, 5))

    def test_invalid_feedback_is_logged_and_stops(self):
        log, reviewer, revisions = self.run_loop([{'stop': 'false'}])
        self.assertEqual(log['stop_reason'], 'reflection_error')
        self.assertEqual(revisions, [])

    def test_expired_deadline_prevents_request(self):
        log, reviewer, revisions = self.run_loop([], expired=True)
        self.assertEqual(log['stop_reason'], 'time_limit')
        self.assertEqual(reviewer.n_calls, 0)

    def test_evidence_includes_initial_and_refinement_steps(self):
        self.assertEqual(evidence_step_limit({'step_limit': 30, 'reflection_enabled': True}), 180)
