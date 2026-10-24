import copy
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsi import ROOT
from rsi.baselines import set_execution_mode
from rsi.checkpoints import inspect_checkpoint, resume_request, run_lock
from rsi.cli import main
from rsi.common import file_map, read_json, write_json
from rsi.config import load_config
from rsi.runner import run_experiment
from rsi.skills import MAX_FILE_CHARS, SkillStore
from rsi.workflow import run_train_test
from scripts.monitor_experiment import problem_in


def tasks(repo, cid, count):
    return [{"id": f"evo-{cid}-{i:03d}", "benchmark": "swe-evo", "prompt": f"Fix {repo} {i}",
             "instance": {"instance_id": f"evo-{cid}-{i:03d}", "repo": repo,
                          "problem_statement": f"Fix {repo} {i}"},
             "chain": {"id": cid, "index": i, "length": count, "source_instance_id": f"{cid}-{i}"}}
            for i in range(count)]


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.output = self.root / 'experiment'
        self.config = load_config(ROOT / 'configs/qwen.yaml')
        self.config['wandb'] = {'enabled': False}
        self.train, self.test = tasks('owner/train', 'train', 3), tasks('owner/test', 'test', 2)
        self.failure_task = self.train[-1]['id']
        self.seen, self.proposals = [], 0
        self.guide = file_map(ROOT / 'skills/test-and-diagnose')['SKILL.md']
        self.learned = self.guide + '\nPrefer a focused reproduction before editing.\n'
        for fixture in (patch('rsi.runner.run_group', side_effect=self.group),
                        patch('rsi.runner.induce', side_effect=self.propose),
                        patch('rsi.runner.functional_gate', return_value={'passed': True})):
            fixture.start()
            self.addCleanup(fixture.stop)

    def group(self, task, folder, skill, config):
        folder.mkdir(parents=True, exist_ok=False)
        self.seen.append((task['id'], skill.name, config['evolve']))
        if task['id'] == self.failure_task:
            partial = folder / 'rollout-0/commands/partial.txt'
            partial.parent.mkdir(parents=True)
            partial.write_text('preserve this interrupted attempt', encoding='utf-8')
            raise KeyboardInterrupt('simulated interruption')
        write_json(folder / 'score.json', {'success': True})
        return {'task_id': task['id'], 'success': True, 'tokens': 2, 'input_tokens': 1,
                'output_tokens': 1, 'seconds': 1, 'skill_version': skill.name, 'chain': task['chain']}, []

    def propose(self, *args, **kwargs):
        self.proposals += 1
        return {'changes': {'SKILL.md': self.learned} if self.proposals == 1 else {}}

    def interrupted(self):
        with self.assertRaises(KeyboardInterrupt):
            run_train_test(self.train, self.test, self.config, self.output)

    def test_resume_keeps_completed_tasks_skill_history_and_total_metrics(self):
        self.interrupted()
        before = {p: p.read_bytes() for t in self.train[:2] for p in (self.output / 'train' / t['id']).rglob('*') if p.is_file()}
        history = (self.output / 'train/skills/history.jsonl').read_bytes()
        version = read_json(self.output / 'train/skills/current.json')['version']
        original_code = read_json(self.output / 'train/run-info.json')['code_version']
        output, config, groups = resume_request(self.output)
        self.failure_task, self.seen = None, []
        result = run_train_test(*groups, config, output, resume=True)
        self.assertEqual([row[0] for row in self.seen], [self.train[-1]['id']] + [t['id'] for t in self.test])
        self.assertTrue(all(row[1] == version for row in self.seen))
        self.assertEqual([row[2] for row in self.seen], [True, False, False])
        self.assertEqual((self.proposals, result['status']), (3, 'completed'))
        self.assertEqual((result['train']['metrics']['tasks'], result['test']['metrics']['tasks']), (3, 2))
        self.assertEqual(read_json(output / 'run-info.json')['completed_tasks'], 5)
        self.assertEqual((output / 'train/skills/history.jsonl').read_bytes(), history)
        self.assertTrue(all(path.read_bytes() == value for path, value in before.items()))
        archives = list((output / 'train/resume-attempts').glob('*/evo-train-002/rollout-0/commands/partial.txt'))
        self.assertEqual(len(archives), 1)
        self.assertEqual(archives[0].read_text(), 'preserve this interrupted attempt')
        info = read_json(output / 'train/run-info.json')
        self.assertEqual((info['code_version'], info['resume_count']), (original_code, 1))
        self.assertIn('resume_code_version', info)

    def test_resume_during_frozen_test_does_not_repeat_training_or_updates(self):
        self.failure_task = self.test[-1]['id']
        self.interrupted()
        self.failure_task, self.seen = None, []
        output, config, groups = resume_request(self.output)
        result = run_train_test(*groups, config, output, resume=True)
        self.assertEqual([row[0] for row in self.seen], [self.test[-1]['id']])
        self.assertFalse(self.seen[0][2])
        self.assertEqual(self.proposals, 3)
        self.assertEqual(result['status'], 'completed')

    def test_changed_config_or_task_contents_are_rejected_before_execution(self):
        self.interrupted()
        directory = self.output / 'train'
        config = read_json(directory / 'config.json')
        altered = {**config, 'command_timeout': config['command_timeout'] + 1}
        with self.assertRaisesRegex(ValueError, 'configuration'):
            inspect_checkpoint(directory, self.train, altered)
        changed = copy.deepcopy(self.train)
        changed[-1]['instance']['problem_statement'] = 'Different problem'
        with self.assertRaisesRegex(ValueError, 'contents'):
            inspect_checkpoint(directory, changed, config)

    def test_unscored_prefix_cannot_be_silently_skipped(self):
        self.interrupted()
        write_json(self.output / 'train' / self.train[0]['id'] / 'score.json', {'success': None, 'error': 'grading failed'})
        with self.assertRaisesRegex(ValueError, 'official score'):
            resume_request(self.output)

    def test_current_skill_cannot_jump_past_the_committed_history(self):
        self.interrupted()
        directory = self.output / 'train'
        store = SkillStore(directory / 'skills', directory / 'initial-skill/versions' / read_json(directory / 'initial-skill/current.json')['version'])
        candidate, _ = store.propose({'SKILL.md': self.guide + '\nUncommitted rule.'})
        write_json(directory / 'skills/current.json', {'version': candidate.name})
        with self.assertRaisesRegex(ValueError, 'pointer'):
            resume_request(self.output)

    def test_resume_lock_prevents_concurrent_processes_and_releases(self):
        self.output.mkdir()
        with run_lock(self.output):
            with self.assertRaisesRegex(ValueError, 'lock'):
                with run_lock(self.output):
                    self.fail('second lock acquired')
        with run_lock(self.output):
            pass

    def test_cli_resume_loads_saved_configuration_and_skips_the_prefix(self):
        self.interrupted()
        self.failure_task, self.seen = None, []
        with patch('rsi.cli.require_credentials'), patch('rsi.cli.shutil.which', return_value='docker'), \
             patch('rsi.cli.validate_environments'), contextlib.redirect_stdout(io.StringIO()):
            main(['run', '--resume', str(self.output)])
        self.assertEqual([row[0] for row in self.seen], [self.train[-1]['id']] + [t['id'] for t in self.test])

    def test_resume_cannot_override_protocol_or_model(self):
        for flags in (['--model', 'other'], ['--config', 'other.yaml'], ['--fixed-skill'], ['--no-wandb'], ['--execution-mode', 'single']):
            with self.subTest(flags=flags), patch('rsi.cli.load_config') as load, \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['run', '--resume', str(self.output), *flags])
            load.assert_not_called()

    def test_oversized_proposal_is_rejected_without_stopping_the_next_task(self):
        self.failure_task = None
        with patch('rsi.runner.induce', side_effect=[{'changes': {'SKILL.md': 'x' * (MAX_FILE_CHARS + 1)}}, {'changes': {}}, {'changes': {}}]):
            reports = run_experiment(self.train, self.config, self.output)
        update = read_json(self.output / self.train[0]['id'] / 'update.json')
        self.assertEqual(update['decision'], 'checks_failed')
        self.assertFalse(update['accepted'])
        self.assertIsNone(problem_in(Path('update.json'), update))
        self.assertEqual(len(reports), 3)
        self.assertEqual(len({row[1] for row in self.seen}), 1)

    def test_real_updater_failure_remains_an_error(self):
        self.failure_task = None
        with patch('rsi.runner.induce', side_effect=RuntimeError('API unavailable')):
            run_experiment(self.train, self.config, self.output)
        update = read_json(self.output / self.train[0]['id'] / 'update.json')
        self.assertEqual(update['decision'], 'update_error')
        self.assertIn('Skill update failed', problem_in(Path('update.json'), update))

    def test_single_frozen_run_can_resume_without_evolution(self):
        config = set_execution_mode(copy.deepcopy(self.config), 'single')
        config['evolve'] = False
        with self.assertRaises(KeyboardInterrupt):
            run_experiment(self.train, config, self.output)
        self.failure_task, self.seen = None, []
        output, saved, groups = resume_request(self.output)
        reports = run_experiment(groups[0], saved, output, resume=True)
        self.assertEqual([row[0] for row in self.seen], [self.train[-1]['id']])
        self.assertFalse(self.seen[0][2])
        self.assertEqual((len(reports), self.proposals), (3, 0))


if __name__ == '__main__':
    unittest.main()
