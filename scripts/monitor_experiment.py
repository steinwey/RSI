"""Run one experiment with a local watchdog; never restart a failed run."""
import json
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi import ROOT

POLL_SECONDS = 30
MIN_FREE_GIB = 8
IDLE_SECONDS = 1200
SCORING_IDLE_SECONDS = 3000
MONITORED_PATTERNS = ('*/fixed-tools-check.json', '*/evo-*/score.json', '*/evo-*/update.json',
                      '*/evo-*/rollout-*/trajectory.json', '*/evo-*/rollout-*/reflection/summary.json')


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return None


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def tail(path, size=65536):
    try:
        with Path(path).open('rb') as stream:
            stream.seek(max(0, stream.seek(0, 2) - size))
            return stream.read().decode('utf-8', errors='replace')
    except OSError:
        return ''


def problem_in(path, data):
    if path.name == 'score.json' and data.get('success') is None:
        return f'Scoring failed: {data.get("error", "missing score")}'
    if path.name == 'trajectory.json':
        if data.get('error'):
            return f'Rollout failed: {data["error"]}'
        if data.get('reflection', {}).get('stop_reason') == 'reflection_error':
            return f'Reflection failed: {data["reflection"].get("error")}'
        assistants = [m.get('content', '') for m in data.get('messages', []) if m.get('role') == 'assistant']
        if len(assistants) >= 3 and all(not str(m).strip() for m in assistants[-3:]):
            return 'Three consecutive empty solver responses'
    if path.name == 'update.json' and data.get('decision') == 'update_error':
        return f'Skill update failed: {data.get("error")}'
    if path.name == 'fixed-tools-check.json' and data.get('passed') is False:
        return 'Initial fixed-tool check failed'
    if path.parent.name == 'reflection' and path.name == 'summary.json' and data.get('stop_reason') == 'reflection_error':
        return f'Reflection failed: {data.get("error")}'
    return None


def owned_containers(output, stdout):
    """Match exact IDs from this run's log, and exact evaluation directory names."""
    pairs = re.findall(r'Started container (minisweagent-[0-9a-f]+) with ID\s+([0-9a-f]{64})', stdout)
    owned = dict(pairs)
    grading = set()
    for directory in output.glob('*/evo-*/rollout-*/logs/evaluation/*/rsi/*'):
        if directory.is_dir():
            grading.add(f'sweb.eval.{directory.name.lower()}.{directory.parent.parent.name}')
    return owned, grading


def cleanup_containers(output, stdout):
    owned, grading = owned_containers(output, stdout)
    result = subprocess.run(['docker', 'ps', '--no-trunc', '--format', '{{json .}}'],
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=20)
    if result.returncode:
        return [{'error': result.stderr.strip()}]
    targets = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    targets = [item for item in targets
               if owned.get(item['Names']) == item['ID'] or item['Names'] in grading]

    def stop(item):
        record = {'name': item['Names'], 'id': item['ID']}
        # Always stop the owned container, even when snapshotting fails.
        try:
            if item['Names'] in owned:
                directory = output / 'interrupted-containers' / item['Names']
                directory.mkdir(parents=True, exist_ok=True)
                snapshot = subprocess.run(['docker', 'exec', item['ID'], 'bash', '-lc',
                                           'cd /testbed && git add -A && git diff --cached HEAD'],
                                          capture_output=True, timeout=20)
                (directory / 'patch.diff').write_bytes(snapshot.stdout)
                (directory / 'snapshot.stderr.txt').write_bytes(snapshot.stderr)
                record['snapshot_exit'] = snapshot.returncode
        except Exception as exc:
            record['snapshot_error'] = str(exc)
        finally:
            try:
                stopped = subprocess.run(['docker', 'stop', '--time', '3', item['ID']],
                                         capture_output=True, timeout=20)
                record['stop_exit'] = stopped.returncode
            except Exception as exc:
                record['stop_error'] = str(exc)
        return record

    with ThreadPoolExecutor(max_workers=5) as pool:
        return list(pool.map(stop, targets))


def mark_stopped(output, reason):
    from rsi.run_logs import update_run
    from rsi.tracking import summary_metrics
    state = read_json(output / 'workflow.json')
    if state:
        state.update(status='interrupted', error=reason, stopped_at=now())
        phase = state.get('phase')
        if phase in ('train', 'test'):
            state[phase]['status'] = 'interrupted'
            state[phase]['metrics'] = summary_metrics(read_json(output / phase / 'summary.json') or [])
        if state.get('test', {}).get('status') == 'pending':
            state['test']['status'] = 'skipped'
        write_json(output / 'workflow.json', state)
    summaries = []
    for phase in ('train', 'test'):
        directory = output / phase
        rows = read_json(directory / 'summary.json') or []
        summaries.extend(rows)
        info = read_json(directory / 'run-info.json')
        if info and info.get('status') == 'running':
            update_run(directory, status='interrupted', error=reason, stopped_at=now(), completed_tasks=len(rows))
    if (output / 'run-info.json').exists():
        update_run(output, status='interrupted', error=reason, stopped_at=now(), completed_tasks=len(summaries),
                   scored_tasks=sum(r.get('success') is not None for r in summaries))


def monitor(resume=None):
    work = ROOT / '_work'
    work.mkdir(exist_ok=True)
    previous = read_json(work / 'active-parallel-launch.json')
    if previous and previous.get('status') in ('starting', 'running', 'stopping'):
        raise RuntimeError('Another run is recorded as active; inspect it before launching')
    job = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S') + '__monitored-parallel__' + os.urandom(3).hex()
    output = Path(resume).expanduser().resolve() if resume else ROOT / 'runs' / job
    if resume and not output.is_relative_to((ROOT / 'runs').resolve()):
        raise ValueError('Only resume runs within this workspace')
    launch_path = work / (job + '.launch.json')
    status_path = work / (job + '.monitor.json')
    stdout_path, stderr_path = work / (job + '.stdout.log'), work / (job + '.stderr.log')
    command = [sys.executable, '-u', '-X', 'utf8', '-m', 'rsi.cli', 'run',
               '--config', str(ROOT / 'configs/qwen.yaml'),
               '--tasks', str(ROOT / 'tasks/splits/evoarena-frozen-v3/train.json'),
               '--test-tasks', str(ROOT / 'tasks/splits/evoarena-frozen-v3/test.json'),
               '--execution-mode', 'parallel', '--output', str(output), '--label', 'monitored-parallel', '--no-wandb']
    if resume:
        from rsi.checkpoints import resume_request
        _, saved_config, groups = resume_request(output)
        if len(groups) != 2 or saved_config.get('execution_mode') != 'parallel':
            raise ValueError('This monitor requires a parallel train/test workflow')
        command = [sys.executable, '-u', '-X', 'utf8', '-m', 'rsi.cli', 'run', '--resume', str(output)]
    launch = {'status': 'starting', 'started_at': now(), 'output': str(output), 'stdout': str(stdout_path),
              'stderr': str(stderr_path), 'monitor_pid': os.getpid(), 'monitor_status': str(status_path),
              'executable': command[0], 'arguments': command[1:], 'execution_mode': 'parallel',
              'training_tasks': 70, 'test_tasks': 45,
              'resumed_from': str(output) if resume else None,
              'monitor_policy': {'poll_seconds': POLL_SECONDS, 'min_free_gib': MIN_FREE_GIB,
                                 'idle_seconds': IDLE_SECONDS, 'scoring_idle_seconds': SCORING_IDLE_SECONDS}}
    process = None
    reason = None
    seen = {p: (p.stat().st_mtime_ns, p.stat().st_size)
            for pattern in MONITORED_PATTERNS for p in output.glob(pattern)} if resume else {}
    previous_cleanup = {p: p.stat().st_mtime_ns for p in output.glob('*/evo-*/cleanup-errors.log')} if resume else {}
    status = {'status': 'starting', 'started_at': now(), 'output': str(output)}

    def save():
        status['checked_at'] = now()
        write_json(status_path, status)
        launch['status'] = status['status']
        for path in (launch_path, work / 'active-parallel-launch.json'):
            write_json(path, launch)

    try:
        with stdout_path.open('wb') as out, stderr_path.open('wb') as err:
            process = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            launch['pid'] = process.pid
            status.update(status='running', pid=process.pid, monitor_pid=os.getpid())
            save()
            print(json.dumps(launch, ensure_ascii=False), flush=True)
            while True:
                stdout = tail(stdout_path)
                stderr = tail(stderr_path)
                state = read_json(output / 'workflow.json') or {}
                status.update(phase=state.get('phase', 'preflight'), disk_free_gib=round(shutil.disk_usage(ROOT).free / 2**30, 2))
                status['completed_tasks'] = sum(len(read_json(output / p / 'summary.json') or []) for p in ('train', 'test'))
                markers = re.findall(r'\[\d+/\d+\] (evo-[A-Za-z0-9-]+)', stdout)
                active = output / state.get('phase', 'train') / markers[-1] if markers else None
                status['current_task'] = markers[-1] if markers else None
                paths = [stdout_path, stderr_path]
                if active and active.exists():
                    for pattern in ('rollout-*/commands/index.jsonl', 'rollout-*/reflection/*.json',
                                    'rollout-*/trajectory.json', '*.json', 'rollout-*/logs/evaluation/**/run_instance.log'):
                        paths.extend(active.glob(pattern))
                latest = max((p.stat().st_mtime for p in paths if p.exists()), default=time.time())
                status['idle_seconds'] = round(max(0, time.time() - latest))
                for pattern in MONITORED_PATTERNS:
                    for path in output.glob(pattern):
                        stamp = (path.stat().st_mtime_ns, path.stat().st_size)
                        if seen.get(path) == stamp:
                            continue
                        data = read_json(path)
                        if data is not None:
                            seen[path] = stamp
                            problem = problem_in(path, data)
                            if problem:
                                reason = f'{path.relative_to(output)}: {problem}'
                                break
                    if reason:
                        break
                if not reason and any(token in stderr for token in (
                        'Traceback (most recent call last):', 'Exception in thread', 'Fatal Python error', 'No space left on device')):
                    reason = 'Uncaught runtime error in stderr; see preserved stderr log'
                if not reason and any(p.stat().st_size and previous_cleanup.get(p) != p.stat().st_mtime_ns
                                      for p in output.glob('*/evo-*/cleanup-errors.log')):
                    reason = 'Container cleanup failed; see preserved cleanup-errors.log'
                if not reason and status['disk_free_gib'] < MIN_FREE_GIB:
                    reason = f'Disk free space below {MIN_FREE_GIB} GiB'
                progress_lines = [line for line in stdout.splitlines() if re.match(r'\[\d\d:\d\d:\d\d\]', line)]
                scoring = bool(progress_lines and 'scoring' in progress_lines[-1])
                idle_limit = SCORING_IDLE_SECONDS if scoring else IDLE_SECONDS
                if not reason and status['idle_seconds'] > idle_limit:
                    reason = f'No execution/log progress for {status["idle_seconds"]} seconds'
                code = process.poll()
                if reason:
                    break
                if code is not None:
                    state = read_json(output / 'workflow.json') or state
                    if code == 0 and state.get('status') == 'completed':
                        status.update(status='completed', exit_code=code, finished_at=now())
                        save()
                        return 0
                    reason = f'Experiment process exited with code {code}; workflow status={state.get("status")}'
                    break
                save()
                time.sleep(POLL_SECONDS)
    except BaseException as exc:
        reason = f'Watchdog failed: {type(exc).__name__}: {exc}'
    finally:
        if reason:
            status.update(status='stopping', reason=reason)
            # Stop API work even if writing status fails (e.g. full disk).
            try:
                save()
            except Exception as exc:
                status['status_write_error'] = str(exc)
            if process is not None and process.poll() is None:
                try:
                    killed = subprocess.run(['taskkill.exe', '/PID', str(process.pid), '/T', '/F'],
                                            capture_output=True, timeout=30)
                    status['taskkill_exit'] = killed.returncode
                    process.wait(timeout=5)
                except Exception as exc:
                    status['process_stop_error'] = str(exc)
            try:
                status['containers'] = cleanup_containers(output, stdout_path.read_text(encoding='utf-8', errors='replace'))
            except Exception as exc:
                status['cleanup_error'] = str(exc)
            try:
                mark_stopped(output, reason)
            except Exception as exc:
                status['metadata_error'] = str(exc)
            status.update(status='stop_failed' if status.get('process_stop_error') else 'stopped', stopped_at=now())
            try:
                save()
            except Exception as exc:
                status['status_write_error'] = str(exc)
            print(json.dumps(status, ensure_ascii=False), flush=True)
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path, help='Continue this interrupted workflow with saved settings')
    raise SystemExit(monitor(parser.parse_args().resume))
