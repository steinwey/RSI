"""Resume pinned SWE-Chain-Evo metadata and image caching; no model calls."""
import concurrent.futures
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsi.common import read_json, write_json
from rsi.evo_swe import (DATASET, REVISION, _fetch, hub_endpoint,
                         prepare_tasks, prepare_images)

DATA = ROOT / 'datasets/evoarena-swe'
STATE = DATA / 'cache-progress.json'


def fetch(path):
    for attempt in range(3):
        try:
            return _fetch(path)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def cache_chain(cid):
    directory = DATA / 'chains' / cid
    manifest = json.loads(fetch(f'chains/{cid}/chain.json'))
    write_json(directory / 'chain.json', manifest)
    for entry in manifest['instances']:
        filename = f"instances/{entry['version_chain_index']:03d}-{entry['instance_id']}.json"
        target = directory / filename
        if target.exists():
            continue
        value = json.loads(fetch(f'chains/{cid}/{filename}'))
        write_json(target, value)
    print(f"Metadata ready: {cid}", flush=True)
    return cid


def main():
    state = {'status': 'metadata', 'dataset': DATASET, 'revision': REVISION,
             'ready_images': 0, 'errors': []}
    write_json(STATE, state)
    try:
        url = f'{hub_endpoint()}/api/datasets/{DATASET}/tree/{REVISION}/chains'
        with urllib.request.urlopen(url, timeout=60) as response:
            entries = json.load(response)
        chains = sorted(e['path'].split('/')[-1] for e in entries if e['type'] == 'directory')
        if len(chains) != 50:
            raise ValueError(f'Expected 50 chains at pinned revision, got {len(chains)}')
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(cache_chain, chains))
        write_json(DATA / 'dataset-source.json', {'dataset': DATASET, 'revision': REVISION})
        tasks = prepare_tasks(DATA, chains, ROOT / 'tasks/evoarena-all.json')
        unique = {t['instance']['image']: t for t in tasks}
        state.update(status='images', chains=len(chains), task_slots=len(tasks),
                     total_images=len(unique))
        write_json(STATE, state)
        for image, task in unique.items():
            state['current_image'] = image
            write_json(STATE, state)
            for drive in ('C:/', str(ROOT.anchor)):
                if shutil.disk_usage(drive).free < 25 * 1024**3:
                    state['status'] = 'stopped_low_disk'
                    write_json(STATE, state)
                    print(f'Stopped: less than 25 GiB free on {drive}', flush=True)
                    return
            try:
                prepare_images([task], DATA / 'image-cache')
                state['ready_images'] += 1
            except Exception as exc:
                state['errors'].append({'image': image, 'error': f'{type(exc).__name__}: {exc}'})
                print(f'Image failed: {image}: {exc}', flush=True)
            write_json(STATE, state)
        state['status'] = 'completed' if not state['errors'] else 'completed_with_errors'
        write_json(STATE, state)
    except Exception as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(STATE, state)
        raise


if __name__ == '__main__':
    main()
