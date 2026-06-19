from feature import FeatureAgent
import json
from pathlib import Path

import numpy as np


USE_SHARDS = False

# Tune this if you want bigger or smaller shard files.
SHARD_SAMPLE_LIMIT = 90000

INPUT_FILE = 'data/data.txt'
OUTPUT_DIR = 'data'

obs = [[] for _ in range(4)]
actions = [[] for _ in range(4)]
matchid = -1
l = []
base_dir = Path(__file__).resolve().parent

if USE_SHARDS:
    shard_id = 0
    shard_sample_count = 0
    shard_match_counts = []
    shard_obs = []
    shard_glob = []
    shard_mask = []
    shard_act = []
    shard_manifest = []


def _resolve_local_path(path):
    path = Path(path)
    return path if path.is_absolute() else base_dir / path


def _save_npz(output_dir, name, **payload):
    output_path = _resolve_local_path(output_dir)
    output_path.mkdir(parents = True, exist_ok = True)
    np.savez(output_path / f'{name}.npz', **payload)


def _save_json(output_dir, name, payload):
    output_path = _resolve_local_path(output_dir)
    output_path.mkdir(parents = True, exist_ok = True)
    with open(output_path / f'{name}.json', 'w', encoding = 'utf-8') as f:
        json.dump(payload, f)


def _open_input_file(input_file):
    return open(_resolve_local_path(input_file), encoding = 'UTF-8')


def filterData():
    global obs
    global actions
    newobs = [[] for _ in range(4)]
    newactions = [[] for _ in range(4)]
    for i in range(4):
        for j, o in enumerate(obs[i]):
            if o['action_mask'].sum() > 1:  # ignore states with single valid action (Pass)
                newobs[i].append(o)
                newactions[i].append(actions[i][j])
    obs = newobs
    actions = newactions


def _materialize_match():
    assert [len(x) for x in obs] == [len(x) for x in actions], 'obs actions not matching!'
    match_samples = sum([len(x) for x in obs])
    l.append(match_samples)
    payload = {
        'obs': np.stack([x['observation'] for i in range(4) for x in obs[i]]).astype(np.int8),
        'glob': np.stack([x['global'] for i in range(4) for x in obs[i]]).astype(np.int8),
        'mask': np.stack([x['action_mask'] for i in range(4) for x in obs[i]]).astype(np.int8),
        'act': np.array([x for i in range(4) for x in actions[i]]),
    }
    return match_samples, payload


def _clear_match_buffers():
    for x in obs:
        x.clear()
    for x in actions:
        x.clear()


def _flush_shard():
    global shard_id
    global shard_sample_count
    global shard_match_counts
    global shard_obs
    global shard_glob
    global shard_mask
    global shard_act
    global shard_manifest

    if shard_sample_count == 0:
        return

    payload = {
        'obs': np.concatenate(shard_obs, axis = 0),
        'glob': np.concatenate(shard_glob, axis = 0),
        'mask': np.concatenate(shard_mask, axis = 0),
        'act': np.concatenate(shard_act, axis = 0),
        'match_counts': np.array(shard_match_counts, dtype = np.int32),
    }
    _save_npz(OUTPUT_DIR, f'shard_{shard_id:05d}', **payload)
    print('Saved shard %d with %d samples and %d matches.' % (shard_id, shard_sample_count, len(shard_match_counts)))
    shard_manifest.append({
        'file': f'shard_{shard_id:05d}.npz',
        'samples': int(shard_sample_count),
        'matches': int(len(shard_match_counts)),
        'match_counts': [int(x) for x in shard_match_counts],
    })
    shard_id += 1
    shard_sample_count = 0
    shard_match_counts = []
    shard_obs = []
    shard_glob = []
    shard_mask = []
    shard_act = []


def _append_match_to_shard(match_samples, payload):
    global shard_sample_count
    global shard_match_counts
    global shard_obs
    global shard_glob
    global shard_mask
    global shard_act

    if shard_sample_count and shard_sample_count + match_samples > SHARD_SAMPLE_LIMIT:
        _flush_shard()

    shard_sample_count += match_samples
    shard_match_counts.append(match_samples)
    shard_obs.append(payload['obs'])
    shard_glob.append(payload['glob'])
    shard_mask.append(payload['mask'])
    shard_act.append(payload['act'])

    if shard_sample_count >= SHARD_SAMPLE_LIMIT:
        _flush_shard()


f = _open_input_file(INPUT_FILE)
try:
    line = f.readline()
    while line:
        t = line.split()
        if len(t) == 0:
            line = f.readline()
            continue
        if t[0] == 'Match':
            agents = [FeatureAgent(i) for i in range(4)]
            matchid += 1
            if matchid % 128 == 0:
                print('Processing match %d %s...' % (matchid, t[1]))
        elif t[0] == 'Wind':
            for agent in agents:
                agent.request2obs(line)
        elif t[0] == 'Player':
            p = int(t[1])
            if t[2] == 'Deal':
                agents[p].request2obs(' '.join(t[2:]))
            elif t[2] == 'Draw':
                for i in range(4):
                    if i == p:
                        obs[p].append(agents[p].request2obs(' '.join(t[2:])))
                        actions[p].append(0)
                    else:
                        agents[i].request2obs(' '.join(t[:3]))
            elif t[2] == 'Play':
                actions[p].pop()
                actions[p].append(agents[p].response2action(' '.join(t[2:])))
                for i in range(4):
                    if i == p:
                        agents[p].request2obs(line)
                    else:
                        obs[i].append(agents[i].request2obs(line))
                        actions[i].append(0)
                curTile = t[3]
            elif t[2] == 'Chi':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Chi %s %s' % (curTile, t[3])))
                for i in range(4):
                    if i == p:
                        obs[p].append(agents[p].request2obs('Player %d Chi %s' % (p, t[3])))
                        actions[p].append(0)
                    else:
                        agents[i].request2obs('Player %d Chi %s' % (p, t[3]))
            elif t[2] == 'Peng':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Peng %s' % t[3]))
                for i in range(4):
                    if i == p:
                        obs[p].append(agents[p].request2obs('Player %d Peng %s' % (p, t[3])))
                        actions[p].append(0)
                    else:
                        agents[i].request2obs('Player %d Peng %s' % (p, t[3]))
            elif t[2] == 'Gang':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Gang %s' % t[3]))
                for i in range(4):
                    agents[i].request2obs('Player %d Gang %s' % (p, t[3]))
            elif t[2] == 'AnGang':
                actions[p].pop()
                actions[p].append(agents[p].response2action('AnGang %s' % t[3]))
                for i in range(4):
                    if i == p:
                        agents[p].request2obs('Player %d AnGang %s' % (p, t[3]))
                    else:
                        agents[i].request2obs('Player %d AnGang' % p)
            elif t[2] == 'BuGang':
                actions[p].pop()
                actions[p].append(agents[p].response2action('BuGang %s' % t[3]))
                for i in range(4):
                    if i == p:
                        agents[p].request2obs('Player %d BuGang %s' % (p, t[3]))
                    else:
                        obs[i].append(agents[i].request2obs('Player %d BuGang %s' % (p, t[3])))
                        actions[i].append(0)
            elif t[2] == 'Hu':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Hu'))
            # Deal with Ignore clause
            if t[2] in ['Peng', 'Gang', 'Hu']:
                for k in range(5, 15, 5):
                    if len(t) > k:
                        p = int(t[k + 1])
                        if t[k + 2] == 'Chi':
                            actions[p].pop()
                            actions[p].append(agents[p].response2action('Chi %s %s' % (curTile, t[k + 3])))
                        elif t[k + 2] == 'Peng':
                            actions[p].pop()
                            actions[p].append(agents[p].response2action('Peng %s' % t[k + 3]))
                        elif t[k + 2] == 'Gang':
                            actions[p].pop()
                            actions[p].append(agents[p].response2action('Gang %s' % t[k + 3]))
                        elif t[k + 2] == 'Hu':
                            actions[p].pop()
                            actions[p].append(agents[p].response2action('Hu'))
                    else:
                        break
        elif t[0] == 'Score':
            filterData()
            if USE_SHARDS:
                match_samples, payload = _materialize_match()
                _append_match_to_shard(match_samples, payload)
            else:
                match_samples, payload = _materialize_match()
                _save_npz(OUTPUT_DIR, '%d' % matchid, **payload)
            _clear_match_buffers()
        line = f.readline()
finally:
    f.close()

if USE_SHARDS:
    _flush_shard()
    _save_json(OUTPUT_DIR, 'count', {
        'schema': 'sharded-v1',
        'shard_sample_limit': SHARD_SAMPLE_LIMIT,
        'total_matches': len(l),
        'total_samples': int(sum(l)),
        'shards': shard_manifest,
    })
else:
    _save_json(OUTPUT_DIR, 'count', l)
