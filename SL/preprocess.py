from feature import FeatureAgent
import argparse
import json
from pathlib import Path

import numpy as np

from augmentation import sample_variant, transform_batch


INPUT_FILE = 'data/data.txt'
OUTPUT_DIR = 'data'
SPLIT_RATIO = 0.95
AUGMENT_SAFE_MATCHES = True
AUGMENT_COPIES = 1
AUGMENT_SEED = 20240618
CHUNK_SAMPLE_LIMIT = 500000
RISKY_FAN_NAMES = {
    '绿一色',
    '推不倒',
}

obs = [[] for _ in range(4)]
actions = [[] for _ in range(4)]
matchid = -1
augmented_matches = 0
augmented_payloads = 0
risky_matches = 0
base_dir = Path(__file__).resolve().parent
rng = np.random.default_rng(seed = AUGMENT_SEED)
split_manifests = {
    'train': {'schema': 'local-chunk-v1', 'chunk_sample_limit': CHUNK_SAMPLE_LIMIT, 'total_samples': 0, 'chunks': []},
    'valid': {'schema': 'local-chunk-v1', 'chunk_sample_limit': CHUNK_SAMPLE_LIMIT, 'total_samples': 0, 'chunks': []},
}
split_output_ids = {
    'train': 0,
    'valid': 0,
}
chunk_buffers = {
    'train': {'obs': [], 'glob': [], 'mask': [], 'act': []},
    'valid': {'obs': [], 'glob': [], 'mask': [], 'act': []},
}
chunk_sample_counts = {
    'train': 0,
    'valid': 0,
}


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


def _parse_args():
    parser = argparse.ArgumentParser(description = 'Preprocess MahjongGB SL data into local train/valid npz chunks.')
    parser.add_argument('--input-file', type = str, default = INPUT_FILE)
    parser.add_argument('--output-dir', type = str, default = OUTPUT_DIR)
    parser.add_argument('--split-ratio', type = float, default = SPLIT_RATIO)
    parser.add_argument('--augment-copies', type = int, default = AUGMENT_COPIES, help = 'Augmented copies per safe training match. 0 disables augmentation; 1 means original + one transformed copy.')
    parser.add_argument('--augment-seed', type = int, default = AUGMENT_SEED)
    parser.add_argument('--chunk-sample-limit', type = int, default = CHUNK_SAMPLE_LIMIT)
    return parser.parse_args()


def _apply_args(args):
    global INPUT_FILE
    global OUTPUT_DIR
    global SPLIT_RATIO
    global AUGMENT_SAFE_MATCHES
    global AUGMENT_COPIES
    global AUGMENT_SEED
    global CHUNK_SAMPLE_LIMIT
    global rng

    if args.augment_copies < 0:
        raise ValueError('--augment-copies must be >= 0')
    if not 0 < args.split_ratio < 1:
        raise ValueError('--split-ratio must be between 0 and 1')

    INPUT_FILE = args.input_file
    OUTPUT_DIR = args.output_dir
    SPLIT_RATIO = args.split_ratio
    AUGMENT_COPIES = args.augment_copies
    AUGMENT_SAFE_MATCHES = AUGMENT_COPIES > 0
    AUGMENT_SEED = args.augment_seed
    CHUNK_SAMPLE_LIMIT = args.chunk_sample_limit
    rng = np.random.default_rng(seed = AUGMENT_SEED)
    for manifest in split_manifests.values():
        manifest['chunk_sample_limit'] = CHUNK_SAMPLE_LIMIT


def _prepare_output_dirs():
    for split_name in ['train', 'valid']:
        split_dir = _resolve_local_path(Path(OUTPUT_DIR) / split_name)
        split_dir.mkdir(parents = True, exist_ok = True)
        for old_npz in split_dir.glob('*.npz'):
            old_npz.unlink()
        count_path = split_dir / 'count.json'
        if count_path.exists():
            count_path.unlink()


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
    payload = {
        'obs': np.stack([x['observation'] for i in range(4) for x in obs[i]]).astype(np.int8),
        'glob': np.stack([x['global'] for i in range(4) for x in obs[i]]).astype(np.float32),
        'mask': np.stack([x['action_mask'] for i in range(4) for x in obs[i]]).astype(np.int8),
        'act': np.array([x for i in range(4) for x in actions[i]]),
    }
    return match_samples, payload


def _clear_match_buffers():
    for x in obs:
        x.clear()
    for x in actions:
        x.clear()


def _has_risky_fan(fan_description):
    return any(name in fan_description for name in RISKY_FAN_NAMES)


def _flush_chunk(split_name):
    sample_count = chunk_sample_counts[split_name]
    if sample_count == 0:
        return
    chunk_id = split_output_ids[split_name]
    filename = 'chunk_%05d.npz' % chunk_id
    payload = {
        key: np.concatenate(chunk_buffers[split_name][key], axis = 0)
        for key in chunk_buffers[split_name]
    }
    _save_npz(Path(OUTPUT_DIR) / split_name, filename[:-4], **payload)
    split_manifests[split_name]['chunks'].append({'file': filename, 'samples': int(sample_count)})
    split_manifests[split_name]['total_samples'] += int(sample_count)
    split_output_ids[split_name] = chunk_id + 1
    chunk_sample_counts[split_name] = 0
    for values in chunk_buffers[split_name].values():
        values.clear()


def _append_payload_to_chunk(split_name, payload):
    sample_count = len(payload['act'])
    if sample_count == 0:
        return
    for key in chunk_buffers[split_name]:
        chunk_buffers[split_name][key].append(payload[key])
    chunk_sample_counts[split_name] += sample_count
    if chunk_sample_counts[split_name] >= CHUNK_SAMPLE_LIMIT:
        _flush_chunk(split_name)


def _merge_payloads(first, second):
    return {
        'obs': np.concatenate([first['obs'], second['obs']], axis = 0),
        'glob': np.concatenate([first['glob'], second['glob']], axis = 0),
        'mask': np.concatenate([first['mask'], second['mask']], axis = 0),
        'act': np.concatenate([first['act'], second['act']], axis = 0),
    }


_apply_args(_parse_args())
_prepare_output_dirs()
f = _open_input_file(INPUT_FILE)
try:
    fan_description = ''
    total_matches = sum(1 for line in f if line.startswith('Match '))
    split_matchid = int(total_matches * SPLIT_RATIO)
    f.seek(0)
    line = f.readline()
    while line:
        t = line.split()
        if len(t) == 0:
            line = f.readline()
            continue
        if t[0] == 'Match':
            agents = [FeatureAgent(i) for i in range(4)]
            matchid += 1
            fan_description = ''
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
        elif t[0] == 'Fan':
            fan_description = line
        elif t[0] == 'Score':
            was_augmented = False
            filterData()
            _, payload = _materialize_match()
            risky_fan = _has_risky_fan(fan_description)
            split_name = 'train' if matchid < split_matchid else 'valid'
            if split_name == 'train' and AUGMENT_SAFE_MATCHES and not risky_fan:
                augmented = []
                for _ in range(AUGMENT_COPIES):
                    variant = sample_variant(rng, honor_weight = 1.0)
                    augmented.append(transform_batch(payload, variant))
                for augmented_payload in augmented:
                    payload = _merge_payloads(payload, augmented_payload)
                was_augmented = True
                augmented_payloads += len(augmented)
            if risky_fan:
                risky_matches += 1
            if was_augmented:
                augmented_matches += 1
            _append_payload_to_chunk(split_name, payload)
            _clear_match_buffers()
        line = f.readline()
finally:
    f.close()

_flush_chunk('train')
_flush_chunk('valid')
_save_json(Path(OUTPUT_DIR) / 'train', 'count', split_manifests['train'])
_save_json(Path(OUTPUT_DIR) / 'valid', 'count', split_manifests['valid'])
print('Saved %d train chunk npz files and %d valid chunk npz files.' % (split_output_ids['train'], split_output_ids['valid']))
print('Saved %d train samples and %d valid samples.' % (split_manifests['train']['total_samples'], split_manifests['valid']['total_samples']))
print('Augmented %d train matches with %d extra transformed copies; kept %d risky-fan matches unaugmented.' % (augmented_matches, augmented_payloads, risky_matches))
