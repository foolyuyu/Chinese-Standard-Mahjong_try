from feature import FeatureAgent
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from augmentation import sample_variant, transform_batch


INPUT_FILE = 'data/data.txt'
OUTPUT_DIR = 'data'
SPLIT_RATIO = 0.95
AUGMENT_SAFE_MATCHES = True
AUGMENT_SEED = 20240618
CHUNK_SAMPLE_LIMIT = 500000
BATCH_MATCHES = 32
RISKY_FAN_NAMES = {
    '绿一色',
    '推不倒',
}

base_dir = Path(__file__).resolve().parent
split_manifests = {}
split_output_ids = {}
chunk_buffers = {}
chunk_sample_counts = {}


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


def _prepare_output_dirs(output_dir):
    for split_name in ['train', 'valid']:
        split_dir = _resolve_local_path(Path(output_dir) / split_name)
        split_dir.mkdir(parents = True, exist_ok = True)
        for old_npz in split_dir.glob('*.npz'):
            old_npz.unlink()
        count_path = split_dir / 'count.json'
        if count_path.exists():
            count_path.unlink()


def _init_output_state(chunk_sample_limit):
    global split_manifests
    global split_output_ids
    global chunk_buffers
    global chunk_sample_counts
    split_manifests = {
        'train': {'schema': 'local-chunk-v2', 'chunk_sample_limit': chunk_sample_limit, 'total_samples': 0, 'chunks': []},
        'valid': {'schema': 'local-chunk-v2', 'chunk_sample_limit': chunk_sample_limit, 'total_samples': 0, 'chunks': []},
    }
    split_output_ids = {'train': 0, 'valid': 0}
    chunk_buffers = {
        'train': _empty_payload_lists(),
        'valid': _empty_payload_lists(),
    }
    chunk_sample_counts = {'train': 0, 'valid': 0}


def _empty_match_buffers():
    return [[] for _ in range(4)], [[] for _ in range(4)]


def _empty_payload_lists():
    return {'suit_obs': [], 'honor_obs': [], 'glob': [], 'mask': [], 'act': []}


def _filter_data(obs, actions):
    newobs = [[] for _ in range(4)]
    newactions = [[] for _ in range(4)]
    for i in range(4):
        for j, o in enumerate(obs[i]):
            if o['action_mask'].sum() > 1:  # ignore states with single valid action (Pass)
                newobs[i].append(o)
                newactions[i].append(actions[i][j])
    return newobs, newactions


def _materialize_match(obs, actions):
    assert [len(x) for x in obs] == [len(x) for x in actions], 'obs actions not matching!'
    samples = [x for i in range(4) for x in obs[i]]
    if not samples:
        return _empty_payload()
    return {
        'suit_obs': np.stack([x['suit_observation'] for x in samples]).astype(np.int8),
        'honor_obs': np.stack([x['honor_observation'] for x in samples]).astype(np.int8),
        'glob': np.stack([x['global'] for x in samples]).astype(np.float32),
        'mask': np.stack([x['action_mask'] for x in samples]).astype(np.int8),
        'act': np.array([x for i in range(4) for x in actions[i]], dtype = np.int64),
    }


def _empty_payload():
    return {
        'suit_obs': np.empty((0, 148, 3, 9), dtype = np.int8),
        'honor_obs': np.empty((0, 148, 7), dtype = np.int8),
        'glob': np.empty((0, FeatureAgent.GLOBAL_SIZE), dtype = np.float32),
        'mask': np.empty((0, FeatureAgent.ACT_SIZE), dtype = np.int8),
        'act': np.empty((0,), dtype = np.int64),
    }


def _has_risky_fan(fan_description):
    return any(name in fan_description for name in RISKY_FAN_NAMES)


def _payload_len(payload):
    return len(payload['act'])


def _append_payload_list(payload_lists, payload):
    if _payload_len(payload) == 0:
        return
    for key in payload_lists:
        payload_lists[key].append(payload[key])


def _concat_payload_list(payload_lists):
    if not payload_lists['act']:
        return _empty_payload()
    return {
        key: np.concatenate(payload_lists[key], axis = 0)
        for key in payload_lists
    }


def _flush_chunk(split_name, output_dir):
    sample_count = chunk_sample_counts[split_name]
    if sample_count == 0:
        return
    chunk_id = split_output_ids[split_name]
    filename = 'chunk_%05d.npz' % chunk_id
    payload = _concat_payload_list(chunk_buffers[split_name])
    _save_npz(Path(output_dir) / split_name, filename[:-4], **payload)
    split_manifests[split_name]['chunks'].append({'file': filename, 'samples': int(sample_count)})
    split_manifests[split_name]['total_samples'] += int(sample_count)
    split_output_ids[split_name] = chunk_id + 1
    chunk_sample_counts[split_name] = 0
    chunk_buffers[split_name] = _empty_payload_lists()


def _append_payload_to_chunk(split_name, payload, output_dir, chunk_sample_limit):
    sample_count = _payload_len(payload)
    if sample_count == 0:
        return
    _append_payload_list(chunk_buffers[split_name], payload)
    chunk_sample_counts[split_name] += sample_count
    if chunk_sample_counts[split_name] >= chunk_sample_limit:
        _flush_chunk(split_name, output_dir)


def _merge_payloads(first, second):
    return {
        'suit_obs': np.concatenate([first['suit_obs'], second['suit_obs']], axis = 0),
        'honor_obs': np.concatenate([first['honor_obs'], second['honor_obs']], axis = 0),
        'glob': np.concatenate([first['glob'], second['glob']], axis = 0),
        'mask': np.concatenate([first['mask'], second['mask']], axis = 0),
        'act': np.concatenate([first['act'], second['act']], axis = 0),
    }


def _process_match(matchid, lines, split_matchid, augment_safe_matches, augment_seed):
    obs, actions = _empty_match_buffers()
    agents = None
    curTile = None
    fan_description = ''

    for line in lines:
        t = line.split()
        if len(t) == 0:
            continue
        if t[0] == 'Match':
            agents = [FeatureAgent(i) for i in range(4)]
            fan_description = ''
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

    obs, actions = _filter_data(obs, actions)
    payload = _materialize_match(obs, actions)
    risky_fan = _has_risky_fan(fan_description)
    split_name = 'train' if matchid < split_matchid else 'valid'
    was_augmented = False
    if split_name == 'train' and augment_safe_matches and not risky_fan:
        rng = np.random.default_rng(seed = augment_seed + matchid)
        variant = sample_variant(rng, honor_weight = 1.0)
        payload = _merge_payloads(payload, transform_batch(payload, variant))
        was_augmented = True
    return split_name, payload, was_augmented, risky_fan


def _process_match_batch(args):
    match_batch, split_matchid, augment_safe_matches, augment_seed = args
    payload_lists = {'train': _empty_payload_lists(), 'valid': _empty_payload_lists()}
    augmented_matches = 0
    risky_matches = 0
    for matchid, lines in match_batch:
        split_name, payload, was_augmented, risky_fan = _process_match(
            matchid,
            lines,
            split_matchid,
            augment_safe_matches,
            augment_seed,
        )
        _append_payload_list(payload_lists[split_name], payload)
        augmented_matches += int(was_augmented)
        risky_matches += int(risky_fan)
    return {
        'train': _concat_payload_list(payload_lists['train']),
        'valid': _concat_payload_list(payload_lists['valid']),
        'augmented_matches': augmented_matches,
        'risky_matches': risky_matches,
        'processed_matches': len(match_batch),
    }


def _count_matches(input_file):
    with _open_input_file(input_file) as f:
        return sum(1 for line in f if line.startswith('Match '))


def _iter_match_batches(input_file, batch_matches):
    match_batch = []
    current_lines = []
    matchid = -1
    with _open_input_file(input_file) as f:
        for line in f:
            if line.startswith('Match '):
                if current_lines:
                    match_batch.append((matchid, current_lines))
                    if len(match_batch) >= batch_matches:
                        yield match_batch
                        match_batch = []
                matchid += 1
                current_lines = [line]
            else:
                current_lines.append(line)
        if current_lines:
            match_batch.append((matchid, current_lines))
        if match_batch:
            yield match_batch


def _consume_result(result, output_dir, chunk_sample_limit):
    for split_name in ['train', 'valid']:
        _append_payload_to_chunk(split_name, result[split_name], output_dir, chunk_sample_limit)


def _default_workers():
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count // 2 or 1))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', default = INPUT_FILE)
    parser.add_argument('--output-dir', default = OUTPUT_DIR)
    parser.add_argument('--split-ratio', type = float, default = SPLIT_RATIO)
    parser.add_argument('--num-workers', type = int, default = _default_workers())
    parser.add_argument('--batch-matches', type = int, default = BATCH_MATCHES)
    parser.add_argument('--chunk-sample-limit', type = int, default = CHUNK_SAMPLE_LIMIT)
    parser.add_argument('--augment-seed', type = int, default = AUGMENT_SEED)
    parser.add_argument('--no-augment', action = 'store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    _prepare_output_dirs(args.output_dir)
    _init_output_state(args.chunk_sample_limit)

    total_matches = _count_matches(args.input_file)
    split_matchid = int(total_matches * args.split_ratio)
    augment_safe_matches = not args.no_augment
    processed_matches = 0
    augmented_matches = 0
    risky_matches = 0
    worker_count = max(1, args.num_workers)

    print('Preprocess %d matches with %d worker(s), batch_matches=%d, split_ratio=%.4f.' % (
        total_matches,
        worker_count,
        args.batch_matches,
        args.split_ratio,
    ))

    batch_iter = (
        (match_batch, split_matchid, augment_safe_matches, args.augment_seed)
        for match_batch in _iter_match_batches(args.input_file, args.batch_matches)
    )
    if worker_count == 1:
        for work_item in batch_iter:
            result = _process_match_batch(work_item)
            _consume_result(result, args.output_dir, args.chunk_sample_limit)
            processed_matches += result['processed_matches']
            augmented_matches += result['augmented_matches']
            risky_matches += result['risky_matches']
            if processed_matches % 128 == 0 or processed_matches == total_matches:
                print('Processed %d/%d matches...' % (processed_matches, total_matches))
    else:
        with Pool(processes = worker_count) as pool:
            for result in pool.imap_unordered(_process_match_batch, batch_iter):
                _consume_result(result, args.output_dir, args.chunk_sample_limit)
                processed_matches += result['processed_matches']
                augmented_matches += result['augmented_matches']
                risky_matches += result['risky_matches']
                if processed_matches % 128 == 0 or processed_matches == total_matches:
                    print('Processed %d/%d matches...' % (processed_matches, total_matches))

    _flush_chunk('train', args.output_dir)
    _flush_chunk('valid', args.output_dir)
    _save_json(Path(args.output_dir) / 'train', 'count', split_manifests['train'])
    _save_json(Path(args.output_dir) / 'valid', 'count', split_manifests['valid'])
    print('Saved %d train chunk npz files and %d valid chunk npz files.' % (split_output_ids['train'], split_output_ids['valid']))
    print('Saved %d train samples and %d valid samples.' % (split_manifests['train']['total_samples'], split_manifests['valid']['total_samples']))
    print('Augmented %d train matches; kept %d risky-fan matches unaugmented.' % (augmented_matches, risky_matches))


if __name__ == '__main__':
    main()
