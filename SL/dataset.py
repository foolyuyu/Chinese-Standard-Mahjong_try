from torch.utils.data import Dataset
from bisect import bisect_right
from pathlib import Path
from contextlib import contextmanager

import numpy as np


def _resolve_local_path(path, base_dir):
    path = Path(path)
    return path if path.is_absolute() else Path(base_dir) / path


def load_json(path, base_dir = None):
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    import json
    with open(_resolve_local_path(path, base_dir)) as f:
        return json.load(f)


@contextmanager
def load_npz(path, base_dir = None):
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    with np.load(_resolve_local_path(path, base_dir)) as d:
        yield d


def resolve_data_path(data_dir, relative_name):
    return str(_resolve_local_path(Path(data_dir) / relative_name, Path(__file__).resolve().parent))


def load_manifest(data_dir):
    return load_json(resolve_data_path(data_dir, 'count.json'))


def load_shard_arrays(data_dir, shard_file):
    with load_npz(resolve_data_path(data_dir, shard_file)) as d:
        return {key: d[key] for key in d.files}


class MahjongGBDataset(Dataset):

    OBS_CHANNELS = 148
    GLOBAL_SIZE = 10

    def __init__(self, begin = 0, end = 1, augment = False, data_dir = None):
        import json
        base_dir = Path(__file__).resolve().parent
        self.data_dir = str(data_dir) if data_dir is not None else str(base_dir / 'data')
        self.augment = augment

        with open(Path(self.data_dir) / 'count.json') as f:
            self.match_samples = json.load(f)

        self.total_matches = len(self.match_samples)
        self.total_samples = sum(self.match_samples)
        self.begin = int(begin * self.total_matches)
        self.end = int(end * self.total_matches)
        self.match_samples = self.match_samples[self.begin : self.end]
        self.matches = len(self.match_samples)
        self.samples = sum(self.match_samples)

        t = 0
        for i in range(self.matches):
            a = self.match_samples[i]
            self.match_samples[i] = t
            t += a

        self.cache = {'obs': [], 'glob': [], 'mask': [], 'act': []}
        for i in range(self.matches):
            if i % 128 == 0:
                print('loading', i)

            match_id = i + self.begin
            with np.load(Path(self.data_dir) / f'{match_id}.npz') as d:
                self._append_match(d)

    def _append_match(self, d):
        obs = d['obs']
        if obs.shape[1] != self.OBS_CHANNELS:
            padded = np.zeros((obs.shape[0], self.OBS_CHANNELS, obs.shape[2], obs.shape[3]), dtype = obs.dtype)
            padded[:, :obs.shape[1]] = obs
            obs = padded
        self.cache['obs'].append(obs)
        if 'glob' in d:
            self.cache['glob'].append(d['glob'])
        else:
            self.cache['glob'].append(np.zeros((obs.shape[0], self.GLOBAL_SIZE), dtype = np.int8))
        self.cache['mask'].append(d['mask'])
        self.cache['act'].append(d['act'])

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        match_id = bisect_right(self.match_samples, index, 0, self.matches) - 1
        sample_id = index - self.match_samples[match_id]
        return self.cache['obs'][match_id][sample_id], self.cache['glob'][match_id][sample_id], self.cache['mask'][match_id][sample_id], self.cache['act'][match_id][sample_id]
