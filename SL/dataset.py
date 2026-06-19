from torch.utils.data import Dataset
from pathlib import Path

import numpy as np


class MahjongGBDataset(Dataset):

    OBS_CHANNELS = 148
    GLOBAL_SIZE = 23

    def __init__(self, begin = 0, end = 1, data_dir = None):
        import json
        base_dir = Path(__file__).resolve().parent
        self.data_dir = str(data_dir) if data_dir is not None else str(base_dir / 'data')

        with open(Path(self.data_dir) / 'count.json') as f:
            manifest = json.load(f)

        if isinstance(manifest, dict):
            entries = manifest.get('chunks', [])
            self.files = [entry['file'] for entry in entries]
            self.match_samples = [int(entry['samples']) for entry in entries]
        else:
            self.files = ['%d.npz' % i for i in range(len(manifest))]
            self.match_samples = manifest

        self.total_matches = len(self.match_samples)
        self.total_samples = sum(self.match_samples)
        self.begin = int(begin * self.total_matches)
        self.end = int(end * self.total_matches)
        self.files = self.files[self.begin : self.end]
        self.match_samples = self.match_samples[self.begin : self.end]
        self.matches = len(self.match_samples)
        self.samples = sum(self.match_samples)

        self.cache = {'suit_obs': [], 'honor_obs': [], 'glob': [], 'mask': [], 'act': []}
        for i in range(self.matches):
            if i % 128 == 0:
                print('loading', i)

            with np.load(Path(self.data_dir) / self.files[i]) as d:
                self._append_match(d)

        for key in self.cache:
            self.cache[key] = np.concatenate(self.cache[key], axis = 0)
        self.samples = len(self.cache['act'])

    def _append_match(self, d):
        if 'suit_obs' in d and 'honor_obs' in d:
            suit_obs = d['suit_obs']
            honor_obs = d['honor_obs']
        else:
            obs = d['obs'].reshape(d['obs'].shape[0], d['obs'].shape[1], 36)
            suit_obs = obs[:, :, :27].reshape((obs.shape[0], obs.shape[1], 3, 9))
            honor_obs = obs[:, :, 27:34]
        if suit_obs.shape[1] != self.OBS_CHANNELS:
            padded = np.zeros((suit_obs.shape[0], self.OBS_CHANNELS, suit_obs.shape[2], suit_obs.shape[3]), dtype = suit_obs.dtype)
            padded[:, :suit_obs.shape[1]] = suit_obs
            suit_obs = padded
        if honor_obs.shape[1] != self.OBS_CHANNELS:
            padded = np.zeros((honor_obs.shape[0], self.OBS_CHANNELS, honor_obs.shape[2]), dtype = honor_obs.dtype)
            padded[:, :honor_obs.shape[1]] = honor_obs
            honor_obs = padded
        self.cache['suit_obs'].append(suit_obs[:, :self.OBS_CHANNELS, :3, :9])
        self.cache['honor_obs'].append(honor_obs[:, :self.OBS_CHANNELS, :7])
        if 'glob' in d:
            self.cache['glob'].append(d['glob'])
        else:
            self.cache['glob'].append(np.zeros((suit_obs.shape[0], self.GLOBAL_SIZE), dtype = np.float32))
        self.cache['mask'].append(d['mask'])
        self.cache['act'].append(d['act'])

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        sample = {
            'suit_obs': self.cache['suit_obs'][index],
            'honor_obs': self.cache['honor_obs'][index],
            'glob': self.cache['glob'][index],
            'mask': self.cache['mask'][index],
            'act': self.cache['act'][index],
        }
        return sample['suit_obs'], sample['honor_obs'], sample['glob'], sample['mask'], sample['act']
