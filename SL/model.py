# Model part
import torch
from torch import nn


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        nn.Module.__init__(self)
        self._block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(channels)
        )
        self._relu = nn.ReLU()

    def forward(self, x):
        return self._relu(x + self._block(x))


class CNNModel(nn.Module):

    OBS_CHANNELS = 148
    GLOBAL_SIZE = 10
    RES_CHANNELS = 128
    RES_BLOCKS = 6
    HEAD_SIZE = 512

    def __init__(self):
        nn.Module.__init__(self)
        self._tower = nn.Sequential(
            nn.Conv2d(self.OBS_CHANNELS, 192, 3, 1, 1, bias = False),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, self.RES_CHANNELS, 3, 1, 1, bias = False),
            nn.BatchNorm2d(self.RES_CHANNELS),
            nn.ReLU(),
            *[ResidualBlock(self.RES_CHANNELS) for _ in range(self.RES_BLOCKS)],
            nn.ReLU(),
            nn.Flatten()
        )
        self._global_tower = nn.Sequential(
            nn.Linear(self.GLOBAL_SIZE, 64),
            nn.ReLU()
        )
        self._tower_head = nn.Sequential(
            nn.Linear(self.RES_CHANNELS * 4 * 9 + 64, self.HEAD_SIZE),
            nn.ReLU(),
            nn.Linear(self.HEAD_SIZE, 235)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _pad_obs(self, obs):
        if obs.size(1) == self.OBS_CHANNELS:
            return obs
        if obs.size(1) > self.OBS_CHANNELS:
            return obs[:, :self.OBS_CHANNELS]
        pad = obs.new_zeros((obs.size(0), self.OBS_CHANNELS - obs.size(1), obs.size(2), obs.size(3)))
        return torch.cat([obs, pad], dim = 1)

    def _global_tensor(self, input_dict, obs):
        glob = input_dict["obs"].get("global")
        if glob is None:
            return obs.new_zeros((obs.size(0), self.GLOBAL_SIZE))
        return glob.float()

    def forward(self, input_dict):
        self.train(mode = input_dict.get("is_training", False))
        obs = input_dict["obs"]["observation"].float()
        obs = self._pad_obs(obs)
        hidden = self._tower(obs)
        glob = self._global_tensor(input_dict, obs)
        hidden = torch.cat([hidden, self._global_tower(glob)], dim = 1)
        action_logits = self._tower_head(hidden)
        action_mask = input_dict["obs"]["action_mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        return action_logits + inf_mask
