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


class MLPResidualBlock(nn.Module):

    def __init__(self, size):
        nn.Module.__init__(self)
        self._block = nn.Sequential(
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, size)
        )
        self._relu = nn.ReLU()

    def forward(self, x):
        return self._relu(x + self._block(x))


class CNNModel(nn.Module):

    OBS_CHANNELS = 148
    GLOBAL_SIZE = 23
    RES_CHANNELS = 128
    RES_BLOCKS = 6
    HONOR_SIZE = 256
    HONOR_BLOCKS = 2
    HEAD_SIZE = 512

    def __init__(self):
        nn.Module.__init__(self)
        self._suit_tower = nn.Sequential(
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
        self._honor_tower = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.OBS_CHANNELS * 7, self.HONOR_SIZE),
            nn.ReLU(),
            *[MLPResidualBlock(self.HONOR_SIZE) for _ in range(self.HONOR_BLOCKS)]
        )
        self._global_tower = nn.Sequential(
            nn.Linear(self.GLOBAL_SIZE, 64),
            nn.ReLU()
        )
        self._tower_head = nn.Sequential(
            nn.Linear(self.RES_CHANNELS * 3 * 9 + self.HONOR_SIZE + 64, self.HEAD_SIZE),
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

    def _split_obs(self, input_dict):
        obs_dict = input_dict["obs"]
        suit_obs = obs_dict.get("suit_observation")
        honor_obs = obs_dict.get("honor_observation")
        if suit_obs is not None and honor_obs is not None:
            return self._pad_suit_obs(suit_obs.float()), self._pad_honor_obs(honor_obs.float())
        obs = obs_dict["observation"].float()
        obs = self._pad_legacy_obs(obs)
        suit_obs = obs[:, :, :3, :]
        honor_obs = obs[:, :, 3, :7]
        return suit_obs, honor_obs

    def _pad_suit_obs(self, suit_obs):
        if suit_obs.size(1) < self.OBS_CHANNELS:
            pad = suit_obs.new_zeros((suit_obs.size(0), self.OBS_CHANNELS - suit_obs.size(1), suit_obs.size(2), suit_obs.size(3)))
            suit_obs = torch.cat([suit_obs, pad], dim = 1)
        return suit_obs[:, :self.OBS_CHANNELS, :3, :9]

    def _pad_honor_obs(self, honor_obs):
        if honor_obs.size(1) < self.OBS_CHANNELS:
            pad = honor_obs.new_zeros((honor_obs.size(0), self.OBS_CHANNELS - honor_obs.size(1), honor_obs.size(2)))
            honor_obs = torch.cat([honor_obs, pad], dim = 1)
        return honor_obs[:, :self.OBS_CHANNELS, :7]

    def _pad_legacy_obs(self, obs):
        if obs.size(1) < self.OBS_CHANNELS:
            pad = obs.new_zeros((obs.size(0), self.OBS_CHANNELS - obs.size(1), obs.size(2), obs.size(3)))
            obs = torch.cat([obs, pad], dim = 1)
        return obs[:, :self.OBS_CHANNELS, :4, :9]

    def _global_tensor(self, input_dict, reference):
        glob = input_dict["obs"].get("global")
        if glob is None:
            return reference.new_zeros((reference.size(0), self.GLOBAL_SIZE))
        if glob.size(1) == self.GLOBAL_SIZE:
            return glob.float()
        if glob.size(1) > self.GLOBAL_SIZE:
            return glob[:, :self.GLOBAL_SIZE].float()
        pad = glob.new_zeros((glob.size(0), self.GLOBAL_SIZE - glob.size(1)))
        return torch.cat([glob, pad], dim = 1).float()

    def forward(self, input_dict):
        self.train(mode = input_dict.get("is_training", False))
        suit_obs, honor_obs = self._split_obs(input_dict)
        suit_hidden = self._suit_tower(suit_obs)
        honor_hidden = self._honor_tower(honor_obs)
        glob = self._global_tensor(input_dict, suit_obs)
        hidden = torch.cat([suit_hidden, honor_hidden, self._global_tower(glob)], dim = 1)
        action_logits = self._tower_head(hidden)
        action_mask = input_dict["obs"]["action_mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        return action_logits + inf_mask
