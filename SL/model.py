# Model part
import torch
from torch import nn


class CNNModel(nn.Module):

    OBS_CHANNELS = 148
    GLOBAL_SIZE = 10

    def __init__(self):
        nn.Module.__init__(self)
        self._tower = nn.Sequential(
            nn.Conv2d(self.OBS_CHANNELS, 128, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Flatten()
        )
        self._global_tower = nn.Sequential(
            nn.Linear(self.GLOBAL_SIZE, 32),
            nn.ReLU(True)
        )
        self._tower_head = nn.Sequential(
            nn.Linear(64 * 4 * 9 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 235)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

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
