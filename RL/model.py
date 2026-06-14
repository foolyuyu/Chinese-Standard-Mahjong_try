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
        self._logits = nn.Sequential(
            nn.Linear(64 * 4 * 9 + 32, 256),
            nn.ReLU(True),
            nn.Linear(256, 235)
        )
        self._value_branch = nn.Sequential(
            nn.Linear(64 * 4 * 9 + 32, 256),
            nn.ReLU(True),
            nn.Linear(256, 1)
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
        glob = input_dict.get("global")
        if glob is None:
            return obs.new_zeros((obs.size(0), self.GLOBAL_SIZE))
        return glob.float()

    def forward(self, input_dict):
        obs = input_dict["observation"].float()
        obs = self._pad_obs(obs)
        hidden = self._tower(obs)
        glob = self._global_tensor(input_dict, obs)
        glob_hidden = self._global_tower(glob)
        hidden = torch.cat([hidden, glob_hidden], dim = 1)
        logits = self._logits(hidden)
        mask = input_dict["action_mask"].float()
        inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
        masked_logits = logits + inf_mask
        value_hidden = self._value_branch[0](hidden)
        value_hidden = self._value_branch[1](value_hidden)
        try:
            value = self._value_branch[2](value_hidden)
        except RuntimeError as e:
            # Work around a known CPU matmul backend issue on some aarch64 builds
            # when Linear outputs one channel.
            if value_hidden.device.type == 'cpu' and 'primitive descriptor' in str(e):
                w = self._value_branch[2].weight
                b = self._value_branch[2].bias
                value = torch.sum(value_hidden * w, dim = 1, keepdim = True)
                if b is not None:
                    value = value + b.view(1, 1)
            else:
                raise
        return masked_logits, value
