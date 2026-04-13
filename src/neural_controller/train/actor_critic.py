import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.mean(x)) * 2.0
        std = torch.exp(self.log_std.clamp(-20, 2))
        return mean, std

class Critic(nn.Module):
    def __init__(self, global_state_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(global_state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.value(x)


def try_load_actor_checkpoint(actor: Actor, path: str, map_location=None) -> bool:
    """Load actor weights only if the checkpoint matches the current observation size."""
    if not os.path.isfile(path):
        return False
    state = torch.load(path, map_location=map_location)
    w = state.get("fc1.weight")
    if w is None:
        print(f"Skip load {path}: checkpoint has no fc1.weight")
        return False
    if w.shape[1] != actor.fc1.in_features:
        print(
            f"Skip load {path}: checkpoint obs_dim={w.shape[1]}, "
            f"current env obs_dim={actor.fc1.in_features}. "
            "Align config (e.g. num_dynamic_obs) with the run that produced the checkpoint, "
            "or remove the old file and retrain."
        )
        return False
    actor.load_state_dict(state, strict=True)
    print(f"Loaded existing model from {path}")
    return True