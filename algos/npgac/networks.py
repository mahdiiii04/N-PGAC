import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torch.distributions import Beta
from torchrl.modules import ProbabilisticActor

class BetaParams(nn.Module):

    def __init__(self, hidden_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),
        )

        self.softplus = nn.Softplus()

    def forward(self, observation):
        raw =  self.net(observation)
        conc = self.softplus(raw) + 1.0
        return conc[...,0:1], conc[..., 1:2]

def make_actor(agent_idx, hidden_size=32, device=None):

    module = TensorDictModule(
        BetaParams(hidden_size=hidden_size),
        in_keys=[("agents", str(agent_idx), "observation")],
        out_keys=[
            ("agents", str(agent_idx), "concentration1"),
            ("agents", str(agent_idx), "concentration0"),
        ],
    )

    actor = ProbabilisticActor(
        module=module,
        in_keys={
            "concentration1" : ("agents", str(agent_idx), "concentration1"),
            "concentration0" : ("agents", str(agent_idx), "concentration0"),
        },
        out_keys=[("agents", str(agent_idx), "action")],
        distribution_class=Beta,
        return_log_prob=True,
        log_prob_key=("agents", str(agent_idx), "sample_log_prob"),
        default_interaction_type="random",
    )

    if device is not None:
        actor = actor.to(device)
    return actor

class JointQHead(nn.Module):

    def __init__(self, num_agents=2, hidden_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_agents, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, joint_strategy):
        return self.net(joint_strategy).squeeze(-1)

class Phi(nn.Module):

    def __init__(self, num_agents=2, hidden_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_agents, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, joint_strategy):
        return self.net(joint_strategy).squeeze(-1)

def replace_agent_coord(joint_strategy, agent_idx, new_value):
    out = joint_strategy.clone()
    out[:, agent_idx] = new_value
    return out
    