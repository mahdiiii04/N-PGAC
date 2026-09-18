import torch as th
from tensordict.nn import TensorDictModule
from torch import nn
from torch.distributions import Beta
from torchrl.modules import ProbabilisticActor

from algos.npgac.networks import BetaParams


def make_actor(agent_idx, hidden_size=32, device=None):

    module = TensorDictModule(
        BetaParams(hidden_size=hidden_size),
        in_keys=["observation"],
        out_keys=["concentration1", "concentration0"],
    )

    actor = ProbabilisticActor(
        module=module,
        in_keys={"concentration1": "concentration1",
                 "concentration0": "concentration0"},
        out_keys=["action"],
        distribution_class=Beta,
        return_log_prob=True,
        log_prob_key="sample_log_prob",
        default_interaction_type="random",
    )

    if device is not None:
        actor = actor.to(device)
    return actor

class CentralizedValueNet(nn.Module):
    def __init__(self, num_agents=2, hidden_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_agents, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state):
        return self.net(state)