import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase

class AlphaMatrixGameEnv(EnvBase):
    """
               B1                        B2
    A1    (1, 1-2a)              (1-2a, (a+1)/2)
    A2    ((1-3a)/2, (7-3a)/4)   (a, 2-3a)
 
    alpha=0: pure NE (A1,B1). alpha=1: matching pennies (zero-sum).
    """

    def __init__(self, alpha=0.0, batch_size=1000, device=None):
        super().__init__(device=device, batch_size=torch.Size([batch_size]))
        self.num_agents = 2
        self.set_alpha(alpha)

        self.observation_spec = Composite(
            observation=Unbounded(shape=(*self.batch_size, self.num_agents, 1)),
            shape=self.batch_size,
        )

        self.action_spec = Composite(
            action=Bounded(
                low=0.0, high=1.0,
                shape=(*self.batch_size, self.num_agents, 1),
                dtype=torch.float32,
            ),
            shape=self.batch_size,
        )

        self.reward_spec = Composite(
            reward=Unbounded(shape=(*self.batch_size, self.num_agents, 1)),
            shape=self.batch_size,
        )

        self.done_spec = Composite(
            done=Bounded(low=0, high=1, shape=(*self.batch_size, 1), dtype=torch.bool),
            terminated=Bounded(low=0, high=1, shape=(*self.batch_size, 1), dtype=torch.bool),
            shape=self.batch_size,
        )

    def set_alpha(self, alpha):

        a = float(alpha)
        self.alpha = a
        self.payoff_A = torch.tensor(
            [[1.0, 1.0 - 2.0 * a], [(1.0 - 3.0 * a) / 2.0, a]], dtype=torch.float32
        )
        self.payoff_B = torch.tensor(
            [[1.0 - 2.0 * a, (a + 1.0) / 2.0], [(7.0 - 3.0 * a) / 4.0, 2.0 - 3.0 * a]], dtype=torch.float32
        )

    def _reset(self, tensordict=None, **kwargs):

        B = self.batch_size
        return TensorDict(
            {
                "observation": torch.ones(*B, self.num_agents, 1, device=self.device),
                "done": torch.zeros(*B, dtype=torch.bool, device=self.device),
                "terminated": torch.zeros(*B, dtype=torch.bool, device=self.device),
            },
            batch_size=B,
            device=self.device,
        )

    def _step(self, tensordict):

        action = tensordict["action"]
        p = action[..., 0, 0].clamp(0.0, 1.0)
        q = action[..., 1, 0].clamp(0.0, 1.0)

        pA = torch.stack([1.0 - p, p], dim=-1)
        pB = torch.stack([1.0 - q, q], dim=-1)

        payoff_A = self.payoff_A.to(action.device)
        payoff_B = self.payoff_B.to(action.device)

        r_A = torch.einsum("...i,ij,...j->...", pA, payoff_A, pB)
        r_B = torch.einsum("...i,ij,...j->...", pA, payoff_B, pB)

        reward = torch.stack([r_A, r_B], dim=-1).unsqueeze(-1)

        B = self.batch_size
        return TensorDict(
            {
                "observation": torch.ones(*B, self.num_agents, 1, device=self.device),
                "reward": reward,
                "done": torch.ones(*B, 1, dtype=torch.bool, device=self.device),
                "terminated": torch.ones(*B, 1, dtype=torch.bool, device=self.device),
            },
            batch_size=B,
            device=self.device,
        )

    def _set_seed(self, seed):
        if seed is not None:
            torch.manual_seed(seed)

