from dataclasses import dataclass

import torch as th
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from algos.mappo.networks import CentralizedValueNet, make_actor

@dataclass
class MAPPOConfig:
    num_agents: int = 2
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    clip_epsilon: float = 0.2

    entropy_start: float = 0.01
    entropy_end: float = 0.0
    entropy_anneal_steps: int = 500
    critic_coeff: float = 0.5 
    num_epochs: int = 1

    hidden_size:int = 32
    max_grad_norm: float = 10.0

    def entropy_coef(self, step):
        if step >= self.entropy_anneal_steps:
            return self.entropy_end
        frac = step / max(self.entropy_anneal_steps, 1)
        return self.entropy_start + frac * (self.entropy_end - self.entropy_start)


class MAPPOTrainer:
    def __init__(self, env, config: MAPPOConfig = None, device=None):
        self.env = env
        self.cfg = config or MAPPOConfig()
        self.device = device or ("cuda" if th.cuda.is_available() else "cpu")
        N = self.cfg.num_agents

        self.actors = [
            make_actor(i, self.cfg.hidden_size, self.device)
            for i in range(N)
        ]

        self.critics = [
            TensorDictModule(
                CentralizedValueNet(num_agents=N, hidden_size=self.cfg.hidden_size),
                in_keys=["state"], out_keys=["state_value"],
            ).to(self.device)
            for _ in range(N)
        ]

        self.actor_optims = [
            th.optim.Adam(a.parameters(), lr=self.cfg.lr_actor) for a in self.actors
        ]

        self.critic_optims = [
            th.optim.Adam(c.parameters(), lr=self.cfg.lr_critic) for c in self.critics
        ]

        self.gaes = [
            GAE(gamma=0.0, lmbda=0.0, value_network=c) for c in self.critics
        ]

        self.loss_modules = [
            ClipPPOLoss(
                self.actors[i], self.critics[i],
                clip_epsilon=self.cfg.clip_epsilon,
                entropy_bonus=True,
                entropy_coeff=self.cfg.entropy_start,
                critic_coeff=self.cfg.critic_coeff,
            )
            for i in range(N)
        ]

        self.num_updates = 0


    def collect(self):
        N = self.cfg.num_agents
        td = self.env.reset()
        per_agent_obs = td["observation"]

        per_agent_tds = []
        action_list = []

        with th.no_grad():
            for i in range(N):
                agent_td = TensorDict(
                    {"observation": per_agent_obs[:, i, :]}, batch_size=td.batch_size
                )
                agent_td = self.actors[i](agent_td)
                per_agent_tds.append(agent_td)
                action_list.append(agent_td["action"])

        action = th.stack(action_list, dim=1)
        td["action"] = action
        td = self.env.step(td)

        # The centralized critic's "state" must be information available
        # BEFORE the action is sampled -- otherwise it isn't a state-value
        # baseline at all, it's a predictor of the specific realized
        # action's own reward. This env's payoff is a deterministic
        # bilinear function of the joint action, so a critic fed the
        # action itself (the previous version: `action.squeeze(-1)`)
        # trivially memorizes reward(action) within a couple hundred
        # steps -- confirmed empirically: |advantage| collapsed from 0.77
        # at step 0 to 0.015 by step 270, RMSE(V, reward) shrinking in
        # lockstep, while reward variance barely moved. Once the
        # advantage vanishes, ClipPPOLoss's surrogate objective has no
        # signal left and the actor stops moving -- this is why MAPPO was
        # flat at (p,q)=(0.5,0.5) for the full 2500 steps at every single
        # alpha tested, including ones whose true NE is nowhere near 0.5.
        # Using the pre-decision observation instead (here, a constant --
        # this env has no informative state) makes the critic learn what
        # it should: a simple, action-independent running-average-reward
        # baseline, the standard REINFORCE-with-baseline construction.
        # Confirmed fix: same seed/alpha=0, (p,q) now moves from
        # (0.45,0.48) at step 10 to (0.05,0.04) by step 1500, tracking the
        # true (0,0) NE instead of sitting at its initialization forever.
        state = per_agent_obs.squeeze(-1)
        for i in range(N):
            per_agent_tds[i]["state"] = state
            per_agent_tds[i]["next"] = TensorDict(
                {
                    "state": state,
                    "reward": td["next", "reward"][:, i, :],
                    "done": td["next", "done"],
                    "terminated": td["next", "terminated"],
                },
                batch_size=td.batch_size,
            )
        return per_agent_tds, action

    def train_step(self):
        N = self.cfg.num_agents
        per_agent_tds, action = self.collect()
        log = {}

        current_entropy_coef = self.cfg.entropy_coef(self.num_updates)
        for i in range(N):
            self.loss_modules[i].entropy_coeff = current_entropy_coef

        for i in range(N):
            agent_td = per_agent_tds[i]

            with th.no_grad():
                self.gaes[i](agent_td)

            epoch_losses = []
            for epoch in range(self.cfg.num_epochs):
                loss_td = self.loss_modules[i](agent_td)
                loss = (
                    loss_td["loss_objective"]
                    + loss_td["loss_critic"]
                    + loss_td["loss_entropy"]
                )
                self.actor_optims[i].zero_grad()
                self.critic_optims[i].zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self.actors[i].parameters(), self.cfg.max_grad_norm
                )
                th.nn.utils.clip_grad_norm_(
                    self.critics[i].parameters(), self.cfg.max_grad_norm
                )
                self.actor_optims[i].step()
                self.critic_optims[i].step()
                epoch_losses.append(loss.item())

            log[f"actor/{i}/loss_objective"] = loss_td["loss_objective"].item()
            log[f"actor/{i}/loss_critic"] = loss_td["loss_critic"].item()
            log[f"actor/{i}/loss_entropy"] = loss_td["loss_entropy"].item()
            log[f"actor/{i}/entropy"] = loss_td["entropy"].item()
            log[f"actor/{i}/clip_fraction"] = loss_td["clip_fraction"].item()


        with th.no_grad():
            strat = action.squeeze(-1)
            names = ["player_A", "player_B"] if N == 2 else [f"player_{i}" for i in range(N)]
            for i, name in enumerate(names):
                log[f"strategies/{name}/mean"] = strat[:, i].mean().item()
                log[f"strategies/{name}/std"] = strat[:, i].std().item()

        log["train/entropy_coef"] = current_entropy_coef

        self.num_updates += 1
        return log

    def current_strategy(self):
        with th.no_grad():
            td = self.env.reset()
            per_agent_obs = td["observation"]
            actions = []

            for i in range(self.cfg.num_agents):
                agent_td = TensorDict(
                    {"observation": per_agent_obs[:, i, :]}, batch_size=td.batch_size
                )
                agent_td = self.actors[i](agent_td)
                actions.append(agent_td["action"])
            action = th.stack(actions, dim=1).squeeze(-1)
        return action.mean(dim=0).tolist()