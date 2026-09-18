from dataclasses import dataclass

import torch as th

from algos.npgac.losses import (
    AdaptiveKMax,
    actor_losses,
    closeness_and_trsut,
    phi_naive_loss,
    phi_td_loss,
    q_regression_loss,
)
from algos.npgac.networks import JointQHead, Phi, make_actor

@dataclass
class NPGACConfig:
    num_agents: int = 2
    lr_actor: float = 1e-4
    lr_q: float = 1e-3
    lr_phi: float = 1e-3

    beta: float = 5.0
    k_max_fraction: float = 0.5
    k_max_ema_decay: float = 0.99

    entropy_start: float = 0.03
    entropy_end: float = 0.003
    entropy_anneal_steps: int = 1000

    actor_max_grad_norm: float = 10.0
    critic_max_grad_norm: float = 10.0
    phi_max_grad_norm: float = 10.0
    hidden_size: int = 32
    batch_size: int = 512

    def entropy_coef(self, step):
        """Linear anneal from entropy_start to entropy_end over
        entropy_anneal_steps, then held at entropy_end.
        """
        if step >= self.entropy_anneal_steps:
            return self.entropy_end
        frac = step / max(self.entropy_anneal_steps, 1)
        return self.entropy_start + frac * (self.entropy_end - self.entropy_start)

class NPGACTrainer:
    def __init__(self, env, config:NPGACConfig = None, device=None):
        self.env = env
        self.cfg = config or NPGACConfig()
        self.device = device or ("cuda" if th.cuda.is_available() else "cpu")
        N = self.cfg.num_agents

        self.actors = [
            make_actor(i, hidden_size=self.cfg.hidden_size, device=self.device)
            for i in range(N)
        ]

        self.q_heads = [
            JointQHead(num_agents=N, hidden_size=self.cfg.hidden_size).to(self.device)
            for _ in range(N)
        ]

        self.phi = Phi(num_agents=N, hidden_size=self.cfg.hidden_size).to(self.device)

        self.actor_optims = [
            th.optim.Adam(a.parameters(), lr=self.cfg.lr_actor) for a in self.actors
        ]

        self.q_optims = [
            th.optim.Adam(q.parameters(), lr=self.cfg.lr_q) for q in self.q_heads
        ]

        self.phi_optim = th.optim.Adam(self.phi.parameters(), lr=self.cfg.lr_phi)

        self.k_max_tracker = AdaptiveKMax(
            k_max_fraction=self.cfg.k_max_fraction,
            ema_decay=self.cfg.k_max_ema_decay,
        )

        self.num_updates = 0

    def collect(self):
        td = self.env.reset()

        for i in range(self.cfg.num_agents):
            td["agents", str(i), "observation"] = td["observation"][:, i, :]

        for actor in self.actors:
            td = actor(td)
        action = th.stack(
            [td["agents", str(i), "action"] for i in range(self.cfg.num_agents)],
            dim=1
        )
        td["action"] = action
        td = self.env.step(td)
        return td

    def train_step(self):
        td = self.collect()
        N = self.cfg.num_agents
        log = {}

        # Q_i update
        q_loss, per_agent_q_loss = q_regression_loss(self.q_heads, td, N)
        for i in range(N):
            self.q_optims[i].zero_grad()
            per_agent_q_loss[i].backward(retain_graph=True)
            grad_norm = th.nn.utils.clip_grad_norm_(
                self.q_heads[i].parameters(), self.cfg.critic_max_grad_norm
            )
            self.q_optims[i].step()
            log[f"q/{i}/loss"] = per_agent_q_loss[i].item()
            log[f"q/{i}/grad_norm"] = grad_norm.item()
        log[f"q/loss_mean"] = q_loss.item()

        # Phi update
        td_loss = phi_td_loss(self.phi, td, N)
        naive_loss, phi_residuals = phi_naive_loss(
            self.phi, self.q_heads, self.actors, td, N
        )
        phi_loss = td_loss + self.cfg.beta * naive_loss

        self.phi_optim.zero_grad()
        phi_loss.backward(retain_graph=True)
        phi_grad_norm = th.nn.utils.clip_grad_norm_(
            self.phi.parameters(), self.cfg.phi_max_grad_norm
        )
        self.phi_optim.step()

        log["phi/td_loss"] = td_loss.item()
        log["phi/naive_loss"] = naive_loss.item()
        log["phi/loss_total"] = phi_loss.item()
        log["phi/grad_norm"] = phi_grad_norm.item()

        # Actor update
        with th.no_grad():
            _, fresh_residuals = phi_naive_loss(
                self.phi, self.q_heads, self.actors, td, N
            )

        k_hat_now, _ = closeness_and_trsut(fresh_residuals, self.k_max_tracker.k_max)
        self.k_max_tracker.update(k_hat_now)

        current_entropy_coef = self.cfg.entropy_coef(self.num_updates)
        actor_out = actor_losses(
            self.actors, self.q_heads, self.phi, td, N, self.k_max_tracker.k_max,
            phi_residuals=fresh_residuals, entropy_coef=current_entropy_coef,
        )

        for i in range(N):
            self.actor_optims[i].zero_grad()
            actor_out.per_agent_loss[i].backward(retain_graph=(i < N - 1))
            grad_norm = th.nn.utils.clip_grad_norm_(
                self.actors[i].parameters(), self.cfg.actor_max_grad_norm
            )
            self.actor_optims[i].step()
            log[f"actor/{i}/loss"] = actor_out.per_agent_loss[i].item()
            log[f"actor/{i}/grad_norm"] = grad_norm.item()
            log[f"actor/{i}/lambda"] = actor_out.lam[i].item()
            log[f"actor/{i}/k_hat"] = actor_out.k_hat[i].item()
            log[f"actor/{i}/entropy"] = actor_out.entropy[i].item()

        log["train/k_max"] = self.k_max_tracker.k_max
        log["train/entropy_coef"] = current_entropy_coef

        with th.no_grad():
            action = td["action"].squeeze(-1)
            names = ["player_A", "player_B"] if N == 2 else [f"player_{i}" for i in range(N)]
            for i, name in enumerate(names):
                log[f"strategies/{name}/mean"] = action[:, i].mean().item()
                log[f"strategies/{name}/std"] = action[:, i].std().item()

        self.num_updates += 1
        return log

    def current_strategy(self, num_samples=256):

        with th.no_grad():
            td = self.env.reset()
            for i in range(self.cfg.num_agents):
                td["agents", str(i), "observation"] = td["observation"][:, i, :]

            for actor in self.actors:
                td = actor(td)
            action = th.stack(
                [td["agents", str(i), "action"] for i in range(self.cfg.num_agents)],
                dim=1,
            ).squeeze(-1)
        return action.mean(dim=0).tolist()


