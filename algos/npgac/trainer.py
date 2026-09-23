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

    beta: float = 1.0

    k_max_fraction: float = 1.5
    # ^ was 0.5. AdaptiveKMax used to gate k_hat against a fraction of its
    # OWN live, pooled-across-agents EMA -- once k_hat stabilized at any
    # level v, that EMA -> v, so k_max -> k_max_fraction * v. With
    # k_max_fraction < 1 this guaranteed k_hat/k_max > 1 (lambda -> 0) at
    # ANY stable k_hat, independent of whether the game is actually
    # near-potential -- see AdaptiveKMax's docstring in losses.py. Now
    # that k_max is (a) per-agent, (b) calibrated once and frozen instead
    # of chasing k_hat forever, and (c) compared against a
    # perturbation-normalized k_hat that should roughly stabilize rather
    # than decay, k_max_fraction instead means "how many multiples of the
    # calibrated reference scale still count as trustworthy." Treat this
    # default as a starting point to sweep, not a known-good value --
    # after re-running, check that lambda/k_hat settle somewhere in (0, 1)
    # for the near-potential alphas instead of saturating at 0 or 1.
    k_max_ema_decay: float = 0.99
    k_max_calibration_steps: int = 200
    # ^ new. Number of train_step()s over which each agent's k_max
    # reference is estimated (an EMA of that agent's OWN k_hat, never
    # pooled with the other agent's); frozen after this many updates. 200
    # is chosen to sit early in the entropy_anneal_steps=1000 schedule
    # (entropy_coef(200) ~= 0.025, vs. entropy_start=0.03), i.e. while the
    # policy is still close to its initial exploration level, so the
    # reference reflects genuine early disagreement rather than an
    # already-sharpened policy.
    phi_probe_eps: float = 1e-4
    # ^ new. Floor added to the squared counterfactual perturbation when
    # normalizing k_hat (see closeness_and_trsut), so the normalization
    # doesn't blow up if a'_i - a_i is ~0 for a batch element.

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
            num_agents=N,
            k_max_fraction=self.cfg.k_max_fraction,
            ema_decay=self.cfg.k_max_ema_decay,
            calibration_steps=self.cfg.k_max_calibration_steps,
            device=self.device,
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
        naive_loss, phi_residuals, phi_deltas = phi_naive_loss(
            self.phi, self.q_heads, self.actors, td, N
        )
        phi_loss = td_loss + self.cfg.beta * naive_loss

        self.phi_optim.zero_grad()
        phi_loss.backward(retain_graph=True)
        phi_grad_norm = th.nn.utils.clip_grad_norm_(
            self.phi.parameters(), self.cfg.phi_max_grad_norm
        )
        self.phi_optim.step()

        log["phi/loss_td"] = td_loss.item()
        log["phi/loss_naive"] = naive_loss.item()
        log["phi/loss_total"] = phi_loss.item()
        log["phi/beta"] = self.cfg.beta
        # ^ phi/loss_td and phi/loss_naive were previously named
        # "phi/td_loss" / "phi/naive_loss" -- silently mismatched against
        # run_matrix_benchmark.py's metrics(), which has always read
        # "phi/loss_td" / "phi/loss_naive" (matching the already-correct
        # "phi/loss_total"). That meant phi_loss_td/phi_loss_naive were
        # always None in every curve file ever written, and
        # plot_phi_loss_vs_step's figure only ever showed the total line.
        # beta is new: phi_loss = loss_td + beta*loss_naive, and with
        # beta != 1 (NPGACConfig's default is 5.0) the raw, unweighted
        # loss_naive line doesn't show what's actually driving phi's
        # gradient -- logging beta lets the plot show beta*loss_naive too.
        log["phi/grad_norm"] = phi_grad_norm.item()

        # Actor update
        with th.no_grad():
            _, fresh_residuals, fresh_deltas = phi_naive_loss(
                self.phi, self.q_heads, self.actors, td, N
            )

        k_hat_now, _ = closeness_and_trsut(
            fresh_residuals, fresh_deltas, self.k_max_tracker.k_max,
            eps=self.cfg.phi_probe_eps,
        )
        self.k_max_tracker.update(k_hat_now)
        # ^ once, full [N] vector -- NOT looped/indexed per agent (e.g.
        # `for i in range(N): self.k_max_tracker.update(k_hat_now[i])`).
        # See AdaptiveKMax.update's docstring in losses.py for why that
        # would break both the per-agent calibration and the calibration
        # step count.

        current_entropy_coef = self.cfg.entropy_coef(self.num_updates)
        actor_out = actor_losses(
            self.actors, self.q_heads, self.phi, td, N, self.k_max_tracker.k_max,
            phi_residuals=fresh_residuals, phi_deltas=fresh_deltas,
            entropy_coef=current_entropy_coef,
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
            log[f"actor/{i}/k_max"] = self.k_max_tracker.k_max[i].item()
            log[f"actor/{i}/probe_delta_sq"] = (fresh_deltas[i] ** 2).mean().item()
            # ^ diagnostic: mean squared counterfactual perturbation size
            # for this agent this step. If lambda/k_hat trend toward 0
            # alongside this also shrinking, that's the entropy-collapse
            # confound (expected in pure-strategy/corner-NE regimes) --
            # not necessarily "this isn't a potential game." If this stays
            # roughly flat while k_hat still trends down, that's a
            # genuine closeness signal.

        log["train/k_max"] = self.k_max_tracker.k_max.tolist()
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