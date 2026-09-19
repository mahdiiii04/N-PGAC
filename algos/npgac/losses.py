from dataclasses import dataclass

import torch as th

from algos.npgac.networks import replace_agent_coord


def joint_strategy_from_td(td, num_agents):
    return th.cat(
        [td["agents", str(i), "action"] for i in range(num_agents)], dim=-1
    )  # [B, num_agents]

def per_agent_reward(td, agent_idx):
    return td["next", "reward"][:, agent_idx, 0]  # [B]

################## Critic (Q_i) Loss #####################

def q_regression_loss(q_heads, td, num_agents):
    joint_strategy = joint_strategy_from_td(td, num_agents)
    per_agent_losses = []
    for i in range(num_agents):
        q_sa = q_heads[i](joint_strategy)
        r_i = per_agent_reward(td, i)
        per_agent_losses.append(th.mean((q_sa - r_i) ** 2))
    return th.stack(per_agent_losses).mean(), per_agent_losses

################## Potential (Phi) Loss ##################

def phi_td_loss(phi, td, num_agents):
    joint_strategy = joint_strategy_from_td(td, num_agents)
    rewards = th.stack([per_agent_reward(td, i) for i in range(num_agents)], dim=0)
    rho = rewards.mean(dim=0)
    phi_sa = phi(joint_strategy)
    return th.mean((phi_sa - rho) ** 2)


def sample_counterfactual_strategies(actors, td, num_agents):
    """Returns (cf_strategies, deltas).

    cf_strategies[i]: joint action with agent i's coordinate replaced by a
        fresh sample a_i' ~ pi_i.
    deltas[i]: (a_i' - a_i), the actual size of the counterfactual
        perturbation used to produce cf_strategies[i].

    deltas is new -- it's needed by closeness_and_trsut to normalize the
    naive-loss residual by the probe size. Without it, k_hat shrinks
    toward 0 purely because pi_i's entropy anneals down / sharpens over
    training (a'_i -> a_i), independent of whether the underlying game is
    actually close to a potential game: since Q_i and phi are smooth,
    delta_Q ~= grad(Q_i)*delta_a and delta_phi ~= grad(phi)*delta_a both
    shrink proportionally to delta_a, so their difference shrinks too even
    when grad(phi) - grad(Q_i) hasn't gone away.
    """
    joint_strategy = joint_strategy_from_td(td, num_agents)
    cf_strategies = []
    deltas = []
    for i, actor in enumerate(actors):
        fresh_td = actor(td.select(("agents", str(i), "observation")).clone())
        a_i_prime = fresh_td["agents", str(i), "action"].squeeze(-1) # [B]
        a_i = joint_strategy[:, i]
        deltas.append(a_i_prime - a_i)
        cf_strategies.append(replace_agent_coord(joint_strategy, i, a_i_prime))
    return cf_strategies, deltas

def phi_naive_loss(phi, q_heads, actors, td, num_agents,
                   cf_strategies=None, deltas=None, detach_q=True):

    joint_strategy = joint_strategy_from_td(td, num_agents)
    if cf_strategies is None or deltas is None:
        cf_strategies, deltas = sample_counterfactual_strategies(actors, td, num_agents)

    phi_sa = phi(joint_strategy)

    def q_call(q, s):
        out = q(s)
        return out.detach() if detach_q else out

    q_sa = th.stack(
        [q_call(q_heads[i], joint_strategy) for i in range(num_agents)], dim=0
    )

    residuals = []
    for i in range(num_agents):
        phi_cf = phi(cf_strategies[i])
        q_cf_i = q_call(q_heads[i], cf_strategies[i])
        delta_phi = phi_cf - phi_sa
        delta_q = q_cf_i - q_sa[i]
        residuals.append(delta_phi - delta_q)

    residuals = th.stack(residuals, dim=0) # [N, B]
    return th.mean(residuals ** 2), residuals, deltas

class AdaptiveKMax:
    """Per-agent trust-gate threshold.

    The previous implementation pooled both agents into a single scalar
    (via k_hat.mean()) and re-derived k_max every step as a fraction of
    that same, continuously-updating quantity's own EMA:

        k_max = max(k_max_fraction * EMA(k_hat.mean()), floor)

    This is self-defeating: once k_hat stabilizes at ANY level v (whether
    or not that level reflects genuine potential-game structure), the EMA
    converges to v too, so k_max -> k_max_fraction * v. For
    k_max_fraction < 1 this guarantees k_hat / k_max > 1 for every agent
    whose k_hat is near its own steady value -- lambda is driven to a
    permanent 0 as an artifact of the definition, not of game structure.
    Pooling across agents compounds this: whichever agent's raw k_hat
    happens to run numerically larger (for reasons unrelated to
    potential-ness -- payoff scale, how fast its own policy sharpens)
    permanently zeroes its own trust while also setting the bar for the
    other agent.

    This version:
      - keeps one EMA per agent (k_hat is never pooled across agents), and
      - only updates that EMA during an initial `calibration_steps`
        window, then freezes it -- so k_max reflects a fixed reference
        scale (roughly: "how much naive-loss residual did this agent show
        early in training, before its policy had sharpened much") rather
        than perpetually chasing k_hat's own ongoing decay or rise.

    k_max_fraction now means "how many multiples of the calibrated
    reference scale still count as trustworthy" -- with the residual
    normalization in closeness_and_trsut (see below) k_hat should roughly
    stabilize once Q_i/phi have converged, rather than trend to 0, so
    values >= 1 make sense here (unlike the old self-referential design,
    where <1 was required to ever produce nonzero lambda at all).
    """

    def __init__(self, num_agents, k_max_fraction=1.5, ema_decay=0.99,
                 floor=1e-3, calibration_steps=200, device=None):
        self.num_agents = num_agents
        self.k_max_fraction = k_max_fraction
        self.ema_decay = ema_decay
        self.floor = floor
        self.calibration_steps = calibration_steps
        self.ema_ref = th.zeros(num_agents, device=device)
        self.n_updates = 0

    def update(self, k_hat):
        """k_hat: [num_agents] tensor, this step's per-agent closeness
        estimate. Call ONCE per training step with the full vector.

        Do not call this inside a `for i in range(N)` loop with
        `k_hat[i]`: k_hat[i] is a scalar, so `(1 - ema_decay) * k_hat[i]`
        would broadcast across the *entire* ema_ref vector on each
        iteration, bleeding one agent's calibration into every other
        agent's reference (re-introducing the pooling bug this class is
        meant to remove). It would also advance self.n_updates N times per
        real training step, silently shrinking calibration_steps by a
        factor of N relative to what NPGACConfig says.
        """
        with th.no_grad():
            self.n_updates += 1
            if self.n_updates <= self.calibration_steps:
                self.ema_ref = (
                    self.ema_decay * self.ema_ref
                    + (1 - self.ema_decay) * k_hat.detach().to(self.ema_ref.device)
                )
            # after calibration_steps, ema_ref is frozen: k_max stops
            # chasing k_hat's own live value.

    @property
    def k_max(self):
        return th.clamp(self.k_max_fraction * self.ema_ref, min=self.floor)


def closeness_and_trsut(residuals, deltas, k_max, eps=1e-4):
    """(k_hat, lam), both shape [N].

    k_hat is now normalized by the squared counterfactual perturbation
    size (E[(a_i' - a_i)^2]) rather than the raw residual magnitude, so it
    estimates the actual local gradient mismatch
    |grad_a(phi) - grad_a(Q_i)| instead of
    (gradient mismatch) * (how far we happened to perturb) -- the latter
    shrinks toward 0 for free as the policy's entropy anneals down over
    training, regardless of whether the game is close to a potential game.
    `eps` guards against blow-up when a'_i - a_i is ~0 for a batch element
    (a very peaked policy).

    residuals: [N, B] -- (delta_phi - delta_q) per agent, per batch
        element, as returned by phi_naive_loss.
    deltas: list of N tensors, each [B] -- (a_i' - a_i) for that agent, as
        returned by sample_counterfactual_strategies / phi_naive_loss.
    k_max: [N] per-agent trust threshold (AdaptiveKMax.k_max).
    """
    with th.no_grad():
        delta_sq = th.stack([d ** 2 for d in deltas], dim=0)  # [N, B]
        normalized = residuals ** 2 / (delta_sq + eps)
        k_hat = th.sqrt(th.mean(normalized, dim=-1))  # [N]
        lam = 1.0 - th.clamp(k_hat / k_max, 0.0, 1.0)
    return k_hat, lam


############### Actor losses #######################

@dataclass
class ActorLossOutputs:
    per_agent_loss: list
    lam: th.Tensor
    k_hat: th.Tensor
    entropy: th.Tensor

def actor_losses(actors, q_heads, phi, td, num_agents,
                 k_max, phi_residuals=None, phi_deltas=None, entropy_coef=0.0):
    joint_strategy = joint_strategy_from_td(td, num_agents)

    if phi_residuals is None or phi_deltas is None:
        _, phi_residuals, phi_deltas = phi_naive_loss(
            phi, q_heads, actors, td, num_agents
        )

    k_hat, lam = closeness_and_trsut(phi_residuals, phi_deltas, k_max)

    cf_strategies, _ = sample_counterfactual_strategies(actors, td, num_agents)

    per_agent_loss = []
    entropies = []

    for i, actor in enumerate(actors):
        log_prob_i = actor.log_prob(td).squeeze(-1)

        with th.no_grad():
            q_sa_i = q_heads[i](joint_strategy)
            q_baseline_i = q_heads[i](cf_strategies[i])
            A_i = q_sa_i - q_baseline_i

            phi_sa = phi(joint_strategy)
            phi_baseline = phi(cf_strategies[i])
            A_phi_i = phi_sa - phi_baseline

        mixed_advantage = (1 -lam[i]) * A_i + lam[i] * A_phi_i
        pg_loss_i = -(log_prob_i * mixed_advantage).mean()

        dist_i = actor.get_dist(td)
        entropy_i = dist_i.entropy().mean()
        loss_i = pg_loss_i - entropy_coef * entropy_i

        per_agent_loss.append(loss_i)
        entropies.append(entropy_i.detach())

    return ActorLossOutputs(
        per_agent_loss=per_agent_loss, lam=lam,
        k_hat=k_hat, entropy=th.stack(entropies),
    )