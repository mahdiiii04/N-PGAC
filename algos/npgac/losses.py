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
    joint_strategy = joint_strategy_from_td(td, num_agents)
    cf_strategies = []
    for i, actor in enumerate(actors):
        obs_i = td["agents", str(i), "observation"]
        fresh_td = actor(td.select(("agents", str(i), "observation")).clone())
        a_i_prime = fresh_td["agents", str(i), "action"].squeeze(-1) # [B]
        cf_strategies.append(replace_agent_coord(joint_strategy, i, a_i_prime))
    return cf_strategies

def phi_naive_loss(phi, q_heads, actors, td, num_agents, 
                   cf_strategies=None, detach_q=True):

    joint_strategy = joint_strategy_from_td(td, num_agents)
    if cf_strategies is None:
        cf_strategies = sample_counterfactual_strategies(actors, td, num_agents)

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
    return th.mean(residuals ** 2), residuals

class AdaptiveKMax:
    def __init__(self, k_max_fraction=0.5, ema_decay=0.99,
                 floor=1e-3, init=0.15):
        self.k_max_fraction = k_max_fraction
        self.ema_decay = ema_decay
        self.floor = floor
        self.ema_max = init

    def update(self, k_hat):
        with th.no_grad():
            batch_stat = k_hat.mean().item()
            self.ema_max = self.ema_decay * self.ema_max + (1 - self.ema_decay) * batch_stat

    @property
    def k_max(self):
        return max(self.k_max_fraction * self.ema_max, self.floor)

def closeness_and_trsut(residuals, k_max):
    with th.no_grad():
        k_hat = th.sqrt(th.mean(residuals ** 2, dim=-1))
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
                 k_max, phi_residuals=None, entropy_coef=0.0):
    joint_strategy = joint_strategy_from_td(td, num_agents)

    if phi_residuals is None:
        _, phi_residuals = phi_naive_loss(phi, q_heads, actors, td, num_agents)

    k_hat, lam = closeness_and_trsut(phi_residuals, k_max)

    cf_strategies = sample_counterfactual_strategies(actors, td, num_agents)

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

