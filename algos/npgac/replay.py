import copy

import torch as th
from tensordict import TensorDict


class ReplayBuffer:
    """Stores (observation, action, reward) transitions across steps so
    Q_i and phi can be trained on a broader, more temporally-diverse slice
    of joint-action space than whatever the current, increasingly narrow
    (as policy entropy collapses over training) on-policy batch alone
    provides.

    This is unusually cheap to do *correctly* in this environment:
    Q_i(a) := E[r_i | a] (q_regression_loss's target) and phi's TD target
    rho(a) := mean_i r_i(a) (phi_td_loss's target) are both deterministic
    functions of the joint action alone -- this env has no real state and
    every episode is exactly one step -- so replaying old (a, r) pairs
    introduces NO off-policy bias into either regression target. This is
    unlike typical off-policy value-function learning, where a stored
    target implicitly depends on which policy generated it; here it
    doesn't, so there's no importance-weighting/staleness correction
    needed for these two losses specifically.

    Deliberately NOT used for phi_naive_loss (the counterfactual-matching
    term, for either its role in phi's own training or the actor/trust-
    gate's k_hat computation): that loss compares a joint action against a
    FRESH counterfactual resample from the CURRENT policy, and pairing a
    replayed (possibly very old, wide-entropy) base action with a fresh,
    already-sharpened counterfactual sample would produce a perturbation
    size that doesn't reflect any single point in training, muddying
    exactly the local sensitivity phi_naive_loss (and by extension k_hat)
    is trying to measure. Confining replay to the two loss terms that
    remain unbiased under it keeps that risk out of the trust-gate's own
    inputs, which is the part of this codebase most sensitive to subtle
    calibration issues (see AdaptiveKMax / closeness_and_trsut in
    losses.py).

    Stores raw tensors (not TensorDicts) for compactness; sample() returns
    a TensorDict shaped exactly like NPGACTrainer.collect()'s output, so
    q_regression_loss and phi_td_loss work completely unchanged on
    replayed data -- no changes to losses.py were needed for this.
    """

    def __init__(self, capacity, num_agents, device=None):
        self.capacity = capacity
        self.num_agents = num_agents
        self.device = device
        self.observation = th.zeros(capacity, num_agents, 1, device=device)
        self.action = th.zeros(capacity, num_agents, 1, device=device)
        self.reward = th.zeros(capacity, num_agents, 1, device=device)
        self.size = 0
        self.ptr = 0

    def push(self, td):
        """td: a batch shaped like NPGACTrainer.collect()'s output --
        action: [B, num_agents, 1], ("next","reward"): [B, num_agents, 1],
        ("agents", str(i), "observation"): [B, 1] each. Always detaches:
        replayed data must never carry gradient history back into the
        actor parameters that produced it at collection time.
        """
        B = td.batch_size[0]
        obs = th.stack(
            [td["agents", str(i), "observation"] for i in range(self.num_agents)],
            dim=1,
        ).detach()
        action = td["action"].detach()
        reward = td["next", "reward"].detach()

        idx = (self.ptr + th.arange(B, device=self.device)) % self.capacity
        self.observation[idx] = obs.to(self.device) if self.device else obs
        self.action[idx] = action.to(self.device) if self.device else action
        self.reward[idx] = reward.to(self.device) if self.device else reward

        self.ptr = (self.ptr + B) % self.capacity
        self.size = min(self.size + B, self.capacity)

    def __len__(self):
        return self.size

    def sample(self, batch_size):
        """Returns a TensorDict shaped like collect()'s output. Caller is
        responsible for checking len(buffer) > 0 first (sampling from an
        empty buffer raises, same as indexing an empty tensor would)."""
        n = min(batch_size, self.size)
        idx = th.randint(0, self.size, (n,), device=self.device)
        obs = self.observation[idx]
        action = self.action[idx]
        reward = self.reward[idx]

        # Both the top-level "action" (what collect() hands to
        # env.step()) and the per-agent nested ("agents", i, "action")
        # (what joint_strategy_from_td actually reads -- written by each
        # actor's out_keys in networks.py's make_actor) are populated:
        # they're the same data, just shaped two ways, and losses.py's
        # functions read the nested form.
        data = {"action": action}
        for i in range(self.num_agents):
            data["agents", str(i), "observation"] = obs[:, i, :]
            data["agents", str(i), "action"] = action[:, i, :]
        td = TensorDict(data, batch_size=[n], device=self.device)
        td["next"] = TensorDict({"reward": reward}, batch_size=[n], device=self.device)
        return td


def make_target(live_net):
    """A frozen (requires_grad=False) deep copy of live_net, parameter-
    identical at creation -- the standard target-network starting point."""
    target = copy.deepcopy(live_net)
    for p in target.parameters():
        p.requires_grad_(False)
    return target


def soft_update(target_net, live_net, tau):
    """Polyak-average target_net's parameters toward live_net's:
    target <- tau*live + (1-tau)*target. Call once per step, after each
    live-network gradient step.

    This env has no bootstrapping at all (single-step episodes, gamma
    never appears in q_regression_loss/phi_td_loss), so target networks
    here aren't serving the classical DDPG/TD3 role of stabilizing a
    self-referential bootstrap target -- there isn't one. They're serving
    a different, narrower purpose specific to this codebase (see
    Welcome_file.md: "pi_i, Q_i, phi_w, lambda_i are co-adapting
    simultaneously"): within one train_step(), Q_i/phi get updated and
    are then IMMEDIATELY read again, same step, to compute the actor's
    advantage and the trust gate's k_hat -- so that signal is always
    evaluated against parameters that just moved out from under it.
    Target copies give the actor/trust-gate a stable, slowly-moving
    reference instead, decoupling that co-adaptation loop.
    """
    with th.no_grad():
        for t_param, l_param in zip(target_net.parameters(), live_net.parameters()):
            t_param.data.mul_(1.0 - tau).add_(l_param.data, alpha=tau)