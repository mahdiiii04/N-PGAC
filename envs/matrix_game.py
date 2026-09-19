import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase

# Sentinel alpha value used by run_matrix_benchmark.py / eval/ground_truth.py /
# eval/regret.py to mean "use CommonInterestMatrixGameEnv instead of
# AlphaMatrixGameEnv(alpha=...)". -1 is unambiguous since AlphaMatrixGameEnv's
# family is only ever swept over alpha in [0, 1]. Defined once here (the
# "owning" module for both envs) and imported everywhere else that needs to
# check for it, rather than re-hardcoding the literal -1 in three files.
COMMON_INTEREST_ALPHA = -1.0

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


class CommonInterestMatrixGameEnv(AlphaMatrixGameEnv):
    """A genuinely common-interest (fully cooperative) 2x2 matrix game:
    both players share an IDENTICAL payoff matrix, so the true potential
    function is exactly the shared reward itself and Q_A == Q_B == phi*
    up to network fitting error.

    This is the correct upper-bound sanity check for N-PGAC's trust gate,
    and AlphaMatrixGameEnv's alpha=0 does NOT already cover it: at
    alpha=0, payoff_A = [[1,1],[0.5,0]] and payoff_B = [[1,0.5],[1.75,2]]
    -- different matrices that merely happen to share the same pure NE.
    No alpha in [0, 1] makes payoff_A == payoff_B (matching the (1,1)
    entry alone forces alpha=0, which already breaks the (1,2) entry), so
    alpha=0 is "general-sum, close-ish to potential" but never an actual
    potential game -- see eval/ground_truth.py and the NePPO paper
    (arXiv:2603.06977) itself, which states this plainly ("At alpha=0, the
    game is a general-sum game...") and proves it formally in Appendix E.

    run_matrix_benchmark.py selects this env instead of AlphaMatrixGameEnv
    whenever alpha == COMMON_INTEREST_ALPHA (-1); eval/ground_truth.py and
    eval/regret.py special-case that same sentinel so regret is computed
    against this env's real payoff matrix rather than extrapolating the
    alpha-family formula to a negative alpha (which would be meaningless).

    Default payoff is a simple, non-degenerate coordination game with two
    strict pure NE, (A1,B1)=2 and (A2,B2)=1, plus one interior mixed NE at
    (p,q) = (2/3, 2/3) (verified numerically via eval.regret.regret,
    which gives exactly 0 there and at both pure NE, and > 0 elsewhere)
    -- deliberately asymmetric payoffs so a coincidental symmetry can't
    mask a bug, even though the resulting mixed NE happens to be
    symmetric in (p, q). Pass `payoff=` to use a different common matrix
    (note eval/regret.py's payoffs() only knows about DEFAULT_PAYOFF, so
    regret against a custom payoff must be computed separately if you
    override this).
    """

    DEFAULT_PAYOFF = [[2.0, 0.0], [0.0, 1.0]]

    def __init__(self, payoff=None, batch_size=1000, device=None):
        self._payoff_override = payoff
        super().__init__(alpha=0.0, batch_size=batch_size, device=device)

    def set_alpha(self, alpha):
        # `alpha` is accepted only for signature-compatibility with
        # AlphaMatrixGameEnv.__init__ (which calls self.set_alpha(alpha));
        # it has no effect here -- this env is a single fixed common-
        # payoff game, not a member of the alpha family.
        self.alpha = COMMON_INTEREST_ALPHA
        M = self._payoff_override
        if M is None:
            M = self.DEFAULT_PAYOFF
        M = torch.as_tensor(M, dtype=torch.float32)
        self.payoff_A = M.clone()
        self.payoff_B = M.clone()  # identical -- genuinely common interest