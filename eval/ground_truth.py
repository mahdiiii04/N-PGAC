"""
Closed-form Nash equilibrium of the alpha-game family, across
the full alpha in [0, 1] range.

alpha in [0, 1/5]    : pure NE (A1, B1)  -> (p, q) = (0, 0)
alpha in (1/5, 1/3]  : pure NE (A1, B2)  -> (p, q) = (0, 1)
alpha in (1/3, 1]    : mixed NE, p(A1) = (9a-1)/(19a-3), p(B1) = 2(3a-1)/(9a-1)
                        -> (p, q) = (1 - p(A1), 1 - p(B1))
                        since our (p, q) convention is P(second action) =
                        P(A2)/P(B2), matching matrix_game_env.py, while
                        the mixed-NE formula is stated in P(first action).

At alpha=1/3 exactly, (A1,B2) is still a valid pure NE (player A is
indifferent between A1/A2 against q=1 at precisely this alpha); for
alpha > 1/3 it is not (A2 strictly dominates against q=1), forcing the
equilibrium to become mixed. The mixed-NE formula's p(A1) jumps from 0 (at
the pure-NE boundary) to (9*(1/3)-1)/(19*(1/3)-3) = 0.6 -- i.e. (p,q) jumps
discontinuously from (0,1) to (0.4,1.0) at alpha=1/3, while q stays
continuous at 1.

alpha == COMMON_INTEREST_ALPHA (-1, see envs/matrix_game.py) is a separate
sentinel, not part of this closed form: it selects
CommonInterestMatrixGameEnv instead of the alpha family. See
ground_truth_pq's and regime's docstrings/branches below.
"""

import numpy as np

from envs.matrix_game import COMMON_INTEREST_ALPHA

ALPHA_BREAK_1 = 1.0 / 5.0 # pure (A1,B1) -> pure (A1, B2) boundary
ALPHA_BREAK_2 = 1.0 / 3.0 # pure (A1,B2) -> mixed NE boundary

def mixed_ne_p_first_action(alpha):
    """Valid for alpha > 1/3"""
    p_A1 = (9.0 * alpha - 1.0) / (19.0 * alpha - 3.0)
    p_B1 = 2 * (3.0 * alpha - 1.0) / (9.0 * alpha - 1.0)
    return p_A1, p_B1

def ground_truth_pq(alpha):
    """(p, q) = (P(A2), P(B2)) at the NE for a given alpha,
    matching matrix_game_env.py's action convention.

    Returns (None, None) for alpha == COMMON_INTEREST_ALPHA: unlike every
    alpha in the [0, 1] family, CommonInterestMatrixGameEnv's default
    payoff has THREE Nash equilibria -- (0,0), (1,1), and the mixed
    (1/3, 2/3) -- so there is no single closed-form "the" ground truth to
    compare a trained (p, q) against; picking one arbitrarily would
    penalize training runs that correctly converge to a different, equally
    valid equilibrium. Callers (see run_matrix_benchmark.py's print, and
    plot_ne_curve.py's plot_metric_vs_step, which already skips drawing
    the reference line when gt is None) must handle this.
    """
    if alpha == COMMON_INTEREST_ALPHA:
        return None, None
    if alpha <= ALPHA_BREAK_1:
        return 0.0, 0.0
    elif alpha <= ALPHA_BREAK_2:
        return 0.0, 1.0
    else:
        p_A1, p_B1 = mixed_ne_p_first_action(alpha)
        return 1.0 - p_A1, 1.0 - p_B1

def ground_truth_curve(num_points):
    alphas = np.linspace(0.0, 1.0, num_points)
    p = np.empty_like(alphas)
    q = np.empty_like(alphas)
    for i, a in enumerate(alphas):
        p[i], q[i] = ground_truth_pq(a)
    return alphas, p, q

def regime(alpha):
    if alpha == COMMON_INTEREST_ALPHA:
        return "common interest"
    if alpha <= ALPHA_BREAK_1:
        return "pure (A1, B1)"
    elif alpha <= ALPHA_BREAK_2:
        return "pure (A1, B2)"
    else:
        return "mixed"