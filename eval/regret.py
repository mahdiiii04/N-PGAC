import numpy as np

# eval/regret.py depends on envs/matrix_game.py (not the other way around,
# so no import cycle) so that the common-interest payoff used for regret
# has one source of truth (CommonInterestMatrixGameEnv.DEFAULT_PAYOFF)
# rather than a second hardcoded copy here that could silently drift out
# of sync with the actual environment being trained on.
from envs.matrix_game import COMMON_INTEREST_ALPHA, CommonInterestMatrixGameEnv

def payoffs(alpha):
    """(A, B) payoff matrices for a given alpha.

    alpha == COMMON_INTEREST_ALPHA (-1) is a sentinel, not part of the
    alpha family's closed form (which is only meaningful for alpha in
    [0, 1]): it returns CommonInterestMatrixGameEnv's actual, identical
    A == B payoff instead of extrapolating the alpha-family formula
    below to a negative alpha, which would just be two more mismatched,
    meaningless matrices.
    """
    if alpha == COMMON_INTEREST_ALPHA:
        M = np.array(CommonInterestMatrixGameEnv.DEFAULT_PAYOFF)
        return M, M.copy()
    a = float(alpha)
    A = np.array([[1.0, 1.0 - 2.0 * a], [(1.0 - 3.0 * a) / 2.0, a]])
    B = np.array(
        [[1.0 - 2.0 * a, (a + 1.0) / 2.0], [(7.0 - 3.0 * a) / 4.0, 2.0 - 3.0 * a]]
    )
    return A, B

def regret(p, q, alpha):
    """(max_regret, regret_A, regret_B, J_A, J_B) of profile (p, q) = the
    pair of P(second action), matching matrix_game_env.py's convention.
    """
    A, B = payoffs(alpha)
    pA = np.array([1.0 - p, p])
    pB = np.array([1.0 - q, q])
    J_A = pA @ A @ pB
    J_B = pA @ B @ pB
    r_A = (A @ pB).max() - J_A
    r_B = (pA @ B).max() - J_B
    return max(r_A, r_B), r_A, r_B, J_A, J_B