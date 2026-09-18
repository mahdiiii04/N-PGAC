import numpy as np

def payoffs(alpha):
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