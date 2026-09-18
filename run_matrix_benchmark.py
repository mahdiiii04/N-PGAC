import json
import os
import time

import torch as th

from algos.mappo.trainer import MAPPOConfig, MAPPOTrainer
from algos.npgac.trainer import NPGACConfig, NPGACTrainer
from envs.matrix_game import AlphaMatrixGameEnv
from eval.ground_truth import ground_truth_pq, regime
from eval.regret import regret


ALPHAS = [
    0.0, 0.1, 0.2,       # pure (A1, B1)
    0.25, 0.3, 0.33,     # pure (A1, B2)
    0.34, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  # mixed
    1.0,                 # zero-sum
]

SEEDS = [0, 1, 2]
TRAIN_STEPS = 2500
BATCH_SIZE = 512
LOG_EVERY = 1  # record metrics every N training steps

OUTPUT_DIR = "output"
CURVES_DIR = os.path.join(OUTPUT_DIR, "curves")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "benchmark_results.json")


TRAINERS = {
    "npgac": (NPGACTrainer, NPGACConfig),
    "mappo": (MAPPOTrainer, MAPPOConfig),
}


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

def metrics(p, q, alpha, train_log=None, num_agents=2):
    """Per-logged-step metric dict. `train_log` is the dict returned by
    trainer.train_step() for THIS step (None at step 0, before any update
    has happened) -- used to pull out method-internal diagnostics that
    aren't recoverable from (p, q) alone:
      - lambda_i, k_hat_i (per agent): only present for N-PGAC (the trust
        gate/closeness estimate; MAPPO has no potential function, so these
        keys are simply absent from MAPPO's train_log and end up as None
        here -- checked explicitly with .get() rather than assuming the
        key exists, so this function works unmodified for either algo).
      - phi_loss_td, phi_loss_naive, phi_loss_total: same -- N-PGAC only.
    Keeping them as None (not omitting the keys) for MAPPO rows means every
    curve entry has a consistent schema regardless of algo, which matters
    for any downstream plotting code that reads both algos' curves the
    same way.
    """
    r_max, r_A, r_B, J_A, J_B = regret(p, q, alpha)
    out = {
        "p": float(p), "q": float(q),
        "regret": float(r_max), "regret_A": float(r_A), "regret_B": float(r_B),
        "J_A": float(J_A), "J_B": float(J_B),
    }

    train_log = train_log or {}
    for i in range(num_agents):
        out[f"lambda_{i}"] = train_log.get(f"actor/{i}/lambda")
        out[f"k_hat_{i}"] = train_log.get(f"actor/{i}/k_hat")

    out["phi_loss_td"] = train_log.get("phi/loss_td")
    out["phi_loss_naive"] = train_log.get("phi/loss_naive")
    out["phi_loss_total"] = train_log.get("phi/loss_total")

    return out


def run_algo(algo, alpha, seed, train_steps=None, log_every=None):
    """Train `algo` and return (p, q, history) where history is a list of
    per-step metric dicts."""
    train_steps = TRAIN_STEPS if train_steps is None else train_steps
    log_every = LOG_EVERY if log_every is None else log_every

    th.manual_seed(seed)
    env = AlphaMatrixGameEnv(alpha=alpha, batch_size=BATCH_SIZE)
    trainer_cls, config_cls = TRAINERS[algo]
    trainer = trainer_cls(env, config_cls())
    num_agents = config_cls().num_agents

    history = []

    # step 0 = initialisation, before any gradient update -- no train_log
    # yet, so lambda/k_hat/phi_loss fields are correctly None here for
    # both algorithms (nothing has been "trusted" or fitted yet).
    p, q = trainer.current_strategy()
    history.append({
        "step": 0,
        **metrics(p, q, alpha, train_log=None, num_agents=num_agents),
    })

    for step in range(1, train_steps + 1):
        train_log = trainer.train_step()
        if step % log_every == 0 or step == train_steps:
            p, q = trainer.current_strategy()
            history.append({
                "step": step,
                **metrics(p, q, alpha, train_log=train_log, num_agents=num_agents),
            })

    p, q = trainer.current_strategy()
    return p, q, history


def run_npgac(alpha, seed, **kw):
    return run_algo("npgac", alpha, seed, **kw)


def run_mappo(alpha, seed, **kw):
    return run_algo("mappo", alpha, seed, **kw)


RUNNERS = {
    "npgac": run_npgac,
    "mappo": run_mappo,
}


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #

def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def save_json(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)  # atomic, so a crash mid-write can't corrupt the file


def load_results():
    return load_json(RESULTS_PATH, {})


def save_results(results):
    save_json(RESULTS_PATH, results)


def curves_path(alpha):
    return os.path.join(CURVES_DIR, f"alpha={alpha}.json")


def load_curves(alpha):
    return load_json(curves_path(alpha), {})


def save_curves(alpha, curves):
    save_json(curves_path(alpha), curves)


def cell_key(algo, alpha, seed):
    return f"{algo}|alpha={alpha}|seed={seed}"


def curve_key(algo, seed, step):
    return f"{algo}|seed={seed}|step={step}"


# --------------------------------------------------------------------------- #
# sweep
# --------------------------------------------------------------------------- #

def run_sweep(alphas=ALPHAS, seeds=SEEDS, algorithms=("npgac", "mappo"),
              train_steps=None, log_every=None):
    results = load_results()

    total = len(alphas) * len(seeds) * len(algorithms)
    done = 0
    start = time.time()

    for algo in algorithms:
        for alpha in alphas:
            curves = load_curves(alpha)

            for seed in seeds:
                key = cell_key(algo, alpha, seed)
                prefix = f"{algo}|seed={seed}|"
                has_curve = any(k.startswith(prefix) for k in curves)

                # only skip if both the summary cell and its curve are on disk
                if key in results and has_curve:
                    done += 1
                    continue

                p, q, history = run_algo(
                    algo, alpha, seed,
                    train_steps=train_steps, log_every=log_every,
                )
                r_max, r_A, r_B, J_A, J_B = regret(p, q, alpha)
                gt_p, gt_q = ground_truth_pq(alpha)

                results[key] = {
                    "algo": algo, "alpha": alpha, "seed": seed,
                    "p": p, "q": q,
                    "regret": r_max, "regret_A": r_A, "regret_B": r_B,
                    "gt_p": gt_p, "gt_q": gt_q, "regime": regime(alpha),
                }

                for entry in history:
                    curves[curve_key(algo, seed, entry["step"])] = {
                        "algo": algo, "alpha": alpha, "seed": seed, **entry,
                    }

                save_results(results)
                save_curves(alpha, curves)

                done += 1
                elapsed = time.time() - start
                print(
                    f"[{done}/{total}] {algo} alpha={alpha} seed={seed}: "
                    f"(p,q)=({p:.4f},{q:.4f}) regret={r_max:.4f} "
                    f"(gt=({gt_p:.4f},{gt_q:.4f}))  [{elapsed:.0f}s elapsed]"
                )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", type=float, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--algo", type=str, choices=["npgac", "mappo", "both"], default="both")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None,
                        help="record training-curve metrics every N steps (default: 1)")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir
    CURVES_DIR = os.path.join(OUTPUT_DIR, "curves")
    RESULTS_PATH = os.path.join(OUTPUT_DIR, "benchmark_results.json")
    os.makedirs(CURVES_DIR, exist_ok=True)

    algos = ("npgac", "mappo") if args.algo == "both" else (args.algo,)
    run_sweep(
        alphas=args.alphas or ALPHAS,
        seeds=args.seeds or SEEDS,
        algorithms=algos,
        train_steps=args.steps,
        log_every=args.log_every,
    )