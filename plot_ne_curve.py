"""Plot the closed-form ground-truth NE curve across alpha in [0,1]
(three regions: pure (A1,B1), pure (A1,B2), mixed -- see
eval/ground_truth.py), with N-PGAC's and MAPPO's learned strategies from
run_matrix_benchmark.py's sweep overlaid as points (mean +/- std across
seeds), plus a separate regret-vs-alpha panel.

Reads run_matrix_benchmark.py's output tree:

    output/benchmark_results.json      final (p, q, regret) per algo|alpha|seed
    output/curves/alpha=<a>.json       per-step metrics per algo|seed|step

    python plot_ne_curve.py                          # NE/regret vs alpha, plus
                                                     # one metrics-vs-step
                                                     # figure per alpha
    python plot_ne_curve.py --curve-alphas 0.5 1.0 --logy
    python plot_ne_curve.py --no-curves              # skip the per-step figures
    python plot_ne_curve.py --output-dir runs/ablation
    python plot_ne_curve.py --results other_run.json
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from envs.matrix_game import COMMON_INTEREST_ALPHA
from eval.ground_truth import (
    ALPHA_BREAK_1,
    ALPHA_BREAK_2,
    ground_truth_curve,
    ground_truth_pq,
    regime,
)


ALGO_STYLE = {
    "npgac": dict(color="#d62728", marker="o", label="N-PGAC"),
    "mappo": dict(color="#1f77b4", marker="s", label="MAPPO"),
}


# --------------------------------------------------------------------------- #
# summary results (one point per algo|alpha|seed)
# --------------------------------------------------------------------------- #

def load_and_aggregate(path):
    with open(path) as f:
        raw = json.load(f)

    # group by (algo, alpha) -> list of cell dicts across seeds
    grouped = defaultdict(list)
    for cell in raw.values():
        grouped[(cell["algo"], cell["alpha"])].append(cell)

    agg = defaultdict(dict)
    for (algo, alpha), cells in grouped.items():
        p = np.array([c["p"] for c in cells])
        q = np.array([c["q"] for c in cells])
        r = np.array([c["regret"] for c in cells])
        agg[algo][alpha] = {
            "p_mean": p.mean(), "p_std": p.std(),
            "q_mean": q.mean(), "q_std": q.std(),
            "regret_mean": r.mean(), "regret_std": r.std(),
            "n_seeds": len(cells),
        }
    return agg


def split_common_interest(agg):
    """agg, as returned by load_and_aggregate, mixes the alpha in [0, 1]
    family with the COMMON_INTEREST_ALPHA (-1) sentinel cell(s) under one
    dict. Returns (agg_alpha_family, agg_common_interest): the sentinel
    can't sit on plot_strategy_curve/plot_regret_curve's 0..1 alpha axis
    next to a ground-truth NE curve that isn't defined for it (see
    eval.ground_truth.ground_truth_pq's docstring -- multiple valid NE,
    no single target), so it must never reach those two functions. Use
    agg_common_interest with print_common_interest_summary instead.
    """
    agg_alpha_family = defaultdict(dict)
    agg_common_interest = defaultdict(dict)
    for algo, by_alpha in agg.items():
        for alpha, stats in by_alpha.items():
            target = agg_common_interest if alpha == COMMON_INTEREST_ALPHA else agg_alpha_family
            target[algo][alpha] = stats
    return agg_alpha_family, agg_common_interest


def print_common_interest_summary(agg_common_interest):
    """Console summary (mean +/- std over seeds) for the common-interest
    sentinel cell, analogous to what the alpha-vs-NE figure shows visually
    for the rest of the sweep but has no plot of its own here -- the real
    per-step signal for this cell (does lambda_i actually approach ~1) is
    in the lambda/k_hat-vs-step figure that make_all_curve_figures already
    produces for every alpha, sentinel included, via the existing
    per-step-curve machinery below (nothing algo-specific needed there).
    """
    if not agg_common_interest:
        return
    print("\ncommon-interest sanity check (alpha = "
          f"{COMMON_INTEREST_ALPHA:g}), final (p, q, regret), mean +/- std over seeds:")
    for algo, by_alpha in agg_common_interest.items():
        stats = by_alpha[COMMON_INTEREST_ALPHA]
        print(
            f"  {algo:6s}: p={stats['p_mean']:.4f}+/-{stats['p_std']:.4f}  "
            f"q={stats['q_mean']:.4f}+/-{stats['q_std']:.4f}  "
            f"regret={stats['regret_mean']:.4f}+/-{stats['regret_std']:.4f}  "
            f"(n={stats['n_seeds']})"
        )
    print("  (see figures/lambda_khat_alpha=-1.png for whether lambda_i "
          "actually approaches ~1 over training)")


def plot_strategy_curve(agg, ax, coord, coord_label):
    """coord: 'p' or 'q' -- which player's strategy to plot against alpha."""
    alphas_gt, p_gt, q_gt = ground_truth_curve(500)
    gt_curve = p_gt if coord == "p" else q_gt

    region1 = alphas_gt <= ALPHA_BREAK_1
    region2 = (alphas_gt > ALPHA_BREAK_1) & (alphas_gt <= ALPHA_BREAK_2)
    region3 = alphas_gt > ALPHA_BREAK_2
    for region in (region1, region2, region3):
        ax.plot(alphas_gt[region], gt_curve[region], color="black", linewidth=2,
                 zorder=1, label="_nolegend_")
    ax.plot([], [], color="black", linewidth=2, label="Ground truth NE")

    for algo, style in ALGO_STYLE.items():
        if algo not in agg:
            continue
        alphas = sorted(agg[algo].keys())
        means = [agg[algo][a][f"{coord}_mean"] for a in alphas]
        stds = [agg[algo][a][f"{coord}_std"] for a in alphas]
        ax.errorbar(
            alphas, means, yerr=stds, fmt=style["marker"], color=style["color"],
            label=style["label"], markersize=6, capsize=3, zorder=2,
            linestyle="none",
        )

    ax.axvline(ALPHA_BREAK_1, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.axvline(ALPHA_BREAK_2, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(coord_label)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.02, 1.02)


def plot_regret_curve(agg, ax):
    for algo, style in ALGO_STYLE.items():
        if algo not in agg:
            continue
        alphas = sorted(agg[algo].keys())
        means = [agg[algo][a]["regret_mean"] for a in alphas]
        stds = [agg[algo][a]["regret_std"] for a in alphas]
        ax.errorbar(
            alphas, means, yerr=stds, fmt=style["marker"] + "-", color=style["color"],
            label=style["label"], markersize=6, capsize=3, linewidth=1,
        )
    ax.axvline(ALPHA_BREAK_1, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.axvline(ALPHA_BREAK_2, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("max regret")
    ax.set_xlim(-0.02, 1.02)


def make_figure(results_path, out_path):
    agg = load_and_aggregate(results_path)
    agg, agg_common_interest = split_common_interest(agg)
    print_common_interest_summary(agg_common_interest)

    if not agg:
        # e.g. a sweep run with only --alphas -1: nothing alpha-family to
        # plot on this panel (its x-axis and ground-truth overlay don't
        # apply to the common-interest sentinel -- see
        # split_common_interest). Skip it rather than save an empty
        # 3-panel figure with legend warnings for panels with no data.
        print("no alpha-family (non-common-interest) results to plot "
              f"the NE-curve summary from; skipping {out_path}")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    plot_strategy_curve(agg, axes[0], "p", r"$p$ = P(player A plays A2)")
    plot_strategy_curve(agg, axes[1], "q", r"$q$ = P(player B plays B2)")
    plot_regret_curve(agg, axes[2])

    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].set_title("Player A strategy vs. ground-truth NE")
    axes[1].set_title("Player B strategy vs. ground-truth NE")
    axes[2].set_title("Max regret vs. alpha")
    axes[2].legend(loc="upper left", fontsize=9)

    fig.suptitle(
        r"N-PGAC vs. MAPPO on the $\alpha$-game family "
        "(dotted lines: NE regime boundaries at $\\alpha=1/5, 1/3$)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    print(f"saved to {out_path}")
    return fig


# --------------------------------------------------------------------------- #
# training curves (one series per algo|seed|step, one file per alpha)
# --------------------------------------------------------------------------- #

def ensure_parent(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def discover_curve_files(curves_dir, only_alphas=None):
    """Return [(alpha, path), ...] sorted by alpha."""
    found = []
    for path in glob.glob(os.path.join(curves_dir, "alpha=*.json")):
        alpha = alpha_of_curve_file(path)
        if alpha is None:
            continue
        if only_alphas is not None and not any(
            abs(alpha - a) < 1e-9 for a in only_alphas
        ):
            continue
        found.append((alpha, path))
    return sorted(found)


def alpha_of_curve_file(path):
    """Prefer the alpha stored inside the file; fall back to the filename."""
    try:
        with open(path) as f:
            raw = json.load(f)
        for entry in raw.values():
            return float(entry["alpha"])
    except (OSError, ValueError, KeyError, AttributeError):
        pass
    match = re.search(r"alpha=([-\d.eE+]+)\.json$", os.path.basename(path))
    return float(match.group(1)) if match else None


def load_curve_file(path):
    """(algo, seed) -> {step: entry}"""
    with open(path) as f:
        raw = json.load(f)
    series = defaultdict(dict)
    for entry in raw.values():
        series[(entry["algo"], entry["seed"])][entry["step"]] = entry
    return series


def aggregate_over_seeds(series, field):
    """algo -> (steps, mean, std), averaged over the steps all seeds share.

    Steps where `field` is None (or missing) for any seed's run are
    excluded: this happens for lambda_i/k_hat_i/phi_loss_* at step 0
    (before any train_step() has run) for every algo, and for those same
    fields at every step for MAPPO (which has no potential function). If a
    seed's run has None at a step, that step is not "common" across seeds
    for the purposes of this field, so it is dropped from the intersection
    -- this keeps `values.mean(axis=0)` from ever seeing a None and
    crashing/silently propagating NaN.
    """
    by_algo = defaultdict(list)
    for (algo, _seed), by_step in series.items():
        by_algo[algo].append(by_step)

    out = {}
    for algo, seed_runs in by_algo.items():
        # steps present in every seed's run AND where `field` isn't None there
        valid_per_run = [
            {s for s, entry in run.items() if entry.get(field) is not None}
            for run in seed_runs
        ]
        common = set.intersection(*valid_per_run) if valid_per_run else set()
        steps = np.array(sorted(common))
        if steps.size == 0:
            continue
        values = np.array([[run[s][field] for s in steps] for run in seed_runs])
        out[algo] = (steps, values.mean(axis=0), values.std(axis=0))
    return out


def plot_metric_vs_step(series, ax, field, ylabel, gt=None, logy=False,
                         color=None, label=None, linestyle="-"):
    """Plot one field vs. training step, averaged over seeds per algo.

    `color`/`label`/`linestyle` let a caller override ALGO_STYLE's
    defaults -- used by plot_lambda_khat_vs_step / plot_phi_loss_vs_step
    to draw several lines (one per agent, or one per loss term) on the
    same axes without them all colliding on the same algo color.
    """
    agg = aggregate_over_seeds(series, field)
    for algo, style in ALGO_STYLE.items():
        if algo not in agg:
            continue
        steps, mean, std = agg[algo]
        line_color = color or style["color"]
        line_label = label or style["label"]
        ax.plot(steps, mean, color=line_color, label=line_label,
                linewidth=1.5, zorder=2, linestyle=linestyle)
        ax.fill_between(steps, mean - std, mean + std, color=line_color,
                        alpha=0.15, linewidth=0, zorder=1)

    if gt is not None:
        ax.axhline(gt, color="black", linestyle="--", linewidth=1.5,
                   label="Ground truth NE", zorder=3)

    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)


def make_curve_figure(curve_path, alpha, out_path, logy=False):
    series = load_curve_file(curve_path)
    if not series:
        print(f"skipping {curve_path}: no entries")
        return None

    n_seeds = len({seed for _algo, seed in series})
    gt_p, gt_q = ground_truth_pq(alpha)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    plot_metric_vs_step(series, axes[0], "p", r"$p$ = P(player A plays A2)", gt=gt_p)
    plot_metric_vs_step(series, axes[1], "q", r"$q$ = P(player B plays B2)", gt=gt_q)
    plot_metric_vs_step(series, axes[2], "regret", "max regret", logy=logy)

    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_ylim(-0.05, 1.05)
    if not logy:
        axes[2].axhline(0, color="black", linewidth=0.8, alpha=0.4)

    axes[0].set_title("Player A strategy vs. training step")
    axes[1].set_title("Player B strategy vs. training step")
    axes[2].set_title("Max regret vs. training step")
    axes[0].legend(loc="best", fontsize=9)
    axes[2].legend(loc="best", fontsize=9)

    fig.suptitle(
        rf"Training curves at $\alpha={alpha:g}$ ({regime(alpha)}) "
        f"-- mean $\\pm$ std over {n_seeds} seed(s)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    print(f"saved to {out_path}")
    return fig


# --------------------------------------------------------------------------- #
# N-PGAC-specific diagnostics: lambda_i / k_hat_i and phi loss components
# --------------------------------------------------------------------------- #

# One color per agent index -- distinct from ALGO_STYLE's algo colors, since
# this figure is N-PGAC only (MAPPO has no lambda/k_hat/phi) and needs to
# distinguish agents, not algorithms.
AGENT_COLORS = ["#2ca02c", "#9467bd", "#8c564b", "#e377c2"]


def num_agents_in_series(series):
    """Infer how many agents were logged by checking which lambda_i keys
    exist on any entry (works whether or not entries are all None, since
    the key itself is always present per run_matrix_benchmark.py's
    metrics() -- see that file's docstring on keeping a consistent schema).
    """
    for by_step in series.values():
        for entry in by_step.values():
            n = 0
            while f"lambda_{n}" in entry:
                n += 1
            if n > 0:
                return n
    return 0


def npgac_only_series(series):
    """Filter a (algo, seed) -> {step: entry} series down to N-PGAC's
    entries only -- the lambda/k_hat/phi figures are meaningless for MAPPO
    (always None there) so there is nothing useful to plot for it.
    """
    return {k: v for k, v in series.items() if k[0] == "npgac"}


def plot_lambda_khat_vs_step(series, alpha, out_path):
    """One figure, two panels: lambda_i(step) and k_hat_i(step), one line
    per agent, mean +/- std over seeds. N-PGAC only (see
    npgac_only_series's docstring) -- if no N-PGAC runs are present for
    this alpha, nothing is written.
    """
    npgac_series = npgac_only_series(series)
    if not npgac_series:
        return None
    n_agents = num_agents_in_series(npgac_series)
    if n_agents == 0:
        # e.g. only step-0 entries logged, where lambda/k_hat are always
        # None by construction -- nothing meaningful to plot yet.
        return None

    n_seeds = len({seed for _algo, seed in npgac_series})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for i in range(n_agents):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        plot_metric_vs_step(
            npgac_series, axes[0], f"lambda_{i}", r"$\lambda_i$",
            color=color, label=f"agent {i}",
        )
        plot_metric_vs_step(
            npgac_series, axes[1], f"k_hat_{i}", r"$\hat{k}_i$",
            color=color, label=f"agent {i}",
        )

    axes[0].set_ylim(-0.05, 1.05)
    axes[0].axhline(0, color="black", linewidth=0.8, alpha=0.3)
    axes[0].axhline(1, color="black", linewidth=0.8, alpha=0.3)
    axes[1].axhline(0, color="black", linewidth=0.8, alpha=0.3)

    axes[0].set_title(r"Trust weight $\lambda_i$ vs. training step")
    axes[1].set_title(r"Closeness estimate $\hat{k}_i$ vs. training step")
    axes[0].legend(loc="best", fontsize=9)
    axes[1].legend(loc="best", fontsize=9)

    fig.suptitle(
        rf"N-PGAC trust gate at $\alpha={alpha:g}$ ({regime(alpha)}) "
        f"-- mean $\\pm$ std over {n_seeds} seed(s)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    print(f"saved to {out_path}")
    return fig


def plot_phi_loss_vs_step(series, alpha, out_path, logy=True):
    """One figure: phi_loss_td, phi_loss_naive, beta*phi_loss_naive, and
    phi_loss_total vs. training step, mean +/- std over seeds. N-PGAC
    only. logy defaults to True since these losses typically span more
    than one order of magnitude over training (unlike strategy/regret,
    which are bounded in [0,1] and don't usually need it).

    beta*phi_loss_naive is plotted alongside the raw phi_loss_naive
    because phi_loss_total = phi_loss_td + beta*phi_loss_naive (see
    algos/npgac/trainer.py's train_step): with beta far from 1
    (NPGACConfig's default is 5.0), the raw, unweighted naive-loss line
    can look small relative to loss_td/loss_total while still being what
    actually dominates phi's gradient -- plotting only the raw value would
    misrepresent which term is driving training.
    """
    npgac_series = npgac_only_series(series)
    if not npgac_series:
        return None

    # Derive the weighted term once, per (algo, seed) series, before
    # aggregating -- aggregate_over_seeds only knows how to average an
    # existing field, not compute a new one.
    for by_step in npgac_series.values():
        for entry in by_step.values():
            naive = entry.get("phi_loss_naive")
            beta = entry.get("phi_beta")
            entry["phi_loss_naive_weighted"] = (
                naive * beta if naive is not None and beta is not None else None
            )

    n_seeds = len({seed for _algo, seed in npgac_series})
    loss_fields = [
        ("phi_loss_td", "#d62728", r"$\mathcal{L}_\phi^{TD}$"),
        ("phi_loss_naive", "#1f77b4", r"$\mathcal{L}_\phi^{naive}$ (raw)"),
        ("phi_loss_naive_weighted", "#9467bd", r"$\beta\mathcal{L}_\phi^{naive}$ (weighted)"),
        ("phi_loss_total", "#2ca02c", r"$\mathcal{L}_\phi$ (total)"),
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    any_plotted = False
    for field, color, label in loss_fields:
        agg = aggregate_over_seeds(npgac_series, field)
        if "npgac" not in agg:
            continue
        any_plotted = True
        steps, mean, std = agg["npgac"]
        ax.plot(steps, mean, color=color, label=label, linewidth=1.5)
        ax.fill_between(steps, mean - std, mean + std, color=color,
                        alpha=0.15, linewidth=0)
    if not any_plotted:
        plt.close(fig)
        return None

    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title(r"$\phi$ loss components vs. training step")
    ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        rf"N-PGAC potential loss at $\alpha={alpha:g}$ ({regime(alpha)}) "
        f"-- mean $\\pm$ std over {n_seeds} seed(s)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    print(f"saved to {out_path}")
    return fig


def make_all_curve_figures(curves_dir, figures_dir, only_alphas=None, logy=False,
                            no_diagnostics=False):
    files = discover_curve_files(curves_dir, only_alphas)
    if not files:
        if not os.path.isdir(curves_dir):
            print(
                f"no per-step metrics: {curves_dir} does not exist.\n"
                "  These are written by run_matrix_benchmark.py; re-run the sweep\n"
                "  to fill them in (cells whose curves are missing are re-run)."
            )
        elif only_alphas is not None:
            available = [a for a, _ in discover_curve_files(curves_dir)]
            print(
                f"no curve files in {curves_dir} for alphas {only_alphas}.\n"
                f"  available: {available}"
            )
        else:
            print(f"no curve files found in {curves_dir}")
        return

    for alpha, path in files:
        out_path = os.path.join(figures_dir, f"curve_alpha={alpha:g}.png")
        fig = make_curve_figure(path, alpha, out_path, logy=logy)
        if fig is not None:
            plt.close(fig)

        if no_diagnostics:
            continue

        series = load_curve_file(path)
        lam_path = os.path.join(figures_dir, f"lambda_khat_alpha={alpha:g}.png")
        fig_lam = plot_lambda_khat_vs_step(series, alpha, lam_path)
        if fig_lam is not None:
            plt.close(fig_lam)

        phi_path = os.path.join(figures_dir, f"phi_loss_alpha={alpha:g}.png")
        fig_phi = plot_phi_loss_vs_step(series, alpha, phi_path)
        if fig_phi is not None:
            plt.close(fig_phi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="output",
                        help="run_matrix_benchmark.py's output directory")
    parser.add_argument("--results", type=str, default=None,
                        help="override the summary json path")
    parser.add_argument("--curves-dir", type=str, default=None,
                        help="override the per-alpha curve directory")
    parser.add_argument("--out", type=str, default=None,
                        help="override the NE-curve figure path")
    parser.add_argument("--figures-dir", type=str, default=None,
                        help="where figures are written (default <output-dir>/figures)")
    parser.add_argument("--no-curves", action="store_true",
                        help="skip the per-alpha metrics-vs-training-step figures")
    parser.add_argument("--no-diagnostics", action="store_true",
                        help="skip the N-PGAC-only lambda/k_hat and phi-loss figures")
    parser.add_argument("--curve-alphas", type=float, nargs="+", default=None,
                        help="restrict --curves to these alphas (default: all found)")
    parser.add_argument("--logy", action="store_true",
                        help="log scale on the regret-vs-step axis")
    args = parser.parse_args()

    results_path = args.results or os.path.join(args.output_dir, "benchmark_results.json")
    curves_dir = args.curves_dir or os.path.join(args.output_dir, "curves")
    figures_dir = args.figures_dir or os.path.join(args.output_dir, "figures")
    out_path = args.out or os.path.join(figures_dir, "ne_curve.png")

    make_figure(results_path, out_path)

    if not args.no_curves:
        make_all_curve_figures(
            curves_dir, figures_dir,
            only_alphas=args.curve_alphas, logy=args.logy,
            no_diagnostics=args.no_diagnostics,
        )