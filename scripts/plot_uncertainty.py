"""Plot per-token uncertainty collected by examples/libero/collect_uncertainty.py.

Runs in the main (JAX) venv, which has matplotlib. Reads the flat columnar .npz and writes:

  <prefix>_trajectories.png   per-inference-step feature over time, one line per episode,
                              colored by success/failure -- the "temporal" view.
  <prefix>_positivity.png     histograms of alpha_min / alpha_sum: is the LogTokU top-K
                              evidence actually positive on this checkpoint? (If not, AU/EU
                              are meaningless regardless of what they plot.)
  <prefix>_pooled.png         per-episode pooled scalar (max over the episode) split by
                              outcome -- the "single threshold" view CP relies on.

Usage:
    uv run scripts/plot_uncertainty.py --npz data/libero/uncertainty/libero10_task0.npz
"""

import dataclasses
import pathlib

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro

# Features shown as time series / pooled. alpha_min & alpha_sum are diagnostics (own figure).
TRAJECTORY_FEATURES = ("entropy", "neg_logp", "au", "eu")
SUCCESS_COLOR, FAILURE_COLOR = "#2ca02c", "#d62728"


@dataclasses.dataclass
class Args:
    npz: str
    # Output prefix; defaults to the npz path without extension.
    out_prefix: str | None = None


def _per_infer_step(data, reduce):
    """Collapse tokens to one value per (episode, inference step) with `reduce` (np.mean/np.max).

    Returns {episode_id: {"x": infer_step[K], feature: value[K]}} sorted by inference step.
    """
    feats = data["features"]
    names = [str(n) for n in data["feature_names"]]
    ep = data["episode_id"]
    step = data["infer_step"]
    keys = ep.astype(np.int64) * (step.max() + 1) + step  # unique per (episode, step)
    out = {}
    for k in np.unique(keys):
        m = keys == k
        e, s = int(ep[m][0]), int(step[m][0])
        rec = out.setdefault(e, {"x": [], **{n: [] for n in names}})
        rec["x"].append(s)
        for i, n in enumerate(names):
            rec[n].append(float(reduce(feats[m, i])))
    for rec in out.values():
        order = np.argsort(rec["x"])
        for key in rec:
            rec[key] = np.asarray(rec[key])[order]
    return out, names


def plot_trajectories(data, path):
    success = data["episode_success"]
    # neg_logp is a "most surprising token" signal -> max; the rest are averaged over the chunk.
    per_step, _ = _per_infer_step(data, np.mean)
    per_step_max, _ = _per_infer_step(data, np.max)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, feat in zip(axes.ravel(), TRAJECTORY_FEATURES, strict=True):
        src = per_step_max if feat == "neg_logp" else per_step
        for e, rec in src.items():
            ok = bool(success[e])
            ax.plot(
                rec["x"],
                rec[feat],
                color=SUCCESS_COLOR if ok else FAILURE_COLOR,
                alpha=0.7,
                lw=1.3,
            )
        ax.set_title(f"{feat}  ({'max' if feat == 'neg_logp' else 'mean'} over chunk)")
        ax.set_xlabel("inference step")
        ax.grid(alpha=0.3)
    handles = [
        plt.Line2D([], [], color=SUCCESS_COLOR, label="success"),
        plt.Line2D([], [], color=FAILURE_COLOR, label="failure"),
    ]
    fig.legend(handles=handles, loc="upper right")
    fig.suptitle(f"{data['task_description']}  (task {int(data['task_id'])}, {len(success)} episodes)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_positivity(data, path):
    feats = data["features"]
    names = [str(n) for n in data["feature_names"]]
    amin = feats[:, names.index("alpha_min")]
    asum = feats[:, names.index("alpha_sum")]
    frac_bad = float(np.mean(amin <= 0.0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(amin, bins=60, color="#1f77b4")
    axes[0].axvline(0.0, color="k", ls="--")
    axes[0].set_title(f"alpha_min (smallest top-K logit)\n{frac_bad * 100:.1f}% of tokens <= 0 -> AU/EU invalid there")
    axes[0].set_xlabel("alpha_min")
    axes[1].hist(asum, bins=60, color="#1f77b4")
    axes[1].set_title("alpha_sum (total top-K evidence)")
    axes[1].set_xlabel("alpha_sum")
    for ax in axes:
        ax.set_ylabel("tokens")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return frac_bad


def plot_pooled(data, path):
    """Per-episode max of each feature, split by outcome -- can one number separate success/failure?"""
    success = data["episode_success"]
    per_step_max, names = _per_infer_step(data, np.max)
    fig, axes = plt.subplots(1, len(TRAJECTORY_FEATURES), figsize=(4 * len(TRAJECTORY_FEATURES), 4))
    for ax, feat in zip(axes, TRAJECTORY_FEATURES, strict=True):
        succ_vals, fail_vals = [], []
        for e, rec in per_step_max.items():
            (succ_vals if success[e] else fail_vals).append(float(np.max(rec[feat])))
        ax.boxplot(
            [succ_vals or [np.nan], fail_vals or [np.nan]],
            tick_labels=["success", "failure"],
            showfliers=False,
        )
        for i, vals in enumerate((succ_vals, fail_vals), start=1):
            if vals:
                ax.scatter(
                    np.full(len(vals), i) + np.random.uniform(-0.05, 0.05, len(vals)), vals, color="k", alpha=0.5, s=15
                )
        ax.set_title(f"episode max {feat}")
        ax.grid(alpha=0.3)
    fig.suptitle("Pooled scalar per episode (the view CP collapses to)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(args: Args) -> None:
    data = dict(np.load(args.npz, allow_pickle=True))
    prefix = args.out_prefix or str(pathlib.Path(args.npz).with_suffix(""))

    plot_trajectories(data, f"{prefix}_trajectories.png")
    frac_bad = plot_positivity(data, f"{prefix}_positivity.png")
    plot_pooled(data, f"{prefix}_pooled.png")

    n = len(data["episode_success"])
    n_succ = int(np.sum(data["episode_success"]))
    print(f"episodes: {n}  ({n_succ} success / {n - n_succ} failure)")
    print(
        f"tokens with alpha_min <= 0: {frac_bad * 100:.1f}%  "
        f"({'AU/EU trustworthy' if frac_bad < 0.01 else 'AU/EU suspect -- see positivity plot'})"
    )
    print(f"wrote {prefix}_trajectories.png, {prefix}_positivity.png, {prefix}_pooled.png")


if __name__ == "__main__":
    main(tyro.cli(Args))
