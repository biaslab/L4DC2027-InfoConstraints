"""Numbers of Sec. V-B of main.tex, from the records of compare.py.

    python experiment-arm/analysis.py

The RMSE of an episode is that of the two joint angles from the joint
configuration of the current target over all steps of the four rounds; the
tables average it over the seeds. The probing weight of the dual MPC with
the Beta input cost and the accuracy of MPC-X are tuned on the held-out
seeds 100-105 (results/compare_tuning) by the mean RMSE, separately under
a known and an unknown polarity. Prints
  - the RMSE per condition for the controllers of the paper, with paired
    bootstrap 95% intervals of the differences to CBFE over the seeds;
  - the RMSE per round under unknown polarity;
  - the activity of the information constraint and the KKT multipliers;
  - the RMSE of the dual MPC over its probing weights (test seeds).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import compare as C  # noqa: E402
import reaching as R  # noqa: E402

CBFE = "CBFE-ARX, KKT β, predictive floor, next node"
CONDITIONS = ("no change", "unknown polarity")
NAMES = {"no change": "known polarity", "unknown polarity": "unknown polarity"}
BOOTSTRAP = 10_000


def episode_rows(records) -> pd.DataFrame:
    rows = []
    for rec in records:
        if rec["condition"] not in CONDITIONS:
            continue
        ep, s = rec["episode"], rec["settings"]
        n = s.steps_per_target
        q_goal = R.joint_targets(R.TARGETS, ep.params)
        target = (np.arange(len(ep.actions)) // n) % len(R.TARGETS)
        error = ep.ys[:, :2] - q_goal[target]
        per_round = error.reshape(-1, len(R.TARGETS) * n, 2)
        rows.append(dict(agent=rec["agent"], condition=rec["condition"], seed=rec["seed"],
                         rmse=float(np.sqrt(np.mean(error ** 2))),
                         rounds=np.sqrt(np.mean(per_round ** 2, axis=(1, 2))),
                         beta=ep.beta, plan_time=ep.plan_time))
    return pd.DataFrame(rows)


def tuned(tuning: pd.DataFrame, prefix: str, condition: str) -> str:
    """The label with the lowest mean RMSE on the held-out seeds."""
    sub = tuning[(tuning.condition == condition) & tuning.agent.str.startswith(prefix)]
    return sub.groupby("agent").rmse.mean().idxmin()


def interval(d: np.ndarray, rng) -> str:
    means = d[rng.integers(0, len(d), (BOOTSTRAP, len(d)))].mean(axis=1)
    return f"{d.mean():+.3f} [{np.quantile(means, 0.025):+.3f}, {np.quantile(means, 0.975):+.3f}]"


def main() -> None:
    pd.set_option("display.width", 200)
    tuning = episode_rows(C.load(C.HERE / "results" / "compare_tuning"))
    test = episode_rows(C.load())
    rng = np.random.default_rng(0)

    print("tuned on held-out seeds 100-105 (mean RMSE):")
    choices = {}
    for condition in CONDITIONS:
        for prefix in ("dual MPC, Beta cost", "MPC-X"):
            if not tuning.agent.str.startswith(prefix).any():
                continue
            label = tuned(tuning, prefix, condition)
            choices[(prefix, condition)] = label
            curve = (tuning[(tuning.condition == condition) & tuning.agent.str.startswith(prefix)]
                     .groupby("agent").rmse.mean().round(3).to_dict())
            print(f"  {NAMES[condition]:16s} {label:32s} {curve}")
    print()

    agents = {CBFE: "CBFE", "CBFE-ARX, β = 0": "cautious (γ = 0)", "CBFE-ARX, β = 1 (EFE)": "EFE (γ = 1)"}
    for (prefix, condition), label in choices.items():
        tuned_with = [NAMES[c] for (pre, c), lab in choices.items() if pre == prefix and lab == label]
        agents[label] = f"{label}, tuned with {' and '.join(tuned_with)}"
    rmse = test.pivot_table(index=["condition", "seed"], columns="agent", values="rmse")
    print("RMSE [rad], mean over 10 seeds, and paired difference to CBFE [95% interval]")
    for condition in CONDITIONS:
        print(f"  {NAMES[condition]}")
        for label, name in agents.items():
            if label not in rmse:
                continue
            values = rmse.loc[condition, label].to_numpy()
            cell = "" if label == CBFE else interval(rmse.loc[condition, CBFE].to_numpy() - values, rng)
            print(f"    {name:48s} {values.mean():.3f}  median {np.median(values):.3f}  {cell}")
    print()

    print("RMSE per round under unknown polarity [rad]")
    for label, name in agents.items():
        sub = test[(test.agent == label) & (test.condition == "unknown polarity")]
        if len(sub):
            print(f"    {name:48s}", np.round(np.mean(np.stack(sub.rounds.to_numpy()), axis=0), 3))
    print()

    for condition in CONDITIONS:
        sub = test[(test.agent == CBFE) & (test.condition == condition)]
        beta = np.stack(sub.beta.to_numpy()).reshape(len(sub), 4, -1)
        active = beta > 0
        print(f"CBFE, {NAMES[condition]}: active on {np.round(100 * active.mean(axis=(0, 2)), 1)}% of the "
              f"steps per round; first step active in {100 * active[:, 0, 0].mean():.0f}% of the seeds")
        if active.any():
            values = beta[active]
            print(f"    multipliers: median {np.median(values):.3g}, 90% quantile "
                  f"{np.quantile(values, 0.9):.3g}, max {values.max():.3g}")
        times = 1e3 * np.concatenate(sub.plan_time.to_numpy())
        print(f"    planning time per step: median {np.median(times):.1f} ms, "
              f"95% quantile {np.quantile(times, 0.95):.1f} ms")
    print()

    curve = test[test.agent.str.startswith("dual MPC, Beta cost")].copy()
    curve["rho"] = curve.agent.str.split("= ").str[-1].astype(float)
    print("dual MPC with the Beta input cost on the test seeds: mean RMSE per probing weight")
    print(curve.pivot_table(index="rho", columns="condition", values="rmse").round(3).to_string())


if __name__ == "__main__":
    main()
