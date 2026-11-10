"""Panels of the result figures, from the records of compare.py in
results/compare, written to figures/:

    fig_arm_rmse.pdf       RMSE of the joint angles per round under unknown
                           polarity (main.tex, Fig. 3b)
    fig_arm_polarity.pdf   targets reached per round under unknown polarity
    fig_arm_activity.pdf   steps on which the information constraint of the
                           CBFE controller is active, per round and condition
    fig_arm_payload.pdf    targets reached per round with the payload rise
                           (main-extended.tex only)
    legend_arm.pdf         the legend of this row in main-extended.tex

The legends of the figures of main.tex come from figures/paper_style.py.

    python experiment-arm/plot_paper.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "figures")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

import compare as C  # noqa: E402
import paper_style as S  # noqa: E402
import reaching as R  # noqa: E402

FIGURES = HERE / "figures"
# labels of compare.py -> agents of paper_style. The dual MPC was tuned to
# λ = 0 on held-out seeds of the payload-rise protocol; λ = 100 = 2 γ_max is
# the probing weight of the saturated multiplier (the paper calls λ ρ).
AGENT = {
    "CBFE-ARX, KKT β": "kkt",
    "CBFE-ARX, β = 0": "gamma0",
    "CBFE-ARX, β = 1 (EFE)": "gamma1",
    "dual MPC, λ = 0": "dual",
    "dual MPC, λ = 100": "dual_100",
    "MPC-X, γ = 0.0004": "mpcx",
    "PD": "pd",
    "PILCO": "pilco",
    "TD-MPC2, 50k steps": "tdmpc2",
}
ORDER = ["tdmpc2", "pilco", "pd", "gamma0", "dual", "mpcx", "dual_100", "kkt"]
CONDITIONS = {"no change": "known polarity", f"rise to {C.NEW_PAYLOAD} kg": "payload rise",
              "unknown polarity": "unknown polarity"}
# agents of the RMSE panel
RMSE_ORDER = ["dual", "mpcx", "kkt"]
# the conditions in the activity panel (the payload rise behaves as the known polarity)
ACTIVITY_STYLE = {"known polarity": dict(marker="o", ls="-"), "unknown polarity": dict(marker="^", ls=":")}


def per_round():
    df = C.per_round([r for r in C.load() if r["agent"] in AGENT])
    df["agent"] = df.agent.map(AGENT)
    df["condition"] = df.condition.map(CONDITIONS)
    return df


def rmse_per_round(records) -> pd.DataFrame:
    """RMSE of the joint angles from the mean of the goal prior, over the
    steps of a round and the two joints, per agent, condition, seed and
    round. The goal prior sits at the joint configuration of the current
    target (reaching.joint_targets with the plant's parameters)."""
    rows = []
    for rec in records:
        if rec["agent"] not in AGENT or rec["condition"] not in ("no change", "unknown polarity"):
            continue
        ep, n = rec["episode"], rec["settings"].steps_per_target
        q_goal = R.joint_targets(R.TARGETS, ep.params)
        target = (np.arange(len(ep.actions)) // n) % len(R.TARGETS)
        error = (ep.ys[:, :2] - q_goal[target]).reshape(-1, len(R.TARGETS) * n, 2)
        rows.extend(dict(agent=AGENT[rec["agent"]], condition=CONDITIONS[rec["condition"]],
                         seed=rec["seed"], round=r, rmse=float(np.sqrt(np.mean(e ** 2))))
                    for r, e in enumerate(error, start=1))
    return pd.DataFrame(rows)


def plot_rmse(rmse, path) -> None:
    mean = rmse[rmse.condition == "unknown polarity"].pivot_table(
        index="agent", columns="round", values="rmse", aggfunc="mean")
    fig, ax = S.panel()
    for key in RMSE_ORDER:
        S.plot(ax, key, mean.columns, mean.loc[key])
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2])
    ax.set_yticklabels(["0.5", "1", "2"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set(xlabel="round", ylim=(0.4, 2.6), xticks=[1, 2, 3, 4])
    ax.tick_params(axis="y", pad=1.5)               # as in the integrator's RMSE panel
    ax.set_ylabel("RMSE [rad]", labelpad=2)
    S.save(fig, path)


CBFE = "CBFE-ARX, KKT β, predictive floor, next node"
CONDITION_STYLE = {"no change": dict(ls="-", mfc=None, label="known"),
                   "unknown polarity": dict(ls="--", mfc="white", label="unknown")}


def plot_sensitivity(records, path) -> None:
    """Episode RMSE of the joint angles of the dual MPC with the Beta input
    cost over its probing weight, relative to that of CBFE, per polarity
    condition. The weight 0 coincides with 10 and is omitted on the
    logarithmic axis."""
    rows = []
    for rec in records:
        if rec["condition"] not in CONDITION_STYLE:
            continue
        if not (rec["agent"] == CBFE or rec["agent"].startswith("dual MPC, Beta cost")):
            continue
        ep, n = rec["episode"], rec["settings"].steps_per_target
        q_goal = R.joint_targets(R.TARGETS, ep.params)
        target = (np.arange(len(ep.actions)) // n) % len(R.TARGETS)
        rows.append(dict(agent=rec["agent"], condition=rec["condition"],
                         rmse=float(np.sqrt(np.mean((ep.ys[:, :2] - q_goal[target]) ** 2))),
                         beta=ep.beta))
    df = pd.DataFrame(rows)
    mean = df.pivot_table(index="agent", columns="condition", values="rmse", aggfunc="mean")
    beta = np.concatenate(df[(df.agent == CBFE)].beta.to_numpy())
    fig, ax = S.panel(S.WIDE_PANEL_SIZE)
    S.sensitivity_axes(ax, rho_cbfe=2 * np.median(beta[beta > 0]))
    weights = [w for w in C.DUAL_WEIGHTS if w > 0]
    color = S.AGENTS["dual"]["color"]
    for condition, style in CONDITION_STYLE.items():
        style = dict(style)
        mfc = style.pop("mfc") or color
        ratio = [mean.loc[f"dual MPC, Beta cost, λ = {w:g}", condition] / mean.loc[CBFE, condition]
                 for w in weights]
        ax.plot(weights, ratio, color=color, marker="s", ms=2.8, mfc=mfc, mec=color if mfc == "white"
                else "white", mew=0.6, lw=1.0, zorder=4, **style)
    ax.legend(loc="upper center", bbox_to_anchor=(0.57, 1.0), frameon=False, handlelength=1.4,
              labelspacing=0.15, borderaxespad=0.1, fontsize=5.6, title="arm, polarity",
              title_fontsize=5.8)
    S.save(fig, path)


def plot_reached(df, condition: str, path, underlay: bool = False) -> None:
    reached = df[df.condition == condition].pivot_table(index="agent", columns="round",
                                                        values="reached", aggfunc="mean")
    fig, ax = S.panel()
    for key in ORDER:
        # under unknown polarity, gamma = 0 and the tuned dual MPC coincide;
        # the wider gamma = 0 line shows beneath
        S.plot(ax, key, reached.columns, 100 * reached.loc[key],
               **(S.UNDERLAY.get(key, {}) if underlay else {}))
    ax.set(xlabel="round", ylabel="reached [%]", ylim=(-4, 104), xticks=[1, 2, 3, 4])
    S.save(fig, path)


def plot_activity(df, path) -> None:
    kkt = df[df.agent == "kkt"].pivot_table(index="condition", columns="round", values="active",
                                           aggfunc="mean")
    fig, ax = S.panel()
    color = S.AGENTS["kkt"]["color"]
    for condition, style in ACTIVITY_STYLE.items():
        ax.plot(kkt.columns, 100 * kkt.loc[condition], color=color, ms=3.4, mec="white", mew=0.4,
                lw=1.2, label=condition, **style)
    ax.set(xlabel="round", ylabel="active [%]", ylim=(-4, 104), xticks=[1, 2, 3, 4])
    ax.legend(loc="upper right", frameon=False, handlelength=2.2, labelspacing=0.2,
              borderaxespad=0.2, fontsize=6)
    S.save(fig, path)


def main() -> None:
    S.apply()
    FIGURES.mkdir(exist_ok=True)
    records = C.load()
    plot_sensitivity(records, FIGURES / "fig_arm_sensitivity.pdf")
    rmse = rmse_per_round(records)
    plot_rmse(rmse, FIGURES / "fig_arm_rmse.pdf")
    print("RMSE of the joint angles per round [rad]:\n",
          rmse.pivot_table(index=["condition", "agent"], columns="round", values="rmse",
                           aggfunc="mean").round(3))
    df = per_round()
    plot_reached(df, "unknown polarity", FIGURES / "fig_arm_polarity.pdf", underlay=True)
    plot_reached(df, "payload rise", FIGURES / "fig_arm_payload.pdf")
    plot_activity(df, FIGURES / "fig_arm_activity.pdf")
    S.legend_strip(["kkt", "gamma0", "dual", "dual_100", "mpcx", "pd", "pilco", "tdmpc2"],
                   FIGURES / "legend_arm.pdf")
    missed = df.assign(missed=4 * (1 - df.reached)).pivot_table(
        index="agent", columns="condition", values="missed", aggfunc="sum")
    print("targets missed of 160:\n", missed.loc[ORDER[::-1]].round(0))


if __name__ == "__main__":
    main()
