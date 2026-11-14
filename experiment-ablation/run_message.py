"""Validation of the message passing planner of CBFE (``agents/cbfe_quad/
messages.py``) against the gradient planner (L-BFGS-B) on the quadcopter of
``ablation.py``, at the nominal IMU noise:

    python experiment-ablation/run_message.py shadow   # data/message_shadow.pkl
    python experiment-ablation/run_message.py test     # data/message_test.pkl
    python experiment-ablation/run_message.py report

``shadow`` flies info-CBFE with the gradient planner on the test seeds
1000-1009, as ``ablation.run_job`` does, and at every step it also plans
with the message planner from the same belief and the same warm start. It
records both first inputs, multipliers, floors, information and planning
times, so that the two planners compare on identical beliefs. ``test`` flies
info-CBFE and no-info-CBFE with the message planner in closed loop on the
test seeds 1000-1099, whose G compares with the flights of the gradient
planner in data/episodes.pkl. G is the metric of main.tex: the goal cost on
the true position plus half the precision-weighted prediction error of the
filter.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import numpy as np  # noqa: E402

import ablation as A  # noqa: E402
from agents.cbfe_quad.agent import GRAVITY  # noqa: E402

NOMINAL = 2.0
SHADOW_SEEDS = range(1000, 1010)
TEST_SEEDS = range(1000, 1100)
MESSAGE = {"CBFE (messages)": dict(caution=True, constrained=True, planner="message"),
           "no constraint (messages)": dict(caution=True, constrained=False, planner="message")}
GRADIENT = {"CBFE (messages)": "CBFE", "no constraint (messages)": "no constraint"}


def shadow_job(seed: int) -> dict:
    """One flight of info-CBFE with the gradient planner, as in
    ``ablation.run_job``, with the message planner on the same beliefs."""
    bench = A.Benchmark(imu_noise=NOMINAL)
    d = A.draw(bench, seed)
    agent = A.make("CBFE", bench, d["target"], d["lean"])
    twin = A.make("CBFE", bench, d["target"], d["lean"], MESSAGE["CBFE (messages)"])
    T, dt = bench.T, bench.dt
    p, v = np.zeros((T + 1, 3)), np.zeros((T + 1, 3))
    p[0] = d["start"]
    agent.observe(position=p[0] + bench.position_noise * d["position"][0])
    gravity = np.array([0.0, 0.0, GRAVITY])
    rec = {key: [] for key in ("u", "u_mp", "multiplier", "multiplier_mp", "floor", "floor_mp", "info", "info_mp",
                               "seconds", "seconds_mp", "plans", "plans_mp", "sweeps_mp", "J", "J_mp")}
    noise, goal = agent._noise(), np.asarray(agent.config.goal, float)
    for t in range(T):
        beliefs = agent.beliefs()
        twin.mean, twin.covariance = agent.mean.copy(), agent.covariance.copy()
        twin._previous_plan = None if agent._previous_plan is None else agent._previous_plan.copy()
        start = time.perf_counter()
        rec["u_mp"].append(twin.plan())
        rec["seconds_mp"].append(time.perf_counter() - start)
        start = time.perf_counter()
        u = agent.plan()
        rec["seconds"].append(time.perf_counter() - start)
        rec["u"].append(u)
        for key, src in (("multiplier", "last_multiplier"), ("floor", "last_floor"), ("info", "last_information"),
                         ("plans", "last_plans")):
            rec[key].append(getattr(agent, src))
            rec[key + "_mp"].append(getattr(twin, src))
        rec["sweeps_mp"].append(twin.last_sweeps)
        # free energy J of each plan at its own multiplier (equal multipliers where the constraint is inactive)
        for key, planner in (("J", agent), ("J_mp", twin)):
            rec[key].append(float(agent.objective_fn(planner._previous_plan, *beliefs, goal, planner.last_multiplier,
                                                     noise)[0]))
        f = (A.specific_force(u, v[t], d["wind"][t + 1], d["k"], agent.config, d["psi"], d["efficiency"][t + 1],
                              d["bias"][t + 1]) + bench.force_noise * d["force"][t])
        a = f - gravity
        v[t + 1] = v[t] + dt * a
        p[t + 1] = p[t] + dt * v[t] + 0.5 * dt**2 * a
        agent.update(u, p[t + 1] + bench.position_noise * d["position"][t + 1], f + bench.imu_noise * d["imu"][t])
    return dict(seed=seed, **{key: np.array(value) for key, value in rec.items()})


def run_shadow(processes=None) -> list:
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    with ProcessPoolExecutor(processes, mp_context=get_context("forkserver")) as pool:
        return list(pool.map(shadow_job, SHADOW_SEEDS))


def flight_G(r: dict) -> float:
    return float(np.mean(r["true_cost"] + 0.5 * r["error"]))


def interval(d: np.ndarray) -> str:
    rng = np.random.default_rng(0)
    m = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    return f"{d.mean():+.1f} [{np.quantile(m, 0.025):+.1f}, {np.quantile(m, 0.975):+.1f}]"


def report() -> None:
    path = A.DATA / "message_shadow.pkl"
    if path.exists():
        runs = A.load(path)
        du = np.concatenate([np.max(np.abs(r["u"] - r["u_mp"]), axis=1) for r in runs])
        active, active_mp = (np.concatenate([r[k] for r in runs]) > 0.0 for k in ("multiplier", "multiplier_mp"))
        both = active & active_mp
        ratio = np.concatenate([r["multiplier_mp"] / np.where(r["multiplier"] > 0, r["multiplier"], np.nan)
                                for r in runs])[both]
        info = np.concatenate([r["info_mp"] - r["info"] for r in runs])
        floor = np.concatenate([r["floor_mp"] - r["floor"] for r in runs])
        print(f"shadow, {len(runs)} flights, {len(du)} steps on identical beliefs:")
        print(f"  first input, max |u_mp - u|: median {np.median(du):.1e}, 95% {np.quantile(du, 0.95):.1e}, "
              f"max {du.max():.1e}; below 1e-4 on {np.mean(du < 1e-4):.0%}, below 1e-3 on {np.mean(du < 1e-3):.0%}")
        if all("J" in r for r in runs):
            gap = np.concatenate([r["J_mp"] - r["J"] for r in runs])
            J = np.concatenate([r["J"] for r in runs])
            inactive = ~np.concatenate([(r["multiplier"] > 0) | (r["multiplier_mp"] > 0) for r in runs])
            apart = inactive & (du >= 1e-4)
            print(f"  inactive steps whose first inputs differ by 1e-4 or more: {apart.sum()} of {inactive.sum()}; "
                  f"there J(messages) - J(gradient) has median {np.median(gap[apart]):+.2e} and max "
                  f"{np.max(gap[apart] / np.abs(J[apart])):+.1e} relative; messages lower on "
                  f"{np.mean(gap[apart] < 0):.0%}")
        print(f"  constraint active: gradient {active.mean():.1%}, messages {active_mp.mean():.1%}, "
              f"same status on {np.mean(active == active_mp):.1%} of the steps")
        print(f"  multiplier ratio (messages / gradient) where both active: median {np.nanmedian(ratio):.4f}, "
              f"range [{np.nanmin(ratio):.3f}, {np.nanmax(ratio):.3f}]")
        print(f"  information of the first input, messages - gradient: max |.| {np.nanmax(np.abs(info)):.1e} nats;"
              f" floor: max |.| {np.nanmax(np.abs(floor)):.1e} nats")
        seconds = np.concatenate([r["seconds"] for r in runs])
        seconds_mp = np.concatenate([r["seconds_mp"] for r in runs])
        sweeps = np.concatenate([r["sweeps_mp"] for r in runs])
        plans = np.concatenate([r["plans_mp"] for r in runs])
        print(f"  time per step: gradient median {1e3 * np.median(seconds):.0f} ms, "
              f"messages median {1e3 * np.median(seconds_mp):.0f} ms (max {np.max(seconds_mp):.1f} s); "
              f"sweeps per plan: median {np.median(sweeps / np.maximum(plans, 1)):.0f}")
    path = A.DATA / "message_test.pkl"
    if path.exists():
        import plot_over_time as O
        stored = O.metric()
        runs = A.load(path)
        print(f"closed loop, test seeds, sigma_f = {NOMINAL:g}:")
        for label, gradient in GRADIENT.items():
            sel = sorted((r for r in runs if r["label"] == label), key=lambda r: r["seed"])
            if not sel:
                continue
            g_mp = np.array([flight_G(r) for r in sel])
            g_grad = np.array([stored[(gradient, NOMINAL, r["seed"])].mean() for r in sel])
            firsts = first_inputs(gradient, [r["seed"] for r in sel])
            first = np.array([np.max(np.abs(r["u"][0] - u0)) for r, u0 in zip(sel, firsts)])
            print(f"  {label}: G {g_mp.mean():.0f} (gradient {g_grad.mean():.0f}), messages - gradient "
                  f"{interval(g_mp - g_grad)}, flights within 1% of G: {np.mean(np.abs(g_mp / g_grad - 1) < 0.01):.0%},"
                  f" first input max |du| median {np.median(first):.1e}, "
                  f"ms/step median {np.median([1e3 * r['wall'] / len(r['u']) for r in sel]):.0f}")
        cbfe = {r["seed"]: flight_G(r) for r in runs if r["label"] == "CBFE (messages)"}
        cautious = {r["seed"]: flight_G(r) for r in runs if r["label"] == "no constraint (messages)"}
        seeds = sorted(set(cbfe) & set(cautious))
        if seeds:
            print(f"  with messages, no-info-CBFE - info-CBFE: "
                  f"{interval(np.array([cautious[s] - cbfe[s] for s in seeds]))}")


def first_inputs(label: str, seeds) -> list:
    """First inputs of the stored flights of the gradient planner."""
    stored = {(r["label"], r["seed"]): r["u"][0] for r in A.load(A.DATA / "episodes.pkl")
              if r["imu_noise"] == NOMINAL}
    return [stored[(label, s)] for s in seeds]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("shadow", "test", "report"))
    parser.add_argument("--processes", type=int, default=None)
    args = parser.parse_args()
    start = time.perf_counter()
    if args.mode == "shadow":
        A.save(run_shadow(args.processes), A.DATA / "message_shadow.pkl")
    elif args.mode == "test":
        jobs = [dict(label=label, spec=spec, bench=A.Benchmark(imu_noise=NOMINAL), seed=seed)
                for label, spec in MESSAGE.items() for seed in TEST_SEEDS]
        A.save(A.run_many(jobs, args.processes), A.DATA / "message_test.pkl")
    if args.mode != "report":
        print(f"{args.mode}: {time.perf_counter() - start:.0f} s")
    report()


if __name__ == "__main__":
    main()
