"""Pretrain TD-MPC2 on the nominal arm.

The ARX agents start from a prior fitted to excitation data of the nominal
arm, i.e., the arm with its uncertain parameters at their prior means.
TD-MPC2 cannot learn from the 90 effective observations of that prior, so
it trains on the same nominal arm for many episodes instead: 48 s episodes
from the start configuration, with a random end-effector target in the
workspace every 3 s, the torque limit and control period of the comparison
settings, and exploration noise. The random actions of the seed phase have
amplitude 0.3, as in the thermal-stage experiments, because full-torque
random actions spin the arm through several turns. The checkpoints include the replay buffer,
so that the agent can keep learning online during the evaluation.

Run in an environment with torch:

    python experiment-arm/train_tdmpc2.py --steps 20000 --checkpoints 5000,10000,20000

``--resume`` continues from a checkpoint up to ``--steps`` steps in total.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

import numpy as np  # noqa: E402

from controllers import joint_error, tdmpc2_obs, tdmpc2_reward  # noqa: E402
import reaching as R  # noqa: E402
from agents.common import ActionScaler  # noqa: E402
from envs.robot_arm import ArmModel, ArmParams, RobotArm, inverse_kinematics  # noqa: E402


def random_joint_target(rng, params, r_min=0.3, r_max=0.9):
    """Elbow-up joint angles of an end-effector target drawn uniformly from
    the annulus r_min <= |y| <= r_max."""
    r = np.sqrt(rng.uniform(r_min**2, r_max**2))
    phi = rng.uniform(-np.pi, np.pi)
    return inverse_kinematics(np.array([r * np.cos(phi), r * np.sin(phi)]), params, sign=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--checkpoints", type=str, default="5000,10000,20000")
    parser.add_argument("--out", type=Path, default=HERE / "results" / "tdmpc2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    from agents.tdmpc2 import TDMPC2, TDMPC2Config

    s = R.Settings()
    nominal = ArmModel(ArmParams(), R.ARM_CONFIG).params
    config = R.arm_config(s)
    scaler = ActionScaler.symmetric(config.u_max, 2)
    checkpoints = {int(c) for c in args.checkpoints.split(",") if c}
    args.out.mkdir(parents=True, exist_ok=True)
    log, step, episode, t0 = [], 0, 0, time.perf_counter()
    if args.resume is None:
        agent = TDMPC2(TDMPC2Config(obs_dim=6, act_dim=2, seed=args.seed, device="cpu",
                                    seed_action_scale=0.3))
    else:
        agent = TDMPC2.load(args.resume, device="cpu")
        step = agent.step_count
        episode = step // (16 * s.steps_per_target)
        log_file = args.out / "train_log.csv"
        if log_file.exists():
            log = [tuple(row) for row in np.atleast_2d(np.loadtxt(log_file, delimiter=",", skiprows=1))
                   if row[0] <= step]
    # A resumed run draws new targets: the stream is seeded with the start step.
    rng = np.random.default_rng(args.seed + step)
    while step < args.steps:
        plant = RobotArm(nominal, config, seed=args.seed * 100_000 + episode)
        p = plant.observe()
        y = np.r_[p, 0.0, 0.0]
        agent.buffer.new_episode()
        returns, errors = [], []
        for _ in range(16):                                   # 16 targets of 3 s
            q_goal = random_joint_target(rng, nominal)
            agent.reset_plan()
            for _ in range(s.steps_per_target):
                obs = tdmpc2_obs(y, q_goal)
                a = np.clip(np.asarray(agent.act(obs), float), -1.0, 1.0)
                for _ in range(s.substeps):
                    p_next = plant.step(scaler.to_physical(a))
                y_next = np.r_[p_next, (p_next - p) / s.dt]
                r = tdmpc2_reward(y_next, a, q_goal)
                agent.store(obs, a.astype(np.float32), r, tdmpc2_obs(y_next, q_goal))
                returns.append(r)
                errors.append(np.linalg.norm(joint_error(y_next[:2], q_goal)))
                p, y, step = p_next, y_next, step + 1
                if step in checkpoints:
                    agent.save(args.out / f"tdmpc2_{step}.pt")
                if step >= args.steps:
                    break
            if step >= args.steps:
                break
        episode += 1
        log.append((step, np.mean(returns), np.mean(errors), np.mean(np.array(errors) < 0.02)))
        print(f"episode {episode:3d}, step {step:6d}: mean reward {log[-1][1]:7.3f}, "
              f"mean joint error {log[-1][2]:.3f} rad, within 0.02 rad {log[-1][3]:.2f}, "
              f"{time.perf_counter() - t0:6.0f} s", flush=True)
        np.savetxt(args.out / "train_log.csv", np.array(log), delimiter=",",
                   header="step,mean_reward,mean_joint_error,frac_within_tol", comments="")


if __name__ == "__main__":
    main()
