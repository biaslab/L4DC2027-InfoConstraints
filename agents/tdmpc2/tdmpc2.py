"""TD-MPC2 (Hansen, Su & Wang, ICLR 2024) — single-task, state-based.

Faithful compact reimplementation of the algorithm in
literature/ICLR-2024-td-mpc2-scalable-robust-world-models-for-continuous-
control-Paper-Conference.pdf, sized for low-dimensional control:

  * implicit world model: encoder h, latent dynamics d, reward R,
    Q-ensemble (5), policy prior p — MLPs with LayerNorm + Mish,
  * SimNorm simplicial latent normalization,
  * discrete regression (two-hot, 101 bins) in symlog space for reward
    and value, cross-entropy losses,
  * joint-embedding consistency loss with temporal weighting lambda^t,
  * EMA target Q-ensemble, TD-targets via min of two subsampled targets,
  * MPPI planner (512 samples, 24 policy-prior samples, 64 elites,
    temperature 0.5, 6 iterations, horizon 3) with warm-started mean.

Hyperparameters follow Table 8 of the paper unless noted; architecture
sizes are reduced (latent 128, MLP 256) to match the 7-D observation of
the thermal-stage task.

The agent interface is:
    a = agent.act(obs, eval_mode=False)   # a in [-1, 1]^{D_a}
    agent.store(obs, a, reward, next_obs)  # also triggers 1 update step
    agent.save(path) / TDMPC2.load(path)   # weights + replay buffer
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TDMPC2Config:
    obs_dim: int = 7
    act_dim: int = 2

    # Architecture
    latent_dim: int = 128
    mlp_dim: int = 256
    num_q: int = 5
    num_bins: int = 101
    vmin: float = -10.0            # bin support in symlog space
    vmax: float = +10.0
    simnorm_dim: int = 8           # V (group size)
    dropout: float = 0.01          # Q-function dropout

    # Planning (MPPI)
    horizon: int = 3
    mppi_iterations: int = 6
    num_samples: int = 512
    num_pi_trajs: int = 24
    num_elites: int = 64
    min_std: float = 0.05
    max_std: float = 2.0
    temperature: float = 0.5

    # Optimization
    batch_size: int = 256
    lr: float = 3e-4
    enc_lr_scale: float = 0.3
    grad_clip_norm: float = 20.0
    discount: float = 0.995
    rho: float = 0.5               # temporal coefficient (lambda)
    consistency_coef: float = 20.0
    reward_coef: float = 0.1
    value_coef: float = 0.1
    entropy_coef: float = 1e-4
    tau: float = 0.01              # EMA rate (momentum 0.99)
    log_std_min: float = -10.0
    log_std_max: float = 2.0

    # Interaction
    buffer_capacity: int = 1_000_000
    seed_steps: int = 1000
    seed_action_scale: float = 1.0  # uniform seed actions in +/- this
    updates_per_step: int = 1      # update-to-data ratio
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Discrete regression helpers (symlog two-hot, paper Sec. 3.1)
# ---------------------------------------------------------------------------

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x: torch.Tensor, cfg: TDMPC2Config) -> torch.Tensor:
    """Two-hot encoding of scalar targets in symlog space. x: (B,)."""
    x = symlog(x).clamp(cfg.vmin, cfg.vmax)
    bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    idx = (x - cfg.vmin) / bin_size
    idx_low = idx.floor().long().clamp(0, cfg.num_bins - 1)
    idx_high = (idx_low + 1).clamp(0, cfg.num_bins - 1)
    w_high = idx - idx_low.float()
    target = torch.zeros(x.shape[0], cfg.num_bins, device=x.device)
    target.scatter_(1, idx_low.unsqueeze(1), (1 - w_high).unsqueeze(1))
    target.scatter_add_(1, idx_high.unsqueeze(1), w_high.unsqueeze(1))
    return target


def from_logits(logits: torch.Tensor, cfg: TDMPC2Config) -> torch.Tensor:
    """Expected value of the discrete distribution, mapped back with
    symexp. logits: (..., num_bins) -> (...)."""
    bins = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins,
                          device=logits.device)
    return symexp((logits.softmax(-1) * bins).sum(-1))


def soft_ce(logits: torch.Tensor, target_scalar: torch.Tensor,
            cfg: TDMPC2Config) -> torch.Tensor:
    """Cross-entropy against the two-hot encoded scalar target. (B,)."""
    target = two_hot(target_scalar, cfg)
    return -(target * logits.log_softmax(-1)).sum(-1)


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class SimNorm(nn.Module):
    """Simplicial normalization (paper Eq. 5)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.view(*shape[:-1], -1, self.dim)
        x = x.softmax(-1)
        return x.view(*shape)


def mlp(in_dim: int, hidden: int, out_dim: int, n_hidden: int = 2,
        out_act: Optional[nn.Module] = None, dropout: float = 0.0
        ) -> nn.Sequential:
    """MLP with LayerNorm + Mish intermediate layers (paper Sec. 3.1)."""
    layers = []
    d = in_dim
    for i in range(n_hidden):
        layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.Mish()]
        if dropout > 0 and i == 0:
            layers.append(nn.Dropout(dropout))
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class WorldModel(nn.Module):
    def __init__(self, cfg: TDMPC2Config):
        super().__init__()
        self.cfg = cfg
        Z, A, H = cfg.latent_dim, cfg.act_dim, cfg.mlp_dim
        self.encoder = mlp(cfg.obs_dim, H, Z, out_act=SimNorm(cfg.simnorm_dim))
        self.dynamics = mlp(Z + A, H, Z, out_act=SimNorm(cfg.simnorm_dim))
        self.reward = mlp(Z + A, H, cfg.num_bins)
        self.qs = nn.ModuleList([
            mlp(Z + A, H, cfg.num_bins, dropout=cfg.dropout)
            for _ in range(cfg.num_q)])
        self.policy = mlp(Z, H, 2 * A)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def next(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.dynamics(torch.cat([z, a], -1))

    def reward_pred(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.reward(torch.cat([z, a], -1))

    def pi(self, z: torch.Tensor, eval_mode: bool = False):
        """Squashed-Gaussian policy prior. Returns (action, log_prob)."""
        cfg = self.cfg
        mu, log_std = self.policy(z).chunk(2, -1)
        log_std = cfg.log_std_min + 0.5 * (cfg.log_std_max - cfg.log_std_min) \
            * (torch.tanh(log_std) + 1)
        if eval_mode:
            return torch.tanh(mu), None
        eps = torch.randn_like(mu)
        pre_tanh = mu + eps * log_std.exp()
        a = torch.tanh(pre_tanh)
        # log-prob with tanh correction
        log_prob = (-0.5 * (eps ** 2) - log_std
                    - 0.5 * np.log(2 * np.pi)).sum(-1)
        log_prob = log_prob - torch.log(1 - a ** 2 + 1e-6).sum(-1)
        return a, log_prob

    def q_logits(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Stacked logits of all Q heads: (num_q, B, num_bins)."""
        za = torch.cat([z, a], -1)
        return torch.stack([q(za) for q in self.qs])

    def q_value(self, z: torch.Tensor, a: torch.Tensor,
                target_model: Optional["WorldModel"] = None,
                mode: str = "avg") -> torch.Tensor:
        """Scalar Q estimate. mode 'min2': min of two random heads
        (TD-targets); 'avg': ensemble mean (planning/policy loss)."""
        model = target_model if target_model is not None else self
        logits = model.q_logits(z, a)
        values = from_logits(logits, self.cfg)          # (num_q, B)
        if mode == "min2":
            i, j = np.random.choice(self.cfg.num_q, 2, replace=False)
            return torch.minimum(values[i], values[j])
        return values.mean(0)


class RunningScale:
    """Moving (5%, 95%) percentile scale for the policy loss."""

    def __init__(self, tau: float = 0.01):
        self.value = 1.0
        self.tau = tau

    def update(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            q05 = torch.quantile(x.float(), 0.05)
            q95 = torch.quantile(x.float(), 0.95)
            scale = float(q95 - q05)
            self.value = (1 - self.tau) * self.value + self.tau * max(scale, 1.0)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.value


# ---------------------------------------------------------------------------
# Replay buffer (sequence sampling, episode-boundary aware)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, cfg: TDMPC2Config):
        self.cfg = cfg
        cap = cfg.buffer_capacity
        self.obs = np.zeros((cap, cfg.obs_dim), dtype=np.float32)
        self.act = np.zeros((cap, cfg.act_dim), dtype=np.float32)
        self.rew = np.zeros(cap, dtype=np.float32)
        self.next_obs = np.zeros((cap, cfg.obs_dim), dtype=np.float32)
        self.episode = np.zeros(cap, dtype=np.int64)
        self.idx = 0
        self.full = False
        self._episode_id = 0

    def __len__(self) -> int:
        return self.cfg.buffer_capacity if self.full else self.idx

    def new_episode(self) -> None:
        self._episode_id += 1

    def add(self, obs, act, rew, next_obs) -> None:
        i = self.idx
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.episode[i] = self._episode_id
        self.idx = (self.idx + 1) % self.cfg.buffer_capacity
        if self.idx == 0:
            self.full = True

    def sample_seq(self, batch_size: int, horizon: int, device: str):
        """Sample sequences of `horizon` consecutive transitions that do
        not cross an episode boundary. Returns tensors:
        obs (H+1, B, D_o), act (H, B, D_a), rew (H, B)."""
        n = len(self)
        starts = np.random.randint(0, n - horizon, size=2 * batch_size)
        # Reject sequences crossing an episode boundary.
        ok = (self.episode[starts] == self.episode[starts + horizon - 1])
        starts = starts[ok][:batch_size]
        while starts.shape[0] < batch_size:
            extra = np.random.randint(0, n - horizon,
                                      size=2 * batch_size)
            ok = (self.episode[extra] == self.episode[extra + horizon - 1])
            starts = np.concatenate([starts, extra[ok]])[:batch_size]
        idx = starts[None, :] + np.arange(horizon)[:, None]   # (H, B)
        obs = torch.as_tensor(self.obs[idx], device=device)
        act = torch.as_tensor(self.act[idx], device=device)
        rew = torch.as_tensor(self.rew[idx], device=device)
        next_last = torch.as_tensor(self.next_obs[idx[-1]],
                                    device=device)
        obs = torch.cat([obs, next_last.unsqueeze(0)], 0)     # (H+1, B, D)
        return obs, act, rew


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TDMPC2:
    def __init__(self, cfg: TDMPC2Config):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.model = WorldModel(cfg).to(self.device)
        self.target = WorldModel(cfg).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)

        enc_params = list(self.model.encoder.parameters())
        other_params = [p for n, p in self.model.named_parameters()
                        if not n.startswith("encoder.")
                        and not n.startswith("policy.")]
        self.optim = torch.optim.Adam([
            {"params": enc_params, "lr": cfg.lr * cfg.enc_lr_scale},
            {"params": other_params, "lr": cfg.lr},
        ])
        self.pi_optim = torch.optim.Adam(
            self.model.policy.parameters(), lr=cfg.lr)

        self.buffer = ReplayBuffer(cfg)
        self.scale = RunningScale()
        self.step_count = 0
        self._prev_mean: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Acting (MPPI planning, paper Sec. 3.2)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(self, obs: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        cfg = self.cfg
        if self.step_count < cfg.seed_steps and not eval_mode:
            s = cfg.seed_action_scale
            return np.random.uniform(-s, s, cfg.act_dim).astype(np.float32)

        obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        z0 = self.model.encode(obs_t)                        # (1, Z)
        H, A = cfg.horizon, cfg.act_dim

        # Policy-prior trajectories
        pi_actions = torch.empty(H, cfg.num_pi_trajs, A, device=self.device)
        z = z0.repeat(cfg.num_pi_trajs, 1)
        for t in range(H):
            a, _ = self.model.pi(z)
            pi_actions[t] = a
            z = self.model.next(z, a)

        # MPPI over the remaining samples
        mean = torch.zeros(H, A, device=self.device)
        if self._prev_mean is not None:
            mean[:-1] = self._prev_mean[1:]
        std = cfg.max_std * torch.ones(H, A, device=self.device)

        for _ in range(cfg.mppi_iterations):
            n_rand = cfg.num_samples - cfg.num_pi_trajs
            noise = torch.randn(H, n_rand, A, device=self.device)
            actions = (mean.unsqueeze(1) + std.unsqueeze(1) * noise
                       ).clamp(-1, 1)
            actions = torch.cat([actions, pi_actions], dim=1)  # (H, N, A)

            # Evaluate returns
            z = z0.repeat(actions.shape[1], 1)
            G = torch.zeros(actions.shape[1], device=self.device)
            disc = 1.0
            for t in range(H):
                G = G + disc * from_logits(
                    self.model.reward_pred(z, actions[t]), cfg)
                z = self.model.next(z, actions[t])
                disc *= cfg.discount
            a_term, _ = self.model.pi(z)
            G = G + disc * self.model.q_value(z, a_term, mode="avg")

            # Elite refit
            elite_idx = torch.topk(G, cfg.num_elites).indices
            elite_G = G[elite_idx]
            elite_actions = actions[:, elite_idx]
            w = torch.softmax(cfg.temperature * (
                elite_G - elite_G.max()), dim=0)
            mean = (w[None, :, None] * elite_actions).sum(1)
            var = (w[None, :, None]
                   * (elite_actions - mean.unsqueeze(1)) ** 2).sum(1)
            std = var.sqrt().clamp(cfg.min_std, cfg.max_std)

        self._prev_mean = mean
        if eval_mode:
            # Deterministic: execute the refined mean action.
            return mean[0].clamp(-1, 1).cpu().numpy()
        # Sample one elite trajectory by weight; take its first action
        # plus exploration noise.
        pick = torch.multinomial(w, 1).item()
        a = elite_actions[0, pick]
        a = (a + std[0] * torch.randn_like(a)).clamp(-1, 1)
        return a.cpu().numpy()

    def reset_plan(self) -> None:
        """Forget the warm-started plan (waypoint/trial boundary)."""
        self._prev_mean = None

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def store(self, obs, act, rew, next_obs) -> None:
        self.buffer.add(obs, act, rew, next_obs)
        self.step_count += 1
        if self.step_count >= self.cfg.seed_steps:
            if self.step_count == self.cfg.seed_steps:
                # Burn-in: bring the model up to speed on seed data.
                for _ in range(self.cfg.seed_steps):
                    self.update()
            else:
                for _ in range(self.cfg.updates_per_step):
                    self.update()

    def update(self) -> dict:
        cfg = self.cfg
        obs, act, rew = self.buffer.sample_seq(
            cfg.batch_size, cfg.horizon, self.device)     # see ReplayBuffer

        # Targets
        with torch.no_grad():
            next_z = self.model.encode(obs[1:])            # (H, B, Z)
            td_targets = torch.empty(cfg.horizon, cfg.batch_size,
                                     device=self.device)
            for t in range(cfg.horizon):
                a_next, _ = self.model.pi(next_z[t])
                q_next = self.model.q_value(next_z[t], a_next,
                                            target_model=self.target,
                                            mode="min2")
                td_targets[t] = rew[t] + cfg.discount * q_next

        # Latent rollout
        z = self.model.encode(obs[0])
        consistency_loss, reward_loss, value_loss = 0.0, 0.0, 0.0
        rho_t = 1.0
        for t in range(cfg.horizon):
            r_logits = self.model.reward_pred(z, act[t])
            q_logits = self.model.q_logits(z, act[t])      # (num_q, B, bins)
            z = self.model.next(z, act[t])
            consistency_loss = consistency_loss + rho_t * F.mse_loss(
                z, next_z[t].detach())
            reward_loss = reward_loss + rho_t * soft_ce(
                r_logits, rew[t], cfg).mean()
            value_loss = value_loss + rho_t * sum(
                soft_ce(q_logits[k], td_targets[t], cfg).mean()
                for k in range(cfg.num_q)) / cfg.num_q
            rho_t *= cfg.rho

        total = (cfg.consistency_coef * consistency_loss
                 + cfg.reward_coef * reward_loss
                 + cfg.value_coef * value_loss)
        self.optim.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       cfg.grad_clip_norm)
        self.optim.step()

        # Policy prior update (on the rolled-out latents, grads to pi only)
        with torch.no_grad():
            zs = [self.model.encode(obs[0])]
            for t in range(cfg.horizon - 1):
                zs.append(self.model.next(zs[-1], act[t]))
        pi_loss, rho_t = 0.0, 1.0
        for t, z_t in enumerate(zs):
            a_pi, log_pi = self.model.pi(z_t)
            q_pi = self.model.q_value(z_t, a_pi, mode="avg")
            self.scale.update(q_pi)
            pi_loss = pi_loss + rho_t * (
                cfg.entropy_coef * log_pi - self.scale(q_pi)).mean()
            rho_t *= cfg.rho
        self.pi_optim.zero_grad(set_to_none=True)
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(),
                                       cfg.grad_clip_norm)
        self.pi_optim.step()

        # EMA target update
        with torch.no_grad():
            for p, tp in zip(self.model.parameters(),
                             self.target.parameters()):
                tp.lerp_(p, cfg.tau)

        return {"consistency": float(consistency_loss.detach()),
                "reward": float(reward_loss.detach()),
                "value": float(value_loss.detach()),
                "pi": float(pi_loss.detach())}

    # ------------------------------------------------------------------
    # Persistence (weights + replay buffer carried across phases)
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        path = Path(path)
        n = len(self.buffer)
        torch.save({
            "cfg": asdict(self.cfg),
            "model": self.model.state_dict(),
            "target": self.target.state_dict(),
            "optim": self.optim.state_dict(),
            "pi_optim": self.pi_optim.state_dict(),
            "scale": self.scale.value,
            "step_count": self.step_count,
            "buffer": {
                "obs": self.buffer.obs[:n],
                "act": self.buffer.act[:n],
                "rew": self.buffer.rew[:n],
                "next_obs": self.buffer.next_obs[:n],
                "episode": self.buffer.episode[:n],
                "episode_id": self.buffer._episode_id,
            },
        }, str(path))

    @classmethod
    def load(cls, path, device: Optional[str] = None) -> "TDMPC2":
        ckpt = torch.load(str(path), map_location=device or "cpu",
                          weights_only=False)
        cfg = TDMPC2Config(**ckpt["cfg"])
        if device is not None:
            cfg.device = device
        agent = cls(cfg)
        agent.model.load_state_dict(ckpt["model"])
        agent.target.load_state_dict(ckpt["target"])
        agent.optim.load_state_dict(ckpt["optim"])
        agent.pi_optim.load_state_dict(ckpt["pi_optim"])
        agent.scale.value = ckpt["scale"]
        agent.step_count = ckpt["step_count"]
        buf = ckpt["buffer"]
        n = buf["obs"].shape[0]
        agent.buffer.obs[:n] = buf["obs"]
        agent.buffer.act[:n] = buf["act"]
        agent.buffer.rew[:n] = buf["rew"]
        agent.buffer.next_obs[:n] = buf["next_obs"]
        agent.buffer.episode[:n] = buf["episode"]
        agent.buffer.idx = n % cfg.buffer_capacity
        agent.buffer.full = n >= cfg.buffer_capacity
        agent.buffer._episode_id = buf["episode_id"]
        return agent
