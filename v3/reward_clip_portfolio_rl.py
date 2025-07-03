"""
Reward Clipping Portfolio RL — v1.0
===================================
Fully‑working end‑to‑end script (weekly rebalancing, 25‑asset universe, GPU‑safe).

* Commodities: **GLD** (gold), **SLV** (silver), **USO** (crude oil)
* Broad ETFs: SPY, QQQ, DIA, IWM, IEF, TLT, UUP, XLK, XLF, XLE
* Mega‑cap stocks: AAPL, MSFT, AMZN, GOOGL, META, NVDA, TSLA,
  JPM, JNJ, UNH, BRK‑B
* FX hedge: KRW=X

Quick start
-----------
```bash
pip install yfinance pandas numpy gymnasium torch
python reward_clip_portfolio_rl.py --epochs 400 --lookback 8
```
"""
from __future__ import annotations
import argparse
import datetime as dt
from dataclasses import dataclass
from typing import List

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf

# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------


def _select_close(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame (dates × tickers) of close prices, agnostic to
    'Adj Close' vs 'Close' Yahoo column naming."""
    if isinstance(raw.columns, pd.MultiIndex):
        closes = {}
        for ticker in raw.columns.levels[0]:
            sub = raw[ticker]
            if "Adj Close" in sub.columns:
                closes[ticker] = sub["Adj Close"]
            elif "Close" in sub.columns:
                closes[ticker] = sub["Close"]
            else:
                closes[ticker] = sub.select_dtypes("number").iloc[:, 0]
        df = pd.concat(closes, axis=1)
        df.columns.name = None
    else:
        ticker = raw.columns[0] if isinstance(raw.columns[0], str) else "T0"
        if "Adj Close" in raw.columns:
            df = raw[["Adj Close"]].rename(columns={"Adj Close": ticker})
        elif "Close" in raw.columns:
            df = raw[["Close"]].rename(columns={"Close": ticker})
        else:
            df = raw.select_dtypes("number").iloc[:, [0]].rename(
                columns=lambda _: ticker)
    return df


def load_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data and return *aligned* close‑price DataFrame."""
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
    )

    df = _select_close(raw)
    df = df.dropna(axis=1, how="all")
    if df.empty:
        raise ValueError(
            "No price data downloaded; check tickers or date range.")

    first_valids = [df[col].first_valid_index() for col in df.columns]
    common_start = max(d for d in first_valids if d is not None)
    df = df.loc[common_start:].dropna()
    return df


def make_weekly_returns(df: pd.DataFrame) -> pd.DataFrame:
    wk_prices = df.resample("W-FRI").last()
    return np.log(wk_prices / wk_prices.shift(1)).dropna()

# ---------------------------------------------------------------------------
# Gym environment
# ---------------------------------------------------------------------------


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, returns: pd.DataFrame, lookback: int = 8):
        super().__init__()
        self.returns = returns.astype(np.float32)
        self.assets = returns.columns.tolist()
        self.n_assets = len(self.assets)
        self.lookback = lookback

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(lookback * self.n_assets,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        self._start = self.lookback  # first index with full window
        self._idx = self._start
        self._w = np.full(self.n_assets, 1 / self.n_assets, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self._idx = self._start
        self._w.fill(1 / self.n_assets)
        return self._obs(), {}

    def step(self, action: np.ndarray):
        # ensure weights sum to 1 (Dirichlet sample should already satisfy)
        self._w = action / action.sum()

        r = float(np.dot(self.returns.iloc[self._idx], self._w))
        self._idx += 1
        terminated = self._idx >= len(self.returns)
        return self._obs(), r, terminated, False, {}

    def _obs(self):
        window = self.returns.iloc[self._idx - self.lookback: self._idx]
        return window.values.flatten()

# ---------------------------------------------------------------------------
# Actor‑Critic network & Reward Clipping loss
# ---------------------------------------------------------------------------


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(
            ), nn.Linear(128, 128), nn.ReLU()
        )
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        z = self.backbone(x)
        return self.actor(z), self.critic(z)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class TrainCfg:
    epochs: int = 400
    gamma: float = 0.99
    lam: float = 0.95
    clip_low: float = -0.1
    lr: float = 3e-4


def gae(rewards, values, gamma, lam, device):
    adv = []
    g = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        g = delta + gamma * lam * g
        adv.insert(0, g)
    adv = torch.tensor(adv, dtype=torch.float32, device=device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # standardize
    return adv


def train(env: PortfolioEnv, cfg: TrainCfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ActorCritic(env.observation_space.shape[0], env.n_assets).to(device)
    opt = optim.Adam(net.parameters(), lr=cfg.lr)

    for ep in range(cfg.epochs):
        s, _ = env.reset()
        done = False
        states, acts, rews, vals = [], [], [], []
        while not done:
            t = torch.tensor(s, dtype=torch.float32, device=device)
            logits, v = net(t)
            dist = torch.distributions.Dirichlet(torch.softmax(logits, dim=-1))
            w = dist.sample()

            s2, r, done, *_ = env.step(w.detach().cpu().numpy())

            states.append(t)
            acts.append(w)
            rews.append(r)
            vals.append(v.squeeze())
            s = s2
        vals.append(torch.zeros((), device=device))

        vals_t = torch.stack(vals)
        adv = gae(rews, vals_t, cfg.gamma, cfg.lam, device)
        ret = adv * vals_t[:-1].std() + vals_t[:-1]  # scale back
        adv_clipped = torch.clamp_min(adv, cfg.clip_low)

        logits_b = torch.stack([net(s)[0] for s in states])
        dists = torch.distributions.Dirichlet(torch.softmax(logits_b, dim=-1))
        logp = dists.log_prob(torch.stack(acts))
        pol_loss = -(logp * adv_clipped.detach()).mean()

        val_pred = torch.stack([net(s)[1] for s in states]).squeeze()
        val_loss = 0.5 * (ret.detach() - val_pred).pow(2).mean()

        loss = pol_loss + val_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (ep + 1) % 25 == 0 or ep == 0:
            cagr = (np.exp(np.sum(rews)) ** (52 / len(rews)) - 1) * 100
            print(
                f"Ep {ep+1}/{cfg.epochs}  loss={loss.item():.4f}  CAGR={cagr:.2f}%"
            )

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cli():
    ap = argparse.ArgumentParser(
        description="Reward Clipping Portfolio RL — weekly rebalancing"
    )
    ap.add_argument("--start", default="2006-01-01",
                    help="earliest price date to download")
    ap.add_argument("--end",   default=str(dt.date.today()),
                    help="last price date (default: today)")
    ap.add_argument("--epochs",   type=int, default=400,
                    help="number of training epochs")
    ap.add_argument("--lookback", type=int, default=8,
                    help="observation window in **weeks**")
    ap.add_argument("--device", choices=["cpu", "cuda"],
                    default="cuda" if torch.cuda.is_available() else "cpu",
                    help="force CPU or CUDA")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = cli()

    TICKERS = [
        # Broad‐market & sector ETFs
        "SPY", "QQQ", "DIA", "IWM",
        "IEF", "TLT", "UUP",
        "XLK", "XLF", "XLE",
        "GLD", "SLV", "USO",
        # Mega-cap individual stocks
        "AAPL", "MSFT", "AMZN", "GOOGL", "META",
        "NVDA", "TSLA", "JPM", "JNJ", "UNH", "BRK-B",
        # FX hedge
        "KRW=X",
    ]

    # 1. Price data → weekly log returns
    prices = load_prices(TICKERS, args.start, args.end)
    returns = np.log(prices.resample("W-FRI").last()
                     / prices.resample("W-FRI").last().shift(1)).dropna()

    # 2. Environment
    env = PortfolioEnv(returns, lookback=args.lookback)

    # 3. Train
    cfg = TrainCfg(epochs=args.epochs)
    train(env, cfg)
