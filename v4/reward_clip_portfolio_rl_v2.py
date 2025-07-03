# reward_clip_portfolio_rl_v2.py
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Literal
from dataclasses import dataclass
import random
import math
import json
import os
import datetime as dt
import argparse
import pandas as pd
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.dirichlet import Dirichlet

# from types import ModuleType

# # NumPy 2.x 에서는 `_broadcast_shape`가 사라짐 → 가짜 함수로 덮어쓴다
# if not hasattr(np.lib.stride_tricks, "_broadcast_shape"):
#     def _broadcast_shape(*args):     # 실제로는 새 API broadcast_shapes 사용
#         return np.broadcast_shapes(*args)
#     # stride_tricks가 모듈이라 setattr 가능
#     setattr(np.lib.stride_tricks, "_broadcast_shape", _broadcast_shape)

np.set_printoptions(precision=4, suppress=True)
# import vectorbt as vbt


# ─────────────────────────────────────────────────────────────────────────────
# 0. Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Data utilities
# ─────────────────────────────────────────────────────────────────────────────


def load_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=False, progress=False, group_by="ticker")
    # pick Adj Close/Close
    if isinstance(raw.columns, pd.MultiIndex):
        closes = {t: (raw[t]["Adj Close"] if "Adj Close" in raw[t]
                      else raw[t]["Close"]) for t in raw.columns.levels[0]}
        df = pd.concat(closes, axis=1)
    else:  # single
        col = "Adj Close" if "Adj Close" in raw else "Close"
        df = raw[[col]].rename(columns={col: tickers[0]})
    df = df.dropna(axis=1, how="all")
    # align on common start
    common_start = max(df[c].first_valid_index() for c in df)
    return df.loc[common_start:]


def make_features(df: pd.DataFrame, freq: Literal["M","W"]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    resampled = df.resample("ME").last() if freq == "M" else df.resample("W-FRI").last()

    ret  = np.log(resampled / resampled.shift(1))
    ema20, ema60 = resampled.ewm(span=20).mean().pct_change(), resampled.ewm(span=60).mean().pct_change()
    vol  = ret.rolling(20).std()

    # ─── Build feature matrix ──────────────────────────────────────────────
    feats = pd.concat({"ret": ret, "ema20": ema20, "ema60": ema60, "vol": vol}, axis=1)
    # columns: (feature, ticker)  →  (ticker, feature)
    feats = feats.swaplevel(0, 1, axis=1).sort_index(axis=1)
    # flatten ("AAPL_ret", "AAPL_ema20", …) so env gets a simple vector
    feats.columns = [f"{t}_{f}" for t, f in feats.columns]

    feats = feats.dropna().astype(np.float32)
    y     = ret.loc[feats.index].astype(np.float32)      # align targets
    return feats, y


# ─────────────────────────────────────────────────────────────────────────────
# 2. Gym Environment
# ─────────────────────────────────────────────────────────────────────────────


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame,
                 lookback: int = 3, eps_floor: float = -0.4,
                 reward_weights: Tuple[float, float, float] = (1.0, 0.0, 0.0)):
        super().__init__()
        self.X, self.y = X.astype(np.float32), y.astype(np.float32)
        self.n_assets = y.shape[1]
        self.lookback = lookback
        self.eps_floor = eps_floor
        self.w_ret, self.w_sharpe, self.w_anti = reward_weights

        feat_dim = self.X.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(lookback*feat_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=0., high=1., shape=(self.n_assets,), dtype=np.float32)

        self._start = lookback
        self._idx = self._start
        self._w = np.full(self.n_assets, 1/self.n_assets, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._idx = self._start
        self._w.fill(1/self.n_assets)
        return self._obs(), {}

    def step(self, action: np.ndarray):
        self._w = action / action.sum()
        r_raw = float(np.dot(self.y.iloc[self._idx], self._w))
        r_clipped = max(r_raw, self.eps_floor)

        # optional extra rewards
        sharpe = r_raw / (self.y.iloc[self._idx].std() + 1e-8)
        antibias = -np.abs(self._w - 1/self.n_assets).mean()

        total_r = (self.w_ret * r_clipped +
                   self.w_sharpe * sharpe +
                   self.w_anti * antibias)

        self._idx += 1
        term = self._idx >= len(self.y)
        return self._obs(), total_r, term, False, {}

    def _obs(self):
        win = self.X.iloc[self._idx-self.lookback:self._idx]
        return win.values.flatten()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Networks
# ─────────────────────────────────────────────────────────────────────────────


class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, n_actions))

    def forward(self, x): return self.net(x)


class ValueNet(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(),
                                 nn.Linear(256, 1))

    def forward(self, x): return self.net(x).squeeze(-1)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Training (Actor-Only or PPO)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TrainCfg:
    algo: Literal["actor", "ppo"] = "actor"
    epochs: int = 600
    gamma: float = 0.99
    lam: float = 0.95
    lr: float = 3e-4
    entropy_coef: float = 1e-2
    clip_ratio: float = 0.2


def gae(rew, val, gamma, lam, device):
    adv, g = [], 0.0
    for t in reversed(range(len(rew))):
        delta = rew[t] + gamma*val[t+1] - val[t]
        g = delta + gamma*lam*g
        adv.insert(0, g)
    adv = torch.tensor(adv, device=device)
    return (adv-adv.mean())/(adv.std()+1e-8)


def train(env: PortfolioEnv, cfg: TrainCfg, device: str):
    state_dim = env.observation_space.shape[0]
    pol = PolicyNet(state_dim, env.n_assets).to(device)
    val = ValueNet(state_dim).to(device) if cfg.algo == "ppo" else None
    params = list(pol.parameters()) + (list(val.parameters()) if val else [])
    opt = optim.Adam(params, lr=cfg.lr)

    for ep in range(cfg.epochs):
        s, _ = env.reset()
        S, A, R, V = [], [], [], []
        done = False
        while not done:
            t = torch.tensor(s, dtype=torch.float32, device=device)
            logits = pol(t)
            dist = Dirichlet(torch.softmax(logits, -1))
            w = dist.sample()
            s2, r, done, *_ = env.step(w.cpu().numpy())

            S.append(t)
            A.append(w)
            R.append(r)
            if val:
                V.append(val(t).unsqueeze(0))
            s = s2
        if val:
            V.append(torch.zeros(1, device=device))
            V_t = torch.cat(V)

        # ─── Losses
        if cfg.algo == "actor":
            # REINFORCE + reward clipping already applied
            returns = torch.tensor(R, device=device).float()
            loss = -(torch.stack([Dirichlet(torch.softmax(pol(s), -1)).log_prob(a)
                                  for s, a in zip(S, A)]) * returns).mean()
        else:  # PPO
            V_t = torch.cat(V)
            adv = gae(R, V_t, cfg.gamma, cfg.lam, device)
            ret = adv * V_t[:-1].std() + V_t[:-1]

            logits_old = torch.stack([pol(s) for s in S])
            dist_old = Dirichlet(torch.softmax(logits_old, -1))
            logp_old = dist_old.log_prob(torch.stack(A)).detach()

            logits_new = logits_old.detach().clone()  # recompute later
            # single gradient step (on-policy)
            logits_new = torch.stack([pol(s) for s in S])
            dist_new = Dirichlet(torch.softmax(logits_new, -1))
            logp_new = dist_new.log_prob(torch.stack(A))

            ratio = torch.exp(logp_new - logp_old)
            surr1 = ratio*adv
            surr2 = torch.clamp(ratio, 1-cfg.clip_ratio, 1+cfg.clip_ratio)*adv
            pol_loss = -torch.min(surr1, surr2).mean() - \
                cfg.entropy_coef*dist_new.entropy().mean()

            val_pred = torch.stack([val(s) for s in S])
            val_loss = 0.5*(ret.detach() - val_pred).pow(2).mean()
            loss = pol_loss + val_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (ep+1) % 25 == 0 or ep == 0:
            cagr = (math.exp(sum(R))**(12/len(R))-1)*100
            print(
                f"Ep {ep+1}/{cfg.epochs}  loss={loss.item():.4f}  CAGR={cagr:.2f}%")

    return pol  # trained policy

# ─────────────────────────────────────────────────────────────────────────────
# 5. Back-test helpers
# ─────────────────────────────────────────────────────────────────────────────


def rollout_policy(pol, env: PortfolioEnv, device) -> pd.Series:
    s, _ = env.reset()
    done = False
    equity = [0.0]
    while not done:
        t = torch.tensor(s, dtype=torch.float32, device=device)
        w = Dirichlet(torch.softmax(pol(t), -1)).sample().cpu().numpy()
        s, r, done, *_ = env.step(w)
        equity.append(equity[-1]+r)
    return pd.Series(equity[1:], index=env.y.index[:len(equity)-1])


def equal_weight(y: pd.DataFrame) -> pd.Series:
    ew = (y.mean(axis=1)).cumsum()
    return ew


def mean_variance(df_price: pd.DataFrame, y: pd.DataFrame) -> pd.Series:
    w = np.repeat(1/df_price.shape[1], df_price.shape[1])
    equity = [0.]
    for i in range(36, len(y)):  # need 3y lookback
        hist = y.iloc[i-36:i]
        cov = np.cov(hist.T)
        inv = np.linalg.pinv(cov)
        w = inv.sum(1)/inv.sum()
        r = np.dot(y.iloc[i], w/w.sum())
        equity.append(equity[-1]+r)
    idx = y.index[36:]
    return pd.Series(equity[1:], index=idx)

# ─────────────────────────────────────────────────────────────────────────────
# 6. CLI
# ─────────────────────────────────────────────────────────────────────────────


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="US_CORE",
                    help="comma-sep list or predefined name")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end",   default=str(dt.date.today()))
    ap.add_argument("--train_end", default="2022-12-31")
    ap.add_argument("--freq", choices=["M", "W"], default="M")
    ap.add_argument("--lookback", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--algo", choices=["actor", "ppo"], default="actor")
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


PRESETS = {
    "US_CORE": [
        "SPY", "QQQ", "DIA", "IWM", "GLD", "SLV", "USO", "IEF", "TLT", "UUP",
        "XLK", "XLF", "XLE", "VNQ", "HYG",
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "JNJ", "UNH", "BRK-B"
    ],
    "US_EXT": [  # S&P 500 top30 by mcap (예시) + 위 ETF
        "AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "GOOG", "BRK-B", "TSLA", "UNH",
        "JPM", "V", "JNJ", "WMT", "XOM", "MA", "LLY", "PG", "HD", "BAC", "CVX", "ABBV", "MRK",
        "SPY", "QQQ", "IEF", "TLT", "GLD", "USO", "UUP"
    ]
}

# ─────────────────────────────────────────────────────────────────────────────
# 7. Entry
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = cli()
    tickers = PRESETS.get(args.tickers, args.tickers.split(","))
    prices = load_prices(tickers, args.start, args.end)

    X, y = make_features(prices, args.freq)
    X_train, X_test = X.loc[:args.train_end], X.loc[args.train_end:]
    y_train, y_test = y.loc[:args.train_end], y.loc[args.train_end:]

    env_train = PortfolioEnv(X_train, y_train, args.lookback)
    device = torch.device(args.device)
    cfg = TrainCfg(algo=args.algo, epochs=args.epochs)
    policy = train(env_train, cfg, device)

    # out-of-sample evaluation
    env_test = PortfolioEnv(X_test, y_test, args.lookback)
    rl_eq = rollout_policy(policy, env_test, device)
    ew_eq = equal_weight(y_test)
    # mv_eq = mean_variance(prices.loc[y_test.index], y_test)


# pip install yfinance pandas numpy gymnasium torch vectorbt
# python reward_clip_portfolio_rl_v2.py --tickers US_CORE --algo ppo
