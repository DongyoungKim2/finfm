# reward_clip_portfolio_sb3.py

from __future__ import annotations
import argparse, datetime as dt, math, os, random
from typing import List, Tuple, Literal

import numpy as np
import pandas as pd
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ───────────────────────────────────
# 1. Data helpers
# ───────────────────────────────────
def load_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, auto_adjust=False,
                      progress=False, group_by="ticker")
    if isinstance(raw.columns, pd.MultiIndex):
        closes = {t: (raw[t]["Adj Close"] if "Adj Close" in raw[t]
                      else raw[t]["Close"]) for t in raw.columns.levels[0]}
        df = pd.concat(closes, axis=1)
    else:
        col  = "Adj Close" if "Adj Close" in raw else "Close"
        df   = raw[[col]].rename(columns={col: tickers[0]})
    df = df.dropna(axis=1, how='all')
    common_start = max(df[c].first_valid_index() for c in df)
    return df.loc[common_start:]

def make_features(df: pd.DataFrame, freq: Literal["M","W"]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    resampled = df.resample("ME").last() if freq == "M" else df.resample("W-FRI").last()
    ret   = np.log(resampled / resampled.shift(1))
    ema20 = resampled.ewm(span=20).mean().pct_change()
    ema60 = resampled.ewm(span=60).mean().pct_change()
    vol   = ret.rolling(20).std()
    feats = pd.concat({"ret": ret, "ema20": ema20, "ema60": ema60, "vol": vol}, axis=1)
    feats = feats.swaplevel(0,1, axis=1).sort_index(axis=1)
    feats.columns = [f"{t}_{f}" for t,f in feats.columns]
    feats = feats.dropna().astype(np.float32)
    y     = ret.loc[feats.index].astype(np.float32)
    return feats, y

# ───────────────────────────────────
# 2. Gym Environment
# ───────────────────────────────────
class PortfolioEnv(gym.Env):
    metadata = {"render_modes":[]}
    def __init__(self, X:pd.DataFrame, y:pd.DataFrame,
                 lookback:int=3, eps_floor:float=-0.4,
                 reward_weights:Tuple[float,float,float]=(1.,0.,0.)):
        super().__init__()
        self.X, self.y = X.astype(np.float32), y.astype(np.float32)
        self.n_assets  = y.shape[1]
        self.lookback  = lookback
        self.eps_floor = eps_floor
        self.w_ret, self.w_sharpe, self.w_anti = reward_weights

        feat_dim = self.X.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(lookback*feat_dim,),
                                            dtype=np.float32)
        # raw weights 0-1
        self.action_space = spaces.Box(low=0., high=1.,
                                       shape=(self.n_assets,),
                                       dtype=np.float32)
        self._start = lookback
        self._idx   = self._start
        self._w     = np.full(self.n_assets, 1/self.n_assets, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._idx = self._start
        self._w.fill(1/self.n_assets)
        return self._obs(), {}

    def step(self, action: np.ndarray):
        w_raw = np.clip(action, 0., 1.)
        self._w = w_raw / (w_raw.sum() + 1e-8)   # normalize
        r_raw   = float(np.dot(self.y.iloc[self._idx], self._w))
        r_clip  = max(r_raw, self.eps_floor)

        sharpe  = r_raw / (self.y.iloc[self._idx].std() + 1e-8)
        antib   = -np.abs(self._w - 1/self.n_assets).mean()
        reward  = (self.w_ret   * r_clip  +
                   self.w_sharpe* sharpe  +
                   self.w_anti  * antib)

        self._idx += 1
        terminated = self._idx >= len(self.y)
        return self._obs(), reward, terminated, False, {}

    def _obs(self):
        win = self.X.iloc[self._idx-self.lookback : self._idx]
        return win.values.flatten()

# ───────────────────────────────────
# 3. Back-test helpers
# ───────────────────────────────────

def rollout(model, env: PortfolioEnv) -> pd.Series:
    # 1) Reset the raw env and grab the obs
    obs, _ = env.reset()           # obs.shape == (312,)

    done = False
    equity = [0.0]

    while not done:
        # 2) Make it a batch of size 1 → shape (1, 312)
        batched = obs[np.newaxis, ...]
        # 3) Predict on that batch
        action, _ = model.predict(batched, deterministic=False)
        # 4) Squeeze back to shape (n_assets,)
        action = action[0]

        # 5) Step the raw env with your action
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        equity.append(equity[-1] + reward)

    # Return a series of cumulative rewards aligned to your test index
    return pd.Series(equity[1:], index=env.y.index[:len(equity)-1])

def equal_weight(y:pd.DataFrame) -> pd.Series:
    return y.mean(axis=1).cumsum()

# ───────────────────────────────────
# 4. CLI & main
# ───────────────────────────────────
PRESETS = {
    "US_CORE": [
        "SPY","QQQ","DIA","IWM","GLD","SLV","USO","IEF","TLT","UUP",
        "XLK","XLF","XLE","VNQ","HYG",
        "AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","JPM","JNJ","UNH","BRK-B"
    ]
}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="US_CORE")
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end",   default=str(dt.date.today()))
    ap.add_argument("--train_end", default="2024-12-31")
    ap.add_argument("--freq", choices=["M","W"], default="M")
    ap.add_argument("--lookback", type=int, default=3)
    ap.add_argument("--total_steps", type=int, default=400_000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tb_logdir", default=None,
        help="TensorBoard 로그를 남길 디렉터리. >None이면 기록 안 함")
    ap.add_argument("--export", default=None,
        help="학습이 끝난 뒤 SavedModel 을 저장할 디렉터리 경로")
    return ap.parse_args()

def main():
    args    = parse_args()
    tickers = PRESETS.get(args.tickers, args.tickers.split(","))
    prices  = load_prices(tickers, args.start, args.end)

    X, y          = make_features(prices, args.freq)
    X_train, X_te = X.loc[:args.train_end], X.loc[args.train_end:]
    y_train, y_te = y.loc[:args.train_end], y.loc[args.train_end:]

    env_train = DummyVecEnv([lambda: Monitor(
                    PortfolioEnv(X_train, y_train,
                                 lookback=args.lookback))])
    # env_test  = PortfolioEnv(X_te, y_te, lookback=args.lookback)
    X_test_pad = pd.concat([X_train.iloc[-args.lookback:], X_te])
    y_test_pad = pd.concat([y_train.iloc[-args.lookback:], y_te])
    env_test   = PortfolioEnv(X_test_pad, y_test_pad, lookback=args.lookback)

    custom_logger = None
    if args.tb_logdir:
        from stable_baselines3.common.logger import configure
        custom_logger = configure(args.tb_logdir, ["stdout", "tensorboard"])

    model = PPO("MlpPolicy",
                env_train,
                learning_rate=2e-4,
                gamma=0.99,
                gae_lambda=0.95,
                n_steps=2048,
                batch_size=256,
                ent_coef=1e-2,
                clip_range=0.15,
                verbose=1,
                seed=SEED,
                device=args.device)
    if custom_logger:
        model.set_logger(custom_logger)

    print(">> train obs space:", env_train.observation_space.shape)
    print(">> test  obs space:", env_test.observation_space.shape)

    model.learn(total_timesteps=args.total_steps)

    rl_eq = rollout(model, env_test)
    ew_eq = equal_weight(y_te)

    print(f"\nFinal RL equity: {rl_eq.iloc[-1]:.4f}  "
          f"EW equity: {ew_eq.iloc[-1]:.4f}")
    
    if args.export:
        export_path = args.export
        os.makedirs(export_path, exist_ok=True)
        print(f"[+] Exporting policy via TorchScript trace → {export_path}/policy_traced.pt")

        model.policy.to("cpu")
        model.policy.eval()
        dummy = torch.rand(1, env_train.observation_space.shape[0])
        # traced_policy = torch.jit.trace(model.policy, dummy)
        model.policy.to("cuda")
        traced_policy = torch.jit.trace(model.policy, dummy.to("cuda"))
        traced_policy.save(os.path.join(export_path, "policy_traced.pt"))

        print("[✓] TorchScript traced model export complete!")


if __name__ == "__main__":
    main()

# python reward_clip_portfolio_sb3.py --tickers US_CORE --freq M --total_steps 500000 --tb_logdir runs/ppo_clip_4 --export exports/ppo_saved_4
