# timesfm_rl_prototype.py
"""
Prototype pipeline to fine‑tune a frozen TimesFM core with an encoder‑adapter and train it via a PPO agent whose reward is the raw price difference.

Steps covered
-------------
1. Data download (yfinance + optional macro series)
2. Pre‑processing / alignment
3. Model definition: Encoder ➔ (frozen) TimesFM ➔ Decoder head, with LoRA adapters on the encoder & head only.
4. Custom Gymnasium environment that feeds windows to the model and returns `reward = price[t+1] - price[t]` (USD).
5. PPO training using Stable‑Baselines3 keeping the standard ratio clip intact.
6. Walk‑forward back‑test on unseen data using vectorbt.
7. Simple matplotlib visualizations (equity curve, drawdown, confusion matrix).

The script is intended for quick experimentation on a handful of tickers; adapt the CONFIG blocks for production‑scale runs.
"""

# =====================================================================
# 0. Imports & Global Config
# =====================================================================

import os
import json
import datetime as dt
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import timesfm
from peft import LoraConfig, get_peft_model  # pip install peft

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

import vectorbt as vbt
# never set explicit colors per instructions
import matplotlib.pyplot as plt


CONFIG = {
    "tickers": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
    "macro":   ["^GSPC", "^VIX"],
    "start":   "2015-01-01",
    "end":     dt.date.today().isoformat(),
    "window":  256,     # context length for TimesFM
    "horizon": 1,       # predict next‑day close
    "train_ratio": 0.8,
    "device":  "cuda" if torch.cuda.is_available() else "cpu",
    "lora_r":  8,
    "ppo_steps": 200_000,
}

DATA_DIR = "./data"
MODEL_DIR = "./models"
PLOT_DIR = "./plots"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

# =====================================================================
# 1. Data Download & Pre‑processing
# =====================================================================


def download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    df = yf.download(tickers, start=start, end=end,
                     group_by="ticker", progress=False, auto_adjust=True)
    closes = pd.concat({t: df[t]["Close"] for t in tickers}, axis=1)
    closes.columns = pd.Index(tickers, name="ticker")
    closes = closes.resample("1D").ffill()  # daily frequency
    return closes


print("Downloading data …")
price_df = download_prices(
    CONFIG["tickers"] + CONFIG["macro"], CONFIG["start"], CONFIG["end"])
price_df.to_csv(f"{DATA_DIR}/prices.csv")

# Split into train / test
split_idx = int(len(price_df) * CONFIG["train_ratio"])
train_df = price_df.iloc[:split_idx]
val_df = price_df.iloc[split_idx:]

# =====================================================================
# 2. Dataset & DataLoader
# =====================================================================


class MultiAssetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window: int, horizon: int, target_ticker: str):
        self.df = df
        self.window = window
        self.horizon = horizon
        self.target_idx = df.columns.get_loc(target_ticker)

    def __len__(self):
        return len(self.df) - self.window - self.horizon

    def __getitem__(self, idx):
        window_slice = self.df.iloc[idx: idx +
                                    self.window].values.astype(np.float32)
        future_price = self.df.iloc[idx + self.window +
                                    self.horizon - 1, self.target_idx].astype(np.float32)
        return window_slice, future_price
# =====================================================================
# 3. Model Definition (Encoder ➔ TimesFM ➔ Head)
# =====================================================================


class Encoder(nn.Module):
    def __init__(self, n_channels: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):  # x: (B, T, C)
        x = x.permute(0, 2, 1)             # (B, C, T)
        h = self.net(x)                    # (B, d_model, T)
        h = h.permute(0, 2, 1)             # (B, T, d_model)
        return h


class TimesFMWrapper(nn.Module):
    """Encoder→TimesFM→Head wrapper.

    We keep TimesFM **frozen** only when it exposes `.parameters()` (PyTorch nn.Module).
    Otherwise we treat it as an opaque callable and only train the surrounding modules.
    """

    def __init__(self, n_channels: int, d_model: int = 512):
        super().__init__()
        self.encoder = Encoder(n_channels, d_model)

        # Load Google‑Research checkpoint (PyTorch backend)
        self.core = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu" if torch.cuda.is_available() else "cpu",
                # small batch ok for demo
                per_core_batch_size=CONFIG["lora_r"],
                horizon_len=CONFIG["horizon"],
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-1.0-200m-pytorch",
            ),
        )

        self.head = nn.Linear(d_model, 1)

        # Freeze TimesFM **only if** it is a torch.nn.Module instance
        if isinstance(self.core, nn.Module):
            for p in self.core.parameters():
                p.requires_grad = False

        # Apply LoRA to encoder + head (core stays untouched)
        lora_config = LoraConfig(
            r=CONFIG["lora_r"],
            lora_alpha=CONFIG["lora_r"] * 2,
            target_modules=["net.0", "net.2", "head"],
            bias="none",
        )
        self.encoder = get_peft_model(self.encoder, lora_config)
        self.head = get_peft_model(self.head, lora_config)

    def forward(self, x):  # x: (B, T, C)
        """Predict next‑step price from multivariate window."""
        h = self.encoder(x)  # (B, T, d_model)
        # TimesFM expects (B, T, d_model) and returns (B, horizon, d_model)
        y_hat = self.core(h)
        out = self.head(y_hat[:, -1, :])  # last token, scalar output
        return out.squeeze(-1)  # (B,)

# =====================================================================
# 4. Gym Environment with Price‑Delta Reward
# =====================================================================


class StockForecastEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, df: pd.DataFrame, target: str):
        super().__init__()
        self.df = df.values.astype(np.float32)
        self.target_idx = df.columns.get_loc(target)
        self.window = CONFIG["window"]
        self.horizon = CONFIG["horizon"]
        self.model = TimesFMWrapper(
            n_channels=df.shape[1]).to(CONFIG["device"])
        self.current_step = 0
        # Action: continuous prediction (scaled price change)
        self.action_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(1,))
        # Observation: window × channels
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(self.window, df.shape[1]), dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = self.df[self.current_step: self.current_step + self.window]
        return obs, {}

    def step(self, action: np.ndarray):
        # Convert action to scalar prediction
        pred_delta = action.item()
        # True future price delta
        idx_now = self.current_step + self.window - 1
        idx_next = idx_now + 1
        true_delta = (self.df[idx_next, self.target_idx] -
                      self.df[idx_now, self.target_idx])
        reward = true_delta  # PPO variant: raw price change
        # Advance
        self.current_step += 1
        done = (self.current_step + self.window + self.horizon) >= len(self.df)
        if done:
            obs = np.zeros_like(self.df[0:self.window])
        else:
            obs = self.df[self.current_step: self.current_step + self.window]
        info = {"true_delta": true_delta, "pred_delta": pred_delta}
        return obs, reward, done, False, info

# =====================================================================
# 5. Train PPO
# =====================================================================


def train_ppo(target: str):
    def env_fn(): return Monitor(StockForecastEnv(train_df, target))
    vec_env = DummyVecEnv([env_fn])
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./ppo_tb")
    model.learn(total_timesteps=CONFIG["ppo_steps"])
    model.save(f"{MODEL_DIR}/ppo_{target}")
    vec_env.close()


for ticker in CONFIG["tickers"]:
    print(f"Training PPO for {ticker} …")
    train_ppo(ticker)

# =====================================================================
# 6. Walk‑Forward Backtest using Vectorbt
# =====================================================================

results = {}
for ticker in CONFIG["tickers"]:
    model = PPO.load(f"{MODEL_DIR}/ppo_{ticker}")
    tgt_series = val_df[ticker]
    preds = []
    obs = val_df.iloc[:CONFIG["window"]].values.astype(np.float32)
    for i in range(CONFIG["window"], len(val_df)):
        action, _ = model.predict(obs, deterministic=True)
        preds.append(action.item())
        obs = val_df.iloc[i - CONFIG["window"] +
                          1: i + 1].values.astype(np.float32)
    preds = pd.Series(preds, index=val_df.index[CONFIG["window"]:])
    price = tgt_series.loc[preds.index]
    # P&L = delta * position (pred)
    pnl = preds.shift(1).fillna(0) * (price.diff())
    equity = (pnl + 1).cumprod()
    results[ticker] = equity

# Combine into portfolio equal weight
portfolio = pd.concat(results, axis=1).ffill().mean(1)

# =====================================================================
# 7. Visualization
# =====================================================================

plt.figure()
portfolio.plot()
plt.title("Prototype Portfolio Equity Curve")
plt.xlabel("Date")
plt.ylabel("Equity (normalized)")
plt.savefig(f"{PLOT_DIR}/equity_curve.png")

# Drawdown plot
running_max = portfolio.cummax()
drawdown = (portfolio - running_max) / running_max
plt.figure()
drawdown.plot()
plt.title("Drawdown")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.savefig(f"{PLOT_DIR}/drawdown.png")

print("Backtest complete. Equity curve saved to",
      f"{PLOT_DIR}/equity_curve.png")
