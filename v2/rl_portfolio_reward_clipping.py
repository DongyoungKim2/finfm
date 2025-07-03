# rl_portfolio_reward_clipping.py
# --------------------------------
# Requirements:
#   pip install yfinance pandas numpy torch gymnasium stable-baselines3==2.3.0 vectorbt==0.25

import os
import warnings
import math
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import vectorbt as vbt

warnings.filterwarnings("ignore", category=FutureWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# 1. 데이터 다운로드 & 전처리
# ---------------------------------------------------------------------
TICKERS = ["AAPL", "MSFT", "META", "SPY", "BND", "GLD"]
TRAIN_START, TRAIN_END = "2010-01-01", "2019-06-10"
TEST_START,  TEST_END = "2019-07-18", "2022-07-22"   # 검증·테스트용


def load_prices(tickers, start, end):
    px = (
        yf.download(tickers, start=start, end=end,
                    auto_adjust=False, progress=False)
        .loc[:, ("Adj Close", slice(None))]
    )
    # yf 멀티인덱스 (컬럼: 'Adj Close', ticker) → 단순화
    px.columns = px.columns.droplevel(0)
    return px.sort_index()


def pct_returns(prices):
    return prices.ffill().pct_change().dropna()


prices_all = load_prices(TICKERS, TRAIN_START, TEST_END)
returns_all = pct_returns(prices_all)

# 학습·테스트 구분
train_mask = (returns_all.index >= TRAIN_START) & (
    returns_all.index <= TRAIN_END)
test_mask = (returns_all.index >= TEST_START) & (returns_all.index <= TEST_END)
rets_train, rets_test = returns_all[train_mask], returns_all[test_mask]

# ---------------------------------------------------------------------
# 2. Gymnasium 포트폴리오 환경
# ---------------------------------------------------------------------


class PortfolioEnv(gym.Env):
    """On-policy 포트폴리오 환경 (일별 리밸런싱)."""
    metadata = {"render_modes": ["human"]}

    def __init__(self, returns: pd.DataFrame, window: int = 256, fee: float = 0.0005):
        super().__init__()
        self.returns = returns.values.astype(np.float32)
        self.dates = returns.index
        self.window = window
        self.fee = fee
        self.n_assets = returns.shape[1]

        # observation: window × n_assets 실수 행렬
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(window, self.n_assets), dtype=np.float32
        )
        # action: 각 자산별 un-normalized logit
        self.action_space = gym.spaces.Box(
            low=-5.0, high=5.0, shape=(self.n_assets,), dtype=np.float32
        )

        self.reset()

    def _get_state(self):
        # shape (window, n_assets)
        hist = self.returns[self.t - self.window: self.t]
        return hist.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.window    # init position
        self.w_prev = np.full(self.n_assets, 1.0 /
                              self.n_assets, dtype=np.float32)
        return self._get_state(), {}

    def step(self, action):
        # softmax → 포트폴리오 weight (합 1.0)
        w = torch.softmax(torch.tensor(action), dim=-1).cpu().numpy()
        r_t = (w * self.returns[self.t]).sum()

        tr_cost = self.fee * np.abs(w - self.w_prev).sum()
        reward = r_t - tr_cost

        self.w_prev = w.copy()
        self.t += 1
        done = self.t >= len(self.returns)

        return self._get_state(), reward, done, False, {"date": str(self.dates[self.t-1]), "ret": r_t}

    def render(self):
        print(f"t={self.t}, weights={self.w_prev}")

# ---------------------------------------------------------------------
# 3. Reward-Clipping Callback
# ---------------------------------------------------------------------


class RewardClipper(BaseCallback):
    """Actor-Only PPO에서 어드밴티지 클리핑 (ε₁ < 0, ε₂ = +∞)."""

    def __init__(self, eps_neg: float = -0.4, eps_pos: float = float("inf"), **kwargs):
        super().__init__(**kwargs)
        self.eps_neg, self.eps_pos = eps_neg, eps_pos

    def _on_step(self) -> bool:
        # rollout_buffer → advantages 텐서 in-place clamp
        if "advantages" in self.locals:
            self.locals["advantages"].clamp_(
                min=self.eps_neg, max=self.eps_pos)
        return True


# ---------------------------------------------------------------------
# 4. 학습
# ---------------------------------------------------------------------
window = 256
env = PortfolioEnv(rets_train, window=window)

policy_kwargs = dict(net_arch=[128, 128])
model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=1024,
    batch_size=256,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.001,
    vf_coef=0.0,            # Actor-Only
    policy_kwargs=policy_kwargs,
    device=device,
    verbose=1,
)

total_timesteps = 2_000_000
model.learn(total_timesteps=total_timesteps, callback=RewardClipper(-0.4))
model.save("ppo_reward_clip_portfolio.zip")
print("✅ 모델 학습 완료 & 저장!")

# ---------------------------------------------------------------------
# 5. 테스트 구간 백테스트 (VectorBT)
# ---------------------------------------------------------------------


def run_policy(model, returns_df, window):
    """훈련된 모델 → 테스트 가격 → 일별 weight 행렬 생성."""
    env_test = PortfolioEnv(returns_df, window=window)
    obs, _ = env_test.reset()
    weights, dates = [], []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        w = torch.softmax(torch.tensor(action), dim=-1).cpu().numpy()
        weights.append(w)
        dates.append(env_test.dates[env_test.t - 1])
        obs, _, done, _, _ = env_test.step(action)
        if done:
            break

    w_df = pd.DataFrame(weights, index=pd.DatetimeIndex(
        dates), columns=returns_df.columns)
    return w_df


w_test = run_policy(model, rets_test, window)

# VectorBT 백테스트
pf = vbt.Portfolio.from_weights(
    price=prices_all.loc[w_test.index, w_test.columns],   # 테스트 구간 가격
    weights=w_test,
    init_cash=1_000_000,
    freq="D",
    fees=0.0,
)
stats = pf.stats()
print("\n===== Backtest Stats (Test Period) =====")
print(stats)

# 그래프 (주피터/GUI 환경에서 실행 시 표시)
try:
    pf.plot().show()
except Exception as e:
    print("plot skipped:", e)
