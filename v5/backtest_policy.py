# backtest_policy.py

from __future__ import annotations
import argparse, datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import gymnasium as gym
from gymnasium import spaces

# ───────────────────────────────────
# 1. 재사용: load_prices, make_features, PortfolioEnv, equal_weight
# ───────────────────────────────────
def load_prices(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=False, progress=False,
                      group_by="ticker")
    if isinstance(raw.columns, pd.MultiIndex):
        closes = {t: (raw[t]["Adj Close"] if "Adj Close" in raw[t]
                      else raw[t]["Close"])
                  for t in raw.columns.levels[0]}
        df = pd.concat(closes, axis=1)
    else:
        col = "Adj Close" if "Adj Close" in raw else "Close"
        df  = raw[[col]].rename(columns={col: tickers[0]})
    df = df.dropna(axis=1, how="all")
    common_start = max(df[c].first_valid_index() for c in df)
    return df.loc[common_start:]

def make_features(df, freq):
    res = df.resample("ME").last() if freq=="M" else df.resample("W-FRI").last()
    ret   = np.log(res / res.shift(1))
    ema20 = res.ewm(span=20).mean().pct_change()
    ema60 = res.ewm(span=60).mean().pct_change()
    vol   = ret.rolling(20).std()
    feats = pd.concat({"ret":ret, "ema20":ema20, "ema60":ema60, "vol":vol},
                      axis=1).swaplevel(0,1,axis=1).sort_index(axis=1)
    feats.columns = [f"{t}_{f}" for t,f in feats.columns]
    feats = feats.dropna().astype(np.float32)
    y     = ret.loc[feats.index].astype(np.float32)
    return feats, y

class PortfolioEnv(gym.Env):
    metadata={"render_modes":[]}
    def __init__(self, X,y, lookback=3, eps_floor=-0.4,
                 reward_weights=(1.,0.,0.)):
        super().__init__()
        self.X, self.y = X.astype(np.float32), y.astype(np.float32)
        self.n_assets  = y.shape[1]
        self.lookback  = lookback
        self.eps_floor = eps_floor
        self.w_ret, self.w_sharpe, self.w_anti = reward_weights

        feat_dim = self.X.shape[1]
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(lookback*feat_dim,),
                                            dtype=np.float32)
        self.action_space = spaces.Box(0.,1.,shape=(self.n_assets,),
                                       dtype=np.float32)
        self._start = lookback
        self._idx   = self._start
        self._w     = np.full(self.n_assets, 1/self.n_assets,
                              dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._idx = self._start
        self._w.fill(1/self.n_assets)
        return self._obs(), {}

    def step(self, action):
        w_raw = np.clip(action,0.,1.)
        self._w = w_raw/(w_raw.sum()+1e-8)
        r_raw  = float(np.dot(self.y.iloc[self._idx], self._w))
        r_clip = max(r_raw, self.eps_floor)
        sharpe = r_raw/(self.y.iloc[self._idx].std()+1e-8)
        antib  = -np.abs(self._w-1/self.n_assets).mean()
        reward = self.w_ret*r_clip + self.w_sharpe*sharpe + self.w_anti*antib

        self._idx += 1
        done = self._idx >= len(self.y)
        return self._obs(), reward, done, False, {}

    def _obs(self):
        win = self.X.iloc[self._idx-self.lookback:self._idx]
        return win.values.flatten()

def equal_weight(y:pd.DataFrame) -> pd.Series:
    return y.mean(axis=1).cumsum()

# ───────────────────────────────────
# 2. TorchScript 모델 로드 & Rollout
# ───────────────────────────────────
def load_policy(model_path, device):
    # policy = torch.jit.load(model_path, map_location=device)
    device = torch.device("cuda")
    policy = torch.jit.load(model_path, map_location=device).to(device)
    policy.eval()
    return policy

def rollout_traced(policy, env, device):
    obs, _ = env.reset()
    done = False
    equity = [0.0]
    while not done:
        x = torch.from_numpy(obs[None]).to(device)
        with torch.no_grad():
            out = policy(x)
        # TorchScript trace가 (action, …) 형태로 나올 수 있으니
        if isinstance(out, tuple):
            action = out[0].cpu().numpy()[0]
        else:
            action = out.cpu().numpy()[0]
        obs, reward, done, _, _ = env.step(action)
        equity.append(equity[-1] + reward)
    return pd.Series(equity[1:], index=env.y.index[:len(equity)-1])

# ───────────────────────────────────
# 3. 성과 지표 계산
# ───────────────────────────────────
def compute_metrics(eq:pd.Series):
    total = eq.iloc[-1]
    days  = (eq.index[-1] - eq.index[0]).days
    years = days/365.25
    cagr  = (1+total)**(1/years)-1
    daily_ret = eq.diff().fillna(0)
    ann_vol   = daily_ret.std()*np.sqrt(252)
    sharpe    = cagr/(ann_vol+1e-8)
    return cagr, ann_vol, sharpe

# ───────────────────────────────────
# 4. CLI & main
# ───────────────────────────────────
def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--model-path", required=True,
                   help="policy_traced.pt 파일 경로")
    p.add_argument("--tickers", default="US_CORE")
    p.add_argument("--start",     default="2015-01-01")
    p.add_argument("--end",       default=str(dt.date.today()))
    p.add_argument("--train_end", default="2025-07-03")
    p.add_argument("--freq", choices=["M","W"], default="M")
    p.add_argument("--lookback", type=int, default=3)
    return p.parse_args()

PRESETS = {
    "US_CORE":[
        "SPY","QQQ","DIA","IWM","GLD","SLV","USO","IEF","TLT","UUP",
        "XLK","XLF","XLE","VNQ","HYG",
        "AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","JPM",
        "JNJ","UNH","BRK-B"
    ]
}

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.model_path, device)

    # 데이터 준비
    tickers = PRESETS.get(args.tickers, args.tickers.split(","))
    prices  = load_prices(tickers, args.start, args.end)
    X, y    = make_features(prices, args.freq)
    X_tr, X_te = X.loc[:args.train_end], X.loc[args.train_end:]
    y_tr, y_te = y.loc[:args.train_end], y.loc[args.train_end:]

    # 테스트 환경
    X_pad = pd.concat([X_tr.iloc[-args.lookback:], X_te])
    y_pad = pd.concat([y_tr.iloc[-args.lookback:], y_te])
    env_test = PortfolioEnv(X_pad, y_pad, lookback=args.lookback)

    # 백테스트
    rl_eq = rollout_traced(policy, env_test, device)
    ew_eq = equal_weight(y_te)

    # 지표 출력
    rl_c, rl_v, rl_s = compute_metrics(rl_eq)
    ew_c, ew_v, ew_s = compute_metrics(ew_eq)

    print("\n=== Performance Metrics ===")
    print(f"{'Strategy':<12}{'CAGR':>8}{'Vol':>10}{'Sharpe':>10}")
    print(f"{'RL-PPO':<12}{rl_c:>8.2%}{rl_v:>10.2%}{rl_s:>10.2f}")
    print(f"{'EqualWt':<12}{ew_c:>8.2%}{ew_v:>10.2%}{ew_s:>10.2f}")

    # 시각화
    import matplotlib.pyplot as plt
    df = pd.DataFrame({"RL-PPO":rl_eq, "EqualWt":ew_eq})
    plt.figure(figsize=(10,5))
    (df+1).cumprod().plot()
    plt.title("Equity Curve")
    plt.ylabel("Cumulative Return")
    plt.xlabel("Date")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()

# python backtest_policy.py   --model-path exports/ppo_saved_3/policy_traced.pt   --tickers US_CORE   --start 2015-01-01   --end   2025-07-03   --train_end 2025-06-03   --freq M   --lookback 3