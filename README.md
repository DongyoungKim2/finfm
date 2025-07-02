# FinFM Pipeline

This repository provides a minimal starting point for fine-tuning the
[TimesFM](https://github.com/google-research/timesfm) model with PPO for
stock price forecasting.  It contains utilities for data ingestion,
pre‑processing, training and a small FastAPI service for inference.

```
├─ configs/                # Hydra style YAML configs (placeholders)
├─ data/                   # Raw and processed data folders
├─ src/
│   ├─ cli.py              # Entry points: download, preprocess, train, predict
│   ├─ data/               # Data fetcher and preprocessing modules
│   ├─ model/              # TimesFM wrapper and reward function
│   ├─ train/              # PPO trainer and evaluation utilities
│   └─ serving/            # FastAPI app and predictor helper
```

The code focuses on structure rather than full functionality.  Modules
can be extended with real training logic, MLflow tracking and proper
configurations.

## Quick start

Install the required packages:

```bash
pip install yfinance pandas transformers trl fastapi typer
```

Download historical data:

```bash
python -m src.cli download-data \
  --symbols AAPL MSFT \
  --start 2015-01-01 --end 2023-01-01 \
  --output data/raw/prices.csv
```

Preprocess the data:

```bash
python -m src.cli preprocess-data --input-path data/raw/prices.csv
```

Run a dummy training loop (placeholder):

```bash
python -m src.cli train
```

Start the prediction service:

```bash
uvicorn src.serving.app:app --reload
```

This repository serves as a template that matches the layout described
in the blueprint.  Additional components like full PPO training,
MLflow integration and Docker files can be added as needed.
