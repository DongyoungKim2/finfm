# FinFM Pipeline

This repository provides a minimal starting point for fine-tuning the
[TimesFM](https://github.com/google-research/timesfm) model with PPO for
stock price forecasting.  It contains utilities for data ingestion,
pre‑processing, training and a small FastAPI service for inference.

```
├─ configs/                # Hydra YAML configs
├─ data/                   # Raw and processed data folders
├─ src/
│   ├─ cli.py              # Entry points: download, preprocess, build_dataset
│   ├─ data/               # Data pipeline modules
│   ├─ model/              # TimesFM wrapper and reward function
│   ├─ train/              # PPO trainer and evaluation utilities
│   └─ serving/            # FastAPI app and predictor helper
```

The code focuses on structure rather than full functionality.  Modules
can be extended with real training logic, MLflow tracking and proper
configurations.

## Quick start

### 1. Install the dependencies

Create a Python 3.10 environment and install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Download raw data

Fetch historical OHLCV prices and cache them under ``data/raw``:

```bash
python -m src.cli download --config configs/data.yaml
```

### 3. Preprocess and feature engineering

Clean the downloaded prices, compute technical indicators and scale the
features.  The processed arrays are stored under ``data/processed``.

```bash
python -m src.cli preprocess --config configs/data.yaml
```

### 4. Build the training dataset

Create sliding windows of context/target pairs and split them into train
and test sets:

```bash
python -m src.cli build-dataset --config configs/data.yaml
```

### 5. Train the model

Run PPO fine-tuning using the configuration in ``configs/train.yaml``.
Trained checkpoints are saved to the ``checkpoints`` folder and logged to
MLflow if a tracking URI is configured.

```bash
python -m src.train.run_train --config configs/train.yaml ticker=AAPL
```

### 6. Backtest the strategy

Evaluate a trained model on historical data using the built-in threshold
long/short strategy.  Metrics are logged to MLflow and a pickle of the
results is saved for further analysis.

```bash
python -m src.backtest.cli --config configs/backtest.yaml
```

### 7. Run the inference service

Start the FastAPI app to serve predictions.  By default it loads the
latest model from the MLflow ``Staging`` stage.

```bash
uvicorn src.serving.app:app --reload
```

This repository serves as a template that matches the layout described
in the blueprint.  Additional components like full PPO training,
MLflow integration and Docker files can be added as needed.
