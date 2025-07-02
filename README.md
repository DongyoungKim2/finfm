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

Install the required packages:

```bash
pip install -r requirements.txt  # or install via Poetry
```

Download historical data and cache them:

```bash
python -m src.cli download
```

Preprocess the data and create features:

```bash
python -m src.cli preprocess
```

Build sliding windows and dataset splits:

```bash
python -m src.cli build-dataset
```

Start the prediction service:

```bash
uvicorn src.serving.app:app --reload
```

This repository serves as a template that matches the layout described
in the blueprint.  Additional components like full PPO training,
MLflow integration and Docker files can be added as needed.
