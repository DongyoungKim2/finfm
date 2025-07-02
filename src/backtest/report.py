from __future__ import annotations

import mlflow
import pandas as pd


def log_to_mlflow(metrics: dict, df_res: pd.DataFrame, cfg) -> None:
    mlflow.log_metrics(metrics)
    tmp_path = "results.pkl"
    df_res.to_pickle(tmp_path)
    mlflow.log_artifact(tmp_path)
