"""Command line interface for FinFM pipeline."""
from __future__ import annotations

import typer

from src.data.fetcher import download
from src.data.preprocess import preprocess
from src.train.ppo_trainer import train as train_model
from src.serving.predictor import predict

cli = typer.Typer()

@cli.command()
def download_data(
    symbols: str = typer.Option(..., help="Space separated tickers"),
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
    interval: str = typer.Option("1d", help="Data interval"),
    output: str = typer.Option("data/raw/data.csv", help="Output CSV path"),
):
    tickers = symbols.split()
    df = download(tickers, start, end, interval)
    df.to_csv(output, index=False)
    typer.echo(f"Saved {len(df)} rows to {output}")

@cli.command()
def preprocess_data(
    input_path: str = typer.Option("data/raw/data.csv"),
    output_dir: str = typer.Option("data/processed"),
):
    preprocess(input_path, output_dir)

@cli.command()
def train(config: str = typer.Option("configs/train.yaml")):
    train_model(config)

@cli.command()
def predict_cli(
    symbol: str,
    context: str,
    horizon: int = 1,
):
    ctx = [float(x) for x in context.split(',')]
    result = predict(symbol, ctx, horizon)
    typer.echo(str(result))

if __name__ == "__main__":
    cli()
