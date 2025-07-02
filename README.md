# FinFM Data Preparation

This repository contains utilities for downloading and preparing
historical price data for use with time series models like
`timesfm`.

The `data_prep.py` script lists a set of major U.S. stocks and ETFs
and downloads their historical prices using the `yfinance` package.
This data can be used for model training, fine-tuning, and
forecasting.

## Requirements

- Python 3.8+
- [`yfinance`](https://pypi.org/project/yfinance/)
- `pandas`

Install requirements with:

```bash
pip install yfinance pandas
```

## Usage

List major tickers:

```bash
python data_prep.py --list-tickers
```

Download historical data to `data.csv`:

```bash
python data_prep.py --start 2010-01-01 --end 2023-01-01 \
  --interval 1d --output data.csv
```

Update existing data file with the latest prices:

```bash
python data_prep.py --update data.csv
```

The CSV file contains a column for the ticker symbol, the date, and
Open/High/Low/Close/Adj Close/Volume columns returned by `yfinance`.
Use this file to train or fine-tune your `timesfm` models and to fetch
new data for future predictions.
