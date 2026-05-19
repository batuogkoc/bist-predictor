"""Build PyTorch sequence datasets from wide parquet price tables.

This file is intended for modeling a binary classifier that looks at the last
T prices and predicts whether the next price will move higher or lower.

The dataset can be created from a single parquet file and used in two modes:
- all tickers concatenated into one dataset
- only the N'th ticker selected for a smaller dataset

Usage examples:
  python parquet_torch_dataset.py --parquet data/bist/raw/prices_raw.parquet --seq-len 20
  python parquet_torch_dataset.py --parquet data/bist/raw/prices_raw.parquet --seq-len 20 --ticker-index 5
  python parquet_torch_dataset.py --parquet data/bist/raw/prices_raw.parquet --seq-len 20 --ticker SYMBOL
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_price_matrix(
    parquet_path: Union[str, Path], price_level: str = "Close"
) -> pd.DataFrame:
    """Load a wide price parquet file and return a ticker x time price matrix."""
    parquet_path = Path(parquet_path)
    df = pd.read_parquet(parquet_path)

    if isinstance(df.columns, pd.MultiIndex):
        if price_level in df.columns.get_level_values(0):
            df = df.xs(price_level, axis=1, level=0)
        else:
            raise ValueError(
                f"Price level '{price_level}' not found in parquet columns. "
                f"Available levels: {sorted(set(df.columns.get_level_values(0)))}"
            )

    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def series_to_sequences(
    prices: Sequence[float], seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a 1D price series into (X, y) sequences.

    X shape: (num_sequences, seq_len)
    y shape: (num_sequences,)
    """
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1:
        raise ValueError("prices must be a 1D sequence")

    valid_mask = ~np.isnan(prices)
    prices = prices[valid_mask]
    n = len(prices)
    if n <= seq_len:
        return np.empty((0, seq_len), dtype=np.float32), np.empty((0,), dtype=np.int64)

    x = np.stack([prices[i : i + seq_len] for i in range(n - seq_len)], axis=0)
    y = (prices[seq_len:] > prices[seq_len - 1 : -1]).astype(np.int64)
    return x.astype(np.float32), y


class ParquetPriceSequenceDataset(Dataset):
    """A PyTorch dataset for ticker price sequences from a parquet file."""

    def __init__(
        self,
        price_matrix: pd.DataFrame,
        seq_len: int,
        selected_ticker: Optional[Union[int, str]] = None,
    ):
        if selected_ticker is not None:
            price_matrix = self._select_ticker(price_matrix, selected_ticker)

        self.ticker_names = list(price_matrix.columns)
        self.seq_len = seq_len
        self.inputs, self.targets, self.ticker_ids = self._build_dataset(
            price_matrix, seq_len
        )
        if len(self.inputs) == 0:
            raise ValueError(
                f"No valid sequences found with seq_len={seq_len}. "
                "Try a smaller sequence length or a different ticker selection."
            )

    @staticmethod
    def _select_ticker(
        price_matrix: pd.DataFrame, selected_ticker: Union[int, str]
    ) -> pd.DataFrame:
        if isinstance(selected_ticker, int):
            if selected_ticker < 0 or selected_ticker >= len(price_matrix.columns):
                raise IndexError(
                    f"ticker index {selected_ticker} is out of range"
                )
            return price_matrix.iloc[:, [selected_ticker]]

        if selected_ticker not in price_matrix.columns:
            raise KeyError(
                f"ticker '{selected_ticker}' not found. Available tickers: {list(price_matrix.columns)}"
            )
        return price_matrix[[selected_ticker]]

    @staticmethod
    def _build_dataset(
        price_matrix: pd.DataFrame, seq_len: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        inputs = []
        targets = []
        ticker_ids = []

        for ticker_id, ticker_name in enumerate(price_matrix.columns):
            series = price_matrix[ticker_name].to_numpy(dtype=float)
            x, y = series_to_sequences(series, seq_len)
            if x.shape[0] == 0:
                continue

            inputs.append(x)
            targets.append(y)
            ticker_ids.append(np.full(len(y), ticker_id, dtype=np.int64))

        if not inputs:
            return np.empty((0, seq_len), dtype=np.float32), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

        return (
            np.vstack(inputs),
            np.concatenate(targets),
            np.concatenate(ticker_ids),
        )

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        x = torch.from_numpy(self.inputs[index]).float()
        y = torch.tensor(self.targets[index], dtype=torch.long)
        ticker_id = int(self.ticker_ids[index])
        return x, y, ticker_id

    def ticker_for_index(self, index: int) -> str:
        ticker_id = int(self.ticker_ids[index])
        return self.ticker_names[ticker_id]

    def ticker_sample_counts(self) -> dict[str, int]:
        counts = {}
        for idx, name in enumerate(self.ticker_names):
            counts[name] = int((self.ticker_ids == idx).sum())
        return counts

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(seq_len={self.seq_len}, "
            f"samples={len(self)}, tickers={len(self.ticker_names)})"
        )


def build_dataset_from_parquet(
    parquet_path: Union[str, Path],
    seq_len: int,
    price_level: str = "Close",
    ticker_selection: Optional[Union[int, str, Sequence[str]]] = None,
) -> ParquetPriceSequenceDataset:
    df = load_price_matrix(parquet_path, price_level=price_level)

    if ticker_selection is not None and not isinstance(ticker_selection, (int, str)):
        df = df.loc[:, list(ticker_selection)]

    return ParquetPriceSequenceDataset(
        df,
        seq_len=seq_len,
        selected_ticker=ticker_selection if isinstance(ticker_selection, (int, str)) else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a PyTorch dataset from a parquet price file."
    )
    parser.add_argument(
        "--parquet",
        default="data/bist/raw/prices_raw.parquet",
        help="Path to the parquet file containing wide price data.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=20,
        help="Number of input time steps in each sequence.",
    )
    parser.add_argument(
        "--price-level",
        default="Close",
        help="Column level to select from a multi-index parquet file (default: Close).",
    )
    parser.add_argument(
        "--ticker-index",
        type=int,
        help="If set, load only the N'th ticker by integer index.",
    )
    parser.add_argument(
        "--ticker",
        help="If set, load only a specific ticker name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection: Optional[Union[int, str]] = None
    if args.ticker is not None:
        selection = args.ticker
    elif args.ticker_index is not None:
        selection = args.ticker_index

    dataset = build_dataset_from_parquet(
        args.parquet,
        seq_len=args.seq_len,
        price_level=args.price_level,
        ticker_selection=selection,
    )

    print(dataset)
    print(f"Ticker names: {dataset.ticker_names}")
    print(f"Samples per ticker: {dataset.ticker_sample_counts()}")

    if len(dataset) > 0:
        x, y, ticker_id = dataset[0]
        print("\nExample sample:")
        print(f"  input shape: {x.shape}")
        print(f"  label: {y.item()}")
        print(f"  ticker: {dataset.ticker_names[ticker_id]}")
        print(f"  input values: {x.tolist()}")


if __name__ == "__main__":
    main()
