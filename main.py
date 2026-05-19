from parquet_torch_dataset import build_dataset_from_parquet
import pandas as pd

if __name__ == "__main__":
	BIST_RAW_PATH = "data/bist/raw/prices_raw.parquet"
	df = pd.read_parquet(BIST_RAW_PATH)

	print(df.shape)
	dataset = build_dataset_from_parquet(parquet_path=BIST_RAW_PATH, seq_len=10, ticker_selection=None)
	print(len(dataset))


