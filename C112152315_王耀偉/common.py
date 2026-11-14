import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
FEATURES = ["性別", "科系", "口味", "價格"]
TARGET = "中餐"

# 讀取資料集
def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv", index_col = "編號")
    test = pd.read_csv(DATA_DIR / "test.csv", index_col = "編號")
    return train, test


# 自我測試
if __name__ == "__main__":
    train, test = load_data()
    print(train.shape, test.shape)
    print(train[TARGET].value_counts())