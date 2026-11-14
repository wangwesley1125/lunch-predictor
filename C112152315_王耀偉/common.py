import pandas as pd
from pathlib import Path
import json

DATA_DIR = Path(__file__).parent / "data"
FEATURES = ["性別", "科系", "口味", "價格"]
TARGET = "中餐"

# 編碼表
ENCODING = {
    "性別" : {"男" : 0, "女" : 1},
    "科系" : {"資工" : 0, "電子" : 1, "資管" : 2},
    "口味" : {"清淡" : 0, "重口味" : 1},
    "價格" : {"低" : 0, "高" : 1},
    "中餐" : {"飯" : 0, "麵" : 1, "速食" : 2},
}

# 讀取資料集
def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv", index_col = "編號")
    test = pd.read_csv(DATA_DIR / "test.csv", index_col = "編號")
    return train, test

# 編碼函式
def encode(df):
    X = df[FEATURES].copy() # 特徵 (模型的輸入)
    for col in FEATURES:
        X[col] = X[col].map(ENCODING[col])

    y = df[TARGET].map(ENCODING[TARGET]) # 答案 (模型要學著猜的東西)

    if X.isna().any().any() or y.isna().any():
        raise ValueError("資料中有編碼表沒有定義的值，請檢察 CSV 是否有錯字")

    return X, y

# 存檔函式 (提供延伸使用 iOS App)
# ensure_ascii = False : 讓中文正常顯示
# indent = 2 : 讓 JSON 檔案好閱讀
def save_encoding():
    with open(Path(__file__).parent / "encoding.json", "w", encoding = "utf-8") as f:
        json.dump(ENCODING, f, ensure_ascii = False, indent = 2)   

# 自我測試
if __name__ == "__main__":
    train, test = load_data()
    # print(train.shape, test.shape)
    # print(train[TARGET].value_counts())

    X_train, y_train = encode(train)
    X_test, y_test = encode(test)

    print("編碼前:")
    print(train.head(3))
    print("\n編碼後:")
    print(X_train.head(3))
    print(y_train.head(3).tolist())

    save_encoding()
    print("\n已產生 encoding.json")