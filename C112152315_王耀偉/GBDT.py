from sklearn.ensemble import GradientBoostingClassifier
from common import load_data, encode, evaluate

# 讀取資料並編碼資料
train, test = load_data()
X_train, y_train = encode(train)
X_test, y_test = encode(test)

# 建立並訓練模型
# fit() : 訓練，讓模型從 20 筆訓練資料中學習規律
# random_state = 42 : GBDT 內部有一些隨機性，固定這個數字，每次執行結果才會一樣，報告數據才能重現（42 只是慣例，任何數字都可以）
model = GradientBoostingClassifier(random_state = 42)
model.fit(X_train, y_train)

# 預測測試集
# predict() : 預測，模型只看測試集的特徵 X_test，完全看不到答案 y_test，猜完再拿去和答案比對
y_pred = model.predict(X_test)

# 計算四項指標
print("=== GBDT ===")

for name, value in evaluate(y_test, y_pred).items():
    print(f"{name:10s}: {value:.4f}")

# 逐筆對照
labels = {
    0: "飯", 
    1: "麵", 
    2: "速食"
}

print("\n編號 實際 預測")

for idx, true, pred in zip(X_test.index, y_test, y_pred):
    mark = "" if true == pred else " <- 錯誤"
    print(f"{idx:4d} {labels[true]:3s} {labels[pred]:3s}{mark}")