from catboost import CatBoostClassifier
from common import load_data, encode, evaluate

# 讀取資料並編碼資料
train, test = load_data()
X_train, y_train = encode(train)
X_test, y_test = encode(test)

# 建立並訓練模型
# fit() : 訓練，讓模型從 20 筆訓練資料中學習規律
# verbose = 0 : CatBoost 預設會每訓練一棵樹就印一行紀錄，1000 棵樹會洗滿整個畫面，設 0 關掉
# llow_writing_files = False : 不要產生 catboost_info 資料夾
model = CatBoostClassifier(random_state = 42, verbose = 0, allow_writing_files = False)
model.fit(X_train, y_train)

# 預測測試集
# predict() : 預測，模型只看測試集的特徵 X_test，完全看不到答案 y_test，猜完再拿去和答案比對
# ravel() : 讓 y_pred 變成一維陣列
y_pred = model.predict(X_test).ravel()

# 計算四項指標
print("=== CatBoost ===")

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


# ===== 實驗：CatBoost 原生類別處理 =====
# 差別 : 我們一開始餵資料是 X_train，已經被我們轉成 0、1、2 的數字；而這次我們直接餵原始資料。
from common import FEATURES

model2 = CatBoostClassifier(
    random_state = 42,
    verbose = 0,
    allow_writing_files = False,
    cat_features = FEATURES,
)

model2.fit(train[FEATURES], y_train)
y_pred2 = model2.predict(test[FEATURES]).ravel()

print("\n=== CatBoost (原生類別處理) ===")
for name, value in evaluate(y_test, y_pred2).items():
    print(f"{name:10s}: {value:.4f}")