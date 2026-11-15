# 學生對中餐的嗜好：GBDT、XGBoost、CatBoost 分類模型

機器學習期中報告。以學生的性別、科系、口味偏好與價格，預測中餐選擇（飯、麵、速食），
比較三種梯度提升演算法，並計算 Accuracy、Precision、Recall、F1-Score 四項指標。

- 學號：C112152315
- 姓名：王耀偉

## 資料集

| 項目 | 說明 |
|---|---|
| 訓練集 | 20 筆（`data/train.csv`） |
| 測試集 | 10 筆（`data/test.csv`） |
| 特徵 | 性別、科系、口味、價格（皆為類別型） |
| 目標 | 中餐：飯、麵、速食 |

## 專案結構

```
├── data/
│   ├── train.csv
│   └── test.csv
├── common.py           # 共用：讀取資料、特徵編碼、評估指標
├── GBDT.py             # GBDT 模型
├── XGBoost.py          # XGBoost 模型
├── CatGBoost.py        # CatBoost 模型（含原生類別處理實驗）
├── analysis.py         # 三模型特徵重要度比較
├── export_coreml.py    # 匯出 Core ML 模型（iOS 延伸用）
├── encoding.json       # 類別編碼規則，Python 與 App 共用
├── models/
│   └── LunchPredictor.mlmodel
└── requirements.txt
```

## 環境建置

需要 Python 3.12 以上。

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS
source .venv/bin/activate

python -m pip install -r requirements.txt
```

## 執行

```bash
python GBDT.py
python XGBoost.py
python CatGBoost.py
python analysis.py       # 產生 feature_importance.png
python export_coreml.py  # 產生 models/LunchPredictor.mlmodel
```

## 結果

| 模型 | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| GBDT | 0.9000 | 0.9333 | 0.8889 | 0.8963 |
| XGBoost | 0.9000 | 0.9333 | 0.8889 | 0.8963 |
| CatBoost | 0.9000 | 0.9333 | 0.8889 | 0.8963 |

Precision、Recall、F1 皆為 macro 平均。

三個模型都只誤判測試集第 23 筆（男、資工、重口味、高價，實際為麵）。
訓練集第 16 筆的特徵與它完全相同，答案卻是速食，因此任何模型都難以猜對。
測試集每一筆的特徵組合都能在訓練集找到，這也是三個模型表現相同的原因。

特徵重要度顯示口味與價格合計佔 73%–87%，性別影響最小。

## 手算

依課堂講義步驟，以單棵樹手算三種演算法
（初始值 → 梯度與 Hessian → Gain 建樹 → 葉子值 → 預測 → 四項指標），
三者 Accuracy 皆為 0.80。詳見期中報告簡報。