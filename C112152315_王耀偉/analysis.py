import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from common import load_data, encode, FEATURES

import platform
# 預設字型沒有中文，Microsoft JhengHei（微軟正黑體）是 Windows 內建字型，PingFang TC（蘋方）是 Mac 內建字型
plt.rcParams["font.family"] = "PingFang TC" if platform.system() == "Darwin" else "Microsoft JhengHei"

# 準備資料與模型
train, test = load_data()
X_train, y_train = encode(train)

models = {
    "GBDT" : GradientBoostingClassifier(random_state = 42),
    "XGBoost" : XGBClassifier(random_state = 42),
    "CatBoost" : CatBoostClassifier(random_state = 42, verbose = 0, allow_writing_files = False),
}

# 訓練並取出重要度
importance = pd.DataFrame(index = FEATURES)
for name, model in models.items():
    model.fit(X_train, y_train)
    values = pd.Series(model.feature_importances_, index = FEATURES)
    importance[name] = values / values.sum() * 100 # 統一換成百分比

print(importance.round(1))

# 畫圖並存檔
ax = importance.plot.barh(figsize = (8, 5)) # 橫向長條圖，類別名稱在左邊，比較好讀
ax.set_xlabel("重要度 (%) ")
ax.set_title("三種模型的特徵重要度比較")
ax.invert_yaxis() # 讓「性別」排在最上面，順序跟 FEATURES 一致
plt.tight_layout()
plt.savefig(Path(__file__).parent / "feature_importance.png", dpi = 150) # 解析度，數字越大越清晰
plt.show()