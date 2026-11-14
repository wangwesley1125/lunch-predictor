import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier
from common import load_data, encode, save_encoding

OUT_DIR = Path(__file__).parent / "models"
OUT_DIR.mkdir(exist_ok=True)

train, test = load_data()
all_data = pd.concat([train, test])
X, y = encode(all_data)

model = CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False)
model.fit(X, y)

model.save_model(
    str(OUT_DIR / "LunchPredictor.mlmodel"),
    format="coreml",
    export_parameters={"prediction_type": "probability"},
)
save_encoding()
print("已匯出 models/LunchPredictor.mlmodel")

proba = model.predict_proba(X.head(3))
print(pd.DataFrame(proba, columns=["飯", "麵", "速食"], index=X.head(3).index).round(3))