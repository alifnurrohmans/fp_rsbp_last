import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import joblib

# Load dataset
df = pd.read_csv("cyber_dataset.csv")

# Features
X = df[[f"q{i}" for i in range(1, 21)]]

# Labels
label_cols = [
    "offensive", "blue_team", "malware", "forensics",
    "network", "cloud", "appsec", "threatintel", "grc"
]
y = df[label_cols]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model menggunakan XGBoost (multi-label = one-vs-rest)
models = {}
for label in label_cols:
    model = XGBClassifier(
        eval_metric="logloss",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1
    )
    model.fit(X_train, y_train[label])
    models[label] = model

# Evaluasi
for label in label_cols:
    pred = models[label].predict(X_test)
    score = f1_score(y_test[label], pred)
    print(f"{label} F1-score:", score)

# Simpan model
joblib.dump(models, "career_models.pkl")
print("Model saved!")
