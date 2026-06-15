from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base_dir = Path(__file__).resolve().parent
csv_paths = sorted(base_dir.glob("*.csv"))
if not csv_paths:
    raise FileNotFoundError(f"No CSV files found in {base_dir}")

frames = [pd.read_csv(p) for p in csv_paths]
data = pd.concat(frames, ignore_index=True)

data.columns = data.columns.str.strip()
if "Label" not in data.columns:
    raise KeyError("Missing 'Label' column")

data["Label"] = (
    data["Label"].astype(str).str.strip().str.lower().map(lambda s: 0 if s == "benign" else 1)
)

non_numeric = [c for c in data.columns if data[c].dtype == "object"]
data = data.drop(columns=non_numeric, errors="ignore")

X = data.drop("Label", axis=1).replace([np.inf, -np.inf], np.nan).fillna(0)
y = data["Label"].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
}

results = {}
for name, clf in models.items():
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else y_pred
    results[name] = {
        "report": classification_report(y_te, y_pred, output_dict=True),
        "auc": roc_auc_score(y_te, y_prob),
        "cm": confusion_matrix(y_te, y_pred),
    }
    print(f"\n{name}\n{classification_report(y_te, y_pred)}ROC-AUC: {results[name]['auc']}\nConfusion Matrix:\n{results[name]['cm']}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
im = None
for i, (name, res) in enumerate(results.items()):
    ax = axes[i]
    cm = res["cm"]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            ax.text(c, r, cm[r, c], ha="center", va="center", color="red", fontsize=10)
fig.colorbar(im, ax=axes.ravel().tolist()).ax.set_ylabel("Count", rotation=270, labelpad=15)

out_dir = Path("/mnt/d/College/AI-ML/DemoScreenshots")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "CICIDS2017_confusions.png"
plt.savefig(out_path, dpi=200)
print(f"Saved confusion matrices to: {out_path}")
