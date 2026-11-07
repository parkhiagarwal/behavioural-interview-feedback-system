import sys
 
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
 
from features import FEATURES, LABELS
 
MODEL_FILE = "model.pkl"
 
 
def candidates():
    return {
        "Baseline (always the average)": DummyRegressor(strategy="mean"),
        "Ridge regression": make_pipeline(StandardScaler(), Ridge(alpha=3.0)),
        "Random forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0),
    }
 
 
def feature_importance(model):
    """Return a list of importance numbers, one per feature."""
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    return np.abs(model[-1].coef_)  # Ridge inside a pipeline: size of each coefficient
 
 
def train():
    df = pd.read_csv("dataset.csv").dropna()
    if len(df) < 15:
        sys.exit(f"Only {len(df)} samples. Collect at least 15 (ideally 40+) with step5_collect_data.py")
 
    X = df[FEATURES].values
    groups = df["speaker"].values
    n_speakers = df["speaker"].nunique()
    if n_speakers >= 3:
        cv = GroupKFold(n_splits=min(5, n_speakers))
        how = "each speaker is held out in turn, so the test is honest for NEW people"
    else:
        cv = KFold(n_splits=min(5, len(df)), shuffle=True, random_state=0)
        how = ("WARNING: fewer than 3 speakers, so the same person appears in training and "
               "testing and the accuracy will look better than it really is")
    bundle = {"features": FEATURES, "models": {}}
 
    print(f"Training on {len(df)} samples from {n_speakers} speaker(s)")
    print(f"Testing method: {how}\n")
 
    for label in LABELS:
        y = df[label].values
        print(f"=== {label.upper()} (ratings 1-5) ===")
 
        results, fitted = {}, {}
        for name, model in candidates().items():
            pred = cross_val_predict(model, X, y, cv=cv, groups=groups)  # predicted by a model that never saw it
            results[name] = mean_absolute_error(y, pred)
            print(f"  {name:<32} average error: {results[name]:.2f}")
            fitted[name] = model
 
        baseline = results["Baseline (always the average)"]
        best_name = min((k for k in results if not k.startswith("Baseline")), key=results.get)
        best = fitted[best_name].fit(X, y)  # final model learns from ALL the data
 
        gain = (baseline - results[best_name]) / baseline * 100
        if gain > 5:
            print(f"  -> Best: {best_name}. It is {gain:.0f}% more accurate than guessing the average.")
        else:
            print(f"  -> Best: {best_name}, but it is NOT clearly better than guessing the average.")
            print("     Collect more samples, more different speakers, or check your ratings.")
 
        imp = feature_importance(best)
        top = sorted(zip(FEATURES, imp), key=lambda kv: kv[1], reverse=True)[:5]
        print("  Most important features:", ", ".join(f"{k}" for k, _ in top), "\n")
 
        bundle["models"][label] = best
 
    joblib.dump(bundle, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")
 
 
def predict(feature_dict):
    """Use the saved model. Takes the dict from features.extract_features; returns ratings 1-5."""
    bundle = joblib.load(MODEL_FILE)
    x = np.array([[feature_dict[f] for f in bundle["features"]]])
    return {label: float(np.clip(m.predict(x)[0], 1, 5)) for label, m in bundle["models"].items()}
 
 
if __name__ == "__main__":
    train()