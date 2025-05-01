# app.py
import json
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ─── 1) Load & train the same pipeline on startup ────────────────────────
print("⏳ Loading data and training ensemble pipeline...")

# 1a. Load synthetic data
df = pd.read_csv("synthetic_bodos_orders.csv", parse_dates=["order_time"])

# 1b. Time-split
cutoff = pd.Timestamp("2025-07-01")
train = df[df["order_time"] < cutoff]

# 1c. Features & target
CONT = ["sin_hour", "cos_hour", "temp"]
BIN  = ["precip_flag", "weekend", "game_day", "exam_week", "big_event"]
X_train = train[CONT + BIN]
y_train = train["next_order_interval_min"]

# 1d. Build preprocessing + model pipeline
pre = ColumnTransformer([
    ("cont", StandardScaler(), CONT),
    ("bin",  "passthrough",        BIN),
])
rf = RandomForestRegressor(
    n_estimators=150, min_samples_leaf=3,
    random_state=42, n_jobs=-1
)
gb = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05,
    max_depth=3, random_state=42
)

# Here you can swap in whatever ensemble you prefer.
# For simplicity we'll just use GBM; or stack them in a VotingRegressor if you like.
MODEL = Pipeline([("pre", pre), ("model", gb)])
MODEL.fit(X_train, y_train)

print("✅ Model trained and ready to serve.")

# 2) Load wait‐mapping params
PARAMS = json.load(open("wait_params.json"))

def apply_map(interval: float):
    w  = PARAMS["slope"] * interval + PARAMS["intercept"]
    lo = w - 1.96 * PARAMS["sigma"]
    hi = w + 1.96 * PARAMS["sigma"]
    return {"wait": round(w, 2), "lower": round(lo, 2), "upper": round(hi, 2)}

def make_features(ts: datetime):
    h = ts.hour
    return pd.DataFrame([{
        "sin_hour":    np.sin(2 * np.pi * h / 24),
        "cos_hour":    np.cos(2 * np.pi * h / 24),
        "temp":        20,               # placeholder; plug real weather here if you want
        "precip_flag": 0,
        "weekend":     int(ts.weekday() >= 5),
        "game_day":    0,
        "exam_week":   0,
        "big_event":   0,
    }])

# 3) FastAPI wiring
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.get("/wait")
def wait(timestamp: str = Query(..., description="ISO date-time")):
    try:
        ts = datetime.fromisoformat(timestamp.replace("Z", ""))
    except Exception:
        raise HTTPException(400, "Bad timestamp format")
    feats = make_features(ts)
    interval = float(MODEL.predict(feats)[0])
    return apply_map(interval)