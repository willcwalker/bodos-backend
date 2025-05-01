try:
    import numpy.random._pickle as _nprp
    import numpy.random._mt19937 as _mt

    # save the original one-arg ctor
    _orig_ctor = _nprp.__bit_generator_ctor

    def __bit_generator_ctor(bit_generator_name='MT19937'):
        # if they passed us the class, just instantiate it
        if isinstance(bit_generator_name, type) and issubclass(bit_generator_name, _mt.MT19937):
            return bit_generator_name()
        # otherwise delegate back to the original name lookup
        return _orig_ctor(bit_generator_name)

    # override numpy's private ctor
    _nprp.__bit_generator_ctor = __bit_generator_ctor

except Exception:
    pass

import joblib
import json
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

MODEL  = joblib.load("interval_model.pkl")
PARAMS = json.load(open("wait_params.json"))

def apply_map(interval):
    w  = PARAMS["slope"] * interval + PARAMS["intercept"]
    lo = w - 1.96 * PARAMS["sigma"]
    hi = w + 1.96 * PARAMS["sigma"]
    return w, lo, hi

def make_features(ts: datetime):
    h = ts.hour
    return pd.DataFrame([{
        "sin_hour": np.sin(2*np.pi*h/24),
        "cos_hour": np.cos(2*np.pi*h/24),
        "temp": 20,
        "precip_flag": 0,
        "weekend": int(ts.weekday() >= 5),
        "game_day": 0,
        "exam_week": 0,
        "big_event": 0,
    }])

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.get("/wait")
def wait(timestamp: str = Query(...)):
    try:
        ts = datetime.fromisoformat(timestamp.replace("Z",""))
    except Exception:
        raise HTTPException(400, "Bad timestamp")
    interval = float(MODEL.predict(make_features(ts))[0])
    w, lo, hi = apply_map(interval)
    return {"wait": round(w,2), "lower": round(lo,2), "upper": round(hi,2)}