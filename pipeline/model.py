"""Team model: ridge regressions for home margin and game total, plus grading helpers."""
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .features import MARGIN_FEATURES, TOTAL_FEATURES

MARGIN_SD = 13.2
TOTAL_SD = 12.8

# confidence tiers on |model - market| (points); set from the 2013-2019 tuning seasons
SPREAD_TIERS = [(1.5, "Low"), (3.0, "Medium"), (4.5, "High")]
TOTAL_TIERS = [(2.0, "Low"), (3.5, "Medium"), (5.0, "High")]
ML_TIERS = [(0.03, "Low"), (0.05, "Medium"), (0.08, "High")]


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def implied(o):
    if o is None or pd.isna(o):
        return None
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)


def payout(o):
    return o / 100 if o > 0 else 100 / -o


def tier(x, tiers):
    t = "Pass"
    for cut, name in tiers:
        if x >= cut:
            t = name
    return t


class TeamModel:
    def __init__(self, alpha=5.0):
        self.alpha = alpha

    def fit(self, f):
        d = f.dropna(subset=["result"])
        # weight recent seasons more
        w = 0.85 ** (d["season"].max() - d["season"])
        self.m = Ridge(alpha=self.alpha).fit(d[MARGIN_FEATURES].fillna(0), d["result"], sample_weight=w)
        self.t = Ridge(alpha=self.alpha).fit(d[TOTAL_FEATURES].fillna(0), d["total"], sample_weight=w)
        return self

    def predict(self, f):
        out = f.copy()
        out["proj_margin"] = self.m.predict(f[MARGIN_FEATURES].fillna(0))
        out["proj_total"] = self.t.predict(f[TOTAL_FEATURES].fillna(0))
        out["proj_home"] = (out["proj_total"] + out["proj_margin"]) / 2
        out["proj_away"] = (out["proj_total"] - out["proj_margin"]) / 2
        out["p_home"] = [phi(m / MARGIN_SD) for m in out["proj_margin"]]
        return out

    def coefficients(self):
        return ({k: round(float(v), 3) for k, v in zip(MARGIN_FEATURES, self.m.coef_)},
                {k: round(float(v), 3) for k, v in zip(TOTAL_FEATURES, self.t.coef_)})


def leans(r):
    """Spread / total / moneyline leans for one projected game row (market lines may be None)."""
    out = {}
    sp = r.get("spread_line")  # positive = home favored
    if sp is not None and not pd.isna(sp):
        e = r["proj_margin"] - sp
        home = e >= 0
        p_home_cover = phi(e / MARGIN_SD)
        out["spread"] = {"side": "home" if home else "away", "line": float(-sp if home else sp),
                         "edge": round(abs(e), 2), "p": round(p_home_cover if home else 1 - p_home_cover, 3),
                         "conf": tier(abs(e), SPREAD_TIERS)}
    tl = r.get("total_line")
    if tl is not None and not pd.isna(tl):
        e = r["proj_total"] - tl
        p_over = phi(e / TOTAL_SD)
        out["total"] = {"side": "over" if e >= 0 else "under", "line": float(tl), "edge": round(abs(e), 2),
                        "p": round(p_over if e >= 0 else 1 - p_over, 3), "conf": tier(abs(e), TOTAL_TIERS)}
    mh, ma = r.get("home_moneyline"), r.get("away_moneyline")
    if mh is not None and ma is not None and not pd.isna(mh) and not pd.isna(ma):
        ih, ia = implied(mh), implied(ma)
        nv = ih / (ih + ia)
        d = r["p_home"] - nv
        home = d >= 0
        p = r["p_home"] if home else 1 - r["p_home"]
        price = mh if home else ma
        out["ml"] = {"side": "home" if home else "away", "price": float(price), "p": round(p, 3),
                     "market_p": round(nv if home else 1 - nv, 3), "diff": round(abs(d), 3),
                     "ev": round(p * payout(price) - (1 - p), 3), "conf": tier(abs(d), ML_TIERS)}
    return out


def grade(r, lean):
    """Grade a lean dict against a finished game row. Returns {'spread': 'W'|'L'|'P', ...}."""
    g = {}
    if pd.isna(r.get("result")):
        return g
    res, tot = r["result"], r["total"]
    if "spread" in lean:
        s = lean["spread"]
        cover = res - r["spread_line"]  # >0 home covers
        g["spread"] = "P" if cover == 0 else ("W" if (cover > 0) == (s["side"] == "home") else "L")
    if "total" in lean:
        s = lean["total"]
        g["total"] = "P" if tot == s["line"] else ("W" if (tot > s["line"]) == (s["side"] == "over") else "L")
    if "ml" in lean:
        s = lean["ml"]
        g["ml"] = "P" if res == 0 else ("W" if (res > 0) == (s["side"] == "home") else "L")
    return g


def units(result, price=-110):
    return {"W": payout(price), "L": -1.0, "P": 0.0}.get(result, 0.0)
