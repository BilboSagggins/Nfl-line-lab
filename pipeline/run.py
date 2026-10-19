"""Weekly pipeline. Usage: python -m pipeline.run --mode thursday|sunday|grade|local

thursday : retrain, pull lines + player props, record this week's picks, grade old picks
sunday   : refresh game lines only (cheap on API credits); picks stay frozen from Thursday
grade    : no odds calls; grade finished games/props, refresh model + backtest
local    : like thursday but never calls the Odds API (uses nflverse lines)
"""
import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import data, features, odds as oddsmod, props as propmod
from .model import TeamModel, leans, grade, units, implied, payout, MARGIN_SD

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "docs", "data")
HIST_SEASONS = 14
VALUE_EV = 0.02        # min edge vs consensus fair price to flag a "best line" value
PROP_EV = 0.03
PROP_MIN_LEAN = 0.06
def half_points(a, b, spread=True):
    """Approx win-probability value of moving from line a to the better line b (key numbers 3 and 7 count more)."""
    lo, hi = sorted((a, b))
    v, x = 0.0, lo
    while x < hi - 1e-9:
        if spread:
            k = abs(round(x * 2) / 2)
            nxt = abs(round((x + 0.5) * 2) / 2)
            key = 3 in (k, nxt) or 7 in (k, nxt)
            v += (0.045 if 3 in (k, nxt) else 0.02) if key else 0.012
        else:
            v += 0.015
        x += 0.5
    return v if b >= a else -v


def log(*a):
    print("[pipeline]", *a, file=sys.stderr, flush=True)


def jdump(obj, name):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as fh:
        json.dump(obj, fh, indent=1, default=_json_default)


def jload(name, default):
    try:
        with open(os.path.join(OUT, name)) as fh:
            return json.load(fh)
    except Exception:
        return default


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else round(float(o), 4)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return str(o)


def clean(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


# ---------------------------------------------------------------- data + model

def load_all(season):
    games = data.schedules()
    tgs, qbs = [], []
    for y in range(season - HIST_SEASONS + 1, season + 1):
        try:
            p = data.pbp(y, season)
        except Exception as e:  # current-season file can lag in week 1
            log("pbp missing", y, e)
            continue
        a, b = features.play_aggregates(p)
        tgs.append(a)
        qbs.append(b)
    feat = features.build(games[games["season"] >= season - HIST_SEASONS + 1], pd.concat(tgs), pd.concat(qbs))
    ps = []
    for y in range(season - 3, season + 1):
        try:
            ps.append(data.player_stats(y, season))
        except Exception as e:
            log("player stats missing", y, e)
    ps = pd.concat(ps) if ps else pd.DataFrame()
    try:
        inj = pd.read_csv(data._get(f"{data.BASE}/injuries/injuries_{season}.csv",
                                    os.path.join(data.CACHE, f"inj_{season}.csv"), 6))
    except Exception:
        inj = pd.DataFrame()
    return games, feat, ps, inj


def target_week(feat, season):
    up = feat[(feat["season"] == season) & feat["result"].isna()]
    if up.empty:
        return None
    today = pd.Timestamp.now(tz="America/Chicago").tz_localize(None).normalize()
    up = up[up["gameday"] >= today - pd.Timedelta(days=1)]
    if up.empty:
        return None
    # a week with only a leftover Monday game loses to the next full slate
    soon = up[up["gameday"] <= today + pd.Timedelta(days=9)]
    if soon.empty:
        return int(up.sort_values("gameday")["week"].iloc[0])
    counts = soon.groupby("week").size()
    return int(counts[counts == counts.max()].index.min())


# ---------------------------------------------------------------- game board

def team_form(feat, team, before):
    g = feat[((feat["home"] == team) | (feat["away"] == team)) & feat["result"].notna() & (feat["gameday"] < before)]
    g = g.sort_values("gameday").tail(5)
    out = []
    for _, r in g.iterrows():
        home = r["home"] == team
        pf, pa = (r["home_score"], r["away_score"]) if home else (r["away_score"], r["home_score"])
        ats = None
        if not pd.isna(r["spread_line"]):
            cov = (r["result"] - r["spread_line"]) * (1 if home else -1)
            ats = "P" if cov == 0 else ("W" if cov > 0 else "L")
        out.append({"opp": r["away"] if home else r["home"], "home": bool(home), "pf": int(pf), "pa": int(pa),
                    "res": "W" if pf > pa else ("L" if pf < pa else "T"), "ats": ats,
                    "ou": None if pd.isna(r["total_line"]) else ("O" if r["total"] > r["total_line"] else
                                                                  "U" if r["total"] < r["total_line"] else "P")})
    return out


def injuries_for(inj, season, week, team):
    if inj.empty:
        return []
    d = inj[(inj["season"] == season) & (inj["week"] == week) & (inj["team"] == team)]
    d = d[d["report_status"].isin(["Out", "Doubtful", "Questionable"])]
    order = {p: i for i, p in enumerate(["QB", "RB", "WR", "TE", "T", "G", "C", "DE", "DT", "LB", "CB", "S"])}
    d = d.assign(o=d["position"].map(order).fillna(99)).sort_values(["o", "report_status"])
    return [{"name": r["full_name"], "pos": r["position"], "status": r["report_status"],
             "injury": clean(r.get("report_primary_injury"))} for _, r in d.head(12).iterrows()]


def market_value(book, model_row):
    """Line-shopping value: best available price vs the consensus fair (no-vig) price."""
    vals = []
    if "spread" in book and book["spread"].get("fair_home"):
        s = book["spread"]
        for side, best_key, fair in (("home", "best_home", s["fair_home"]), ("away", "best_away", 1 - s["fair_home"])):
            b = s.get(best_key)
            if not b:
                continue
            cons_line = s["home_line"] if side == "home" else -s["home_line"]
            p = fair + half_points(cons_line, b["line"], True)
            ev = p * payout(b["price"]) - (1 - p)
            if ev >= VALUE_EV:
                vals.append({"market": "spread", "side": side, "line": b["line"], "price": b["price"],
                             "book": b["book"], "ev": round(ev, 3), "fair_p": round(p, 3)})
    if "total" in book and book["total"].get("fair_over"):
        t = book["total"]
        for side, best_key, fair in (("over", "best_over", t["fair_over"]), ("under", "best_under", 1 - t["fair_over"])):
            b = t.get(best_key)
            if not b:
                continue
            better = (b["line"] < t["line"]) if side == "over" else (b["line"] > t["line"])
            p = fair + (abs(half_points(t["line"], b["line"], False)) * (1 if better else -1))
            ev = p * payout(b["price"]) - (1 - p)
            if ev >= VALUE_EV:
                vals.append({"market": "total", "side": side, "line": b["line"], "price": b["price"],
                             "book": b["book"], "ev": round(ev, 3), "fair_p": round(p, 3)})
    if "ml" in book and book["ml"].get("fair_home"):
        m = book["ml"]
        for side, best_key, p in (("home", "best_home", m["fair_home"]), ("away", "best_away", 1 - m["fair_home"])):
            b = m.get(best_key)
            if not b:
                continue
            ev = p * payout(b["price"]) - (1 - p)
            if ev >= VALUE_EV:
                vals.append({"market": "ml", "side": side, "line": None, "price": b["price"], "book": b["book"],
                             "ev": round(ev, 3), "fair_p": round(p, 3)})
    return sorted(vals, key=lambda v: -v["ev"])


def build_board(feat, model, season, week, odds_games, inj):
    wk = feat[(feat["season"] == season) & (feat["week"] == week)].copy()
    oddsmap = {(g["away"], g["home"]): g for g in (odds_games or [])}
    # prefer live consensus lines when we have them
    for i, r in wk.iterrows():
        o = oddsmap.get((r["away"], r["home"]))
        if not o:
            continue
        if "spread" in o:
            wk.at[i, "spread_line"] = -o["spread"]["home_line"]
        if "total" in o:
            wk.at[i, "total_line"] = o["total"]["line"]
        if "ml" in o:
            wk.at[i, "home_moneyline"] = o["ml"]["home"]
            wk.at[i, "away_moneyline"] = o["ml"]["away"]
    wk = model.predict(wk)
    ranks = feat[(feat["season"] == season) & (feat["week"] == week)]
    board = []
    for _, r in wk.sort_values(["gameday", "gametime"]).iterrows():
        L = leans(r)
        o = oddsmap.get((r["away"], r["home"]))
        g = {
            "game_id": r["game_id"], "week": int(r["week"]), "gameday": r["gameday"].strftime("%Y-%m-%d"),
            "gametime": clean(r["gametime"]), "home": r["home"], "away": r["away"],
            "kickoff": o["commence"] if o else None,
            "home_qb": clean(r["h_qb_name"]), "away_qb": clean(r["a_qb_name"]),
            "qb_change": {"home": bool(r["h_qb_change"]), "away": bool(r["a_qb_change"])},
            "market": {"home_line": None if pd.isna(r["spread_line"]) else -float(r["spread_line"]),
                       "total": clean(r["total_line"]), "ml_home": clean(r["home_moneyline"]),
                       "ml_away": clean(r["away_moneyline"]),
                       "source": f"{o['books']} sportsbooks (median)" if o else "nflverse consensus"},
            "model": {"home_pts": round(r["proj_home"], 1), "away_pts": round(r["proj_away"], 1),
                      "margin": round(r["proj_margin"], 1), "total": round(r["proj_total"], 1),
                      "p_home": round(r["p_home"], 3)},
            "leans": L,
            "books": {k: v for k, v in (o or {}).items() if k in ("spread", "total", "ml", "by_book")},
            "value": market_value(o, r) if o else [],
            "context": {
                "rest": {"home": clean(r["home_rest"]), "away": clean(r["away_rest"])},
                "roof": clean(r["roof"]), "wind": clean(r["wind"]), "temp": clean(r["temp"]),
                "div_game": bool(r["div_game"]),
                "net_epa": {"home": round(r["h_off_epa"] - r["h_def_epa"], 3),
                            "away": round(r["a_off_epa"] - r["a_def_epa"], 3)},
                "qb_epa": {"home": round(r["h_qb"], 3), "away": round(r["a_qb"], 3)},
                "form": {"home": team_form(feat, r["home"], r["gameday"]),
                         "away": team_form(feat, r["away"], r["gameday"])},
                "injuries": {"home": injuries_for(inj, season, week, r["home"]),
                             "away": injuries_for(inj, season, week, r["away"])},
            },
        }
        board.append(g)
    # team strength table (opponent-adjusted net EPA per play, latest)
    return board


def team_table(feat, season, week):
    wk = feat[(feat["season"] == season) & (feat["week"] == week)]
    rows = []
    for _, r in wk.iterrows():
        for s in ("h", "a"):
            t = r["home"] if s == "h" else r["away"]
            rows.append({"team": t, "off_epa": round(r[f"{s}_off_epa"], 3), "def_epa": round(r[f"{s}_def_epa"], 3),
                         "net": round(r[f"{s}_off_epa"] - r[f"{s}_def_epa"], 3),
                         "pts_for": round(r[f"{s}_pf"], 1), "pts_against": round(r[f"{s}_pa"], 1),
                         "qb": clean(r[f"{s}_qb_name"]), "qb_epa": round(r[f"{s}_qb"], 3)})
    return sorted(rows, key=lambda x: -x["net"])


# ---------------------------------------------------------------- props

def build_props(ps, feat, season, week, prop_lines, board):
    if ps.empty:
        return []
    hist = propmod.history(ps, feat)
    teams = {}
    for g in board:
        teams[g["home"]] = (g["away"], g["game_id"])
        teams[g["away"]] = (g["home"], g["game_id"])
    cur = hist[(hist["season"] == season)].sort_values("week")
    latest = cur.drop_duplicates("player_id", keep="last")
    latest = latest[latest["team"].isin(teams) & latest["position"].isin(propmod.POS)]
    # synthetic upcoming rows
    up = latest[["player_id", "player_display_name", "position", "team"]].copy()
    up["season"], up["week"], up["season_type"] = season, week, "REG"
    up["game_id"] = up["team"].map(lambda t: teams[t][1])
    up["opponent_team"] = up["team"].map(lambda t: teams[t][0])
    it = propmod.implied_totals(feat[(feat["season"] == season) & (feat["week"] == week)])
    up = up.merge(it, on=["game_id", "team"], how="left")
    allrows = pd.concat([hist, up], ignore_index=True)
    allrows = propmod.pregame(allrows)
    model = propmod.PropModel().fit(allrows[(allrows["season"] < season) | (allrows["week"] < week)].dropna(
        subset=["pass_yds"]))
    upc = allrows[(allrows["season"] == season) & (allrows["week"] == week)].copy()
    idx = {oddsmod.norm_name(n): i for i, n in zip(upc.index, upc["player_display_name"])}
    out = []
    lines = prop_lines or []
    # if we have no posted lines, still project top usage players so the page is useful
    if not lines:
        for _, r in upc.iterrows():
            for k, (_, lab, pos) in propmod.STATS.items():
                if r["position"] not in pos or k in ("pass_att", "completions", "rush_att", "rush_rec_yds"):
                    continue
                ew = r.get(f"ew_{k}")
                if pd.isna(ew) or ew <= 0 or (k != "anytime_td" and ew < {"pass_yds": 150, "rush_yds": 25,
                                                                          "rec_yds": 25, "receptions": 2.5,
                                                                          "pass_tds": 0.8}.get(k, 0)):
                    continue
                proj, _ = model.project(upc.loc[[r.name]], k)
                out.append({"player": r["player_display_name"], "team": r["team"], "opp": r["opponent_team"],
                            "pos": r["position"], "stat": k, "label": lab, "proj": round(float(proj[0]), 2),
                            "line": None, "game_id": r["game_id"]})
        return out
    for pl in lines:
        i = idx.get(oddsmod.norm_name(pl["player"]))
        k = pl["stat"]
        rec = {**pl, "label": propmod.STATS[k][1]}
        if i is None:
            rec.update({"matched": False})
            out.append(rec)
            continue
        r = upc.loc[i]
        proj, _ = model.project(upc.loc[[i]], k)
        proj = float(proj[0])
        p_over = model.p_over(proj, None, pl["line"], k)
        p_model = 0.5 + 0.75 * (p_over - 0.5)  # shrink: high-confidence bins ran hot in testing
        fair = pl.get("fair_over")
        p_blend = (0.5 * fair + 0.5 * p_model) if fair else p_model
        ev_o = p_blend * payout(pl["best_over"]["price"]) - (1 - p_blend) if pl.get("best_over") else None
        ev_u = (1 - p_blend) * payout(pl["best_under"]["price"]) - p_blend if pl.get("best_under") else None
        side = "over" if p_model >= 0.5 else "under"
        ev = ev_o if side == "over" else ev_u
        last = allrows[(allrows["player_id"] == r["player_id"]) & allrows[k].notna()].tail(8)
        rec.update({
            "matched": True, "team": r["team"], "opp": r["opponent_team"], "pos": r["position"],
            "game_id": r["game_id"], "proj": round(proj, 2), "p_over_model": round(p_model, 3),
            "p_over_blend": round(p_blend, 3), "ev_over": None if ev_o is None else round(ev_o, 3),
            "ev_under": None if ev_u is None else round(ev_u, 3), "side": side,
            "signal": bool(ev is not None and ev >= PROP_EV and abs(p_model - 0.5) >= PROP_MIN_LEAN and fair),
            "recent": [clean(x) for x in last[k].tolist()],
            "hit_rate": round(float((last[k] > pl["line"]).mean()), 2) if len(last) else None,
        })
        out.append(rec)
    return sorted(out, key=lambda x: -(max(x.get("ev_over") or -1, x.get("ev_under") or -1)))


# ---------------------------------------------------------------- ledger

def record_picks(ledger, board, props_list, season, week):
    have = {p["id"] for p in ledger["picks"]}
    now = datetime.now(timezone.utc).isoformat()

    def add(p):
        if p["id"] not in have:
            p.update({"season": season, "week": week, "made_at": now, "result": None})
            ledger["picks"].append(p)
            have.add(p["id"])

    for g in board:
        gid, L = g["game_id"], g["leans"]
        name = f"{g['away']} @ {g['home']}"
        if "spread" in L and L["spread"]["conf"] in ("Medium", "High"):
            s = L["spread"]
            team = g["home"] if s["side"] == "home" else g["away"]
            add({"id": f"{gid}:model:spread", "game_id": gid, "kind": "model", "market": "spread", "side": s["side"],
                 "line": s["line"], "price": -110, "book": "consensus", "conf": s["conf"], "edge": s["edge"],
                 "desc": f"{team} {s['line']:+g} ({name})"})
        if "total" in L and L["total"]["conf"] in ("Medium", "High"):
            s = L["total"]
            add({"id": f"{gid}:model:total", "game_id": gid, "kind": "model", "market": "total", "side": s["side"],
                 "line": s["line"], "price": -110, "book": "consensus", "conf": s["conf"], "edge": s["edge"],
                 "desc": f"{s['side'].title()} {s['line']} ({name})"})
        for v in g["value"][:2]:
            team = g["home"] if v["side"] == "home" else g["away"] if v["side"] == "away" else None
            if v["market"] == "spread":
                d = f"{team} {v['line']:+g} {int(v['price']):+d} @ {v['book']} ({name})"
            elif v["market"] == "total":
                d = f"{v['side'].title()} {v['line']} {int(v['price']):+d} @ {v['book']} ({name})"
            else:
                d = f"{team} ML {int(v['price']):+d} @ {v['book']} ({name})"
            add({"id": f"{gid}:value:{v['market']}:{v['side']}", "game_id": gid, "kind": "value",
                 "market": v["market"], "side": v["side"], "line": v["line"], "price": v["price"], "book": v["book"],
                 "conf": None, "edge": v["ev"], "desc": d})
    for p in props_list:
        if not p.get("signal"):
            continue
        b = p["best_over"] if p["side"] == "over" else p["best_under"]
        add({"id": f"{p['game_id']}:prop:{oddsmod.norm_name(p['player'])}:{p['stat']}", "game_id": p["game_id"],
             "kind": "prop", "market": p["stat"], "side": p["side"], "line": p["line"], "price": b["price"],
             "book": b["book"], "conf": None, "edge": p["ev_over"] if p["side"] == "over" else p["ev_under"],
             "player": p["player"], "team": p["team"],
             "desc": f"{p['player']} {p['side']} {p['line']} {p['label']} {int(b['price']):+d} @ {b['book']}"})


def grade_ledger(ledger, games, ps):
    g = games.set_index("game_id")
    psn = None
    if not ps.empty:
        psn = ps[["game_id", "player_display_name", "passing_yards", "passing_tds", "attempts", "completions", "rushing_yards", "carries", "receptions", "receiving_yards", "rushing_tds", "receiving_tds"]].copy()
        psn["nn"] = psn["player_display_name"].map(oddsmod.norm_name)
    for p in ledger["picks"]:
        if p.get("result") not in (None, "pending"):
            continue
        if p["game_id"] not in g.index:
            continue
        r = g.loc[p["game_id"]]
        if pd.isna(r["result"]):
            p["result"] = "pending"
            continue
        # closing-line value (nflverse stores closing lines; only final once the game is played)
        if p["market"] == "spread" and not pd.isna(r["spread_line"]):
            close_home = -r["spread_line"]
            close = close_home if p["side"] == "home" else -close_home
            p["close_line"] = float(close)
            p["clv"] = round(p["line"] - close, 2)
        elif p["market"] == "total" and not pd.isna(r["total_line"]):
            p["close_line"] = float(r["total_line"])
            p["clv"] = round((r["total_line"] - p["line"]) if p["side"] == "over" else (p["line"] - r["total_line"]), 2)
        elif p["market"] == "ml" and not pd.isna(r["home_moneyline"]):
            close = r["home_moneyline"] if p["side"] == "home" else r["away_moneyline"]
            p["close_line"] = float(close)
            p["clv"] = round(implied(close) - implied(p["price"]), 3)
        res, tot = r["result"], r["total"]
        if p["kind"] == "prop":
            if psn is None:
                p["result"] = "pending"
                continue
            row = psn[(psn["game_id"] == p["game_id"]) & (psn["nn"] == oddsmod.norm_name(p["player"]))]
            if row.empty:
                # stats can lag a day; void after 4 days
                age = (pd.Timestamp.now() - pd.Timestamp(r["gameday"])).days
                p["result"] = "void" if age > 4 else "pending"
                continue
            cols = propmod.STATS[p["market"]][0]
            val = float(row[cols].fillna(0).sum(axis=1).iloc[0])
            p["actual"] = val
            p["result"] = "P" if val == p["line"] else ("W" if (val > p["line"]) == (p["side"] == "over") else "L")
        elif p["market"] == "spread":
            home_line = p["line"] if p["side"] == "home" else -p["line"]
            cov = res + home_line
            p["result"] = "P" if cov == 0 else ("W" if (cov > 0) == (p["side"] == "home") else "L")
        elif p["market"] == "total":
            p["result"] = "P" if tot == p["line"] else ("W" if (tot > p["line"]) == (p["side"] == "over") else "L")
        elif p["market"] == "ml":
            p["result"] = "P" if res == 0 else ("W" if (res > 0) == (p["side"] == "home") else "L")
        if p["result"] in ("W", "L", "P"):
            p["units"] = round(units(p["result"], p["price"]), 3)
    # summary
    summ = {}
    for p in ledger["picks"]:
        if p.get("result") not in ("W", "L", "P"):
            continue
        for key in (p["kind"], f"{p['kind']}:{p['market'] if p['kind'] != 'prop' else 'prop'}", "all"):
            s = summ.setdefault(key, {"w": 0, "l": 0, "p": 0, "units": 0.0, "clv": [], "n": 0})
            s[{"W": "w", "L": "l", "P": "p"}[p["result"]]] += 1
            s["units"] = round(s["units"] + p.get("units", 0), 3)
            s["n"] += 1
            if p.get("clv") is not None:
                s["clv"].append(p["clv"])
    for s in summ.values():
        s["avg_clv"] = round(float(np.mean(s["clv"])), 3) if s["clv"] else None
        s["beat_close"] = round(float(np.mean([c > 0 for c in s["clv"]])), 3) if s["clv"] else None
        del s["clv"]
    ledger["summary"] = summ
    return ledger


# ---------------------------------------------------------------- backtest

def backtest(feat, season):
    f = feat[(feat["season"] >= season - HIST_SEASONS + 1)]
    first = season - HIST_SEASONS + 4
    rows = []
    for Y in range(first, season + 1):
        tr, te = f[f["season"] < Y], f[(f["season"] == Y) & f["result"].notna() & f["spread_line"].notna()]
        if te.empty:
            continue
        m = TeamModel().fit(tr)
        P = m.predict(te)
        for _, r in P.iterrows():
            L = leans(r)
            G = grade(r, L)
            rows.append({"season": int(r["season"]), "week": int(r["week"]),
                         "s_edge": L.get("spread", {}).get("edge"), "s_conf": L.get("spread", {}).get("conf"),
                         "s_res": G.get("spread"), "t_edge": L.get("total", {}).get("edge"),
                         "t_conf": L.get("total", {}).get("conf"), "t_res": G.get("total"),
                         "m_err": abs(r["proj_margin"] - r["result"]), "k_err": abs(r["spread_line"] - r["result"]),
                         "tm_err": abs(r["proj_total"] - r["total"]), "tk_err": abs(r["total_line"] - r["total"])})
    R = pd.DataFrame(rows)

    def rec(d, col):
        d = d[d[col].isin(["W", "L"])]
        w, l = int((d[col] == "W").sum()), int((d[col] == "L").sum())
        return {"w": w, "l": l, "pct": round(w / (w + l), 3) if w + l else None, "units": round(w * 100 / 110 - l, 1)}

    out = {"generated": datetime.now(timezone.utc).isoformat(), "seasons": [], "tiers": {}}
    for Y, d in R.groupby("season"):
        out["seasons"].append({"season": int(Y), "games": len(d),
                               "margin_mae": round(d["m_err"].mean(), 2), "market_mae": round(d["k_err"].mean(), 2),
                               "total_mae": round(d["tm_err"].mean(), 2), "market_total_mae": round(d["tk_err"].mean(), 2),
                               "spread_med_high": rec(d[d["s_conf"].isin(["Medium", "High"])], "s_res"),
                               "total_med_high": rec(d[d["t_conf"].isin(["Medium", "High"])], "t_res")})
    for t in ["Low", "Medium", "High"]:
        out["tiers"][t] = {"spread": rec(R[R["s_conf"] == t], "s_res"), "total": rec(R[R["t_conf"] == t], "t_res")}
    out["overall"] = {"games": len(R), "margin_mae": round(R["m_err"].mean(), 2),
                      "market_mae": round(R["k_err"].mean(), 2)}
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="local", choices=["thursday", "sunday", "grade", "local"])
    args = ap.parse_args()
    season = data.current_season()
    log("season", season, "mode", args.mode)
    games, feat, ps, inj = load_all(season)
    model = TeamModel().fit(feat[feat["result"].notna() & (feat["season"] >= season - HIST_SEASONS + 1)])
    week = target_week(feat, season)
    latest = jload("latest.json", {})
    ledger = jload("ledger.json", {"picks": []})
    key = os.environ.get("ODDS_API_KEY", "").strip()
    api = oddsmod.OddsAPI(key) if key and args.mode in ("thursday", "sunday") else None
    notes = []

    if week is None:
        notes.append("No upcoming games found — season may be over.")
        board, plist = [], []
    else:
        odds_games, prop_lines = None, None
        if api:
            try:
                raw = api.games()
                odds_games = [oddsmod.summarize_game(g) for g in raw]
                log("odds games", len(odds_games), "credits left", api.remaining)
            except Exception as e:
                notes.append(f"Odds API game lines failed ({e}); using nflverse consensus lines.")
        elif args.mode in ("thursday", "sunday"):
            notes.append("No ODDS_API_KEY secret found — using nflverse consensus lines, no player props.")
        board = build_board(feat, model, season, week, odds_games, inj)
        if api and args.mode == "thursday":
            prop_lines = []
            wanted = {(g["away"], g["home"]) for g in board}
            try:
                evs = [e for e in api.events()
                       if (oddsmod.TEAM.get(e["away_team"]), oddsmod.TEAM.get(e["home_team"])) in wanted]
                for e in evs:
                    try:
                        prop_lines += oddsmod.summarize_props(api.props(e["id"]))
                    except Exception as ex:
                        log("props failed", e["id"], ex)
                log("prop lines", len(prop_lines), "credits left", api.remaining)
            except Exception as e:
                notes.append(f"Odds API props failed ({e}).")
        if args.mode == "sunday" or (args.mode == "grade" and latest.get("week") == week):
            plist = latest.get("props", [])  # keep Thursday's props
        else:
            plist = build_props(ps, feat, season, week, prop_lines, board)
        if args.mode == "thursday":
            record_picks(ledger, board, plist, season, week)

    ledger = grade_ledger(ledger, games, ps)
    bt = jload("backtest.json", None)
    if args.mode in ("grade", "local") or bt is None:
        bt = backtest(feat, season)
        jdump(bt, "backtest.json")
    m_coef, t_coef = model.coefficients()
    latest = {
        "generated": datetime.now(timezone.utc).isoformat(), "mode": args.mode, "season": season, "week": week,
        "credits_remaining": api.remaining if api else latest.get("credits_remaining"),
        "games": board, "props": plist, "teams": team_table(feat, season, week) if week else [],
        "notes": notes, "model": {"margin_coef": m_coef, "total_coef": t_coef, "margin_sd": MARGIN_SD},
    }
    jdump(latest, "latest.json")
    jdump(ledger, "ledger.json")
    log("done: games", len(board), "props", len(plist), "picks", len(ledger["picks"]))


if __name__ == "__main__":
    main()
