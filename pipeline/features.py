"""Turn play-by-play + schedules into pre-game team features (only information available before kickoff)."""
import numpy as np
import pandas as pd

# tuning (chosen on 2013-2019 games only, then frozen for 2020+ testing)
GAME_DECAY = 0.88      # weight on older games each new game (~5.4-game half-life)
SEASON_CARRY = 0.45    # how much of last season's evidence survives the offseason
PRIOR_GAMES = 2.0      # shrink each team toward league average by this many games
QB_DECAY = 0.94
QB_SEASON_CARRY = 0.65
QB_PRIOR_DB = 250.0    # dropbacks of prior shrinkage for QBs
QB_PRIOR_EPA = -0.05   # unknown/backup QB starts below average

TEAM_METRICS = ["off_epa", "def_epa", "off_sr", "def_sr", "off_pass", "def_pass",
                "off_rush", "def_rush", "pf", "pa", "plays"]


def play_aggregates(pbp):
    p = pbp[pbp["season_type"].isin(["REG", "POST"])]
    p = p[p["play_type"].isin(["pass", "run"]) & p["epa"].notna()]
    p = p[(p["qb_kneel"] != 1) & (p["qb_spike"] != 1) & (p["two_point_attempt"] != 1)]
    p = p.assign(pass_epa=np.where(p["pass"] == 1, p["epa"], np.nan),
                 rush_epa=np.where(p["rush"] == 1, p["epa"], np.nan))
    off = p.groupby(["game_id", "posteam"]).agg(
        off_epa=("epa", "mean"), off_sr=("success", "mean"), off_pass=("pass_epa", "mean"),
        off_rush=("rush_epa", "mean"), plays=("epa", "size")).reset_index().rename(columns={"posteam": "team"})
    d = p.groupby(["game_id", "defteam"]).agg(
        def_epa=("epa", "mean"), def_sr=("success", "mean"), def_pass=("pass_epa", "mean"),
        def_rush=("rush_epa", "mean")).reset_index().rename(columns={"defteam": "team"})
    team = off.merge(d, on=["game_id", "team"], how="outer")
    db = p[p["qb_dropback"] == 1].dropna(subset=["passer_player_id"])
    qb = db.groupby(["game_id", "posteam", "passer_player_id"]).agg(
        n=("epa", "size"), epa_sum=("epa", "sum"), name=("passer_player_name", "first")).reset_index()
    return team, qb


class _EW:
    """Exponentially weighted mean with season carry-over and a prior."""
    def __init__(self):
        self.s, self.w = {}, {}

    def decay(self, k, f):
        if k in self.w:
            self.s[k] *= f
            self.w[k] *= f

    def add(self, k, val, weight=1.0):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return
        self.s[k] = self.s.get(k, 0.0) + val * weight
        self.w[k] = self.w.get(k, 0.0) + weight

    def get(self, k, prior, prior_w):
        return (self.s.get(k, 0.0) + prior * prior_w) / (self.w.get(k, 0.0) + prior_w)


def build(games, team_games, qb_games):
    """Return one row per game with home/away pre-game features. Walks games in date order."""
    g = games[games["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])].copy()
    g = g.sort_values(["gameday", "gametime", "game_id"]).reset_index(drop=True)
    tg = team_games.set_index(["game_id", "team"])
    starters = (qb_games.sort_values("n", ascending=False).drop_duplicates(["game_id", "posteam"])
                .set_index(["game_id", "posteam"]))
    qbg = qb_games.groupby("game_id")

    league = {m: _EW() for m in ["off_epa", "off_sr", "off_pass", "off_rush", "pf", "plays"]}
    T = _EW()          # key: (team, metric)
    Q = _EW()          # key: qb id
    last_starter = {}
    last_season = None
    rows = []

    def lg(m):
        return league[m].get("x", {"off_epa": 0.0, "off_sr": 0.45, "off_pass": 0.05, "off_rush": -0.08,
                                    "pf": 22.0, "plays": 62.0}[m], 1.0)

    for _, r in g.iterrows():
        if r["season"] != last_season:
            for k in list(T.w):
                T.decay(k, SEASON_CARRY)
            for k in list(Q.w):
                Q.decay(k, QB_SEASON_CARRY)
            last_season = r["season"]
        h, a = r["home_team"], r["away_team"]
        feat = {"game_id": r["game_id"], "season": r["season"], "week": r["week"], "gameday": r["gameday"],
                "home": h, "away": a}
        means = {m: lg(m) for m in league}
        for side, t in (("h", h), ("a", a)):
            for m in TEAM_METRICS:
                base = {"def_epa": "off_epa", "def_sr": "off_sr", "def_pass": "off_pass", "def_rush": "off_rush",
                        "pa": "pf"}.get(m, m)
                feat[f"{side}_{m}"] = T.get((t, m), means[base], PRIOR_GAMES)
            qid = r.get(f"{'home' if side == 'h' else 'away'}_qb_id")
            if not isinstance(qid, str):
                qid = last_starter.get(t)
            feat[f"{side}_qb"] = Q.get(qid, QB_PRIOR_EPA, QB_PRIOR_DB) if qid else QB_PRIOR_EPA
            feat[f"{side}_qb_change"] = int(bool(qid) and last_starter.get(t) not in (None, qid))
            feat[f"{side}_qb_name"] = r.get(f"{'home' if side == 'h' else 'away'}_qb_name")
        for c in ["spread_line", "total_line", "home_moneyline", "away_moneyline", "result", "total",
                  "home_score", "away_score", "home_rest", "away_rest", "div_game", "roof", "wind", "temp",
                  "location", "gametime", "home_spread_odds", "away_spread_odds", "over_odds", "under_odds"]:
            feat[c] = r.get(c)
        rows.append(feat)

        # ---- update state with this game's results (if played) ----
        if pd.isna(r["result"]):
            continue
        pts = {h: r["home_score"], a: r["away_score"]}
        for t, o in ((h, a), (a, h)):
            for k in [k for k in T.w if k[0] == t]:
                T.decay(k, GAME_DECAY)
            if (r["game_id"], t) in tg.index:
                x = tg.loc[(r["game_id"], t)]
                y = tg.loc[(r["game_id"], o)] if (r["game_id"], o) in tg.index else None
                # opponent-adjust: credit/blame relative to what the opponent usually allows/produces
                for m, opp_m, base in (("off_epa", "def_epa", "off_epa"), ("off_sr", "def_sr", "off_sr"),
                                       ("off_pass", "def_pass", "off_pass"), ("off_rush", "def_rush", "off_rush")):
                    opp_rating = feat[f"{'a' if t == h else 'h'}_{opp_m}"]
                    T.add((t, m), x[m] - (opp_rating - means[base]))
                if y is not None:
                    for m, opp_m, base in (("def_epa", "off_epa", "off_epa"), ("def_sr", "off_sr", "off_sr"),
                                           ("def_pass", "off_pass", "off_pass"), ("def_rush", "off_rush", "off_rush")):
                        opp_rating = feat[f"{'a' if t == h else 'h'}_{opp_m}"]
                        T.add((t, m), x[m] - (opp_rating - means[base]))
                T.add((t, "plays"), x["plays"])
            opp_pa = feat[f"{'a' if t == h else 'h'}_pa"]
            opp_pf = feat[f"{'a' if t == h else 'h'}_pf"]
            T.add((t, "pf"), pts[t] - (opp_pa - means["pf"]))
            T.add((t, "pa"), pts[o] - (opp_pf - means["pf"]))
        for m in league:
            league[m].decay("x", 0.995)
        for t in (h, a):
            if (r["game_id"], t) in tg.index:
                x = tg.loc[(r["game_id"], t)]
                for m in ["off_epa", "off_sr", "off_pass", "off_rush", "plays"]:
                    league[m].add("x", x[m])
            league["pf"].add("x", pts[t])
        if r["game_id"] in qbg.groups:
            for _, q in qbg.get_group(r["game_id"]).iterrows():
                Q.decay(q["passer_player_id"], QB_DECAY)
                Q.add(q["passer_player_id"], q["epa_sum"] / q["n"], q["n"])
        for t in (h, a):
            if (r["game_id"], t) in starters.index:
                last_starter[t] = starters.loc[(r["game_id"], t), "passer_player_id"]

    f = pd.DataFrame(rows)
    return add_model_columns(f)


def add_model_columns(f):
    f = f.copy()
    for m in TEAM_METRICS + ["qb"]:
        f[f"d_{m}"] = f[f"h_{m}"] - f[f"a_{m}"]
    f["neutral"] = (f["location"] == "Neutral").astype(int)
    f["rest_diff"] = (f["home_rest"].fillna(7) - f["away_rest"].fillna(7)).clip(-7, 7)
    f["div_game"] = f["div_game"].fillna(0)
    f["indoor"] = f["roof"].isin(["dome", "closed"]).astype(int)
    f["wind_x"] = np.where(f["indoor"] == 1, 0, f["wind"].fillna(8)).clip(0, 30)
    f["cold"] = np.where(f["indoor"] == 1, 0, (40 - f["temp"].fillna(55)).clip(0, 40))
    # strength = offense minus defense allowed (both opponent-adjusted)
    f["d_net_epa"] = (f["h_off_epa"] - f["h_def_epa"]) - (f["a_off_epa"] - f["a_def_epa"])
    f["d_net_pts"] = (f["h_pf"] - f["h_pa"]) - (f["a_pf"] - f["a_pa"])
    f["s_off_epa"] = f["h_off_epa"] + f["a_off_epa"]
    f["s_def_epa"] = f["h_def_epa"] + f["a_def_epa"]
    f["s_pts"] = f["h_pf"] + f["a_pf"] + f["h_pa"] + f["a_pa"]
    f["s_plays"] = f["h_plays"] + f["a_plays"]
    f["s_qb"] = f["h_qb"] + f["a_qb"]
    f["home_field"] = 1 - f["neutral"]
    return f


MARGIN_FEATURES = ["home_field", "d_net_epa", "d_net_pts", "d_off_pass", "d_def_pass", "d_qb",
                   "h_qb_change", "a_qb_change", "rest_diff", "div_game"]
TOTAL_FEATURES = ["s_off_epa", "s_def_epa", "s_pts", "s_plays", "s_qb", "indoor", "wind_x", "cold"]
