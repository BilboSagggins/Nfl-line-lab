"""Player prop projections: recent production x usage x expected game script (market-implied team total)."""
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

POS = ["QB", "RB", "WR", "TE"]
STATS = {
    # key: (column(s), label, positions)
    "pass_yds": (["passing_yards"], "Passing yards", ["QB"]),
    "pass_tds": (["passing_tds"], "Passing TDs", ["QB"]),
    "pass_att": (["attempts"], "Pass attempts", ["QB"]),
    "completions": (["completions"], "Completions", ["QB"]),
    "rush_yds": (["rushing_yards"], "Rushing yards", ["QB", "RB", "WR"]),
    "rush_att": (["carries"], "Rush attempts", ["RB"]),
    "receptions": (["receptions"], "Receptions", ["RB", "WR", "TE"]),
    "rec_yds": (["receiving_yards"], "Receiving yards", ["RB", "WR", "TE"]),
    "rush_rec_yds": (["rushing_yards", "receiving_yards"], "Rush + rec yards", ["RB", "WR", "TE"]),
    "anytime_td": (["rushing_tds", "receiving_tds"], "Anytime TD", ["RB", "WR", "TE"]),
}
ALPHA = 0.3  # EWM weight on the most recent game


def _stat(df, key):
    cols = STATS[key][0]
    return df[cols].fillna(0).sum(axis=1)


def history(ps, games_feat):
    """ps: weekly player stats (several seasons). games_feat: team-model feature frame (for market lines)."""
    p = ps[ps["season_type"].isin(["REG", "POST"]) & ps["position"].isin(POS)].copy()
    p = p.sort_values(["season", "week"])
    team = p.groupby(["game_id", "team"]).agg(team_att=("attempts", "sum"), team_car=("carries", "sum"),
                                              team_tgt=("targets", "sum")).reset_index()
    p = p.merge(team, on=["game_id", "team"], how="left")
    p["car_share"] = p["carries"] / p["team_car"].replace(0, np.nan)
    p["tgt_share"] = p["targets"] / p["team_tgt"].replace(0, np.nan)
    for k in STATS:
        p[k] = _stat(p, k)
    p = p.merge(implied_totals(games_feat), on=["game_id", "team"], how="left")
    return p


def implied_totals(f):
    g = f[["game_id", "home", "away", "spread_line", "total_line", "h_def_pass", "a_def_pass",
           "h_def_rush", "a_def_rush"]].dropna(subset=["total_line"])
    h = pd.DataFrame({"game_id": g["game_id"], "team": g["home"],
                      "team_total": g["total_line"] / 2 + g["spread_line"].fillna(0) / 2,
                      "opp_def_pass": g["a_def_pass"], "opp_def_rush": g["a_def_rush"]})
    a = pd.DataFrame({"game_id": g["game_id"], "team": g["away"],
                      "team_total": g["total_line"] / 2 - g["spread_line"].fillna(0) / 2,
                      "opp_def_pass": g["h_def_pass"], "opp_def_rush": g["h_def_rush"]})
    return pd.concat([h, a])


def pregame(p):
    """Add pre-game rolling features (shifted so a row only sees earlier games)."""
    p = p.sort_values(["player_id", "season", "week"]).copy()
    grp = p.groupby("player_id")
    p["n_prior"] = grp.cumcount()
    for c in list(STATS) + ["car_share", "tgt_share"]:
        p[f"ew_{c}"] = grp[c].transform(lambda s: s.shift(1).ewm(alpha=ALPHA, ignore_na=True).mean())
    return p


def _X(p, key):
    tt = p["team_total"].fillna(22.0)
    ew = p[f"ew_{key}"].fillna(0)
    passing = key.startswith("pass") or key == "completions" or key in ("receptions", "rec_yds")
    opp = (p["opp_def_pass"] if passing else p["opp_def_rush"]).fillna(0)
    shrink = p["n_prior"].clip(0, 8) / 8
    return pd.DataFrame({"ew": ew, "ew_tt": ew * (tt - 22) / 22, "ew_opp": ew * opp, "ew_shrink": ew * shrink,
                         "tt": tt - 22, "tgt": p["ew_tgt_share"].fillna(0), "car": p["ew_car_share"].fillna(0)})


class PropModel:
    def fit(self, p):
        self.m, self.sd = {}, {}
        for k, (_, _, pos) in STATS.items():
            d = p[p["position"].isin(pos) & (p["n_prior"] >= 2)]
            if k == "anytime_td":
                y = d[k]
            else:
                y = d[k]
            X = _X(d, k)
            self.m[k] = Ridge(alpha=1.0).fit(X, y)
            pred = np.clip(self.m[k].predict(X), 0.05, None)
            # empirical distribution of actual/projection, in 3 projection-size buckets (stats are skewed)
            ratio = (y.values / pred)
            cuts = np.quantile(pred, [1 / 3, 2 / 3])
            b = np.digitize(pred, cuts)
            self.sd[k] = (cuts.tolist(), [np.sort(ratio[b == i]).tolist() for i in range(3)])
        return self

    def project(self, rows, key):
        pred = np.clip(self.m[key].predict(_X(rows, key)), 0, None)
        return pred, pred

    def p_over(self, proj, _unused, line, key):
        if key == "anytime_td":
            return 1 - math.exp(-max(proj, 0.01))
        if key == "pass_tds":
            lam = max(proj, 0.01)
            k = int(math.floor(line))
            return 1 - sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(k + 1))
        cuts, dists = self.sd[key]
        proj = max(proj, 0.05)
        r = np.asarray(dists[int(np.digitize([proj], cuts)[0])])
        thr = line / proj
        return float(1 - np.searchsorted(r, thr, side="right") / len(r))
