"""The Odds API: game lines and player props across US sportsbooks, consensus fair odds, best prices."""
import json
import os
import re
import statistics
import urllib.parse
import urllib.request

API = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"

TEAM = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
PROP_MARKETS = {
    "player_pass_yds": "pass_yds", "player_pass_tds": "pass_tds", "player_rush_yds": "rush_yds",
    "player_reception_yds": "rec_yds", "player_receptions": "receptions", "player_anytime_td": "anytime_td",
}


class OddsAPI:
    def __init__(self, key):
        self.key = key
        self.remaining = None

    def get(self, path, **params):
        params["apiKey"] = self.key
        url = f"{API}{path}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=60) as r:
            self.remaining = r.headers.get("x-requests-remaining", self.remaining)
            return json.loads(r.read().decode())

    def games(self):
        return self.get("/odds", regions="us", markets="h2h,spreads,totals", oddsFormat="american")

    def events(self):
        return self.get("/events")

    def props(self, event_id):
        return self.get(f"/events/{event_id}/odds", regions="us", markets=",".join(PROP_MARKETS),
                        oddsFormat="american")


def implied(o):
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)


def payout(o):
    return o / 100 if o > 0 else 100 / -o


def to_american(p):
    p = min(max(p, 0.01), 0.99)
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def norm_name(s):
    s = re.sub(r"[.'’]", "", str(s).lower())
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", s)).strip()


def consensus(lines):
    """Most common line across books (ties -> the one nearest the median). Always a real, bettable number."""
    from collections import Counter
    c = Counter(lines)
    top = max(c.values())
    cands = [x for x, n in c.items() if n == top]
    med = statistics.median(lines)
    return min(cands, key=lambda x: (abs(x - med), x))


def _fair_pair(prices_a, prices_b):
    """Median no-vig probability of side A from books that list both sides."""
    ps = []
    for a, b in zip(prices_a, prices_b):
        ia, ib = implied(a), implied(b)
        ps.append(ia / (ia + ib))
    return statistics.median(ps) if ps else None


def summarize_game(g):
    """Consensus + best prices for one Odds API game object. Lines are from the home team's perspective."""
    home, away = g["home_team"], g["away_team"]
    books = g.get("bookmakers", [])
    out = {"home": TEAM.get(home, home), "away": TEAM.get(away, away), "commence": g["commence_time"],
           "books": len(books), "odds_id": g["id"], "by_book": {}}
    sp, tot, mlp = [], [], []
    for b in books:
        bb = {}
        for m in b.get("markets", []):
            o = {x["name"]: x for x in m["outcomes"]}
            if m["key"] == "spreads" and home in o and away in o:
                bb["spread"] = {"home": o[home]["point"], "home_price": o[home]["price"], "away_price": o[away]["price"]}
                sp.append(bb["spread"])
            elif m["key"] == "totals" and "Over" in o and "Under" in o:
                bb["total"] = {"line": o["Over"]["point"], "over": o["Over"]["price"], "under": o["Under"]["price"]}
                tot.append(bb["total"])
            elif m["key"] == "h2h" and home in o and away in o:
                bb["ml"] = {"home": o[home]["price"], "away": o[away]["price"]}
                mlp.append(bb["ml"])
        out["by_book"][b["title"]] = bb
    if sp:
        cons = consensus([x["home"] for x in sp])
        at = [x for x in sp if x["home"] == cons] or sp
        fair = _fair_pair([x["home_price"] for x in at], [x["away_price"] for x in at])
        out["spread"] = {"home_line": cons, "fair_home": fair,
                         "best_home": _best(sp, "home", "home_price", out["by_book"], "spread", higher_line=True),
                         "best_away": _best(sp, "home", "away_price", out["by_book"], "spread", higher_line=False)}
    if tot:
        cons = consensus([x["line"] for x in tot])
        at = [x for x in tot if x["line"] == cons] or tot
        fair = _fair_pair([x["over"] for x in at], [x["under"] for x in at])
        out["total"] = {"line": cons, "fair_over": fair,
                        "best_over": _best(tot, "line", "over", out["by_book"], "total", higher_line=False),
                        "best_under": _best(tot, "line", "under", out["by_book"], "total", higher_line=True)}
    if mlp:
        fair = _fair_pair([x["home"] for x in mlp], [x["away"] for x in mlp])
        bh = max(out["by_book"].items(), key=lambda kv: payout(kv[1]["ml"]["home"]) if "ml" in kv[1] else -9)
        ba = max(out["by_book"].items(), key=lambda kv: payout(kv[1]["ml"]["away"]) if "ml" in kv[1] else -9)
        med_h = statistics.median([implied(x["home"]) for x in mlp])
        med_a = statistics.median([implied(x["away"]) for x in mlp])
        out["ml"] = {"fair_home": fair, "home": to_american(med_h), "away": to_american(med_a),
                     "best_home": {"book": bh[0], "price": bh[1]["ml"]["home"]},
                     "best_away": {"book": ba[0], "price": ba[1]["ml"]["away"]}}
    return out


def _best(rows, line_key, price_key, by_book, mkey, higher_line):
    """Best number for a side: most favorable line, then best price at that line."""
    best = None
    for book, bb in by_book.items():
        x = bb.get(mkey)
        if not x:
            continue
        line, price = x[line_key], x[price_key]
        if mkey == "spread" and not higher_line:
            line = -line  # away line
        score = (line if higher_line or mkey == "spread" else -line, payout(price))
        if best is None or score > best[0]:
            best = (score, {"book": book, "line": line, "price": price})
    return best[1] if best else None


def summarize_props(ev):
    """Group an event's prop odds by player + market: consensus line, fair over prob, best over/under."""
    grp = {}
    for b in ev.get("bookmakers", []):
        for m in b.get("markets", []):
            key = PROP_MARKETS.get(m["key"])
            if not key:
                continue
            for o in m["outcomes"]:
                who = o.get("description")
                if not who:
                    continue
                pt = o.get("point", 0.5)
                g = grp.setdefault((who, key), {"player": who, "stat": key, "rows": {}})
                side = "over" if o["name"] in ("Over", "Yes") else "under"
                g["rows"].setdefault((b["title"], pt), {})[side] = o["price"]
    out = []
    for g in grp.values():
        lines = [pt for (_, pt) in g["rows"]]
        cons = consensus(lines)
        at = [(bk, v) for (bk, pt), v in g["rows"].items() if pt == cons]
        both = [v for _, v in at if "over" in v and "under" in v]
        fair = _fair_pair([v["over"] for v in both], [v["under"] for v in both]) if both else None
        if fair is None:
            ones = [v["over"] for _, v in at if "over" in v]
            if ones:  # one-sided market (e.g. anytime TD): strip a typical ~7% hold
                fair = statistics.median([implied(o) for o in ones]) / 1.07
        overs = [(bk, v["over"]) for bk, v in at if "over" in v]
        unders = [(bk, v["under"]) for bk, v in at if "under" in v]
        bo = max(overs, key=lambda x: payout(x[1])) if overs else None
        bu = max(unders, key=lambda x: payout(x[1])) if unders else None
        out.append({"player": g["player"], "stat": g["stat"], "line": cons, "books": len(at),
                    "fair_over": round(fair, 4) if fair else None,
                    "best_over": {"book": bo[0], "price": bo[1]} if bo else None,
                    "best_under": {"book": bu[0], "price": bu[1]} if bu else None})
    return out
