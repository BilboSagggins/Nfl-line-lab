"""Download and cache nflverse data (schedules with closing lines, play-by-play, weekly player stats)."""
import os
import time
import urllib.request

import pandas as pd

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
CACHE = os.environ.get("NLL_CACHE", os.path.join(os.path.dirname(__file__), "..", ".cache"))

PBP_COLS = [
    "game_id", "season", "week", "season_type", "posteam", "defteam", "home_team", "away_team",
    "play_type", "epa", "success", "pass", "rush", "qb_dropback", "qb_kneel", "qb_spike",
    "passer_player_id", "passer_player_name", "wp", "down", "two_point_attempt",
]


def _get(url, path, max_age_hours):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < max_age_hours * 3600:
        return path
    tmp = path + ".part"
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, path)
            return path
        except Exception:
            if attempt == 2:
                if os.path.exists(path):
                    return path  # fall back to stale cache
                raise
            time.sleep(3)
    return path


def current_season(today=None):
    today = today or pd.Timestamp.now(tz="America/Chicago")
    return today.year if today.month >= 8 else today.year - 1


def schedules(max_age_hours=6):
    p = _get(f"{BASE}/schedules/games.csv", os.path.join(CACHE, "games.csv"), max_age_hours)
    g = pd.read_csv(p, low_memory=False)
    g["gameday"] = pd.to_datetime(g["gameday"])
    return g


def pbp(season, current, max_age_hours=6):
    age = max_age_hours if season == current else 24 * 365
    p = _get(f"{BASE}/pbp/play_by_play_{season}.csv.gz", os.path.join(CACHE, f"pbp_{season}.csv.gz"), age)
    return pd.read_csv(p, usecols=PBP_COLS, low_memory=False)


def player_stats(season, current, max_age_hours=6):
    age = max_age_hours if season == current else 24 * 365
    p = _get(f"{BASE}/stats_player/stats_player_week_{season}.csv",
             os.path.join(CACHE, f"ps_{season}.csv"), age)
    return pd.read_csv(p, low_memory=False)
