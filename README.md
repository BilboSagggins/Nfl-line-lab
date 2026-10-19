# NFL Line Lab

A weekly NFL betting research page that updates itself.

**Live page:** `https://bilbosagggins.github.io/Nfl-line-lab/` (after GitHub Pages is turned on — see Setup).

## What it does

| When (Central time) | What runs |
|---|---|
| Thursday 8am | Retrain the team model, pull spreads, totals, moneylines and player props from every US sportsbook, freeze this week's picks |
| Sunday 10am | Refresh game lines (cheap on API credits) |
| Tuesday 9am | Grade last week's picks, refresh the backtest |

The page shows:

- **Best prices** — books offering a better number than the market's fair (no-vig) price
- **Game board** — consensus lines, best line per side with the book, model projection, form, ATS/O-U trends, injuries, rest, weather
- **Player props** — projection vs posted line, model and market probabilities, expected value at the best price, last 8 games
- **Record** — every frozen pick graded automatically, with closing-line value (CLV)
- **Model** — season-by-season backtest against closing lines

## Data

- [nflverse](https://github.com/nflverse/nflverse-data): play-by-play, schedules with closing lines, player stats, injury reports
- [The Odds API](https://the-odds-api.com): live lines and props (key stored as the `ODDS_API_KEY` repo secret)

A full week uses roughly 100 of the free plan's 500 monthly credits.

## Honest expectations

Tested on ~2,850 games from 2016 onward, each season predicted by a model trained only on earlier seasons, the team model is **less accurate than the closing line** and its spread picks hit about 48–50% (break-even is 52.4%). That's normal: closing lines already price in public data. Model leans are shown as context and tracked live. The edges this tool is built around are getting the best number across books, beating the closing line, and softer prop markets — and the Record tab shows whether that's working.

## Setup

1. **Secret:** Settings → Secrets and variables → Actions → `ODDS_API_KEY`.
2. **Pages:** Settings → Pages → Source: *Deploy from a branch* → Branch `main`, folder `/docs`.
3. **First run:** Actions → *Weekly update* → *Run workflow* → mode `thursday`.

## Discord bot

Live scores and betting lines in a Discord channel. Setup guide: [discord/README.md](discord/README.md).

## Run locally

```bash
pip install -r requirements.txt
ODDS_API_KEY=... python -m pipeline.run --mode thursday   # or: local (no API calls)
python -m http.server -d docs 8000
```

## Layout

```
pipeline/   data.py (nflverse download/cache) · features.py (pre-game team features)
            model.py (ridge models, leans, grading) · props.py (player projections)
            odds.py (Odds API, consensus + best prices) · run.py (weekly job)
docs/       index.html (the page) · data/*.json (written by the job)
.github/workflows/weekly.yml
```

For research and entertainment. Bet only what you can afford to lose. 1-800-GAMBLER.
