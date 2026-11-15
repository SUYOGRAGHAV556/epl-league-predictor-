"""Generate SYNTHETIC Premier League style data so the pipeline can be tried end to end.

The results are random draws from made-up team strengths. They are NOT real matches.
Use real football-data.co.uk CSVs in data/raw/ for real predictions.

Usage:  python scripts/make_sample_data.py [--out data/sample] [--seasons 8]
"""
from __future__ import annotations

import argparse
from math import exp, factorial
from pathlib import Path

import numpy as np
import pandas as pd

TEAMS = ["Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", "Chelsea", "Crystal Palace",
         "Everton", "Fulham", "Ipswich", "Leicester", "Liverpool", "Man City", "Man United", "Newcastle",
         "Nott'm Forest", "Southampton", "Tottenham", "West Ham", "Wolves"]


def round_robin(teams: list[str]) -> list[list[tuple[str, str]]]:
    """Double round-robin schedule via the circle method: 38 rounds of 10 matches."""
    t = teams[:]
    n = len(t)
    rounds = []
    for r in range(n - 1):
        pairs = [(t[i], t[n - 1 - i]) for i in range(n // 2)]
        rounds.append(pairs if r % 2 == 0 else [(b, a) for a, b in pairs])
        t = [t[0]] + [t[-1]] + t[1:-1]
    return rounds + [[(b, a) for a, b in rd] for rd in rounds]


def _outcome_probs(lh: float, la: float, max_goals: int = 10) -> tuple[float, float, float]:
    pois = lambda lam: [exp(-lam) * lam ** k / factorial(k) for k in range(max_goals + 1)]
    ph, pa = pois(lh), pois(la)
    h = d = a = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = ph[i] * pa[j]
            if i > j: h += p
            elif i == j: d += p
            else: a += p
    s = h + d + a
    return h / s, d / s, a / s


def generate(n_seasons: int = 8, seed: int = 7, first_season: int = 2018, played_weeks_last: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(TEAMS)
    att, dfn = rng.normal(0, 0.25, n), rng.normal(0, 0.2, n)
    press, poss = rng.normal(0, 1, n), rng.normal(0, 1, n)
    rows = []
    for si in range(n_seasons + 1):                       # last season is partial (fixtures remain)
        season = first_season + si
        att = 0.8 * att + rng.normal(0, 0.1, n)           # strengths drift slowly
        dfn = 0.8 * dfn + rng.normal(0, 0.08, n)
        order = list(rng.permutation(n))
        sched = round_robin([TEAMS[i] for i in order])
        start = pd.Timestamp(f"{season}-08-15")
        for wk, pairs in enumerate(sched, start=1):
            date = start + pd.Timedelta(days=7 * (wk - 1))
            played = not (si == n_seasons and wk > played_weeks_last)
            for home, away in pairs:
                h, a = TEAMS.index(home), TEAMS.index(away)
                lh = 1.45 * exp(att[h] - dfn[a] + 0.12)
                la = 1.15 * exp(att[a] - dfn[h] - 0.05)
                row = {"Date": date.strftime("%d/%m/%Y"), "HomeTeam": home, "AwayTeam": away}
                if played:
                    hg, ag = int(rng.poisson(lh)), int(rng.poisson(la))
                    ph, pd_, pa = _outcome_probs(lh, la)
                    m = 1.05
                    row.update({
                        "FTHG": hg, "FTAG": ag, "FTR": "H" if hg > ag else ("A" if hg < ag else "D"),
                        "HS": rng.poisson(5 + 5 * lh), "AS": rng.poisson(4 + 5 * la),
                        "HST": rng.poisson(1.5 + 1.8 * lh), "AST": rng.poisson(1.2 + 1.8 * la),
                        "HC": rng.poisson(5 + 0.8 * poss[h]), "AC": rng.poisson(4.5 + 0.8 * poss[a]),
                        "HF": rng.poisson(11 - 0.5 * poss[h]), "AF": rng.poisson(11 - 0.5 * poss[a]),
                        "HxG": round(lh * rng.uniform(0.8, 1.2), 2), "AxG": round(la * rng.uniform(0.8, 1.2), 2),
                        "HPoss": round(float(np.clip(50 + 3 * (poss[h] - poss[a]) + rng.normal(0, 3), 25, 75)), 1),
                        "HPPDA": round(float(np.clip(11 - 1.5 * press[h] + rng.normal(0, 1.5), 5, 20)), 1),
                        "APPDA": round(float(np.clip(11 - 1.5 * press[a] + rng.normal(0, 1.5), 5, 20)), 1),
                        "B365H": round(1 / (ph * m), 2), "B365D": round(1 / (pd_ * m), 2), "B365A": round(1 / (pa * m), 2),
                    })
                    row["APoss"] = round(100 - row["HPoss"], 1)
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sample")
    ap.add_argument("--seasons", type=int, default=8, help="number of complete seasons")
    args = ap.parse_args()
    df = generate(args.seasons)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(df["Date"], dayfirst=True)
    for season in sorted(set(np.where(dates.dt.month >= 8, dates.dt.year, dates.dt.year - 1))):
        part = df[np.where(dates.dt.month >= 8, dates.dt.year, dates.dt.year - 1) == season]
        part.to_csv(out / f"SYNTHETIC_{season}.csv", index=False)
    print(f"Wrote synthetic data for {args.seasons + 1} seasons to {out}/ (NOT real results)")


if __name__ == "__main__":
    main()
