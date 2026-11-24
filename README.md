# End-Point Football Scanner

The scanner evaluates three selections on each fixture:

- **1X** — home win or draw
- **12** — either team wins
- **Over 1.5** — at least two total goals

## Actual combined qualification target

The three selections all win together only when the **home team wins and the match has at least two goals**. The scanner therefore trains a separate `ALL3` model against that exact historical outcome and qualifies a fixture only when:

`P_ALL3 > 0.60`

This is a direct joint-event probability model. It does not average or multiply the three independent market probabilities.

## Included training data

Place these workbooks in `data/` (included in this repository):

- `all-euro-data-2025-2026.xlsx`
- `all-euro-data-2026-2027.xlsx`
- `ARG.xlsx`

The loader normalizes the European and Argentina workbook schemas. Historical team form is kept separate by league to avoid mixing teams with identical names across competitions.

## Run locally

```bash
pip install -r requirements.txt
python run_scan.py --upcoming-csv data/upcoming_fixtures_sample.csv --upcoming-only
```

For the full historical evaluation:

```bash
python run_scan.py --leagues ALL
```

The scanner processes matches chronologically and trains the joint model only on earlier settled fixtures. The optimizer begins fitting after its minimum sample requirement; before that it uses its conservative initialized coefficients.

## Important interpretation

- `P_ALL3` is the model's estimated probability that all three selections win together.
- `Qualified` means `P_ALL3 > 0.60` and the league filter passed.
- `Hit_ALL3` is the historical outcome label; it is blank for upcoming fixtures.
- `joint_event_hit_rate_pct` is the observed success rate among qualified settled historical fixtures.
- The 60% threshold is a filter, not a guarantee. Evaluate calibration and out-of-sample performance before relying on it.
- The source workbooks contain historical results and odds, not live upcoming fixtures or executable odds. Supply a separate upcoming CSV for future fixtures.
- Legacy `--min-odd`, bankroll and stake options are retained for command compatibility but do not determine ALL3 qualification or produce betting P&L. No combined market price is inferred from unrelated prices.
