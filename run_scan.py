import argparse
import json
import os
import httpx
from scanner_core import FOCUSED_LEAGUES, load_datasets, run_scanner


def post_telegram(message: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    with httpx.Client(timeout=10.0) as client:
        client.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"})


def main():
    parser = argparse.ArgumentParser(description="Run End-Point Football Scanner")
    parser.add_argument("--date-start", default=None)
    parser.add_argument("--date-end", default=None)
    parser.add_argument("--season", default=None)
    parser.add_argument("--upcoming-csv", default=None)
    parser.add_argument("--upcoming-only", action="store_true")
    parser.add_argument("--min-odd", type=float, default=1.25)
    parser.add_argument("--bankroll", type=float, default=20000.0)
    parser.add_argument("--stake-pct", type=float, default=0.02)
    parser.add_argument("--leagues", default=",".join(FOCUSED_LEAGUES))
    parser.add_argument("--out-csv", default="scanner_picks.csv")
    parser.add_argument("--telegram", action="store_true")
    args = parser.parse_args()

    leagues = None if args.leagues.strip().upper() == "ALL" else [x.strip() for x in args.leagues.split(",") if x.strip()]
    df_all = load_datasets(upcoming_csv_path=args.upcoming_csv)
    picks_df, summary = run_scanner(
        df_all=df_all,
        focused_leagues=leagues,
        min_leg_odd=args.min_odd,
        starting_bankroll=args.bankroll,
        stake_fraction=args.stake_pct,
        date_start=args.date_start,
        date_end=args.date_end,
        season_filter=args.season,
        include_upcoming_only=args.upcoming_only,
    )

    if len(picks_df) > 0:
        picks_df.drop(columns=["Date_dt"], errors="ignore").to_csv(args.out_csv, index=False)

    print(json.dumps(summary, indent=2))

    if args.telegram:
        msg = (
            f"*Football Scanner Alert*\n"
            f"Leagues: `{','.join(summary['focused_leagues'])}`\n"
            f"Qualified Picks: *{summary['qualified_picks']}* (Settled: {summary['settled_picks']}, Upcoming: {summary['upcoming_picks']})\n"
            f"Hit Rate: *{summary['hit_rate_pct']}%* | Stake/Pick: *{summary['stake_unit']:.2f}*\n"
            f"Net P&L: *{summary['net_pnl']:+.2f}* | Bankroll: *{summary['ending_bankroll']:.2f}*"
        )
        post_telegram(msg)


if __name__ == "__main__":
    main()
