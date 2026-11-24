import os
from typing import List, Optional
import httpx
from fastapi import FastAPI, Query
from scanner_core import FOCUSED_LEAGUES, load_datasets, run_scanner

app = FastAPI(title="End-Point Football Scanner", version="1.5.0")


def send_telegram_alert(text: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"})
            return resp.status_code == 200
    except Exception:
        return False


@app.get("/health")
def health():
    return {"status": "ok", "focused_leagues": FOCUSED_LEAGUES}


@app.get("/scan")
def scan_endpoint(
    date_start: Optional[str] = Query(default=None),
    date_end: Optional[str] = Query(default=None),
    season: Optional[str] = Query(default=None),
    upcoming_csv: Optional[str] = Query(default=None),
    upcoming_only: bool = Query(default=False),
    min_leg_odd: float = Query(default=1.25),
    bankroll: float = Query(default=20000.0),
    stake_pct: float = Query(default=0.02),
    leagues: Optional[str] = Query(default=",".join(FOCUSED_LEAGUES)),
    notify_telegram: bool = Query(default=False),
):
    league_list: Optional[List[str]] = (
        None if (leagues is None or leagues.strip().upper() == "ALL") else [x.strip() for x in leagues.split(",") if x.strip()]
    )
    df_all = load_datasets(upcoming_csv_path=upcoming_csv)
    picks_df, summary = run_scanner(
        df_all=df_all,
        focused_leagues=league_list,
        min_leg_odd=min_leg_odd,
        starting_bankroll=bankroll,
        stake_fraction=stake_pct,
        date_start=date_start,
        date_end=date_end,
        season_filter=season,
        include_upcoming_only=upcoming_only,
    )
    if notify_telegram:
        msg = (
            f"*Scanner Run Summary*\n"
            f"Leagues: `{','.join(summary['focused_leagues'])}`\n"
            f"Joint-event picks: *{summary['qualified_picks']}* (Upcoming: *{summary['upcoming_picks']}*) | "
            f"Historical joint hit rate: *{summary['joint_event_hit_rate_pct']}%*\n"
            f"Rule: `P_ALL3 > 60%`"
        )
        send_telegram_alert(msg)

    latest_picks = (
        picks_df.drop(columns=["Date_dt"], errors="ignore").tail(30).to_dict(orient="records")
        if len(picks_df) > 0
        else []
    )
    return {"summary": summary, "picks": latest_picks}
