import math
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from market_optimizer import build_market_features, create_market_optimizers

FOCUSED_LEAGUES = ["I1", "E3", "EC", "E2", "E0", "I2", "B1", "N1"]


def ewma_series(vals: List[float], alpha: float = 0.18) -> Optional[float]:
    if not vals:
        return None
    w = 1.0
    num = 0.0
    den = 0.0
    for x in reversed(vals):
        num += w * x
        den += w
        w *= (1.0 - alpha)
    return num / den if den > 0 else None


def poisson_pmf(k: int, lam: float) -> float:
    lam_clamped = max(0.01, min(15.0, lam))
    return math.exp(-lam_clamped) * (lam_clamped ** k) / math.factorial(k)


def _normalize_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "HomeTeam" not in df.columns and "Home" in df.columns:
        df["HomeTeam"] = df["Home"]
    if "AwayTeam" not in df.columns and "Away" in df.columns:
        df["AwayTeam"] = df["Away"]
    if "odd_H" not in df.columns:
        df["odd_H"] = df.get("AvgH", pd.Series(index=df.index, dtype=float)).fillna(
            df.get("B365H", pd.Series(index=df.index, dtype=float))
        )
    if "odd_D" not in df.columns:
        df["odd_D"] = df.get("AvgD", pd.Series(index=df.index, dtype=float)).fillna(
            df.get("B365D", pd.Series(index=df.index, dtype=float))
        )
    if "odd_A" not in df.columns:
        df["odd_A"] = df.get("AvgA", pd.Series(index=df.index, dtype=float)).fillna(
            df.get("B365A", pd.Series(index=df.index, dtype=float))
        )
    if "odd_O25" not in df.columns:
        df["odd_O25"] = df.get("Avg>2.5", pd.Series(index=df.index, dtype=float)).fillna(
            df.get("B365>2.5", pd.Series(index=df.index, dtype=float))
        )
    return df


def load_datasets(
    euro_25_26_path: str = "data/all-euro-data-2025-2026.xlsx",
    euro_26_27_path: str = "data/all-euro-data-2026-2027 (1).xlsx",
    upcoming_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    frames = []
    for path, tag, order in [
        (euro_25_26_path, "2025-2026", 0),
        (euro_26_27_path, "2026-2027", 1),
    ]:
        if os.path.exists(path):
            xl = pd.ExcelFile(path)
            for sheet in xl.sheet_names:
                df = pd.read_excel(path, sheet_name=sheet)
                df["DivSheet"] = sheet
                df["SeasonTag"] = tag
                df["s_ord"] = order
                df["IsUpcoming"] = False
                frames.append(df)

    if upcoming_csv_path and os.path.exists(upcoming_csv_path):
        up_df = pd.read_csv(upcoming_csv_path)
        if "DivSheet" not in up_df.columns:
            up_df["DivSheet"] = up_df.get("League", "E0")
        up_df["SeasonTag"] = up_df.get("SeasonTag", "UPCOMING")
        up_df["s_ord"] = 2
        up_df["IsUpcoming"] = True
        if "FTHG" not in up_df.columns:
            up_df["FTHG"] = np.nan
        if "FTAG" not in up_df.columns:
            up_df["FTAG"] = np.nan
        frames.append(up_df)

    if not frames:
        raise FileNotFoundError("No dataset files found in expected paths.")

    df_all = pd.concat([_normalize_common_columns(f) for f in frames], ignore_index=True)
    df_all["Date_dt"] = pd.to_datetime(df_all["Date"], errors="coerce")
    df_all = df_all.dropna(subset=["Date_dt", "HomeTeam", "AwayTeam"]).copy()

    has_score = df_all["FTHG"].notna() & df_all["FTAG"].notna()
    is_up = df_all.get("IsUpcoming", pd.Series(False, index=df_all.index)).fillna(False).astype(bool)
    df_all = df_all[has_score | is_up].copy()
    df_all = df_all.sort_values(["s_ord", "Date_dt", "DivSheet", "HomeTeam"]).reset_index(drop=True)

    df_all["actual_1X"] = np.where(has_score, (df_all["FTHG"] >= df_all["FTAG"]).astype(float), np.nan)
    df_all["actual_12"] = np.where(has_score, (df_all["FTHG"] != df_all["FTAG"]).astype(float), np.nan)
    df_all["actual_O15"] = np.where(has_score, ((df_all["FTHG"] + df_all["FTAG"]) >= 2).astype(float), np.nan)
    return df_all


def run_scanner(
    df_all: pd.DataFrame,
    focused_leagues: Optional[List[str]] = None,
    min_leg_odd: float = 1.25,
    starting_bankroll: float = 20000.0,
    stake_fraction: float = 0.02,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    season_filter: Optional[str] = None,
    include_upcoming_only: bool = False,
    recalibrate_every: int = 60,
) -> Tuple[pd.DataFrame, Dict]:
    leagues_set = set(focused_leagues) if focused_leagues else None
    team_hist: Dict[str, List[Dict]] = {}
    div_goals_hist: Dict[str, List[float]] = {}
    optimizers = create_market_optimizers()
    market_train_hist: Dict[str, List[Dict]] = {"1X": [], "12": [], "O1.5": []}
    last_fit_count = 0
    scored_rows: List[Dict] = []

    for _, row in df_all.iterrows():
        h = str(row["HomeTeam"])
        a = str(row["AwayTeam"])
        div = str(row["DivSheet"])
        oh, od, oa = row["odd_H"], row["odd_D"], row["odd_A"]
        o25 = row["odd_O25"]
        has_result = pd.notna(row["FTHG"]) and pd.notna(row["FTAG"])

        settled_n = len(market_train_hist["1X"])
        if settled_n >= 60 and (settled_n - last_fit_count) >= recalibrate_every:
            for m_key, opt in optimizers.items():
                opt.fit(market_train_hist[m_key][-500:])
            last_fit_count = settled_n

        d_hist = div_goals_hist.get(div, [])
        div_avg_tot = float(np.mean(d_hist[-150:])) if len(d_hist) >= 10 else 2.65

        if pd.notna(oh) and pd.notna(od) and pd.notna(oa) and float(oh) > 1.0 and float(od) > 1.0 and float(oa) > 1.0:
            inv_h, inv_d, inv_a = 1.0 / float(oh), 1.0 / float(od), 1.0 / float(oa)
            s_1x2 = inv_h + inv_d + inv_a
            mkt_pH, mkt_pD, mkt_pA = inv_h / s_1x2, inv_d / s_1x2, inv_a / s_1x2
            mkt_1X = mkt_pH + mkt_pD
            mkt_12 = 1.0 - mkt_pD
            o_1X = 1.0 / (inv_h + inv_d)
            o_12 = 1.0 / (inv_h + inv_a)
        else:
            mkt_pH, mkt_pD, mkt_pA = 0.45, 0.27, 0.28
            mkt_1X = mkt_pH + mkt_pD
            mkt_12 = 1.0 - mkt_pD
            o_1X, o_12 = np.nan, np.nan

        est_o15 = (
            max(1.12, min(1.80, 1.0 + 0.19 * (float(o25) ** 1.15)))
            if (pd.notna(o25) and float(o25) > 1.0)
            else np.nan
        )
        mkt_o15 = (1.0 / float(est_o15)) if (pd.notna(est_o15) and float(est_o15) > 1.0) else np.nan

        hh = team_hist.get(h, [])
        ah = team_hist.get(a, [])
        hh_venue = [m for m in hh if m["side"] == "H"][-10:]
        ah_venue = [m for m in ah if m["side"] == "A"][-10:]
        w_h = min(0.60, len(hh) / 18.0)
        w_a = min(0.60, len(ah) / 18.0)

        if hh and ah:
            h_1x_ewma = ewma_series([m["hit_1X"] for m in hh[-15:]], alpha=0.20)
            h_venue_1x = float(np.mean([m["hit_1X"] for m in hh_venue])) if hh_venue else h_1x_ewma
            a_aw_ewma = ewma_series([m["away_win"] for m in ah[-15:]], alpha=0.20)
            a_venue_aw = float(np.mean([m["away_win"] for m in ah_venue])) if ah_venue else a_aw_ewma
            hist_1X = 0.35 * h_1x_ewma + 0.25 * h_venue_1x + 0.40 * (1.0 - (0.5 * a_aw_ewma + 0.5 * a_venue_aw))
            raw_1X = (1.0 - w_h) * mkt_1X + w_h * hist_1X

            hist_12 = 0.5 * ewma_series([m["hit_12"] for m in hh[-15:]], alpha=0.18) + 0.5 * ewma_series(
                [m["hit_12"] for m in ah[-15:]], alpha=0.18
            )
            raw_12 = (1.0 - 0.5 * (w_h + w_a)) * mkt_12 + 0.5 * (w_h + w_a) * hist_12

            h_att = ewma_series([m["xgf"] if m["xgf"] is not None else m["gf"] for m in hh[-15:]], alpha=0.20)
            h_def = ewma_series([m["xga"] if m["xga"] is not None else m["ga"] for m in hh[-15:]], alpha=0.20)
            a_att = ewma_series([m["xgf"] if m["xgf"] is not None else m["gf"] for m in ah[-15:]], alpha=0.20)
            a_def = ewma_series([m["xga"] if m["xga"] is not None else m["ga"] for m in ah[-15:]], alpha=0.20)
            base_half = max(1.0, div_avg_tot / 2.0)
            lam_h = max(0.35, (0.55 * h_att + 0.45 * a_def) * (base_half / 1.32))
            lam_a = max(0.25, (0.55 * a_att + 0.45 * h_def) * (base_half / 1.32))
            lam_tot = lam_h + lam_a
            p_o15_pois = 1.0 - (poisson_pmf(0, lam_tot) + poisson_pmf(1, lam_tot))
            hist_o15 = 0.5 * ewma_series([m["hit_O15"] for m in hh[-15:]], alpha=0.18) + 0.5 * ewma_series(
                [m["hit_O15"] for m in ah[-15:]], alpha=0.18
            )
            raw_O15 = 0.65 * p_o15_pois + 0.35 * hist_o15
        else:
            raw_1X = mkt_1X
            raw_12 = mkt_12
            raw_O15 = (
                min(0.95, max(0.45, 0.72 + 0.12 * ((1.0 / float(o25)) - 0.50)))
                if (pd.notna(o25) and float(o25) > 1.0)
                else 0.72
            )

        div_tempo_norm = div_avg_tot / 2.65
        h_v_1x = float(np.mean([m["hit_1X"] for m in hh_venue])) if hh_venue else np.nan
        a_v_1x = float(np.mean([m["hit_1X"] for m in ah_venue])) if ah_venue else np.nan
        h_v_12 = float(np.mean([m["hit_12"] for m in hh_venue])) if hh_venue else np.nan
        a_v_12 = float(np.mean([m["hit_12"] for m in ah_venue])) if ah_venue else np.nan
        h_v_o15 = float(np.mean([m["hit_O15"] for m in hh_venue])) if hh_venue else np.nan
        a_v_o15 = float(np.mean([m["hit_O15"] for m in ah_venue])) if ah_venue else np.nan

        feat_1X = build_market_features(raw_1X, mkt_1X, h_v_1x, a_v_1x, div_tempo_norm, mkt_pD)
        feat_12 = build_market_features(raw_12, mkt_12, h_v_12, a_v_12, div_tempo_norm, mkt_pD)
        feat_O15 = build_market_features(raw_O15, mkt_o15, h_v_o15, a_v_o15, div_tempo_norm, mkt_pD)

        p1X = optimizers["1X"].predict(feat_1X)
        p12 = optimizers["12"].predict(feat_12)
        pO15 = optimizers["O1.5"].predict(feat_O15)

        edges = []
        if pd.notna(o_1X) and float(o_1X) > 1.0:
            edges.append(p1X - 1.0 / float(o_1X))
        if pd.notna(o_12) and float(o_12) > 1.0:
            edges.append(p12 - 1.0 / float(o_12))
        if pd.notna(est_o15) and float(est_o15) > 1.0:
            edges.append(pO15 - 1.0 / float(est_o15))
        avg_edge = float(np.mean(edges)) if edges else 0.0

        m_outcome = ((p1X - 0.50) + (p12 - 0.50) + (pO15 - 0.50)) / 3.0
        p_min = min(p1X, p12, pO15)
        score = float(np.clip(200.0 * m_outcome + 120.0 * (p_min - 0.50) + 100.0 * avg_edge, 0.0, 100.0))

        legs = {
            "1X": (p1X, o_1X, int(row["actual_1X"]) if has_result else None),
            "12": (p12, o_12, int(row["actual_12"]) if has_result else None),
            "O1.5": (pO15, est_o15, int(row["actual_O15"]) if has_result else None),
        }
        top_leg, (top_p, top_o_raw, top_hit) = max(legs.items(), key=lambda x: x[1][0])
        settle_odd = max(1.10, float(top_o_raw)) if (pd.notna(top_o_raw) and float(top_o_raw) > 1.01) else 1.30
        all3_above_50 = (p1X > 0.50) and (p12 > 0.50) and (pO15 > 0.50)
        qualified = all3_above_50 and (settle_odd >= min_leg_odd) and (leagues_set is None or div in leagues_set)

        scored_rows.append(
            {
                "SeasonTag": row["SeasonTag"],
                "Date": row["Date_dt"].strftime("%Y-%m-%d"),
                "Date_dt": row["Date_dt"],
                "League": div,
                "HomeTeam": h,
                "AwayTeam": a,
                "HasResult": bool(has_result),
                "FTHG": int(row["FTHG"]) if has_result else None,
                "FTAG": int(row["FTAG"]) if has_result else None,
                "P_1X": round(p1X, 4),
                "P_12": round(p12, 4),
                "P_O15": round(pO15, 4),
                "Score": round(score, 2),
                "TopLeg": top_leg,
                "TopLegProb": round(top_p, 4),
                "SettleOdd": round(settle_odd, 3),
                "Qualified": qualified,
                "Hit": top_hit,
            }
        )

        if has_result:
            hxg = float(row["HxG"]) if ("HxG" in row and pd.notna(row["HxG"])) else None
            axg = float(row["AxG"]) if ("AxG" in row and pd.notna(row["AxG"])) else None
            fthg, ftag = float(row["FTHG"]), float(row["FTAG"])
            div_goals_hist.setdefault(div, []).append(fthg + ftag)

            market_train_hist["1X"].append({"features": feat_1X, "hit": int(row["actual_1X"])})
            market_train_hist["12"].append({"features": feat_12, "hit": int(row["actual_12"])})
            market_train_hist["O1.5"].append({"features": feat_O15, "hit": int(row["actual_O15"])})

            team_hist.setdefault(h, []).append(
                {
                    "side": "H",
                    "hit_1X": 1.0 if fthg >= ftag else 0.0,
                    "hit_12": 1.0 if fthg != ftag else 0.0,
                    "hit_O15": 1.0 if (fthg + ftag) >= 2 else 0.0,
                    "away_win": 1.0 if ftag > fthg else 0.0,
                    "gf": fthg,
                    "ga": ftag,
                    "xgf": hxg,
                    "xga": axg,
                }
            )
            team_hist.setdefault(a, []).append(
                {
                    "side": "A",
                    "hit_1X": 1.0 if fthg >= ftag else 0.0,
                    "hit_12": 1.0 if fthg != ftag else 0.0,
                    "hit_O15": 1.0 if (fthg + ftag) >= 2 else 0.0,
                    "away_win": 1.0 if ftag > fthg else 0.0,
                    "gf": ftag,
                    "ga": fthg,
                    "xgf": axg,
                    "xga": hxg,
                }
            )

    out_df = pd.DataFrame(scored_rows)
    if include_upcoming_only:
        out_df = out_df[~out_df["HasResult"]].copy()
    if season_filter:
        out_df = out_df[out_df["SeasonTag"] == season_filter].copy()
    if date_start:
        out_df = out_df[out_df["Date_dt"] >= pd.to_datetime(date_start)].copy()
    if date_end:
        out_df = out_df[out_df["Date_dt"] <= pd.to_datetime(date_end)].copy()

    picks_df = out_df[out_df["Qualified"]].sort_values(["Date_dt", "League", "HomeTeam"]).reset_index(drop=True)
    stake_unit = starting_bankroll * stake_fraction
    bankroll = starting_bankroll
    peak = bankroll
    max_dd = 0.0
    pnls = []
    bankrolls = []

    settled_count = 0
    wins_count = 0
    for _, r in picks_df.iterrows():
        if r["HasResult"] and pd.notna(r["Hit"]):
            settled_count += 1
            hit_val = int(r["Hit"])
            wins_count += hit_val
            pnl = stake_unit * (r["SettleOdd"] - 1.0) if hit_val == 1 else -stake_unit
            bankroll += pnl
            pnls.append(round(pnl, 2))
            bankrolls.append(round(bankroll, 2))
            if bankroll > peak:
                peak = bankroll
            max_dd = max(max_dd, peak - bankroll)
        else:
            pnls.append(None)
            bankrolls.append(round(bankroll, 2))

    if len(picks_df) > 0:
        picks_df["Stake"] = round(stake_unit, 2)
        picks_df["PnL"] = pnls
        picks_df["Bankroll"] = bankrolls

    calibrator_stats = {
        m_k: {
            "samples": opt.fitted_samples,
            "loss": opt.last_loss,
            "w_logit": round(float(opt.theta[0]), 4),
            "w_edge": round(float(opt.theta[1]), 4),
            "w_venue": round(float(opt.theta[2]), 4),
            "w_tempo": round(float(opt.theta[3]), 4),
            "w_draw": round(float(opt.theta[4]), 4),
            "bias": round(float(opt.theta[5]), 4),
        }
        for m_k, opt in optimizers.items()
    }

    summary = {
        "total_matches_in_window": int(len(out_df)),
        "qualified_picks": int(len(picks_df)),
        "settled_picks": int(settled_count),
        "upcoming_picks": int(len(picks_df) - settled_count),
        "wins": int(wins_count),
        "hit_rate_pct": round(float((wins_count / settled_count) * 100.0), 2) if settled_count > 0 else 0.0,
        "starting_bankroll": round(starting_bankroll, 2),
        "stake_unit": round(stake_unit, 2),
        "ending_bankroll": round(bankroll, 2),
        "net_pnl": round(bankroll - starting_bankroll, 2),
        "max_drawdown": round(-max_dd, 2),
        "focused_leagues": sorted(list(leagues_set)) if leagues_set else ["ALL"],
        "min_leg_odd": min_leg_odd,
        "market_calibrators": calibrator_stats,
    }
    return picks_df, summary
