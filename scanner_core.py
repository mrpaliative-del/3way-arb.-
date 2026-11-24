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
    """Normalize European and Argentina workbook schemas to one match schema."""
    df = df.copy()
    aliases = {
        "HomeTeam": ("Home",),
        "AwayTeam": ("Away",),
        "FTHG": ("HG",),
        "FTAG": ("AG",),
        "DivSheet": ("Div", "League"),
    }
    for target, candidates in aliases.items():
        if target not in df.columns:
            for source in candidates:
                if source in df.columns:
                    df[target] = df[source]
                    break

    # Use only observed 1X2 prices. Never manufacture an odds value.
    for target, candidates in {
        "odd_H": ("AvgH", "AvgCH", "B365H", "B365CH"),
        "odd_D": ("AvgD", "AvgCD", "B365D", "B365CD"),
        "odd_A": ("AvgA", "AvgCA", "B365A", "B365CA"),
        "odd_O25": ("Avg>2.5", "B365>2.5"),
    }.items():
        if target not in df.columns:
            df[target] = np.nan
        for source in candidates:
            if source in df.columns:
                df[target] = df[target].where(df[target].notna(), df[source])
    return df


def load_datasets(
    euro_25_26_path: str = "data/all-euro-data-2025-2026.xlsx",
    euro_26_27_path: str = "data/all-euro-data-2026-2027.xlsx",
    arg_path: str = "data/ARG.xlsx",
    upcoming_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    frames = []
    sources = [
        (euro_25_26_path, "2025-2026", 0),
        (euro_26_27_path, "2026-2027", 1),
        (arg_path, "ARG", 0),
    ]
    for path, tag, order in sources:
        if not os.path.exists(path):
            continue
        xl = pd.ExcelFile(path)
        for sheet in xl.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet)
            df["DivSheet"] = df.get("DivSheet", sheet)
            if tag == "ARG":
                # Argentina file carries season by row; retain it for filtering/auditing.
                if "Season" in df.columns:
                    df["SeasonTag"] = "ARG-" + df["Season"].astype(str)
                else:
                    df["SeasonTag"] = "ARG"
            else:
                df["SeasonTag"] = tag
            df["s_ord"] = order
            df["IsUpcoming"] = False
            frames.append(_normalize_common_columns(df))

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
        frames.append(_normalize_common_columns(up_df))

    if not frames:
        raise FileNotFoundError("No dataset files found in expected paths.")

    df_all = pd.concat(frames, ignore_index=True, sort=False)
    df_all = _normalize_common_columns(df_all)
    if "FTHG" not in df_all.columns:
        df_all["FTHG"] = np.nan
    if "FTAG" not in df_all.columns:
        df_all["FTAG"] = np.nan
    df_all["Date_dt"] = pd.to_datetime(df_all["Date"], errors="coerce", dayfirst=True)
    df_all = df_all.dropna(subset=["Date_dt", "HomeTeam", "AwayTeam"]).copy()

    is_up = df_all.get("IsUpcoming", pd.Series(False, index=df_all.index)).fillna(False).astype(bool)
    has_score = df_all["FTHG"].notna() & df_all["FTAG"].notna()
    df_all = df_all[has_score | is_up].copy()
    df_all = df_all.reset_index(drop=True)
    has_score = df_all["FTHG"].notna() & df_all["FTAG"].notna()
    df_all["IsUpcoming"] = is_up.loc[df_all.index].values if len(is_up) == len(df_all) else df_all["IsUpcoming"].fillna(False)
    df_all = df_all.sort_values(["s_ord", "Date_dt", "DivSheet", "HomeTeam"]).reset_index(drop=True)

    has_score = df_all["FTHG"].notna() & df_all["FTAG"].notna()
    df_all["actual_1X"] = np.where(has_score, (df_all["FTHG"] >= df_all["FTAG"]).astype(float), np.nan)
    df_all["actual_12"] = np.where(has_score, (df_all["FTHG"] != df_all["FTAG"]).astype(float), np.nan)
    df_all["actual_O15"] = np.where(has_score, ((df_all["FTHG"] + df_all["FTAG"]) >= 2).astype(float), np.nan)
    # The three selections all win together iff the home team wins and total goals >= 2.
    df_all["actual_ALL3"] = np.where(
        has_score,
        ((df_all["FTHG"] > df_all["FTAG"]) & ((df_all["FTHG"] + df_all["FTAG"]) >= 2)).astype(float),
        np.nan,
    )
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
    div_o15_hist: Dict[str, List[float]] = {}
    optimizers = create_market_optimizers()
    market_train_hist: Dict[str, List[Dict]] = {"1X": [], "12": [], "O1.5": [], "ALL3": []}
    last_fit_count = 0
    last_fit_date = None
    scored_rows: List[Dict] = []

    # Enforce chronological walk-forward evaluation across all countries/seasons.
    df_all = df_all.sort_values(["Date_dt", "s_ord", "DivSheet", "HomeTeam"]).reset_index(drop=True)
    for _, row in df_all.iterrows():
        h = str(row["HomeTeam"])
        a = str(row["AwayTeam"])
        div = str(row["DivSheet"])
        oh, od, oa = row["odd_H"], row["odd_D"], row["odd_A"]
        o25 = row["odd_O25"]
        has_result = pd.notna(row["FTHG"]) and pd.notna(row["FTAG"])

        settled_n = len(market_train_hist["1X"])
        current_date = row["Date_dt"].date()
        if (
            last_fit_date is not None
            and current_date != last_fit_date
            and settled_n >= 60
            and (settled_n - last_fit_count) >= recalibrate_every
        ):
            for m_key, opt in optimizers.items():
                opt.fit(market_train_hist[m_key][-500:])
            last_fit_count = settled_n
        last_fit_date = current_date

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

        # Team names can repeat across countries/leagues; isolate their histories.
        h_key, a_key = f"{div}::{h}", f"{div}::{a}"
        hh = team_hist.get(h_key, [])
        ah = team_hist.get(a_key, [])
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
            league_o15 = div_o15_hist.get(div, [])
            if len(league_o15) >= 10:
                raw_O15 = float(np.mean(league_o15[-150:]))
            elif pd.notna(o25) and float(o25) > 1.0:
                # Conservative prior from the observed Over 2.5 price; this is
                # a model feature only, never presented as an executable O1.5 price.
                raw_O15 = min(0.90, max(0.55, 0.72 + 0.12 * ((1.0 / float(o25)) - 0.50)))
            else:
                raw_O15 = 0.72

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

        # Direct joint-event model. The proxy is only an input feature; ALL3 is
        # trained against the observed joint result, not inferred by multiplying
        # three separately calibrated market probabilities.
        if hh and ah:
            h_win_all = ewma_series([m["team_win"] for m in hh[-15:]], alpha=0.20)
            h_win_venue = float(np.mean([m["home_win"] for m in hh_venue])) if hh_venue else h_win_all
            a_win_all = ewma_series([m["team_win"] for m in ah[-15:]], alpha=0.20)
            a_win_venue = float(np.mean([m["away_win"] for m in ah_venue])) if ah_venue else a_win_all
            h_form_win = h_win_all if h_win_all is not None else mkt_pH
            a_form_win = a_win_all if a_win_all is not None else (1.0 - mkt_pH)
            raw_home_win = (1.0 - w_h) * mkt_pH + w_h * (0.65 * h_win_venue + 0.35 * (1.0 - a_win_venue))
            raw_home_win = 0.85 * raw_home_win + 0.15 * (0.5 * h_form_win + 0.5 * (1.0 - a_form_win))
        else:
            raw_home_win = mkt_pH
        raw_joint = float(np.clip(raw_home_win * raw_O15, 0.01, 0.99))
        joint_market_proxy = float(np.clip(mkt_pH * raw_O15, 0.01, 0.99))
        h_home_win_rate = float(np.mean([m["home_win"] for m in hh_venue])) if hh_venue else np.nan
        a_away_win_rate = float(np.mean([m["away_win"] for m in ah_venue])) if ah_venue else np.nan
        feat_ALL3 = build_market_features(
            raw_joint, joint_market_proxy, h_home_win_rate, a_away_win_rate,
            div_tempo_norm, mkt_pD
        )
        pALL3_raw = optimizers["ALL3"].predict(feat_ALL3)
        # A joint event cannot be more likely than any one of its component
        # events. Cap the direct model output to keep probabilities coherent.
        pALL3 = min(pALL3_raw, p1X, p12, pO15)

        top_leg, top_p = max(
            {"1X": p1X, "12": p12, "O1.5": pO15}.items(),
            key=lambda x: x[1],
        )
        actual_all3 = int(row["actual_ALL3"]) if has_result and pd.notna(row.get("actual_ALL3")) else None
        # No synthetic odds gate: the qualification target is the joint probability.
        qualified = (pALL3 > 0.60) and (leagues_set is None or div in leagues_set)

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
                "P_ALL3": round(pALL3, 4),
                "Score": round(100.0 * pALL3, 2),
                "TopLeg": top_leg,
                "TopLegProb": round(top_p, 4),
                "Qualified": qualified,
                "Hit_ALL3": actual_all3,
                "Hit": actual_all3,
            }
        )

        if has_result:
            hxg = float(row["HxG"]) if ("HxG" in row and pd.notna(row["HxG"])) else None
            axg = float(row["AxG"]) if ("AxG" in row and pd.notna(row["AxG"])) else None
            fthg, ftag = float(row["FTHG"]), float(row["FTAG"])
            div_goals_hist.setdefault(div, []).append(fthg + ftag)
            div_o15_hist.setdefault(div, []).append(1.0 if (fthg + ftag) >= 2 else 0.0)

            market_train_hist["1X"].append({"features": feat_1X, "hit": int(row["actual_1X"])})
            market_train_hist["12"].append({"features": feat_12, "hit": int(row["actual_12"])})
            market_train_hist["O1.5"].append({"features": feat_O15, "hit": int(row["actual_O15"])})
            market_train_hist["ALL3"].append({"features": feat_ALL3, "hit": int(row["actual_ALL3"])})

            team_hist.setdefault(h_key, []).append(
                {
                    "side": "H",
                    "hit_1X": 1.0 if fthg >= ftag else 0.0,
                    "hit_12": 1.0 if fthg != ftag else 0.0,
                    "hit_O15": 1.0 if (fthg + ftag) >= 2 else 0.0,
                    "team_win": 1.0 if fthg > ftag else 0.0,
                    "home_win": 1.0 if fthg > ftag else 0.0,
                    "away_win": 1.0 if ftag > fthg else 0.0,
                    "gf": fthg,
                    "ga": ftag,
                    "xgf": hxg,
                    "xga": axg,
                }
            )
            team_hist.setdefault(a_key, []).append(
                {
                    "side": "A",
                    "hit_1X": 1.0 if ftag >= fthg else 0.0,
                    "hit_12": 1.0 if fthg != ftag else 0.0,
                    "hit_O15": 1.0 if (fthg + ftag) >= 2 else 0.0,
                    "team_win": 1.0 if ftag > fthg else 0.0,
                    "home_win": 1.0 if ftag > fthg else 0.0,
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
    settled_count = int(picks_df["HasResult"].sum()) if len(picks_df) else 0
    wins_count = int(picks_df.loc[picks_df["HasResult"], "Hit_ALL3"].fillna(0).sum()) if settled_count else 0

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
        "joint_event_wins": int(wins_count),
        "joint_event_hit_rate_pct": round(float((wins_count / settled_count) * 100.0), 2) if settled_count > 0 else 0.0,
        "qualification_rule": "P_ALL3 > 0.60",
        "joint_event_definition": "Home win AND total goals >= 2 (all of 1X, 12 and Over 1.5 win)",
        "minimum_joint_probability": 0.60,
        "focused_leagues": sorted(list(leagues_set)) if leagues_set else ["ALL"],
        "min_leg_odd_ignored": True,
        "market_calibrators": calibrator_stats,
    }
    return picks_df, summary
