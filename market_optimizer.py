from typing import Dict, List, Optional
import numpy as np
from scipy.optimize import minimize

MARKETS = ["1X", "12", "O1.5"]


def logit_clip(p: float, eps: float = 1e-5) -> float:
    p_c = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p_c / (1.0 - p_c)))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z_c = np.clip(z, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-z_c))


def build_market_features(
    raw_prob: float,
    mkt_prob: Optional[float],
    home_venue_rate: Optional[float],
    away_venue_rate: Optional[float],
    div_tempo_norm: float,
    draw_prob: float,
) -> np.ndarray:
    """
    5-feature vector per market:
      f0: logit(raw_prob)
      f1: raw_prob - mkt_prob (model edge vs vig-adjusted market)
      f2: home_venue_rate - away_venue_rate (venue performance differential)
      f3: div_tempo_norm - 1.0 (league goal tempo shift vs baseline 2.65)
      f4: draw_prob - 0.27 (draw probability / match tightness shift)
    """
    f0 = logit_clip(raw_prob)
    f1 = (raw_prob - mkt_prob) if (mkt_prob is not None and not np.isnan(mkt_prob)) else 0.0
    h_v = home_venue_rate if (home_venue_rate is not None and not np.isnan(home_venue_rate)) else raw_prob
    a_v = away_venue_rate if (away_venue_rate is not None and not np.isnan(away_venue_rate)) else (1.0 - raw_prob)
    f2 = float(np.clip(h_v - a_v, -1.0, 1.0))
    f3 = float(np.clip(div_tempo_norm - 1.0, -0.5, 0.5))
    f4 = float(np.clip(draw_prob - 0.27, -0.25, 0.25))
    return np.array([f0, f1, f2, f3, f4], dtype=float)


class MarketOptimizerEngine:
    """
    Per-market statistical optimizer using scipy.optimize.minimize (L-BFGS-B)
    on strictly prior settled matches (zero look-ahead leakage).
    Regularized toward market-specific priors to prevent overfitting on noisy windows.
    """

    def __init__(self, market: str, min_samples: int = 60, reg_lambda: float = 0.35):
        self.market = market
        self.min_samples = min_samples
        self.reg_lambda = reg_lambda
        if market == "12":
            self.default_theta = np.array([1.0, 0.30, 0.15, 0.25, -0.45, 0.0], dtype=float)
        elif market == "O1.5":
            self.default_theta = np.array([1.0, 0.30, 0.10, 0.55, -0.35, 0.0], dtype=float)
        else:
            self.default_theta = np.array([1.0, 0.30, 0.25, 0.20, -0.20, 0.0], dtype=float)
        self.theta = self.default_theta.copy()
        self.fitted_samples: int = 0
        self.last_loss: Optional[float] = None

    def fit(self, history: List[Dict]) -> None:
        n = len(history)
        self.fitted_samples = n
        if n < self.min_samples:
            return

        X = np.vstack([item["features"] for item in history])
        y = np.array([float(item["hit"]) for item in history], dtype=float)

        idx = np.arange(n, dtype=float)
        w = np.exp(-0.004 * (n - 1 - idx))
        w = w / np.sum(w) * n

        def objective(params: np.ndarray) -> float:
            weights = params[:5]
            bias = params[5]
            logits = X @ weights + bias
            probs = np.clip(sigmoid(logits), 1e-6, 1.0 - 1e-6)

            logloss = float(np.mean(w * (-(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)))))
            brier = float(np.mean(w * ((probs - y) ** 2)))

            bins = np.linspace(0.0, 1.0, 6)
            cal_err = 0.0
            for b_i in range(5):
                mask = (probs >= bins[b_i]) & (probs < bins[b_i + 1] if b_i < 4 else probs <= bins[b_i + 1])
                if np.any(mask):
                    cal_err += abs(float(np.mean(probs[mask])) - float(np.mean(y[mask])))
            cal_penalty = 0.10 * (cal_err / 5.0)

            reg = self.reg_lambda * float(np.sum((params - self.default_theta) ** 2))
            return 0.65 * logloss + 0.25 * brier + cal_penalty + reg

        bounds = [
            (0.35, 2.50),
            (-1.00, 2.00),
            (-1.00, 1.50),
            (-1.00, 2.00),
            (-1.50, 1.00),
            (-0.75, 0.75),
        ]

        res = minimize(objective, x0=self.theta, method="L-BFGS-B", bounds=bounds)
        if res.success or np.isfinite(res.fun):
            self.theta = res.x
            self.last_loss = round(float(res.fun), 5)

    def predict(self, features: np.ndarray) -> float:
        z = float(features @ self.theta[:5] + self.theta[5])
        return float(np.clip(sigmoid(np.array([z]))[0], 0.01, 0.99))


def create_market_optimizers() -> Dict[str, MarketOptimizerEngine]:
    return {m: MarketOptimizerEngine(m, min_samples=60, reg_lambda=0.35) for m in MARKETS}
