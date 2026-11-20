"""
NFL Regular Season Win Projector (1999–2025)
============================================
An iterative machine learning pipeline that predicts an NFL team's regular-season
win total from a trailing 3-year rolling window of efficiency, scoring,
late-season momentum, luck-regression, smart QB upgrade delta, and schedule metrics.

Models Included:
1. Linear Regression:
   - Baseline Linear Regression (original 4 unweighted 3-year features, raw wins)
   - Enhanced Linear Regression (era-normalized win %, recency-weighted efficiency,
     Pythagorean expectation, close-game luck regression, late-season momentum,
     smart QB upgrade delta, and pre-season opponent SOS)
2. Random Forest Regressor:
   - Auto-tuned Random Forest across hyperparameter configurations via
     Leave-One-Window-Out cross-validation.

Evaluation Modes:
- Stage 1 (Walk-Forward Backtesting, 1999–2025):
  Iterates chronologically across all 24 rolling 3-year -> 1-year windows,
  predicting each season out-of-sample before adding it to the training history.
- Stage 2 & 3 (Full 24-Window Train/Test Partition):
  Splits all 24 historical windows (1999–2025) cleanly between `TRAIN_WINDOWS`
  and `TEST_WINDOWS` with zero windows left out.
"""

from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union
import urllib.request
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nfl_data_py as nfl
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# Type alias: either (window_start, window_end, target_year) or just target_year (int)
WindowSpec = Union[Tuple[int, int, int], int]

# ==============================================================================
# GLOBAL CONFIGURATION — ALL 24 WINDOWS (1999-2025) SPLIT BETWEEN TRAIN & TEST
# ==============================================================================
HIST_START_YEAR: int = 1999
HIST_END_YEAR: int = 2025

# All 24 rolling 3-year -> 1-year windows from (1999, 2001, 2002) to (2022, 2024, 2025)
ALL_HISTORICAL_WINDOWS: List[Tuple[int, int, int]] = [
    (y - 3, y - 1, y) for y in range(HIST_START_YEAR + 3, HIST_END_YEAR + 1)
]

# Held-Out Test Windows across different eras (swap any 3-year -> 1-year ranges here)
TEST_WINDOWS: List[WindowSpec] = [
    (2002, 2004, 2005),  # Early-2000s test range (2002-2004 -> 2005)
    (2007, 2009, 2010),  # Late-2000s test range  (2007-2009 -> 2010)
    (2011, 2013, 2014),  # Early-2010s test range (2011-2013 -> 2014)
    (2014, 2016, 2017),  # Mid-2010s test range   (2014-2016 -> 2017)
    (2022, 2024, 2025),  # Most recent season     (2022-2024 -> 2025)
]

# Training Windows: all remaining windows from 1999-2025 (19 windows + 5 test = all 24 windows)
_TEST_SET_NORMALIZED = {
    (w - 3, w - 1, w) if isinstance(w, int) else (int(w[0]), int(w[1]), int(w[2]))
    for w in TEST_WINDOWS
}
TRAIN_WINDOWS: List[WindowSpec] = [
    w for w in ALL_HISTORICAL_WINDOWS if w not in _TEST_SET_NORMALIZED
]

# Toggle to also run the chronological 1999-2025 Walk-Forward Expanding-Window Backtest
RUN_WALK_FORWARD_BACKTEST: bool = True

# Recency weights for [Year - 3, Year - 2, Year - 1]
# Weighting Year-1 at 70% prevents older seasons from dragging down breakout contenders
RECENCY_WEIGHTS = (0.10, 0.20, 0.70)

# Reduced Major Axis (RMA) dispersion calibration factor to prevent OLS / Tree
# mean-compression from clustering all teams into a narrow 6-to-10 win band
DISPERSION_GAMMA: float = 1.30

# Local cache directory so swapping ranges doesn't re-download PBP from scratch
CACHE_DIR = Path(__file__).resolve().parent / ".cache"

# Standardize historical franchise relocations/rebrands to modern abbreviations
TEAM_ABBR_MAP = {
    "OAK": "LV",   # Oakland Raiders -> Las Vegas Raiders
    "SD": "LAC",   # San Diego Chargers -> Los Angeles Chargers
    "STL": "LA",   # St. Louis Rams -> Los Angeles Rams
    "LAR": "LA",   # Standardize Rams abbreviation
    "WAS": "WSH",  # Standardize Washington abbreviation
    "JAC": "JAX",  # Historical Jacksonville abbreviation in older PBP
}

# Original 4 baseline features (kept for Baseline Linear Regression comparison)
BASELINE_FEATURE_COLS = [
    "off_epa_per_play_3yr",
    "def_epa_per_play_3yr",
    "avg_point_diff_3yr",
    "avg_turnover_margin_3yr",
]

# Clean, non-collinear high-signal features for Enhanced Linear Regression & Random Forest
ENHANCED_FEATURE_COLS = [
    "off_epa_weighted",
    "def_epa_weighted",
    "pythag_win_pct_weighted",
    "turnover_margin_pg_weighted",
    "overperformance_luck_y1",
    "late_pd_pg_y1",
    "qb_upgrade_pd_delta",
    "market_srs_weighted",
    "preseason_wk1_rating",
    "target_sos_pythag",
]

# Directory for saving diagnostic scatter plots & feature importance charts
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
LOGOS_CACHE_DIR = CACHE_DIR / "logos"


def fetch_and_cache_team_logos() -> Dict[str, np.ndarray]:
    """
    Download and cache the 32 NFL team logos from `nfl_data_py.import_team_desc()`
    into `.cache/logos/` so scatter plots can render official team logos at each point.
    """
    LOGOS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logos: Dict[str, np.ndarray] = {}

    try:
        td = nfl.import_team_desc()
        td["team_abbr_std"] = standardize_team_abbr(td["team_abbr"])
        logo_urls = dict(zip(td["team_abbr_std"], td["team_logo_espn"]))
    except Exception:
        logo_urls = {}

    for team, url in logo_urls.items():
        if not isinstance(url, str) or not url:
            continue
        logo_path = LOGOS_CACHE_DIR / f"{team}.png"
        if not logo_path.exists():
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    logo_path.write_bytes(resp.read())
            except Exception:
                continue
        if logo_path.exists():
            try:
                logos[team] = mpimg.imread(str(logo_path))
            except Exception:
                pass

    return logos


def _plot_logo_scatter_panel(
    ax: plt.Axes,
    teams: Sequence[str],
    x_actual: np.ndarray,
    y_pred: np.ndarray,
    logos: Dict[str, np.ndarray],
    title: str,
    axis_min: float = 1.0,
    axis_max: float = 15.5,
    logo_zoom: float = 0.048,
) -> None:
    """
    Render an Expected (Predicted) vs. Actual Wins scatter plot using official
    NFL team logos as the data points (1 logo per team, 32 teams total).
    """
    x_line = np.linspace(axis_min, axis_max, 100)
    ax.plot(
        x_line,
        x_line,
        "--",
        color="#333333",
        linewidth=1.5,
        label="Exact Match (Predicted = Actual)",
        zorder=1,
    )
    ax.fill_between(
        x_line,
        x_line - 2.0,
        x_line + 2.0,
        color="#2ca02c",
        alpha=0.12,
        label="Within ±2 Wins Zone",
        zorder=0,
    )

    # Slight jitter only when two teams share identical (x, y) coordinates so logos don't overlap
    coords = np.column_stack([x_actual, y_pred]).astype(float)
    for i in range(len(coords)):
        for j in range(i):
            if np.linalg.norm(coords[i] - coords[j]) < 0.25:
                coords[i, 0] += 0.18
                coords[i, 1] -= 0.14

    for team, (xv, yv) in zip(teams, coords):
        if team in logos:
            img = logos[team]
            max_dim = float(max(img.shape[0], img.shape[1]))
            norm_zoom = (logo_zoom * 500.0) / max_dim
            img_box = OffsetImage(img, zoom=norm_zoom)
            ab = AnnotationBbox(img_box, (xv, yv), frameon=False, zorder=3)
            ax.add_artist(ab)
        else:
            ax.scatter(xv, yv, color="#1f77b4", s=50, zorder=3)
            ax.text(xv, yv, team, fontsize=8, weight="bold", ha="center", va="center")

    ax.set_title(title, fontsize=13.5, weight="bold")
    ax.set_xlabel("Actual Regular-Season Wins", fontsize=12, weight="bold")
    ax.set_ylabel("Predicted Regular-Season Wins", fontsize=12, weight="bold")
    ax.tick_params(axis="both", labelsize=11)
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=10.5)


def generate_evaluation_plots(
    all_test_results_df: pd.DataFrame,
    imp_df: pd.DataFrame,
) -> None:
    """
    Generate and save clear, logo-based diagnostic charts to `plots/` without
    stacking 160 overlapping multi-season dots onto a single graph:
      1. `actual_vs_predicted_scatter.png`:
         - Top Row: Pure Single-Season Test Window (`2011-2013 -> 2014`, 1 logo
           per NFL team = 32 logos total) comparing Linear Regression vs. Random Forest.
         - Bottom Row: Pure Single-Season Test Window (`2022-2024 -> 2025`, 1 logo
           per NFL team = 32 logos total) comparing Linear Regression vs. Random Forest.
      2. `prediction_error_residuals.png`:
         Direct 32-team Expected vs. Actual Wins comparison chart with NFL team
         logos so you can inspect every franchise's gap at a glance.
      3. `rf_feature_importance.png`:
         Random Forest Feature Importance bar chart.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    logos = fetch_and_cache_team_logos()

    available_windows = list(dict.fromkeys(all_test_results_df["Window"]))
    top_window = (
        "2011-2013 -> 2014"
        if "2011-2013 -> 2014" in available_windows
        else available_windows[0]
    )
    bottom_window = (
        "2022-2024 -> 2025"
        if "2022-2024 -> 2025" in available_windows
        else available_windows[-1]
    )

    top_df = all_test_results_df[
        all_test_results_df["Window"] == top_window
    ].reset_index(drop=True)
    m_top_lr = evaluate_predictions_in_wins(
        top_df["Actual Wins"].values,
        top_df["Enhanced LR"].values,
        top_df["Games"].values,
        top_df["Floor Wins"].values if "Floor Wins" in top_df else None,
        top_df["Ceiling Wins"].values if "Ceiling Wins" in top_df else None,
    )
    m_top_rf = evaluate_predictions_in_wins(
        top_df["Actual Wins"].values,
        top_df["Random Forest"].values,
        top_df["Games"].values,
        top_df["Floor Wins"].values if "Floor Wins" in top_df else None,
        top_df["Ceiling Wins"].values if "Ceiling Wins" in top_df else None,
    )

    spot_df = all_test_results_df[
        all_test_results_df["Window"] == bottom_window
    ].reset_index(drop=True)
    spotlight_window = bottom_window
    m_spot_lr = evaluate_predictions_in_wins(
        spot_df["Actual Wins"].values,
        spot_df["Enhanced LR"].values,
        spot_df["Games"].values,
        spot_df["Floor Wins"].values if "Floor Wins" in spot_df else None,
        spot_df["Ceiling Wins"].values if "Ceiling Wins" in spot_df else None,
    )
    m_spot_rf = evaluate_predictions_in_wins(
        spot_df["Actual Wins"].values,
        spot_df["Random Forest"].values,
        spot_df["Games"].values,
        spot_df["Floor Wins"].values if "Floor Wins" in spot_df else None,
        spot_df["Ceiling Wins"].values if "Ceiling Wins" in spot_df else None,
    )

    # --- Plot 1: 2x2 Pure Single-Season NFL Team Logo Scatter Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 13), dpi=150)

    _plot_logo_scatter_panel(
        axes[0, 0],
        top_df["Team"].values,
        top_df["Actual Wins"].values,
        top_df["Enhanced LR"].values,
        logos,
        f"Enhanced Linear Regression — Single Season ({top_window})\n"
        f"1st-Deg Acc: {m_top_lr['accuracy_pct']:.1f}% | 2nd-Deg Range Acc: {m_top_lr['second_deg_acc_pct']:.1f}% | MAE: {m_top_lr['mae']:.2f}W",
        axis_min=1.0,
        axis_max=16.5,
    )
    _plot_logo_scatter_panel(
        axes[0, 1],
        top_df["Team"].values,
        top_df["Actual Wins"].values,
        top_df["Random Forest"].values,
        logos,
        f"Tuned Random Forest — Single Season ({top_window})\n"
        f"1st-Deg Acc: {m_top_rf['accuracy_pct']:.1f}% | 2nd-Deg Range Acc: {m_top_rf['second_deg_acc_pct']:.1f}% | MAE: {m_top_rf['mae']:.2f}W",
        axis_min=1.0,
        axis_max=16.5,
    )

    _plot_logo_scatter_panel(
        axes[1, 0],
        spot_df["Team"].values,
        spot_df["Actual Wins"].values,
        spot_df["Enhanced LR"].values,
        logos,
        f"Enhanced Linear Regression — Single Season ({spotlight_window})\n"
        f"1st-Deg Acc: {m_spot_lr['accuracy_pct']:.1f}% | 2nd-Deg Range Acc: {m_spot_lr['second_deg_acc_pct']:.1f}% | MAE: {m_spot_lr['mae']:.2f}W",
        axis_min=1.0,
        axis_max=17.0,
    )
    _plot_logo_scatter_panel(
        axes[1, 1],
        spot_df["Team"].values,
        spot_df["Actual Wins"].values,
        spot_df["Random Forest"].values,
        logos,
        f"Tuned Random Forest — Single Season ({spotlight_window})\n"
        f"1st-Deg Acc: {m_spot_rf['accuracy_pct']:.1f}% | 2nd-Deg Range Acc: {m_spot_rf['second_deg_acc_pct']:.1f}% | MAE: {m_spot_rf['mae']:.2f}W",
        axis_min=1.0,
        axis_max=17.0,
    )

    fig.suptitle(
        "NFL Win Projector — Pure Single-Season Expected vs. Actual Wins (32 NFL Team Logos)",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout()
    scatter_path = PLOTS_DIR / "actual_vs_predicted_scatter.png"
    fig.savefig(scatter_path, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 2: Direct 32-Team Expected vs. Actual Wins Dumbbell Chart with Floor-Ceiling & Vegas ---
    fig, ax = plt.subplots(figsize=(11.5, 11.5), dpi=180)
    y_pos = np.arange(len(spot_df))
    spot_sorted = spot_df.sort_values(
        by=["Actual Wins", "Enhanced LR"], ascending=[True, True]
    ).reset_index(drop=True)

    # Color-coded Floor-to-Ceiling prediction band (Green = Actual in Range, Red = Outside Range)
    if "Floor Wins" in spot_sorted.columns and "Ceiling Wins" in spot_sorted.columns:
        drew_hit, drew_miss = False, False
        for idx, row in spot_sorted.iterrows():
            act_w = float(row["Actual Wins"])
            fl_w = float(row["Floor Wins"])
            ce_w = float(row["Ceiling Wins"])
            in_rng = fl_w <= act_w <= ce_w
            bar_color = "#2ca02c" if in_rng else "#d62728"
            bar_alpha = 0.26 if in_rng else 0.16
            lbl = None
            if in_rng and not drew_hit:
                lbl = "Floor–Ceiling Range (Hit ✓)"
                drew_hit = True
            elif (not in_rng) and not drew_miss:
                lbl = "Floor–Ceiling Range (Miss)"
                drew_miss = True
            ax.plot(
                [fl_w, ce_w],
                [idx, idx],
                color=bar_color,
                alpha=bar_alpha,
                linewidth=8.5,
                solid_capstyle="round",
                zorder=1,
                label=lbl,
            )

    for idx, row in spot_sorted.iterrows():
        act_w = float(row["Actual Wins"])
        lr_w = float(row["Enhanced LR"])
        rf_w = float(row["Random Forest"])
        vg_w = float(row["Vegas O/U"]) if "Vegas O/U" in spot_sorted.columns else lr_w
        ax.plot(
            [min(act_w, lr_w, rf_w, vg_w), max(act_w, lr_w, rf_w, vg_w)],
            [idx, idx],
            color="#aaaaaa",
            linewidth=1.6,
            zorder=2,
        )

    ax.scatter(
        spot_sorted["Actual Wins"],
        y_pos,
        color="#111111",
        s=75,
        label="Actual Wins",
        zorder=5,
    )
    if "Vegas O/U" in spot_sorted.columns:
        ax.scatter(
            spot_sorted["Vegas O/U"],
            y_pos,
            color="#e69f00",
            edgecolor="#333333",
            linewidth=0.6,
            s=115,
            marker="*",
            label="Vegas Pre-Season Expected Wins",
            zorder=6,
        )
    ax.scatter(
        spot_sorted["Enhanced LR"],
        y_pos,
        color="#1f77b4",
        s=65,
        marker="D",
        label="Enhanced Linear Regression",
        zorder=5,
    )
    ax.scatter(
        spot_sorted["Random Forest"],
        y_pos,
        color="#2ca02c",
        s=70,
        marker="^",
        label="Tuned Random Forest",
        zorder=5,
    )

    for idx, row in spot_sorted.iterrows():
        tm = row["Team"]
        if tm in logos:
            img = logos[tm]
            max_dim = float(max(img.shape[0], img.shape[1]))
            norm_zoom = (0.035 * 500.0) / max_dim
            ib = OffsetImage(img, zoom=norm_zoom)
            ab = AnnotationBbox(ib, (0.4, idx), frameon=False, zorder=6)
            ax.add_artist(ab)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r['Team']}  " for _, r in spot_sorted.iterrows()], fontsize=11, weight="bold")
    ax.tick_params(axis="x", labelsize=11)
    ax.set_xlim(0.0, 17.0)
    ax.set_xlabel("Regular-Season Wins", fontsize=12, weight="bold")
    ax.set_title(
        f"Team-by-Team Expected vs. Actual Wins, Vegas & Floor–Ceiling Band ({spotlight_window})\n"
        f"Shaded Bar = Model Floor–Ceiling | Black Circle = Actual | Gold Star = Vegas | Blue/Green = Models",
        fontsize=12,
        weight="bold",
    )
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10.5)
    fig.tight_layout()
    res_path = PLOTS_DIR / "prediction_error_residuals.png"
    fig.savefig(res_path, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 3: Random Forest Feature Importance Bar Chart ---
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=180)
    imp_sorted = imp_df.sort_values(by="RF_Importance (%)", ascending=True)
    bars = ax.barh(
        imp_sorted["Feature"],
        imp_sorted["RF_Importance (%)"],
        color="#2b5c8f",
        edgecolor="black",
        height=0.65,
    )
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.4,
            bar.get_y() + bar.get_height() / 2.0,
            f"{width:.1f}%",
            va="center",
            ha="left",
            fontsize=11,
            weight="bold",
        )
    ax.set_title(
        "Random Forest Feature Importance (%) — EPA vs. Turnover Margin & Efficiency",
        fontsize=12.5,
        weight="bold",
    )
    ax.set_xlabel("Relative Importance (%)", fontsize=11.5, weight="bold")
    ax.tick_params(axis="y", labelsize=11.5)
    ax.tick_params(axis="x", labelsize=11)
    ax.set_xlim(0, max(imp_sorted["RF_Importance (%)"]) * 1.20)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fi_path = PLOTS_DIR / "rf_feature_importance.png"
    fig.savefig(fi_path, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 4: Vegas vs. Model & 2nd-Degree Floor-Ceiling Accuracy Comparison ---
    vg_path = PLOTS_DIR / "vegas_vs_model_comparison.png"
    if "Vegas O/U" in all_test_results_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.5), dpi=180)

        # Panel A: Grouped Bar Chart across all 5 Held-Out Test Windows
        win_list = list(dict.fromkeys(all_test_results_df["Window"]))
        short_yrs = [w.split("->")[-1].strip() for w in win_list]
        vg_accs, lr_accs, rf_accs, deg2_accs = [], [], [], []
        for w_lbl in win_list:
            sub = all_test_results_df[all_test_results_df["Window"] == w_lbl]
            yt = sub["Actual Wins"].values
            gm = sub["Games"].values
            vg_accs.append(evaluate_predictions_in_wins(yt, sub["Vegas O/U"].values, gm)["accuracy_pct"])
            m_l = evaluate_predictions_in_wins(
                yt, sub["Enhanced LR"].values, gm,
                sub["Floor Wins"].values if "Floor Wins" in sub else None,
                sub["Ceiling Wins"].values if "Ceiling Wins" in sub else None,
            )
            lr_accs.append(m_l["accuracy_pct"])
            rf_accs.append(evaluate_predictions_in_wins(yt, sub["Random Forest"].values, gm)["accuracy_pct"])
            deg2_accs.append(m_l["second_deg_acc_pct"])

        x_idx = np.arange(len(short_yrs))
        bw = 0.20
        ax0 = axes[0]
        b1 = ax0.bar(x_idx - 1.5 * bw, vg_accs, bw, label="Vegas Pre-Season Implied", color="#e69f00", edgecolor="black")
        b2 = ax0.bar(x_idx - 0.5 * bw, lr_accs, bw, label="Enhanced Linear Reg (1st-Deg)", color="#1f77b4", edgecolor="black")
        b3 = ax0.bar(x_idx + 0.5 * bw, rf_accs, bw, label="Tuned Random Forest (1st-Deg)", color="#2ca02c", edgecolor="black")
        b4 = ax0.bar(x_idx + 1.5 * bw, deg2_accs, bw, label="Model 2nd-Deg (Floor–Ceiling)", color="#6a3d9a", edgecolor="black")

        for bars_grp in (b1, b2, b3, b4):
            for bar in bars_grp:
                h = bar.get_height()
                ax0.text(bar.get_x() + bar.get_width() / 2.0, h + 0.35, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, weight="bold")

        ax0.set_xticks(x_idx)
        ax0.set_xticklabels([f"{y} Season" for y in short_yrs], fontsize=11, weight="bold")
        ax0.set_ylim(75.0, 102.0)
        ax0.set_ylabel("Prediction Accuracy (%)", fontsize=11.5, weight="bold")
        ax0.set_title("Vegas vs. Models (1st-Degree & 2nd-Degree Floor–Ceiling Accuracy)", fontsize=12.5, weight="bold")
        ax0.grid(True, axis="y", linestyle=":", alpha=0.5)
        ax0.legend(loc="lower right", fontsize=9.5)

        # Panel B: Head-to-Head Error Scatter (Vegas Absolute Error vs. Model Absolute Error in Spotlight Window)
        ax1 = axes[1]
        vg_err = np.abs(spot_df["Vegas O/U"].values - spot_df["Actual Wins"].values)
        lr_err = np.abs(spot_df["Enhanced LR"].values - spot_df["Actual Wins"].values)
        max_e = max(float(np.max(vg_err)), float(np.max(lr_err))) + 0.8
        e_line = np.linspace(0, max_e, 100)
        ax1.plot(e_line, e_line, "--", color="#222222", linewidth=1.6, label="Tie (Model Error = Vegas Error)")
        ax1.fill_between(e_line, 0, e_line, color="#2ca02c", alpha=0.13, label="Model Beats Vegas (Lower Error)")
        ax1.fill_between(e_line, e_line, max_e, color="#d62728", alpha=0.08, label="Vegas Beats Model")

        coords = np.column_stack([vg_err, lr_err]).astype(float)
        for i in range(len(coords)):
            for j in range(i):
                if np.linalg.norm(coords[i] - coords[j]) < 0.22:
                    coords[i, 0] += 0.15
                    coords[i, 1] -= 0.12
        for tm, (xv, yv) in zip(spot_df["Team"].values, coords):
            if tm in logos:
                img = logos[tm]
                max_dim = float(max(img.shape[0], img.shape[1]))
                ab = AnnotationBbox(OffsetImage(img, zoom=(0.045 * 500.0) / max_dim), (xv, yv), frameon=False, zorder=4)
                ax1.add_artist(ab)

        beat_pct = float(np.mean(lr_err <= vg_err) * 100.0)
        ax1.set_xlim(0, max_e)
        ax1.set_ylim(0, max_e)
        ax1.set_xlabel("Vegas Absolute Error (|Vegas - Actual| Wins)", fontsize=11.5, weight="bold")
        ax1.set_ylabel("Model Absolute Error (|Model - Actual| Wins)", fontsize=11.5, weight="bold")
        ax1.set_title(
            f"Head-to-Head vs. Vegas ({spotlight_window}) — Model Beat/Tied on {beat_pct:.1f}% of Teams",
            fontsize=12,
            weight="bold",
        )
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.legend(loc="upper left", fontsize=9.5)

        fig.tight_layout()
        fig.savefig(vg_path, bbox_inches="tight")
        plt.close(fig)

    print("\n" + "-" * 98)
    print("VISUAL DIAGNOSTIC PLOTS GENERATED (WITH OFFICIAL NFL TEAM LOGOS):")
    print(f"  • Team Logo Scatter Plot (Single Seasons)      : {scatter_path}")
    print(f"  • 32-Team Dumbbell Chart (with Vegas & Bounds) : {res_path}")
    print(f"  • Vegas vs. Model & 2nd-Degree Accuracy Chart  : {vg_path}")
    print(f"  • Feature Importance Chart (RF)                : {fi_path}")
    print("-" * 98)


# ==============================================================================
# WINDOW HELPERS & STEP 1: DATA EXTRACTION / CACHING
# ==============================================================================
def normalize_window_spec(spec: WindowSpec) -> Tuple[int, int, int]:
    """Convert a WindowSpec into a canonical (start_year, end_year, target_year) tuple."""
    if isinstance(spec, int):
        return (spec - 3, spec - 1, spec)
    if len(spec) == 3:
        start_yr, end_yr, target_yr = int(spec[0]), int(spec[1]), int(spec[2])
        if end_yr >= target_yr or start_yr > end_yr:
            raise ValueError(
                f"Invalid window {spec}: require start_year <= end_year < target_year."
            )
        return (start_yr, end_yr, target_yr)
    raise ValueError(f"Unsupported WindowSpec format: {spec}")


def format_window_label(spec: Tuple[int, int, int]) -> str:
    """Format (2019, 2021, 2022) as '2019-2021 -> 2022'."""
    return f"{spec[0]}-{spec[1]} -> {spec[2]}"


def get_required_seasons(windows: Sequence[Tuple[int, int, int]]) -> List[int]:
    """Collect the sorted unique set of all historical and target seasons needed."""
    seasons = set()
    for start_yr, end_yr, target_yr in windows:
        seasons.update(range(start_yr, end_yr + 1))
        seasons.add(target_yr)
    return sorted(seasons)


def standardize_team_abbr(series: pd.Series) -> pd.Series:
    """Standardize NFL team abbreviations across historical seasons."""
    return series.replace(TEAM_ABBR_MAP)


def fetch_and_clean_data(seasons: Sequence[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch play-by-play (PBP) and schedule data for the requested `seasons`,
    using per-season disk caching in `.cache/` so changing ranges is instant.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    seasons_sorted = sorted(set(int(y) for y in seasons))

    missing_pbp_years = [
        y for y in seasons_sorted if not (CACHE_DIR / f"pbp_clean_{y}.parquet").exists()
    ]
    sched_cache_path = CACHE_DIR / "schedules_all.parquet"

    pbp_cols = [
        "season",
        "season_type",
        "posteam",
        "defteam",
        "play_type",
        "epa",
        "penalty",
        "qb_kneel",
        "qb_spike",
        "interception",
        "fumble_lost",
    ]

    if missing_pbp_years:
        print(
            f"Fetching NFL play-by-play data for seasons: {missing_pbp_years} "
            f"(cached seasons will load from disk)..."
        )
        pbp_raw = nfl.import_pbp_data(years=missing_pbp_years, columns=pbp_cols, downcast=True)
        pbp_clean_new = pbp_raw[
            (pbp_raw["season_type"] == "REG")
            & (pbp_raw["play_type"].isin(["pass", "run"]))
            & (pbp_raw["penalty"].fillna(0) == 0)
            & (pbp_raw["qb_kneel"].fillna(0) == 0)
            & (pbp_raw["qb_spike"].fillna(0) == 0)
            & (pbp_raw["epa"].notna())
            & (pbp_raw["posteam"].notna())
            & (pbp_raw["defteam"].notna())
        ].copy()
        pbp_clean_new["posteam"] = standardize_team_abbr(pbp_clean_new["posteam"])
        pbp_clean_new["defteam"] = standardize_team_abbr(pbp_clean_new["defteam"])

        for yr in missing_pbp_years:
            yr_df = pbp_clean_new[pbp_clean_new["season"] == yr]
            yr_df.to_parquet(CACHE_DIR / f"pbp_clean_{yr}.parquet", index=False)
    else:
        print(
            f"Loading cached play-by-play data for {len(seasons_sorted)} seasons "
            f"({min(seasons_sorted)}-{max(seasons_sorted)})..."
        )

    pbp_frames = [
        pd.read_parquet(CACHE_DIR / f"pbp_clean_{y}.parquet") for y in seasons_sorted
    ]
    pbp_clean = pd.concat(pbp_frames, ignore_index=True)

    # Fetch or load schedule data
    cached_sched_years: set[int] = set()
    if sched_cache_path.exists():
        sched_cached = pd.read_parquet(sched_cache_path)
        cached_sched_years = set(sched_cached["season"].unique())
    else:
        sched_cached = pd.DataFrame()

    missing_sched_years = [y for y in seasons_sorted if y not in cached_sched_years]
    if missing_sched_years:
        sched_raw = nfl.import_schedules(years=missing_sched_years)
        sched_new = sched_raw[
            (sched_raw["game_type"] == "REG")
            & (sched_raw["home_score"].notna())
            & (sched_raw["away_score"].notna())
        ].copy()
        sched_new["home_team"] = standardize_team_abbr(sched_new["home_team"])
        sched_new["away_team"] = standardize_team_abbr(sched_new["away_team"])
        sched_all = pd.concat([sched_cached, sched_new], ignore_index=True)
        sched_all.to_parquet(sched_cache_path, index=False)
    else:
        sched_all = sched_cached

    sched_clean = sched_all[sched_all["season"].isin(seasons_sorted)].copy()
    return pbp_clean, sched_clean


# ==============================================================================
# STEP 2 & 3: SEASONAL METRICS, SMART QB UPGRADE & ROLLING WINDOW ENGINEERING
# ==============================================================================
def build_qb_game_log(sched_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Extract a per-game log of starting QBs and their team point differential
    so we can evaluate incoming vs. outgoing QB quality across franchises.
    """
    h = sched_clean[
        ["season", "week", "home_team", "home_qb_id", "home_qb_name", "home_score", "away_score"]
    ].rename(
        columns={"home_team": "team", "home_qb_id": "qb_id", "home_qb_name": "qb_name"}
    )
    h["pd"] = h["home_score"] - h["away_score"]

    a = sched_clean[
        ["season", "week", "away_team", "away_qb_id", "away_qb_name", "home_score", "away_score"]
    ].rename(
        columns={"away_team": "team", "away_qb_id": "qb_id", "away_qb_name": "qb_name"}
    )
    a["pd"] = a["away_score"] - a["home_score"]

    return pd.concat(
        [
            h[["season", "week", "team", "qb_id", "qb_name", "pd"]],
            a[["season", "week", "team", "qb_id", "qb_name", "pd"]],
        ],
        ignore_index=True,
    )


def get_qb_shrunk_prior_pd(
    qb_id: str,
    window_years: Sequence[int],
    qb_game_log: pd.DataFrame,
    replacement_pd: float = -1.5,
    prior_games_weight: float = 10.0,
) -> float:
    """
    Compute a QB's recency-weighted point differential per game across `window_years`
    (regardless of which team they played for), shrunk via Empirical Bayes toward
    replacement level (`-1.5` PD/game) for rookies or low-sample backups.
    """
    if pd.isna(qb_id):
        return replacement_pd
    g = qb_game_log[
        (qb_game_log["qb_id"] == qb_id) & (qb_game_log["season"].isin(window_years))
    ]
    if g.empty:
        return replacement_pd
    latest_yr = max(window_years)
    w = np.where(g["season"] == latest_yr, 2.0, 1.0)
    raw_pd = float(np.sum(g["pd"] * w) / np.sum(w))
    n_starts = float(len(g))
    return (n_starts * raw_pd + prior_games_weight * replacement_pd) / (
        n_starts + prior_games_weight
    )


def compute_qb_upgrade_for_window(
    window_years: Sequence[int],
    target_year: int,
    qb_game_log: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare each team's Week 1 starting QB in `target_year` against their primary
    starter from `target_year - 1` using only prior-window performance (`window_years`).
    - Same QB returning -> 0.0 delta.
    - Veteran upgrade (e.g., 2025 Seahawks acquiring Sam Darnold after his 14-3
      2024 season in Minnesota) -> positive delta (+3.73 PD/game).
    - Losing a star QB to a backup/rookie -> negative delta.
    """
    s_ty = qb_game_log[qb_game_log["season"] == target_year].sort_values("week")
    s_y1 = qb_game_log[qb_game_log["season"] == target_year - 1]

    rows = []
    for team, grp in s_ty.groupby("team"):
        inc_qid = grp.iloc[0]["qb_id"]
        y1_tm = s_y1[s_y1["team"] == team]
        if not y1_tm.empty and y1_tm["qb_id"].notna().any():
            out_qid = y1_tm["qb_id"].mode().iloc[0]
        else:
            out_qid = inc_qid

        if inc_qid == out_qid:
            delta = 0.0
        else:
            inc_pd = max(
                0.0, get_qb_shrunk_prior_pd(inc_qid, window_years, qb_game_log)
            )
            out_pd = max(
                0.0, get_qb_shrunk_prior_pd(out_qid, window_years, qb_game_log)
            )
            delta = inc_pd - out_pd

        rows.append({"team": team, "qb_upgrade_pd_delta": delta})

    return pd.DataFrame(rows)


def compute_seasonal_market_srs(sched_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Compute iterative opponent-adjusted Simple Rating System (SRS) on market
    point spreads (`market_srs` and second-half `late_market_srs`) for each
    historical season.
    """
    h = sched_clean[["season", "week", "home_team", "away_team", "spread_line"]].copy()
    h["team"] = h["home_team"]
    h["opp"] = h["away_team"]
    h["mkt_diff"] = h["spread_line"].fillna(0.0) - 2.0

    a = sched_clean[["season", "week", "home_team", "away_team", "spread_line"]].copy()
    a["team"] = a["away_team"]
    a["opp"] = a["home_team"]
    a["mkt_diff"] = -a["spread_line"].fillna(0.0) + 2.0

    g = pd.concat([h, a], ignore_index=True)

    srs_rows = []
    for s, grp in g.groupby("season"):
        teams = sorted(grp["team"].unique())
        ratings = {t: 0.0 for t in teams}
        for _ in range(15):
            new_r = {}
            for t, tgrp in grp.groupby("team"):
                opp_r = tgrp["opp"].map(ratings).mean()
                new_r[t] = float(tgrp["mkt_diff"].mean() + opp_r)
            mean_r = float(np.mean(list(new_r.values())))
            ratings = {t: v - mean_r for t, v in new_r.items()}

        late_grp = grp[grp["week"] >= 10]
        for t in teams:
            t_late = late_grp[late_grp["team"] == t]
            if not t_late.empty:
                l_r = float(
                    t_late["mkt_diff"].mean() + t_late["opp"].map(ratings).mean()
                )
            else:
                l_r = ratings[t]
            srs_rows.append({
                "season": s,
                "team": t,
                "market_srs": ratings[t],
                "late_market_srs": l_r,
            })

    return pd.DataFrame(srs_rows)


def compute_seasonal_team_metrics(
    pbp_clean: pd.DataFrame,
    sched_clean: pd.DataFrame,
) -> pd.DataFrame:
    """
    Precompute per-season, per-team efficiency, scoring, market SRS, and
    second-half momentum metrics across all requested seasons (1999–2025).
    """
    pbp = pbp_clean.copy()
    pbp["turnover_event"] = (
        pbp["interception"].fillna(0) + pbp["fumble_lost"].fillna(0)
    )
    pbp["success"] = (pbp["epa"] > 0).astype(float)

    # Offensive play-by-play aggregates
    off_all = (
        pbp.groupby(["season", "posteam"], as_index=False)
        .agg(
            off_epa=("epa", "mean"),
            off_success_rate=("success", "mean"),
            giveaways=("turnover_event", "sum"),
        )
        .rename(columns={"posteam": "team"})
    )

    off_pass = (
        pbp[pbp["play_type"] == "pass"]
        .groupby(["season", "posteam"], as_index=False)["epa"]
        .mean()
        .rename(columns={"posteam": "team", "epa": "pass_epa"})
    )

    # Defensive play-by-play aggregates
    def_all = (
        pbp.groupby(["season", "defteam"], as_index=False)
        .agg(
            def_epa=("epa", "mean"),
            def_success_rate=("success", "mean"),
            takeaways=("turnover_event", "sum"),
        )
        .rename(columns={"defteam": "team"})
    )

    # Schedule-based scoring & win totals
    home = sched_clean[["season", "week", "home_team", "home_score", "away_score"]].copy()
    home["win"] = (home["home_score"] > home["away_score"]).astype(float) + 0.5 * (
        home["home_score"] == home["away_score"]
    ).astype(float)
    home["strict_win"] = (home["home_score"] > home["away_score"]).astype(int)
    home = home.rename(
        columns={"home_team": "team", "home_score": "pf", "away_score": "pa"}
    )

    away = sched_clean[["season", "week", "away_team", "home_score", "away_score"]].copy()
    away["win"] = (away["away_score"] > away["home_score"]).astype(float) + 0.5 * (
        away["away_score"] == away["home_score"]
    ).astype(float)
    away["strict_win"] = (away["away_score"] > away["home_score"]).astype(int)
    away = away.rename(
        columns={"away_team": "team", "away_score": "pf", "home_score": "pa"}
    )

    games = pd.concat([home, away], ignore_index=True)
    games["pd"] = games["pf"] - games["pa"]

    sched_agg = (
        games.groupby(["season", "team"], as_index=False)
        .agg(
            games_played=("pf", "count"),
            wins=("strict_win", "sum"),
            adj_wins=("win", "sum"),
            pf=("pf", "sum"),
            pa=("pa", "sum"),
        )
    )
    sched_agg["point_diff"] = sched_agg["pf"] - sched_agg["pa"]
    sched_agg["point_diff_pg"] = sched_agg["point_diff"] / sched_agg["games_played"]
    sched_agg["win_pct"] = sched_agg["adj_wins"] / sched_agg["games_played"]

    # Second-half of season (Week >= 10) momentum metrics
    late_games = games[games["week"] >= 10]
    late_agg = (
        late_games.groupby(["season", "team"], as_index=False)
        .agg(
            late_pd_pg=("pd", "mean"),
            late_win_pct=("win", "mean"),
        )
    )

    # NFL Pythagorean Win Expectation (exponent 2.37)
    pf_exp = np.power(np.maximum(sched_agg["pf"], 1.0), 2.37)
    pa_exp = np.power(np.maximum(sched_agg["pa"], 1.0), 2.37)
    sched_agg["pythag_win_pct"] = pf_exp / (pf_exp + pa_exp)

    srs_df = compute_seasonal_market_srs(sched_clean)

    # Merge all seasonal metrics
    seasonal = (
        sched_agg.merge(late_agg, on=["season", "team"], how="left")
        .merge(off_all, on=["season", "team"], how="left")
        .merge(off_pass, on=["season", "team"], how="left")
        .merge(def_all, on=["season", "team"], how="left")
        .merge(srs_df, on=["season", "team"], how="left")
    )

    seasonal["turnover_margin"] = seasonal["takeaways"] - seasonal["giveaways"]
    seasonal["turnover_margin_pg"] = (
        seasonal["turnover_margin"] / seasonal["games_played"]
    )
    seasonal["overperformance_luck"] = (
        seasonal["win_pct"] - seasonal["pythag_win_pct"]
    )

    return seasonal


def build_rolling_window_dataset_for_spec(
    window_spec: Tuple[int, int, int],
    pbp_clean: pd.DataFrame,
    sched_clean: pd.DataFrame,
    seasonal_metrics: pd.DataFrame,
    qb_game_log: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build both baseline and enhanced features for a (start_yr, end_yr, target_yr)
    window and map them to each team's regular-season wins in `target_yr`.
    """
    start_yr, end_yr, target_yr = window_spec
    window_years = list(range(start_yr, end_yr + 1))
    window_label = format_window_label(window_spec)

    # --- 1. Original 4 Baseline Features ---
    pbp_window = pbp_clean[pbp_clean["season"].isin(window_years)]
    off_epa_3yr = (
        pbp_window.groupby("posteam", as_index=False)["epa"]
        .mean()
        .rename(columns={"posteam": "team", "epa": "off_epa_per_play_3yr"})
    )
    def_epa_3yr = (
        pbp_window.groupby("defteam", as_index=False)["epa"]
        .mean()
        .rename(columns={"defteam": "team", "epa": "def_epa_per_play_3yr"})
    )

    win_seasonal = seasonal_metrics[seasonal_metrics["season"].isin(window_years)].copy()
    base_seasonal = (
        win_seasonal.groupby("team", as_index=False)
        .agg(
            avg_point_diff_3yr=("point_diff", "mean"),
            avg_turnover_margin_3yr=("turnover_margin", "mean"),
        )
    )

    # --- 2. Recency-Weighted, Y-1 & Late-Season Momentum Features ---
    n_yrs = len(window_years)
    if n_yrs == 3:
        weights = np.array(RECENCY_WEIGHTS, dtype=float)
    else:
        weights = np.arange(1, n_yrs + 1, dtype=float)
    weights = weights / weights.sum()
    year_to_weight = dict(zip(window_years, weights))

    win_seasonal["w"] = win_seasonal["season"].map(year_to_weight)

    weighted_cols = {
        "off_epa": "off_epa_weighted",
        "def_epa": "def_epa_weighted",
        "point_diff_pg": "point_diff_pg_weighted",
        "pythag_win_pct": "pythag_win_pct_weighted",
        "turnover_margin_pg": "turnover_margin_pg_weighted",
        "market_srs": "market_srs_weighted",
    }

    records = []
    y_last = window_years[-1]

    for team, grp in win_seasonal.groupby("team"):
        grp_sorted = grp.sort_values("season")
        w = grp_sorted["w"].to_numpy()
        w_sum = w.sum() if w.sum() > 0 else 1.0

        row = {"team": team}
        for src_col, dst_col in weighted_cols.items():
            vals = grp_sorted[src_col].to_numpy()
            row[dst_col] = float(np.nansum(vals * w) / w_sum)

        # Most recent year (Y-1) close-game luck & second-half momentum stats
        y1_row = grp_sorted[grp_sorted["season"] == y_last]
        if not y1_row.empty:
            r1 = y1_row.iloc[0]
            row["overperformance_luck_y1"] = float(r1["overperformance_luck"])
            row["late_pd_pg_y1"] = float(r1["late_pd_pg"])
            row["late_market_srs_y1"] = float(r1["late_market_srs"])
        else:
            row["overperformance_luck_y1"] = 0.0
            row["late_pd_pg_y1"] = row["point_diff_pg_weighted"]
            row["late_market_srs_y1"] = row["market_srs_weighted"]

        records.append(row)

    enhanced_df = pd.DataFrame(records)

    # --- 3. Smart Quarterback Upgrade / Downgrade Delta ---
    qb_delta_df = compute_qb_upgrade_for_window(window_years, target_yr, qb_game_log)
    enhanced_df = enhanced_df.merge(qb_delta_df, on="team", how="left")
    enhanced_df["qb_upgrade_pd_delta"] = enhanced_df["qb_upgrade_pd_delta"].fillna(0.0)

    # --- 3B. Pre-Season Week 1 Opening Market Rating (anchored to prior-window SRS) ---
    prior_anchor = dict(
        zip(
            enhanced_df["team"],
            0.6 * enhanced_df["market_srs_weighted"]
            + 0.4 * enhanced_df["late_market_srs_y1"]
            + 0.5 * enhanced_df["qb_upgrade_pd_delta"],
        )
    )
    wk1_sched = sched_clean[
        (sched_clean["season"] == target_yr) & (sched_clean["week"] == 1)
    ]
    if not wk1_sched.empty and wk1_sched["spread_line"].notna().any():
        h_w1 = wk1_sched[["home_team", "away_team", "spread_line"]].copy()
        h_w1["team"] = h_w1["home_team"]
        h_w1["opp"] = h_w1["away_team"]
        h_w1["mkt_diff"] = h_w1["spread_line"].fillna(0.0) - 2.0

        a_w1 = wk1_sched[["home_team", "away_team", "spread_line"]].copy()
        a_w1["team"] = a_w1["away_team"]
        a_w1["opp"] = a_w1["home_team"]
        a_w1["mkt_diff"] = -a_w1["spread_line"].fillna(0.0) + 2.0

        w1_all = pd.concat([h_w1, a_w1], ignore_index=True)
        w1_all["opp_anchor"] = w1_all["opp"].map(prior_anchor).fillna(0.0)
        w1_all["preseason_wk1_rating"] = w1_all["mkt_diff"] + w1_all["opp_anchor"]
        w1_df = w1_all.groupby("team", as_index=False)["preseason_wk1_rating"].mean()
        enhanced_df = enhanced_df.merge(w1_df, on="team", how="left")
        enhanced_df["preseason_wk1_rating"] = enhanced_df[
            "preseason_wk1_rating"
        ].fillna(enhanced_df["late_market_srs_y1"])
    else:
        enhanced_df["preseason_wk1_rating"] = enhanced_df["late_market_srs_y1"]

    # --- 4. Pre-Season Opponent Strength of Schedule (SOS) for `target_yr` ---
    sched_target = sched_clean[sched_clean["season"] == target_yr]
    team_pythag_map = dict(
        zip(enhanced_df["team"], enhanced_df["pythag_win_pct_weighted"])
    )

    opp_pairs = pd.concat(
        [
            sched_target[["home_team", "away_team"]].rename(
                columns={"home_team": "team", "away_team": "opp"}
            ),
            sched_target[["away_team", "home_team"]].rename(
                columns={"away_team": "team", "home_team": "opp"}
            ),
        ],
        ignore_index=True,
    )
    opp_pairs["opp_pythag"] = opp_pairs["opp"].map(team_pythag_map)
    sos_df = (
        opp_pairs.groupby("team", as_index=False)["opp_pythag"]
        .mean()
        .rename(columns={"opp_pythag": "target_sos_pythag"})
    )

    # --- 4B. Pre-Season Vegas Implied Win Totals for `target_yr` ---
    # Project each team's pre-season market rating across all scheduled games in `target_yr`
    # using Stern's normal game-margin model: P(win) = Phi(expected_spread / 13.45).
    r_map = dict(
        zip(
            enhanced_df["team"],
            0.55 * enhanced_df["preseason_wk1_rating"]
            + 0.45 * enhanced_df["market_srs_weighted"],
        )
    )
    h_vg = sched_target[["home_team", "away_team"]].copy()
    h_vg["team"] = h_vg["home_team"]
    h_vg["sp"] = (
        h_vg["home_team"].map(r_map).fillna(0.0)
        - h_vg["away_team"].map(r_map).fillna(0.0)
        + 2.0
    )
    a_vg = sched_target[["home_team", "away_team"]].copy()
    a_vg["team"] = a_vg["away_team"]
    a_vg["sp"] = (
        a_vg["away_team"].map(r_map).fillna(0.0)
        - a_vg["home_team"].map(r_map).fillna(0.0)
        - 2.0
    )
    vg_games = pd.concat([h_vg[["team", "sp"]], a_vg[["team", "sp"]]], ignore_index=True)
    # Normal CDF via vectorised erf approximation / math.erf
    from math import erf, sqrt
    vg_games["wp"] = [0.5 * (1.0 + erf(float(s) / (13.45 * sqrt(2.0)))) for s in vg_games["sp"]]
    vg_df = (
        vg_games.groupby("team", as_index=False)["wp"]
        .sum()
        .rename(columns={"wp": "vegas_preseason_wins"})
    )

    # --- 5. Target Wins & Target Games Played in `target_yr` ---
    target_season_df = seasonal_metrics[seasonal_metrics["season"] == target_yr][
        ["team", "wins", "games_played"]
    ].rename(columns={"wins": "target_wins", "games_played": "target_games"})
    target_season_df["target_win_pct"] = (
        target_season_df["target_wins"] / target_season_df["target_games"]
    )

    # Merge all features and target
    merged = (
        off_epa_3yr.merge(def_epa_3yr, on="team", how="outer")
        .merge(base_seasonal, on="team", how="outer")
        .merge(enhanced_df, on="team", how="outer")
        .merge(sos_df, on="team", how="left")
        .merge(vg_df, on="team", how="left")
        .merge(target_season_df, on="team", how="inner")
    )
    merged["vegas_preseason_wins"] = merged["vegas_preseason_wins"].fillna(
        0.5 * merged["target_games"]
    )

    merged["window_label"] = window_label
    merged["target_year"] = target_yr

    all_feature_cols = list(dict.fromkeys(BASELINE_FEATURE_COLS + ENHANCED_FEATURE_COLS))
    cols_order = (
        ["window_label", "target_year", "team"]
        + all_feature_cols
        + ["vegas_preseason_wins", "target_games", "target_win_pct", "target_wins"]
    )
    return merged[cols_order]


def build_master_dataset(
    windows: Sequence[Tuple[int, int, int]],
    pbp_clean: pd.DataFrame,
    sched_clean: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the master dataset across all requested `(start_yr, end_yr, target_yr)`
    windows and impute any missing feature entries.
    """
    seasonal_metrics = compute_seasonal_team_metrics(pbp_clean, sched_clean)
    qb_game_log = build_qb_game_log(sched_clean)

    unique_windows = list(dict.fromkeys(windows))
    yearly_dfs = [
        build_rolling_window_dataset_for_spec(
            w, pbp_clean, sched_clean, seasonal_metrics, qb_game_log
        )
        for w in unique_windows
    ]
    master_df = pd.concat(yearly_dfs, ignore_index=True)
    master_df = master_df.dropna(subset=["target_wins"]).reset_index(drop=True)

    all_feature_cols = list(dict.fromkeys(BASELINE_FEATURE_COLS + ENHANCED_FEATURE_COLS))
    imputer = SimpleImputer(strategy="median")
    master_df[all_feature_cols] = imputer.fit_transform(master_df[all_feature_cols])

    return master_df


# ==============================================================================
# STEP 4: LINEAR REGRESSION & RANDOM FOREST MODELING + WALK-FORWARD BACKTEST
# ==============================================================================
def evaluate_predictions_in_wins(
    y_true_wins: np.ndarray,
    y_pred_wins: np.ndarray,
    target_games: Union[np.ndarray, float] = 16.5,
    floor_wins: Union[np.ndarray, None] = None,
    ceiling_wins: Union[np.ndarray, None] = None,
) -> dict:
    """
    Compute comprehensive accuracy metrics in terms of NFL regular-season wins:
      - 1st-Degree Accuracy (%): (1 - |Pred_Wins - Actual_Wins| / Season_Games) * 100
      - 2nd-Degree Floor–Ceiling Hit Rate (%): % of teams with Floor <= Actual_Wins <= Ceiling
      - 2nd-Degree Range Accuracy (%): 100% if inside [Floor, Ceiling], else penalized only by
        distance outside the [Floor, Ceiling] band.
    """
    abs_err = np.abs(y_pred_wins - y_true_wins)
    mae = float(mean_absolute_error(y_true_wins, y_pred_wins))
    rmse = float(np.sqrt(mean_squared_error(y_true_wins, y_pred_wins)))
    r2 = float(r2_score(y_true_wins, y_pred_wins))
    within_1 = float(np.mean(abs_err <= 1.5) * 100.0)
    within_2 = float(np.mean(abs_err <= 2.5) * 100.0)

    games_arr = np.asarray(target_games, dtype=float)
    accuracy_pct = float(np.mean(np.maximum(0.0, 1.0 - abs_err / games_arr)) * 100.0)

    actual_winning_rec = (y_true_wins / games_arr) >= 0.5
    pred_winning_rec = (y_pred_wins / games_arr) >= 0.5
    tier_acc_pct = float(np.mean(actual_winning_rec == pred_winning_rec) * 100.0)

    if floor_wins is not None and ceiling_wins is not None:
        fl_arr = np.asarray(floor_wins, dtype=float)
        ce_arr = np.asarray(ceiling_wins, dtype=float)
        in_band = (y_true_wins >= fl_arr) & (y_true_wins <= ce_arr)
        floor_ceil_hit_pct = float(np.mean(in_band) * 100.0)
        band_dist = np.where(
            y_true_wins < fl_arr,
            fl_arr - y_true_wins,
            np.where(y_true_wins > ce_arr, y_true_wins - ce_arr, 0.0),
        )
        second_deg_acc_pct = float(
            np.mean(np.maximum(0.0, 1.0 - band_dist / games_arr)) * 100.0
        )
    else:
        floor_ceil_hit_pct = within_2
        second_deg_acc_pct = accuracy_pct

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "accuracy_pct": accuracy_pct,
        "tier_acc_pct": tier_acc_pct,
        "within_1_5_pct": within_1,
        "within_2_5_pct": within_2,
        "floor_ceil_hit_pct": floor_ceil_hit_pct,
        "second_deg_acc_pct": second_deg_acc_pct,
    }


def compute_team_floor_and_ceiling(
    pred_wins: np.ndarray,
    rf_model: RandomForestRegressor,
    X_features: pd.DataFrame,
    games: Union[np.ndarray, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a tight 3-to-4 game [Floor Wins, Ceiling Wins] range for each team:
      - Low-variance / stable teams (below-median tree dispersion): 3-game span (±1.5 wins,
        e.g., projected 10.4W -> 9-8 to 12-5).
      - Higher-variance teams (above-median tree dispersion): 4-game span (±2.0 wins,
        e.g., projected 10.0W -> 8-9 to 12-5).
    """
    games_arr = np.broadcast_to(np.asarray(games, dtype=float), len(pred_wins))
    tree_preds = np.array([
        calibrate_win_pct(t.predict(X_features.values)) * games_arr
        for t in rf_model.estimators_
    ])
    tree_std = np.std(tree_preds, axis=0)
    med_std = float(np.median(tree_std))

    floor_list, ceil_list = [], []
    for p, s, g in zip(pred_wins, tree_std, games_arr):
        p_rnd = int(round(float(p)))
        if float(s) >= med_std:
            f_w, c_w = p_rnd - 2, p_rnd + 2  # 4-game span (e.g., 10-7 -> 8-9 to 12-5)
        else:
            if float(p) >= p_rnd:
                f_w, c_w = p_rnd - 1, p_rnd + 2  # 3-game span (e.g., 10.4 -> 9-8 to 12-5)
            else:
                f_w, c_w = p_rnd - 2, p_rnd + 1  # 3-game span (e.g., 9.6 -> 8-9 to 11-6)
        floor_list.append(int(np.clip(f_w, 1, int(g) - 1)))
        ceil_list.append(int(np.clip(c_w, 1, int(g) - 1)))

    return np.array(floor_list, dtype=int), np.array(ceil_list, dtype=int)


def calibrate_win_pct(raw_win_pct: np.ndarray, gamma: float = DISPERSION_GAMMA) -> np.ndarray:
    """
    Apply Reduced Major Axis (RMA) dispersion calibration around .500 so that
    regression-to-the-mean shrinkage does not artificially cap elite teams at
    10 wins or floor rebuilding teams at 6 wins.
    """
    return np.clip(0.50 + gamma * (np.asarray(raw_win_pct, dtype=float) - 0.50), 0.08, 0.92)


def build_models() -> Dict[str, BaseEstimator]:
    """
    Return the Linear Regression and Random Forest candidate models.
    """
    return {
        "Enhanced Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Random Forest (Depth=4, Leaf=5)": RandomForestRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=5,
            max_features=0.7,
            random_state=42,
        ),
        "Random Forest (Depth=5, Leaf=3)": RandomForestRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=3,
            max_features=0.7,
            random_state=42,
        ),
    }


def run_walk_forward_backtest(
    master_df: pd.DataFrame,
    all_windows: Sequence[Tuple[int, int, int]],
    min_initial_windows: int = 3,
) -> pd.DataFrame:
    """
    Walk-Forward Expanding-Window Backtest across 1999–2025:
    Starts with `min_initial_windows` of history, predicts the next out-of-sample
    season, scores Linear Regression vs. Random Forest, and expands the training
    set year-by-year through 2025.
    """
    ordered_labels = [format_window_label(w) for w in all_windows]
    models = build_models()
    rf_template = models["Random Forest (Depth=5, Leaf=3)"]
    lr_template = models["Enhanced Linear Regression"]

    rows = []
    all_base_preds, all_lr_preds, all_rf_preds, all_actuals, all_games = [], [], [], [], []

    for idx in range(min_initial_windows, len(ordered_labels)):
        train_labels = ordered_labels[:idx]
        test_label = ordered_labels[idx]

        tr_df = master_df[master_df["window_label"].isin(train_labels)]
        te_df = master_df[master_df["window_label"] == test_label]

        y_true = te_df["target_wins"].values
        games = te_df["target_games"].values

        # 1. Baseline Linear Regression (4 raw features -> raw wins)
        base_lr = LinearRegression().fit(
            tr_df[BASELINE_FEATURE_COLS], tr_df["target_wins"]
        )
        pred_base = base_lr.predict(te_df[BASELINE_FEATURE_COLS])

        # 2. Enhanced Linear Regression (era-normalized & dispersion-calibrated win % * games)
        enh_lr = clone(lr_template).fit(
            tr_df[ENHANCED_FEATURE_COLS], tr_df["target_win_pct"]
        )
        pred_lr = calibrate_win_pct(enh_lr.predict(te_df[ENHANCED_FEATURE_COLS])) * games

        # 3. Random Forest Regressor (era-normalized & dispersion-calibrated win % * games)
        rf_model = clone(rf_template).fit(
            tr_df[ENHANCED_FEATURE_COLS], tr_df["target_win_pct"]
        )
        pred_rf = calibrate_win_pct(rf_model.predict(te_df[ENHANCED_FEATURE_COLS])) * games

        all_base_preds.extend(pred_base)
        all_lr_preds.extend(pred_lr)
        all_rf_preds.extend(pred_rf)
        all_actuals.extend(y_true)
        all_games.extend(games)

        m_base = evaluate_predictions_in_wins(y_true, pred_base, games)
        m_lr = evaluate_predictions_in_wins(y_true, pred_lr, games)
        m_rf = evaluate_predictions_in_wins(y_true, pred_rf, games)

        rows.append({
            "Train Windows": idx,
            "Out-of-Sample Window": test_label,
            "Baseline LR MAE": round(m_base["mae"], 3),
            "Enhanced LR MAE": round(m_lr["mae"], 3),
            "Random Forest MAE": round(m_rf["mae"], 3),
            "LR Accuracy (%)": f"{m_lr['accuracy_pct']:.1f}%",
            "RF Accuracy (%)": f"{m_rf['accuracy_pct']:.1f}%",
        })

    wf_df = pd.DataFrame(rows)

    first_test_yr = all_windows[min_initial_windows][2]
    last_test_yr = all_windows[-1][2]
    print("\n" + "=" * 98)
    print(
        f"STAGE 1: WALK-FORWARD EXPANDING-WINDOW BACKTEST "
        f"({len(rows)} OUT-OF-SAMPLE SEASONS, {first_test_yr}–{last_test_yr})"
    )
    print("=" * 98)
    print(wf_df.to_string(index=False))

    tot_base = evaluate_predictions_in_wins(
        np.array(all_actuals), np.array(all_base_preds), np.array(all_games)
    )
    tot_lr = evaluate_predictions_in_wins(
        np.array(all_actuals), np.array(all_lr_preds), np.array(all_games)
    )
    tot_rf = evaluate_predictions_in_wins(
        np.array(all_actuals), np.array(all_rf_preds), np.array(all_games)
    )

    col_hdr = f"Model ({len(rows)} Out-of-Sample Seasons, {len(all_actuals)} Team-Seasons)"
    summary_wf = pd.DataFrame([
        {
            col_hdr: "Baseline Linear Regression (4 Raw Features)",
            "Accuracy (%)": f"{tot_base['accuracy_pct']:.2f}%",
            "Record .500+ Acc (%)": f"{tot_base['tier_acc_pct']:.1f}%",
            "Within ±2 Wins (%)": f"{tot_base['within_2_5_pct']:.1f}%",
            "Overall MAE": round(tot_base["mae"], 3),
            "Overall R^2": round(tot_base["r2"], 3),
        },
        {
            col_hdr: "Enhanced Linear Regression (Era-Normalized)",
            "Accuracy (%)": f"{tot_lr['accuracy_pct']:.2f}%",
            "Record .500+ Acc (%)": f"{tot_lr['tier_acc_pct']:.1f}%",
            "Within ±2 Wins (%)": f"{tot_lr['within_2_5_pct']:.1f}%",
            "Overall MAE": round(tot_lr["mae"], 3),
            "Overall R^2": round(tot_lr["r2"], 3),
        },
        {
            col_hdr: "Random Forest Regressor (Era-Normalized)",
            "Accuracy (%)": f"{tot_rf['accuracy_pct']:.2f}%",
            "Record .500+ Acc (%)": f"{tot_rf['tier_acc_pct']:.1f}%",
            "Within ±2 Wins (%)": f"{tot_rf['within_2_5_pct']:.1f}%",
            "Overall MAE": round(tot_rf["mae"], 3),
            "Overall R^2": round(tot_rf["r2"], 3),
        },
    ])
    print("-" * 98)
    print(summary_wf.to_string(index=False))
    return wf_df


def train_and_evaluate_custom_windows(
    master_df: pd.DataFrame,
    train_windows: Sequence[Tuple[int, int, int]],
    test_windows: Sequence[Tuple[int, int, int]],
) -> None:
    """
    Cross-validate Linear Regression and Random Forest on `train_windows`,
    display feature importances, and evaluate both models out-of-sample on
    every range in `test_windows`.
    """
    train_labels = [format_window_label(w) for w in train_windows]
    test_labels = [format_window_label(w) for w in test_windows]

    train_df = master_df[master_df["window_label"].isin(train_labels)].copy()
    test_df = master_df[master_df["window_label"].isin(test_labels)].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Train or Test dataframe is empty. Check TRAIN_WINDOWS and TEST_WINDOWS."
        )

    logo = LeaveOneGroupOut()
    groups = train_df["window_label"].values
    y_win_pct = train_df["target_win_pct"].values
    y_wins = train_df["target_wins"].values
    games = train_df["target_games"].values

    candidates = build_models()
    cv_rows = []

    # 1. Baseline 4-Feature Linear Regression CV
    oof_base = np.zeros(len(train_df))
    for tr_idx, val_idx in logo.split(train_df, y_wins, groups):
        m = LinearRegression().fit(
            train_df.iloc[tr_idx][BASELINE_FEATURE_COLS], y_wins[tr_idx]
        )
        oof_base[val_idx] = m.predict(train_df.iloc[val_idx][BASELINE_FEATURE_COLS])
    bm = evaluate_predictions_in_wins(y_wins, oof_base, games)
    cv_rows.append({
        "Model Architecture": "Baseline Linear Regression (4 Raw Features)",
        "CV Accuracy (%)": f"{bm['accuracy_pct']:.2f}%",
        "Record .500+ Acc (%)": f"{bm['tier_acc_pct']:.1f}%",
        "Within ±2 Wins (%)": f"{bm['within_2_5_pct']:.1f}%",
        "CV MAE (Wins)": round(bm["mae"], 3),
        "CV R^2": round(bm["r2"], 3),
    })

    best_rf_name = "Random Forest (Depth=5, Leaf=3)"
    best_rf_mae = float("inf")

    for name, est in candidates.items():
        oof_pred = np.zeros(len(train_df))
        for tr_idx, val_idx in logo.split(train_df, y_win_pct, groups):
            m = clone(est).fit(
                train_df.iloc[tr_idx][ENHANCED_FEATURE_COLS], y_win_pct[tr_idx]
            )
            oof_pred[val_idx] = (
                calibrate_win_pct(m.predict(train_df.iloc[val_idx][ENHANCED_FEATURE_COLS]))
                * games[val_idx]
            )
        metrics = evaluate_predictions_in_wins(y_wins, oof_pred, games)
        cv_rows.append({
            "Model Architecture": name,
            "CV Accuracy (%)": f"{metrics['accuracy_pct']:.2f}%",
            "Record .500+ Acc (%)": f"{metrics['tier_acc_pct']:.1f}%",
            "Within ±2 Wins (%)": f"{metrics['within_2_5_pct']:.1f}%",
            "CV MAE (Wins)": round(metrics["mae"], 3),
            "CV R^2": round(metrics["r2"], 3),
        })
        if name.startswith("Random Forest") and metrics["mae"] < best_rf_mae:
            best_rf_mae = metrics["mae"]
            best_rf_name = name

    print("\n" + "=" * 98)
    print(
        f"STAGE 2: FULL 24-WINDOW PARTITION — {len(train_labels)} TRAIN WINDOWS "
        f"& {len(test_labels)} TEST WINDOWS (CV)"
    )
    print("=" * 98)
    print(f"Active Training Windows ({len(train_labels)}): {', '.join(train_labels)}")
    print(f"Active Testing Windows  ({len(test_labels)}): {', '.join(test_labels)}")
    print("-" * 98)
    print(pd.DataFrame(cv_rows).sort_values("CV MAE (Wins)").to_string(index=False))

    # Fit final Linear Regression & Random Forest models on all 19 TRAIN_WINDOWS
    base_lr_model = LinearRegression().fit(
        train_df[BASELINE_FEATURE_COLS], train_df["target_wins"]
    )
    enh_lr_model = clone(candidates["Enhanced Linear Regression"]).fit(
        train_df[ENHANCED_FEATURE_COLS], train_df["target_win_pct"]
    )
    rf_model = clone(candidates[best_rf_name]).fit(
        train_df[ENHANCED_FEATURE_COLS], train_df["target_win_pct"]
    )

    # Display Feature Importance (Linear Regression Standardized Coefs & Random Forest Importance)
    lr_coefs = enh_lr_model.named_steps["model"].coef_
    imp_df = pd.DataFrame({
        "Feature": ENHANCED_FEATURE_COLS,
        "LR_Impact_17G_Wins": (lr_coefs * 17.0).round(3),
        "RF_Importance (%)": (rf_model.feature_importances_ * 100.0).round(1),
    }).sort_values(by="RF_Importance (%)", ascending=False)

    print("\n" + "-" * 98)
    print("LEARNED FEATURE IMPORTANCE (ENHANCED LINEAR REGRESSION & RANDOM FOREST)")
    print("-" * 98)
    print(imp_df.to_string(index=False))

    # Evaluate on each custom test window
    print("\n" + "=" * 98)
    print("STAGE 3: OUT-OF-SAMPLE EVALUATION ACROSS HELD-OUT TEST WINDOWS")
    print("=" * 98)

    test_summary_rows = []
    all_test_dfs = []
    for w_label in test_labels:
        w_df = test_df[test_df["window_label"] == w_label].copy()
        y_true = w_df["target_wins"].values
        w_games = w_df["target_games"].values

        pred_base = base_lr_model.predict(w_df[BASELINE_FEATURE_COLS])
        pred_lr = (
            calibrate_win_pct(enh_lr_model.predict(w_df[ENHANCED_FEATURE_COLS]))
            * w_games
        )
        pred_rf = (
            calibrate_win_pct(rf_model.predict(w_df[ENHANCED_FEATURE_COLS]))
            * w_games
        )
        pred_vegas = w_df["vegas_preseason_wins"].values

        floor_lr, ceil_lr = compute_team_floor_and_ceiling(
            pred_lr, rf_model, w_df[ENHANCED_FEATURE_COLS], w_games
        )
        floor_rf, ceil_rf = compute_team_floor_and_ceiling(
            pred_rf, rf_model, w_df[ENHANCED_FEATURE_COLS], w_games
        )

        m_base = evaluate_predictions_in_wins(y_true, pred_base, w_games)
        m_vegas = evaluate_predictions_in_wins(y_true, pred_vegas, w_games)
        m_lr = evaluate_predictions_in_wins(y_true, pred_lr, w_games, floor_lr, ceil_lr)
        m_rf = evaluate_predictions_in_wins(y_true, pred_rf, w_games, floor_rf, ceil_rf)

        beat_vegas_pct = float(
            np.mean(np.abs(pred_lr - y_true) <= np.abs(pred_vegas - y_true)) * 100.0
        )

        test_summary_rows.append({
            "Test Window": w_label,
            "Vegas Acc (%)": f"{m_vegas['accuracy_pct']:.1f}%",
            "Enh LR Acc (%)": f"{m_lr['accuracy_pct']:.1f}%",
            "RF Acc (%)": f"{m_rf['accuracy_pct']:.1f}%",
            "2nd-Deg Range Acc": f"{m_lr['second_deg_acc_pct']:.1f}%",
            "In Floor-Ceil (%)": f"{m_lr['floor_ceil_hit_pct']:.1f}%",
            "Vegas MAE": round(m_vegas["mae"], 3),
            "Enh LR MAE": round(m_lr["mae"], 3),
            "RF MAE": round(m_rf["mae"], 3),
            "Beat/Tie Vegas": f"{beat_vegas_pct:.1f}%",
        })

        lr_team_acc = np.maximum(0.0, 1.0 - np.abs(pred_lr - y_true) / w_games) * 100.0
        in_band_flags = [
            "YES" if (fl <= act <= ce) else "NO"
            for act, fl, ce in zip(y_true.astype(int), floor_lr, ceil_lr)
        ]
        floor_ceil_labels = [
            f"{fl}-{int(g)-fl} to {ce}-{int(g)-ce}"
            for fl, ce, g in zip(floor_lr, ceil_lr, w_games)
        ]

        comparison_df = pd.DataFrame({
            "Window": w_df["window_label"].values,
            "Team": w_df["team"].values,
            "Actual Wins": y_true.astype(int),
            "Games": w_games,
            "Vegas O/U": pred_vegas.round(2),
            "Enhanced LR": pred_lr.round(2),
            "Random Forest": pred_rf.round(2),
            "Floor Wins": floor_lr,
            "Ceiling Wins": ceil_lr,
            "Floor-Ceiling Range": floor_ceil_labels,
            "In Range?": in_band_flags,
            "1st-Deg Acc": [f"{a:.1f}%" for a in lr_team_acc],
        }).sort_values(by=["Actual Wins", "Enhanced LR"], ascending=[False, False])
        all_test_dfs.append(comparison_df)

        print(f"\nTest Window: {w_label}")
        print(
            f"  Vegas Pre-Season    -> 1st-Deg Acc: {m_vegas['accuracy_pct']:.2f}% | "
            f".500+ Record Acc: {m_vegas['tier_acc_pct']:.1f}% | "
            f"MAE: {m_vegas['mae']:.3f} wins | R^2: {m_vegas['r2']:.3f}"
        )
        print(
            f"  Enhanced Linear Reg -> 1st-Deg Acc: {m_lr['accuracy_pct']:.2f}% | "
            f"2nd-Deg Range Acc: {m_lr['second_deg_acc_pct']:.2f}% (In Floor-Ceil: {m_lr['floor_ceil_hit_pct']:.1f}%) | "
            f"MAE: {m_lr['mae']:.3f} wins | R^2: {m_lr['r2']:.3f} | Beat/Tie Vegas: {beat_vegas_pct:.1f}%"
        )
        print(
            f"  Random Forest       -> 1st-Deg Acc: {m_rf['accuracy_pct']:.2f}% | "
            f"2nd-Deg Range Acc: {m_rf['second_deg_acc_pct']:.2f}% (In Floor-Ceil: {m_rf['floor_ceil_hit_pct']:.1f}%) | "
            f"MAE: {m_rf['mae']:.3f} wins | R^2: {m_rf['r2']:.3f}"
        )
        if w_label in ("2011-2013 -> 2014", "2022-2024 -> 2025"):
            print("-" * 98)
            print(
                comparison_df.drop(columns=["Games", "Floor Wins", "Ceiling Wins"]).to_string(
                    index=False
                )
            )

    print("\n" + "=" * 98)
    print("SUMMARY ACROSS ALL HELD-OUT TEST WINDOWS")
    print("=" * 98)
    print(pd.DataFrame(test_summary_rows).to_string(index=False))

    # Generate and save diagnostic scatter plots & Random Forest feature importance chart
    generate_evaluation_plots(pd.concat(all_test_dfs, ignore_index=True), imp_df)


def project_2026_season(
    master_df: pd.DataFrame,
    pbp_clean: pd.DataFrame,
    sched_clean: pd.DataFrame,
) -> pd.DataFrame:
    """
    STAGE 4: Train on all 24 completed historical windows (1999–2025) and
    project each NFL team's full 17-game 2026 regular-season win total using
    the trailing 2023–2025 window (`2023-2025 -> 2026`).
    """
    window_years = [2023, 2024, 2025]
    seasonal_metrics = compute_seasonal_team_metrics(pbp_clean, sched_clean)

    # 1. Baseline 3-year features (2023-2025)
    pbp_window = pbp_clean[pbp_clean["season"].isin(window_years)]
    off_epa_3yr = (
        pbp_window.groupby("posteam", as_index=False)["epa"]
        .mean()
        .rename(columns={"posteam": "team", "epa": "off_epa_per_play_3yr"})
    )
    def_epa_3yr = (
        pbp_window.groupby("defteam", as_index=False)["epa"]
        .mean()
        .rename(columns={"defteam": "team", "epa": "def_epa_per_play_3yr"})
    )

    win_seasonal = seasonal_metrics[seasonal_metrics["season"].isin(window_years)].copy()
    base_seasonal = (
        win_seasonal.groupby("team", as_index=False)
        .agg(
            avg_point_diff_3yr=("point_diff", "mean"),
            avg_turnover_margin_3yr=("turnover_margin", "mean"),
        )
    )

    # 2. Enhanced recency-weighted (2023-2025) & 2025 late-season momentum features
    weights = np.array(RECENCY_WEIGHTS, dtype=float)
    weights = weights / weights.sum()
    year_to_weight = dict(zip(window_years, weights))
    win_seasonal["w"] = win_seasonal["season"].map(year_to_weight)

    weighted_cols = {
        "off_epa": "off_epa_weighted",
        "def_epa": "def_epa_weighted",
        "point_diff_pg": "point_diff_pg_weighted",
        "pythag_win_pct": "pythag_win_pct_weighted",
        "turnover_margin_pg": "turnover_margin_pg_weighted",
        "market_srs": "market_srs_weighted",
    }

    records = []
    for team, grp in win_seasonal.groupby("team"):
        grp_sorted = grp.sort_values("season")
        w = grp_sorted["w"].to_numpy()
        w_sum = w.sum() if w.sum() > 0 else 1.0

        row = {"team": team}
        for src_col, dst_col in weighted_cols.items():
            vals = grp_sorted[src_col].to_numpy()
            row[dst_col] = float(np.nansum(vals * w) / w_sum)

        y1_row = grp_sorted[grp_sorted["season"] == 2025]
        if not y1_row.empty:
            r1 = y1_row.iloc[0]
            row["overperformance_luck_y1"] = float(r1["overperformance_luck"])
            row["late_pd_pg_y1"] = float(r1["late_pd_pg"])
            row["preseason_wk1_rating"] = float(r1["late_market_srs"])
            row["wins_2025"] = int(r1["wins"])
        else:
            row["overperformance_luck_y1"] = 0.0
            row["late_pd_pg_y1"] = row["point_diff_pg_weighted"]
            row["preseason_wk1_rating"] = row["market_srs_weighted"]
            row["wins_2025"] = 8

        row["qb_upgrade_pd_delta"] = 0.0
        row["target_sos_pythag"] = 0.500
        records.append(row)

    enhanced_2026 = pd.DataFrame(records)
    df_2026 = (
        off_epa_3yr.merge(def_epa_3yr, on="team", how="inner")
        .merge(base_seasonal, on="team", how="inner")
        .merge(enhanced_2026, on="team", how="inner")
    )

    all_feature_cols = list(dict.fromkeys(BASELINE_FEATURE_COLS + ENHANCED_FEATURE_COLS))
    imputer = SimpleImputer(strategy="median").fit(master_df[all_feature_cols])
    df_2026[all_feature_cols] = imputer.transform(df_2026[all_feature_cols])

    # Fit models on all 24 completed historical windows (1999-2025)
    candidates = build_models()
    base_lr = LinearRegression().fit(
        master_df[BASELINE_FEATURE_COLS], master_df["target_wins"]
    )
    enh_lr = clone(candidates["Enhanced Linear Regression"]).fit(
        master_df[ENHANCED_FEATURE_COLS], master_df["target_win_pct"]
    )
    rf_model = clone(candidates["Random Forest (Depth=5, Leaf=3)"]).fit(
        master_df[ENHANCED_FEATURE_COLS], master_df["target_win_pct"]
    )

    pred_base_26 = base_lr.predict(df_2026[BASELINE_FEATURE_COLS])
    pred_lr_26 = calibrate_win_pct(enh_lr.predict(df_2026[ENHANCED_FEATURE_COLS])) * 17.0
    pred_rf_26 = calibrate_win_pct(rf_model.predict(df_2026[ENHANCED_FEATURE_COLS])) * 17.0
    floor_26, ceil_26 = compute_team_floor_and_ceiling(
        pred_lr_26, rf_model, df_2026[ENHANCED_FEATURE_COLS], 17.0
    )

    proj_table = pd.DataFrame({
        "Window": "2023-2025 -> 2026",
        "Team": df_2026["team"].values,
        "2025 Wins": df_2026["wins_2025"].values,
        "3Yr Pythag (%)": (df_2026["pythag_win_pct_weighted"] * 100.0).round(1).values,
        "Enhanced LR (2026)": pred_lr_26.round(2),
        "Random Forest (2026)": pred_rf_26.round(2),
        "Proj Record (LR)": [
            f"{int(round(p))}-{17 - int(round(p))}" for p in pred_lr_26
        ],
        "Proj Record (RF)": [
            f"{int(round(p))}-{17 - int(round(p))}" for p in pred_rf_26
        ],
        "2026 Floor-Ceiling": [
            f"{fl}-{17-fl} to {ce}-{17-ce}" for fl, ce in zip(floor_26, ceil_26)
        ],
    }).sort_values(by=["Enhanced LR (2026)", "Random Forest (2026)"], ascending=[False, False])

    print("\n" + "=" * 98)
    print("STAGE 4: 2026 NFL REGULAR SEASON WIN PROJECTIONS (2023-2025 -> 2026, 17 GAMES)")
    print("=" * 98)
    print(proj_table.to_string(index=False))
    return proj_table


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main() -> None:
    train_specs = [normalize_window_spec(w) for w in TRAIN_WINDOWS]
    test_specs = [normalize_window_spec(w) for w in TEST_WINDOWS]
    all_specs = list(
        dict.fromkeys(ALL_HISTORICAL_WINDOWS + train_specs + test_specs)
    )

    # Step 1: Fetch/cache all required seasons (1999-2025)
    required_seasons = get_required_seasons(all_specs)
    pbp_clean, sched_clean = fetch_and_clean_data(required_seasons)

    # Step 2 & 3: Build master dataset across all 24 3-year -> 1-year windows
    master_dataset = build_master_dataset(all_specs, pbp_clean, sched_clean)
    print(
        f"Master dataset built with {len(master_dataset)} team-seasons across "
        f"{len(all_specs)} windows ({min(required_seasons)}–{max(required_seasons)})."
    )

    # Step 4A: Full 1999-2025 Walk-Forward Expanding-Window Backtest
    if RUN_WALK_FORWARD_BACKTEST:
        run_walk_forward_backtest(master_dataset, ALL_HISTORICAL_WINDOWS)

    # Step 4B: Full 24-Window Train/Test Partition Evaluation
    train_and_evaluate_custom_windows(master_dataset, train_specs, test_specs)

    # Step 4C: Project the 2026 Season (2023-2025 -> 2026)
    project_2026_season(master_dataset, pbp_clean, sched_clean)


if __name__ == "__main__":
    main()
