"""
Two-year forecast demo -- uses the existing dot_forecast_model.py as-is
(no changes to it). Prints FY2028/FY2029 point forecasts + 95% prediction
intervals for all three series, and saves one chart: the historical
Modified series (FY2017-2026) plus the FY2027-2029 forecast with interval
band, in the project's established chart style.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dot_forecast_model as m

# --- Project's established chart style (matches dot_forecast.ipynb) ------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = {
    "adopted": "#2a78d6",
    "modified": "#eb6834",
    "actual": "#1baf7a",
}

for _style in ["seaborn-v0_8-whitegrid", "seaborn-whitegrid"]:
    if _style in plt.style.available:
        plt.style.use(_style)
        break
else:
    plt.style.use("default")


# ---------------------------------------------------------------------------
# 1. Numbers: FY2028 and FY2029 only
# ---------------------------------------------------------------------------
print("#" * 70)
print("# FY2028 & FY2029 forecast (model trained on FY2017-2024)")
print("#" * 70)

m.train_models()

for year in (2028, 2029):
    forecast = m.predict(year)
    print(f"\nFY{year}:")
    print(forecast.to_string())


# ---------------------------------------------------------------------------
# 2. Chart: historical Modified (2017-2026) + FY2027-2029 forecast band
# ---------------------------------------------------------------------------
df = m.load_data()
history = df[df["fiscal_year"] <= 2026].reset_index(drop=True)
FORECAST_YEARS = [2027, 2028, 2029]

forecast_rows = [m.predict(y).loc["modified"] for y in FORECAST_YEARS]
last_hist_year = int(history["fiscal_year"].max())
last_hist_val = history.loc[history["fiscal_year"] == last_hist_year, "modified"].iloc[0]

fx = [last_hist_year] + FORECAST_YEARS
fpoint = [last_hist_val] + [r["point_forecast"] for r in forecast_rows]
flow = [last_hist_val] + [r["lower_95"] for r in forecast_rows]
fhigh = [last_hist_val] + [r["upper_95"] for r in forecast_rows]

color = SERIES["modified"]

fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

ax.plot(
    history["fiscal_year"], history["modified"] / 1e9,
    color=color, linewidth=2.5, marker="o", markersize=8,
    markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.5,
    label="Historical (FY2017-2026)", zorder=3,
)
ax.plot(
    fx, np.array(fpoint) / 1e9,
    color=color, linewidth=2.5, linestyle="--", marker="o", markersize=7,
    markerfacecolor=SURFACE, markeredgecolor=color, markeredgewidth=1.5,
    label="Forecast (FY2027-2029)", zorder=3,
)
ax.fill_between(
    fx, np.array(flow) / 1e9, np.array(fhigh) / 1e9,
    color=color, alpha=0.15, linewidth=0, label="95% prediction interval", zorder=1,
)

ax.axvline(last_hist_year + 0.5, color=BASELINE, linewidth=1.5, linestyle=":", zorder=2)
ax.text(
    last_hist_year + 0.55, 0.97, "Forecast begins",
    transform=ax.get_xaxis_transform(),
    color=INK_SECONDARY, fontsize=12, va="top", ha="left",
)

ax.set_xlabel("Fiscal Year", fontsize=14, color=INK_SECONDARY, labelpad=10)
ax.set_ylabel("Modified Budget ($ Billions)", fontsize=14, color=INK_SECONDARY, labelpad=10)
ax.set_title(
    "DOT Modified Budget: History & 2-Year Forecast (FY2028-2029)",
    fontsize=17, fontweight="bold", color=INK_PRIMARY, pad=20,
)
ax.set_xticks(list(history["fiscal_year"]) + FORECAST_YEARS)
ax.tick_params(labelsize=12, colors=INK_MUTED)

ax.grid(axis="y", color=GRIDLINE, linewidth=1, alpha=0.8)
ax.grid(axis="x", visible=False)
ax.set_axisbelow(True)
for spine_name in ("top", "right"):
    ax.spines[spine_name].set_visible(False)
for spine_name in ("left", "bottom"):
    ax.spines[spine_name].set_color(BASELINE)

ax.legend(fontsize=11, frameon=False, loc="upper left", labelcolor=INK_SECONDARY)

fig.tight_layout(pad=2)

OUT_PATH = "dot_forecast_2yr.png"
fig.savefig(OUT_PATH, dpi=150, facecolor=SURFACE)
print(f"\nSaved chart to {OUT_PATH}")


# ---------------------------------------------------------------------------
# 3. Clean version: no shaded band, matches dot_forecast_three_clean.png's
#    style. 95% interval shown as thin dashed error-bar caps on FY2028 and
#    FY2029 only (not FY2027, per instruction), plus point-value labels.
# ---------------------------------------------------------------------------
error_years = [2028, 2029]
error_point = np.array([m.predict(y).loc["modified", "point_forecast"] for y in error_years]) / 1e9
error_margin = np.array([m.predict(y).loc["modified", "margin"] for y in error_years]) / 1e9

fig2, ax2 = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
ax2.set_facecolor(SURFACE)

# Historical: solid + filled markers
ax2.plot(
    history["fiscal_year"], history["modified"] / 1e9,
    color=color, linewidth=2.5, marker="o", markersize=8,
    markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.5,
    label="Historical (FY2017-2026)", zorder=3,
)
# Forecast: dashed + open markers, anchored at the last historical point
ax2.plot(
    fx, np.array(fpoint) / 1e9,
    color=color, linewidth=2.5, linestyle="--", marker="o", markersize=8,
    markerfacecolor=SURFACE, markeredgecolor=color, markeredgewidth=1.8,
    label="Forecast (FY2027-2029)", zorder=3,
)

# 95% interval: thin dashed error-bar caps, FY2028 & FY2029 only -- no
# shaded band.
ax2.errorbar(
    error_years, error_point, yerr=error_margin,
    fmt="none", ecolor=color, elinewidth=1.3, capsize=7, capthick=1.3,
    linestyle="--", alpha=0.85, zorder=4, label="95% interval (FY2028-29)",
)

# Point-value annotations, FY2028 & FY2029 -- offset sideways (left for
# 2028, right for 2029) rather than straight up, so the label clears its
# own error-bar cap instead of colliding with it.
label_offsets = [(-14, 0, "right"), (14, 0, "left")]
for (yr, val), (dx, dy, ha) in zip(zip(error_years, error_point), label_offsets):
    ax2.annotate(
        f"${val:.2f}B",
        xy=(yr, val), xytext=(dx, dy), textcoords="offset points",
        ha=ha, va="center", fontsize=12, fontweight="bold", color=INK_PRIMARY,
    )

ax2.axvline(last_hist_year + 0.5, color=BASELINE, linewidth=1.5, linestyle=":", zorder=2)
ax2.text(
    last_hist_year + 0.55, 0.97, "Forecast begins",
    transform=ax2.get_xaxis_transform(),
    color=INK_SECONDARY, fontsize=12, va="top", ha="left",
)

ax2.set_xlabel("Fiscal Year", fontsize=14, color=INK_SECONDARY, labelpad=10)
ax2.set_ylabel("Modified Budget ($ Billions)", fontsize=14, color=INK_SECONDARY, labelpad=10)
ax2.set_title(
    "DOT Modified Budget: History & 2-Year Forecast (FY2028-2029)",
    fontsize=17, fontweight="bold", color=INK_PRIMARY, pad=20,
)
ax2.set_xticks(list(history["fiscal_year"]) + FORECAST_YEARS)
ax2.tick_params(labelsize=12, colors=INK_MUTED)

ax2.grid(axis="y", color=GRIDLINE, linewidth=1, alpha=0.8)
ax2.grid(axis="x", visible=False)
ax2.set_axisbelow(True)
for spine_name in ("top", "right"):
    ax2.spines[spine_name].set_visible(False)
for spine_name in ("left", "bottom"):
    ax2.spines[spine_name].set_color(BASELINE)

ax2.legend(fontsize=11, frameon=False, loc="upper left", labelcolor=INK_SECONDARY)

fig2.tight_layout(pad=2)

CLEAN_OUT_PATH = "dot_forecast_2yr_clean.png"
fig2.savefig(CLEAN_OUT_PATH, dpi=150, facecolor=SURFACE)
print(f"Saved clean chart to {CLEAN_OUT_PATH}")


# ---------------------------------------------------------------------------
# 4. Three-series version: Adopted, Modified, Actual -- same structure and
#    style as dot_forecast_three_clean.png (solid+filled history, dashed+
#    open forecast, no bands, no error bars, just clean lines), scoped to
#    the 2-year FY2028-2029 horizon instead of that chart's 5-year one.
# ---------------------------------------------------------------------------
SERIES_COLS = ["adopted", "modified", "actual"]

fig3, ax3 = plt.subplots(figsize=(11, 7), facecolor=SURFACE)
ax3.set_facecolor(SURFACE)

for col in SERIES_COLS:
    series_color = SERIES[col]

    # Historical: solid + filled markers
    ax3.plot(
        history["fiscal_year"], history[col] / 1e9,
        color=series_color, linewidth=2.5, marker="o", markersize=8,
        markerfacecolor=series_color, markeredgecolor=SURFACE, markeredgewidth=2,
        label=col.capitalize(), zorder=3,
    )

    # Forecast: dashed + open markers, anchored at the last historical
    # point so it connects with no visual gap
    col_forecast = [m.predict(y).loc[col, "point_forecast"] for y in FORECAST_YEARS]
    last_hist_val_col = history.loc[history["fiscal_year"] == last_hist_year, col].iloc[0]
    fx_col = [last_hist_year] + FORECAST_YEARS
    fy_col = [last_hist_val_col] + col_forecast
    ax3.plot(
        fx_col, np.array(fy_col) / 1e9,
        color=series_color, linewidth=2.5, linestyle="--", marker="o", markersize=8,
        markerfacecolor=SURFACE, markeredgecolor=series_color, markeredgewidth=2,
        zorder=3,
    )

ax3.axvline(last_hist_year + 0.5, color=BASELINE, linewidth=1.5, linestyle=":", zorder=2)
ax3.text(
    last_hist_year + 0.55, 0.97, "Forecast begins",
    transform=ax3.get_xaxis_transform(),
    color=INK_SECONDARY, fontsize=12, va="top", ha="left",
)

ax3.set_xlabel("Fiscal Year", fontsize=14, color=INK_SECONDARY, labelpad=10)
ax3.set_ylabel("$ Billions", fontsize=14, color=INK_SECONDARY, labelpad=10)
ax3.set_title(
    "DOT Budget: Adopted, Modified & Actual -- History & 2-Year Forecast",
    fontsize=17, fontweight="bold", color=INK_PRIMARY, pad=20,
)
ax3.set_xticks(list(history["fiscal_year"]) + FORECAST_YEARS)
ax3.tick_params(labelsize=12, colors=INK_MUTED)

ax3.grid(axis="y", color=GRIDLINE, linewidth=1, alpha=0.8)
ax3.grid(axis="x", visible=False)
ax3.set_axisbelow(True)
for spine_name in ("top", "right"):
    ax3.spines[spine_name].set_visible(False)
for spine_name in ("left", "bottom"):
    ax3.spines[spine_name].set_color(BASELINE)

ax3.legend(
    fontsize=12, frameon=False, loc="upper left", labelcolor=INK_SECONDARY,
    title="Series", title_fontsize=12,
)

fig3.tight_layout(pad=2)

THREE_SERIES_OUT_PATH = "dot_forecast_2yr_three_clean.png"
fig3.savefig(THREE_SERIES_OUT_PATH, dpi=150, facecolor=SURFACE)
print(f"Saved three-series chart to {THREE_SERIES_OUT_PATH}")
