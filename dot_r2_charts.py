"""
DOT trend fit-quality charts -- uses the existing dot_forecast_model.py
as-is (no changes to it). Two figures:

  FIGURE A: bar chart of each series' R^2 (dot_r2_bars.png)
  FIGURE B: 3-panel fit overlay -- training points + fitted line + R^2
            annotation per series (dot_r2_fit_overlay.png)

Both use the training window models are actually fit on (FY2017-2024, the
model's TRAIN_END_YEAR) -- Figure B's points are exactly the points that
produced each panel's R^2, not the full FY2017-2026 history, since
including the held-out years would show points the line was never fit to
alongside an R^2 that doesn't describe them.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dot_forecast_model as m

# --- Project's established chart style ------------------------------------
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
SERIES_COLS = ["adopted", "modified", "actual"]

for _style in ["seaborn-v0_8-whitegrid", "seaborn-whitegrid"]:
    if _style in plt.style.available:
        plt.style.use(_style)
        break
else:
    plt.style.use("default")

GOOD_FIT_THRESHOLD = 0.80

# --- Fit the models (FY2017-2024) and grab the training data --------------
models = m.train_models()
df = m.load_data()
train_df = df[df["fiscal_year"] <= m.TRAIN_END_YEAR].reset_index(drop=True)

r2_values = {col: models[col]["r2"] for col in SERIES_COLS}
print("\nR^2 by series (FY2017-2024 fit):")
for col in SERIES_COLS:
    print(f"  {col:>9}: {r2_values[col]:.4f}")


# ---------------------------------------------------------------------------
# FIGURE A: R^2 bar chart
# ---------------------------------------------------------------------------
fig_a, ax_a = plt.subplots(figsize=(8, 6), facecolor=SURFACE)
ax_a.set_facecolor(SURFACE)

labels = [c.capitalize() for c in SERIES_COLS]
values = [r2_values[c] for c in SERIES_COLS]
colors = [SERIES[c] for c in SERIES_COLS]

bars = ax_a.bar(
    labels, values, color=colors, width=0.55, zorder=3,
    edgecolor=SURFACE, linewidth=1.5,
)

for bar, val in zip(bars, values):
    ax_a.annotate(
        f"{val:.3f}",
        xy=(bar.get_x() + bar.get_width() / 2, val),
        xytext=(0, 8), textcoords="offset points",
        ha="center", fontsize=13, fontweight="bold", color=INK_PRIMARY,
    )

ax_a.axhline(
    GOOD_FIT_THRESHOLD, color=INK_MUTED, linewidth=1.3, linestyle="--", zorder=2,
)
ax_a.text(
    0.015, GOOD_FIT_THRESHOLD + 0.02,
    f"Good fit (R² = {GOOD_FIT_THRESHOLD:.2f})",
    transform=ax_a.get_yaxis_transform(),  # x in axes fraction, y in data coords
    fontsize=10.5, color=INK_SECONDARY, ha="left", va="bottom", style="italic",
)

ax_a.set_ylim(0, 1)
ax_a.set_ylabel("R²", fontsize=14, color=INK_SECONDARY, labelpad=10)
ax_a.set_title(
    "DOT Trend Fit Quality (R² by Series)",
    fontsize=17, fontweight="bold", color=INK_PRIMARY, pad=20,
)

ax_a.grid(axis="y", color=GRIDLINE, linewidth=1, alpha=0.8)
ax_a.grid(axis="x", visible=False)
ax_a.set_axisbelow(True)
for spine_name in ("top", "right"):
    ax_a.spines[spine_name].set_visible(False)
for spine_name in ("left", "bottom"):
    ax_a.spines[spine_name].set_color(BASELINE)
ax_a.tick_params(labelsize=12, colors=INK_MUTED)

fig_a.tight_layout(pad=2)
fig_a.savefig("dot_r2_bars.png", dpi=150, facecolor=SURFACE)
print("\nSaved Figure A to dot_r2_bars.png")


# ---------------------------------------------------------------------------
# FIGURE B: 3-panel fit overlay -- points + fitted line + R^2 annotation
# ---------------------------------------------------------------------------
fig_b, axes = plt.subplots(1, 3, figsize=(15, 5.5), facecolor=SURFACE)

x_train = train_df["fiscal_year"].values.astype(float)
x_line = np.linspace(x_train.min() - 0.3, x_train.max() + 0.3, 100)

for ax, col in zip(axes, SERIES_COLS):
    ax.set_facecolor(SURFACE)
    color = SERIES[col]
    model = models[col]

    y_train = train_df[col].values.astype(float) / 1e9
    y_line = (model["slope"] * x_line + model["intercept"]) / 1e9

    ax.plot(
        x_line, y_line, color=color, linewidth=2.5, zorder=2,
    )
    ax.scatter(
        x_train, y_train, color=color, s=90, zorder=3,
        edgecolor=SURFACE, linewidth=1.5,
    )

    ax.annotate(
        f"R² = {model['r2']:.3f}",
        xy=(0.05, 0.92), xycoords="axes fraction",
        fontsize=13, fontweight="bold", color=INK_PRIMARY,
        va="top", ha="left",
    )

    ax.set_title(col.capitalize(), fontsize=15, fontweight="bold", color=INK_PRIMARY, pad=10)
    ax.set_xlabel("Fiscal Year", fontsize=11, color=INK_SECONDARY)
    ax.set_xticks(list(x_train.astype(int)))
    ax.tick_params(axis="x", labelrotation=45, labelsize=9, colors=INK_MUTED)
    ax.tick_params(axis="y", labelsize=10, colors=INK_MUTED)
    ax.grid(True, color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    for spine_name in ("left", "bottom"):
        ax.spines[spine_name].set_color(BASELINE)

axes[0].set_ylabel("$ Billions", fontsize=12, color=INK_SECONDARY)

fig_b.suptitle(
    "DOT Trend Fit: Training Points vs. Fitted Line (FY2017-2024)",
    fontsize=17, fontweight="bold", color=INK_PRIMARY, y=1.03,
)

fig_b.tight_layout()
fig_b.savefig("dot_r2_fit_overlay.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
print("Saved Figure B to dot_r2_fit_overlay.png")
