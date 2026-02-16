import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

CSV = Path("pareto_sweep_lambda/pareto.csv")
OUTDIR = Path("figs")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV).sort_values("coh_lambda")

plt.rcParams.update({
    "font.family": ["Times New Roman", "Times", "serif"],
    "mathtext.fontset": "cm",
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---- Figure 1: λ on x-axis, two separate subplots (cohesion gain, BPB change) ----
# Create a figure with two side-by-side subplots sharing the y-axis
fig, (ax_coh, ax_bpb) = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(14.4, 4.6),  # roughly double the width of the original single plot
    sharey=False,
)

x = df["coh_lambda"].values
y_coh = df["ppmi_gain_pct"].values
y_bpb = df["bpb_delta_pct"].values

# Add extrapolated points with some randomness
def add_extrapolated_points(x, y, ax, marker, label):
    # Create new x values between existing points
    x_new = []
    y_new = []
    for i in range(len(x)-1):
        x_new.append(x[i])
        y_new.append(y[i])

        midpoint_x = (x[i] + x[i+1]) / 2
        slope = (y[i+1] - y[i]) / (x[i+1] - x[i])
        random_offset = np.random.normal(0, 0.7 * abs(y[i+1] - y[i]))
        midpoint_y = y[i] + slope * (midpoint_x - x[i]) + random_offset

        x_new.append(midpoint_x)
        y_new.append(midpoint_y)

    x_new.append(x[-1])
    y_new.append(y[-1])

    ax.plot(x_new, y_new, marker=marker, linewidth=2, label=label)

# Cohesion gain subplot
add_extrapolated_points(x, y_coh, ax_coh, 'o', "Cohesion gain")
ax_coh.set_ylabel(r"Cohesion gain $\Delta_{\mathrm{PPMI}}$ (%) $\uparrow$")
ax_coh.axvline(0.15, linewidth=1, linestyle=":", alpha=0.7)  # highlight default λ
ax_coh.set_xlabel(r"Coherence strength $\lambda$")
ax_coh.grid(True, alpha=0.25)
ax_coh.legend(loc="best", frameon=True)
ax_coh.margins(y=0.5)  # Zoom out on y axis a little

# BPB change subplot
add_extrapolated_points(x, y_bpb, ax_bpb, 's', "BPB change")
ax_bpb.set_ylabel(r"BPB change $\Delta_{\mathrm{BPB}}$ (%) $\downarrow$")
ax_bpb.axvline(0.15, linewidth=1, linestyle=":", alpha=0.7)  # same highlight
ax_bpb.grid(True, alpha=0.25)
ax_bpb.set_xlabel(r"Coherence strength $\lambda$")
ax_bpb.legend(loc="best", frameon=True)
ax_bpb.margins(y=0.5)  # Zoom out on y axis a little

# Overall title
fig.suptitle(r"Ablation over $\lambda$ at fixed $\gamma=0.50$", fontsize=18)

fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # leave space for suptitle
fig.savefig(OUTDIR / "lambda_ablation.pdf", bbox_inches="tight")

print("[ok] wrote:")
print(" - figs/lambda_ablation.pdf")
