from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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
    figsize=(14.4, 4.6),
    sharey=False,
)

x = df["coh_lambda"].values
y_coh = df["ppmi_gain_pct"].values
y_bpb = df["bpb_delta_pct"].values

# Cohesion gain subplot
ax_coh.plot(x, y_coh, marker='o', linewidth=2, label="Cohesion gain")
ax_coh.set_ylabel(r"Cohesion gain $\Delta_{\mathrm{PPMI}}$ (%) $\uparrow$")
ax_coh.axvline(0.15, linewidth=1, linestyle=":", alpha=0.7)  # highlight default λ
ax_coh.set_xlabel(r"Coherence strength $\lambda$")
ax_coh.grid(True, alpha=0.25)
ax_coh.legend(loc="best", frameon=True)
ax_coh.margins(y=0.5)  # Zoom out on y axis a little

# BPB change subplot
ax_bpb.plot(x, y_bpb, marker='s', linewidth=2, label="BPB change")
ax_bpb.set_ylabel(r"BPB change $\Delta_{\mathrm{BPB}}$ (%) $\downarrow$")
ax_bpb.axvline(0.15, linewidth=1, linestyle=":", alpha=0.7)  # same highlight
ax_bpb.grid(True, alpha=0.25)
ax_bpb.set_xlabel(r"Coherence strength $\lambda$")
ax_bpb.legend(loc="best", frameon=True)
ax_bpb.margins(y=0.5)  # Zoom out on y axis a little

# Overall title
fig.suptitle(r"Ablation over $\lambda$ at fixed $\beta=0.50$", fontsize=18)

fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # leave space for suptitle
fig.savefig(OUTDIR / "lambda_ablation.pdf", bbox_inches="tight")

print("[ok] wrote:")
print(" - figs/lambda_ablation.pdf")
