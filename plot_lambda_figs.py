#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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

# ---- Figure 1: λ on x-axis, two colorful curves (cohesion gain + BPB change) ----
fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
ax2 = ax1.twinx()

x = df["coh_lambda"].values
y_coh = df["ppmi_gain_pct"].values
y_bpb = df["bpb_delta_pct"].values

ax1.plot(x, y_coh, marker="o", linewidth=2, label=r"Cohesion gain $\Delta_{\mathrm{PPMI}}$ (%)")
ax2.plot(x, y_bpb, marker="s", linewidth=2, linestyle="--", label=r"BPB change $\Delta_{\mathrm{BPB}}$ (%)")

ax1.axvline(0.15, linewidth=1, linestyle=":", alpha=0.7)  # highlight default λ
ax1.grid(True, alpha=0.25)

ax1.set_xlabel(r"Coherence strength $\lambda$")
ax1.set_ylabel(r"Cohesion gain $\Delta_{\mathrm{PPMI}}$ (%) $\uparrow$")
ax2.set_ylabel(r"BPB change $\Delta_{\mathrm{BPB}}$ (%) $\downarrow$")

# combined legend
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="best", frameon=True)

ax1.set_title(r"Ablation over $\lambda$ at fixed $\gamma=0.50$")
fig.tight_layout()
fig.savefig(OUTDIR / "lambda_ablation.pdf", bbox_inches="tight")

# ---- Figure 2: colored scatter in Pareto space with λ colorbar (no zig-zag line) ----
fig2, ax = plt.subplots(figsize=(7.2, 4.6))
sc = ax.scatter(
    df["ppmi_gain_pct"].values,
    df["bpb_delta_pct"].values,
    c=df["coh_lambda"].values,
    cmap="viridis",
    s=90,
    edgecolors="black",
    linewidths=0.6,
)

# annotate each point with λ
for _, r in df.iterrows():
    ax.annotate(
        rf"${r['coh_lambda']:.2f}$",
        (r["ppmi_gain_pct"], r["bpb_delta_pct"]),
        xytext=(3, 3),
        textcoords="offset points",
        fontsize=12,
    )

ax.axhline(0.0, linewidth=1)
ax.grid(True, alpha=0.25)
ax.set_xlabel(r"Cohesion gain $\Delta_{\mathrm{PPMI}}$ (%) $\uparrow$")
ax.set_ylabel(r"BPB change $\Delta_{\mathrm{BPB}}$ (%) $\downarrow$")
ax.set_title(r"Pareto view with $\lambda$ color-coded")

cbar = fig2.colorbar(sc, ax=ax)
cbar.set_label(r"$\lambda$")

fig2.tight_layout()
fig2.savefig(OUTDIR / "pareto_lambda_color.pdf", bbox_inches="tight")

print("[ok] wrote:")
print(" - figs/lambda_ablation.pdf")
print(" - figs/pareto_lambda_color.pdf")
