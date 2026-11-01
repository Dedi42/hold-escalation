"""Figure 1: framework of the decision aid (drawn with matplotlib, fully reproducible).

A two-row flowchart of rounded boxes and straight arrows, so that it can also be
redrawn by hand in any drawing tool:

  row 1 (left to right):  MES data -> Features -> remaining-time models (onset, landmark)
  row 2 (right to left):  Escalation index <- Due dates ;  index -> Select top b -> Replay evaluation

    python -m holdesc.framework_figure --out figures/figure1_framework
"""
from __future__ import annotations
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})
STY = {  # border, fill
    "data": ("#2B5D9C", "#EAF1FB"), "feat": ("#5F6B7A", "#F2F4F7"), "model": ("#C0661A", "#FDEEDD"),
    "group": ("#C0661A", "#FFF8F1"), "index": ("#2E7D4F", "#E6F4EA"), "select": ("#1F3A5F", "#E8EDF5"),
    "eval": ("#6A4C93", "#EFE9F7")}
INK = "#1F2933"


def box(ax, x, y, w, h, kind, title, lines=(), dashed=False, title_size=8.4, line_size=7.6, z=2):
    ec, fc = STY[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", lw=1.2, ec=ec, fc=fc,
                                ls=(0, (4, 2.5)) if dashed else "-", zorder=z))
    n = len(lines)
    step = h / (n + 1.7)
    ax.text(x + w / 2, y + h - 0.95 * step, title, ha="center", va="center", fontsize=title_size,
            fontweight="bold", color=INK, zorder=z + 1)
    for k, t in enumerate(lines):
        ax.text(x + w / 2, y + h - (1.95 + k) * step, t, ha="center", va="center", fontsize=line_size,
                color=INK, zorder=z + 1)


def arrow(ax, p, q, label=None, dy=1.3, color="#222222"):
    ax.annotate("", xy=q, xytext=p, zorder=5,
                arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.25", lw=1.2, color=color,
                                shrinkA=0, shrinkB=0))
    if label:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, label, ha="center", va="bottom", fontsize=7.6,
                style="italic", color=INK, zorder=6)


def draw(path: str):
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.set_xlim(0, 100); ax.set_ylim(0, 43); ax.axis("off")
    # ---- row 1 ----
    box(ax, 1, 25, 17, 13, "data", "MES data", ["hold events", "lot attributes"])
    arrow(ax, (18, 31.5), (24, 31.5))
    box(ax, 24, 25, 18, 13, "feat", "Features", [r"at onset: $x_i$", r"at review $t_k$: $x_i(t_k)$"])
    arrow(ax, (42, 31.5), (48, 31.5))
    box(ax, 48, 21, 51, 21, "group", "Remaining-time models", dashed=True, title_size=8.2, z=1)
    ax.texts[-1].set_position((73.5, 39.8))
    box(ax, 51, 27.5, 21, 9.5, "model", "Onset model", ["LightGBM on log $T$"])
    box(ax, 75, 27.5, 21, 9.5, "model", "Landmark model", ["LightGBM at elapsed $e$"])
    ax.text(73.5, 23.6, r"$\hat m_i(e)=E[\min(T_i,\tau)-e \mid T_i>e,\;x_i]$", ha="center", va="center",
            fontsize=8.2, color=INK, zorder=3)
    # ---- row 2 (flows right to left) ----
    arrow(ax, (73.5, 21), (73.5, 15.5))
    box(ax, 60, 2.5, 27, 13, "index", "Escalation index", [r"$I_i=r_i^{\mathrm{LM}}+r_i^{\mathrm{GBM}}$",
                                                          r"weight $w_j=1+\kappa\,u_j$"])
    box(ax, 90.5, 4.5, 8.5, 9, "data", "Due", ["dates"], title_size=8.0)
    arrow(ax, (90.5, 9), (87, 9), label=r"$u_j$", dy=0.6)
    arrow(ax, (60, 9), (53, 9))
    box(ax, 30, 2.5, 23, 13, "select", "Escalate", [r"$A_k=\mathrm{top}_b\{w_j I_i\}$", "every review $k$"])
    arrow(ax, (30, 9), (23, 9))
    box(ax, 1, 2.5, 22, 13, "eval", "Replay evaluation", ["105 held-out days", "hours saved, lateness"])
    fig.savefig(path + ".png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(path + ".pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="figures/figure1_framework")
    draw(ap.parse_args().out)
