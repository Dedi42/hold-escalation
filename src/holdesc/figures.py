"""Result figures (Figures 2-4). Figure 1 is drawn by holdesc.framework_figure.

Layout rules: two panels of equal size, panel labels (a)/(b) at the same position
above the top-left corner of each panel, legends outside the data area (shared legend
below the panels when it would cover lines), and fonts of at least 7 pt at print size.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False})
STYLE = {"Age (oldest first)": ("#4d4d4d", "o", "-"), "Static p90 rule": ("#9a9a9a", "s", ":"),
         "Kaplan–Meier (D×R)": ("#9467bd", "^", "-."), "Cox PH": ("#8c564b", "x", "--"),
         "Log-normal AFT": ("#bcbd22", "D", ":"), "XGBoost AFT": ("#17becf", "<", ":"),
         "LightGBM (log T)": ("#ff7f0e", "v", "-"), "Landmark GBM": ("#d62728", "d", "-"),
         "Ensemble (proposed)": ("#1f4e9a", "o", "-"), "Ensemble + due-date urgency": ("#2ca02c", "P", "--"),
         "Oracle": ("#000000", "*", "--")}
LABEL = {"LightGBM (log T)": "LightGBM (log $T$)"}


def _panels(figsize=(7.0, 2.7)):
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw=dict(wspace=0.32))
    for ax, lab in zip(axes, ["(a)", "(b)"]):
        ax.text(-0.13, 1.06, lab, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
    return fig, axes


def _save(fig, path):
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(path.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def fig_duration(h: pd.DataFrame, path: str):
    fig, (a, b) = _panels()
    d = np.sort(h.dur_h.values); n = len(d)
    a.loglog(d, 1 - np.arange(n) / n, color="#1f4e9a", lw=1.3, label="holds longer than $t$")
    a.loglog(d, np.cumsum(d[::-1])[::-1] / d.sum(), color="#d62728", lw=1.3, ls="--", label="hold hours in holds longer than $t$")
    a.set_xlabel("Hold duration $t$ (h)"); a.set_ylabel("Share"); a.set_xlim(0.01, 3000); a.set_ylim(1e-4, 1.5)
    a.legend(frameon=False, loc="lower left")
    deps = list(h.department.value_counts().index[:4])
    sel = [deps[0], deps[1], deps[3]] if len(deps) > 3 else deps
    groups = [("All holds", h, "k", "o")] + [(d_, h[h.department == d_], c, m) for d_, c, m in zip(sel, ["#1f4e9a", "#ff7f0e", "#2ca02c"], ["s", "^", "D"])]
    for lab, g, c, m in groups:
        T = np.minimum(g.dur_h.values, 720); xs, ys = [], []
        for e in [0, 0.25, 0.5, 1, 2, 4, 8, 16, 24, 48, 96]:
            s = T[T > e]
            if len(s) >= 30:
                xs.append(max(e, 0.1)); ys.append(np.median(s - e))
        b.loglog(xs, ys, marker=m, ms=3, lw=1.1, color=c, label=lab)
    b.loglog([0.1, 150], [0.1, 150], color="#9a9a9a", lw=0.8, ls=":", label="remaining = elapsed")
    b.set_xlim(0.08, 150); b.set_ylim(0.08, 150)
    b.set_xlabel("Elapsed hold time $e$ (h)"); b.set_ylabel("Median remaining time (h)")
    b.legend(frameon=False, loc="upper left")
    _save(fig, path)


def fig_landmark(M: pd.DataFrame, path: str, models):
    M = M[M.elapsed_h.astype(str) != "pooled"].copy(); M["elapsed_h"] = M.elapsed_h.astype(float)
    fig, (a, b) = _panels((7.0, 2.6))
    for m in models:
        c, mk, ls = STYLE.get(m, ("k", "o", "-")); s = M[M.model == m].sort_values("elapsed_h")
        for ax, col in ((a, "C"), (b, "AUC_8h")):
            ax.semilogx(s.elapsed_h, s[col], color=c, ls=ls, marker=mk, ms=3, lw=1.1, label=LABEL.get(m, m))
    for ax, lab in [(a, "C-index of remaining time"), (b, "AUC, remaining time > 8 h")]:
        ax.axhline(0.5, color="#9a9a9a", lw=0.6)
        ax.set_xlabel("Elapsed hold time $e$ (h)"); ax.set_ylabel(lab)
        ax.set_xticks([0.5, 1, 2, 4, 8, 16, 24, 48]); ax.set_xticklabels(["0.5", "1", "2", "4", "8", "16", "24", "48"]); ax.minorticks_off()
    a.set_ylim(0.40, 0.75); b.set_ylim(0.30, 0.95)
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.2))
    _save(fig, path)


def fig_frontier(P: pd.DataFrame, path: str):
    fig, (a, b) = _panels((7.0, 2.6))
    for m, (c, mk, ls) in STYLE.items():
        s = P[P.policy == m].sort_values("budget")
        if s.empty:
            continue
        a.plot(s.budget, s.saved_h_per_day, color=c, marker=mk, ms=3.5, ls=ls, lw=1.1, label=LABEL.get(m, m))
        if "lateness_removed_h" in s:
            b.plot(s.budget, s.lateness_removed_h / 1000, color=c, marker=mk, ms=3.5, ls=ls, lw=1.1)
    budgets = sorted(P.budget.unique())
    for ax in (a, b):
        ax.set_xscale("log", base=2); ax.set_xticks(budgets); ax.set_xticklabels([str(x) for x in budgets])
        ax.minorticks_off(); ax.set_xlabel("Escalation budget $B$ (holds per day)")
    a.set_ylabel("Hold hours recovered per day"); b.set_ylabel("Lateness removed (10$^3$ h)")
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.26))
    _save(fig, path)
