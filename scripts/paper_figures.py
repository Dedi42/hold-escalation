"""Figures 2-4 of the paper (called by make_paper.py).
    python scripts/paper_figures.py <run_hypotheses output> <run_benchmark output, line A> <run_benchmark output, line B> <out dir>
"""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle

H, BA, BB, OUT = sys.argv[1:5]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#404040", "xtick.color": "#404040", "ytick.color": "#404040"})
ACC, GREY, DARK, LIGHT = "#2F5D8A", "#8C8C8C", "#262626", "#D9D9D9"
COL = {"A": ACC, "B": GREY}; MK = {"A": "o", "B": "s"}
W = 6.3
T = pd.read_csv(f"{H}/hypotheses_summary.csv")
# Optional: map the line names of your own config to A/B, e.g. HOLDESC_LINE_ALIAS="L1=A,L2=B"
ALIAS = dict(kv.split("=", 1) for kv in os.environ.get("HOLDESC_LINE_ALIAS", "").split(",") if "=" in kv)
INV = {v: k for k, v in ALIAS.items()}
_rep = {**ALIAS, **{f"{a}→{b}": f"{ALIAS.get(a, a)}→{ALIAS.get(b, b)}" for a in ALIAS for b in ALIAS if a != b}}
T["line"] = T.line.astype(str).replace(_rep)


def tbl(name, L):
    for p in (f"{H}/tables/{L}_{name}.csv", f"{H}/tables/{INV.get(L, L)}_{name}.csv"):
        try:
            return pd.read_csv(p)
        except FileNotFoundError:
            pass
    raise FileNotFoundError(name)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def tag(ax, t):
    ax.text(-0.02, 1.04, t, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom", ha="right")


# ------------------------------------------------------------------ Figure 2
fig = plt.figure(figsize=(W, 2.35))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.7], wspace=0.32)
ax = fig.add_subplot(gs[0])
for k, L in enumerate(["A", "B"]):
    sh = 100 * float(T[(T.hypothesis == "H1") & (T.line == L)].estimate.iloc[0])
    ax.barh(k, sh, color=COL[L], height=0.55)
    ax.barh(k, 100 - sh, left=sh, color="white", edgecolor=LIGHT, height=0.55, lw=0.6)
    ax.text(sh / 2, k, f"{sh:.0f}%", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
    ax.text(sh + (100 - sh) / 2, k, f"{100 - sh:.0f}%", ha="center", va="center", color=DARK, fontsize=8)
ax.set_yticks([0, 1]); ax.set_yticklabels(["Line A", "Line B"]); ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100]); ax.set_xlabel("Share of hold hours (%)")
ax.set_ylim(1.55, -1.05)
ax.text(37, -0.62, "longest 10% of holds", ha="center", va="center", fontsize=7.5, color=DARK)
ax.text(90, -0.62, "other 90%", ha="center", va="center", fontsize=7.5, color=DARK)
tag(ax, "(a)")
ax = fig.add_subplot(gs[1])
for L in ["A", "B"]:
    r = tbl("residual_life", L); r = r[r.elapsed_h > 0]
    ax.plot(r.elapsed_h, r.median_remaining_h, "-", marker=MK[L], ms=3.2, lw=1.1, color=COL[L], mfc="white" if L == "B" else COL[L], label=f"Line {L}")
x = np.array([0.4, 130]); ax.plot(x, x, ":", color=DARK, lw=0.8, label="$m(e) = e$")
ax.set_xscale("log"); ax.set_yscale("log")
for a_ in (ax.xaxis, ax.yaxis): a_.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
ax.set_xlabel("Elapsed time of the open hold, $e$ (h)"); ax.set_ylabel("Median remaining time, $m(e)$ (h)")
# annotate three points with plain numbers (range over lines)
ra, rb = tbl("residual_life", "A").set_index("elapsed_h"), tbl("residual_life", "B").set_index("elapsed_h")
for e, dy in [(1, 3.2), (8, 3.0), (96, 2.6)]:
    lo, hi = sorted([ra.median_remaining_h[e], rb.median_remaining_h[e]])
    ytxt = hi * dy
    ax.annotate(f"+{lo:.0f}–{hi:.0f} h" if hi >= 10 else f"+{lo:.1f}–{hi:.1f} h", xy=(e, hi if dy > 1 else lo), xytext=(e, ytxt),
                ha="center", fontsize=7.5, color=DARK, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5))
ax.set_ylim(0.5, 900)
ax.legend(frameon=False, loc="upper left", handlelength=1.6, borderaxespad=0.1)
tag(ax, "(b)")
save(fig, "figure2_concentration_residual_life")

# ------------------------------------------------------------------ Figure 3: policy ladder (% of oracle)
B = sorted(int(b) for b in tbl("policies", "A").budget.unique())
rows = []
for L, tgt_tr in [("A", "B→A"), ("B", "A→B")]:
    P = tbl("policies", L).pivot(index="policy", columns="budget", values="h_per_day")
    for b in B:
        o = P.loc["Oracle", b]
        tr = T[(T.hypothesis == "H8a") & (T.line == tgt_tr) & (T.budget == b)]
        if b not in P.columns: continue
        rows.append(dict(line=L, B=b, p90=P.loc["Static p90 rule", b] / o, age=P.loc["Age (oldest first)", b] / o,
                         idx=P.loc["Ensemble", b] / o,
                         transfer=(P.loc["Age (oldest first)", b] * (1 + tr.estimate.iloc[0]) / o) if len(tr) else np.nan))
R = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(W, 2.6))
ys = []
for k, r in R.iterrows():
    y = k + (0.6 if r.line == "B" else 0); ys.append(y)
    vals = [v for v in [r.p90, r.age, r.idx, r.transfer] if pd.notna(v)]
    ax.plot([min(vals) * 100, 100], [y, y], color=LIGHT, lw=0.8, zorder=1)
    ax.plot([r.age * 100, r.idx * 100], [y, y], color=ACC, lw=2.2, zorder=2, solid_capstyle="butt")
    ax.plot(r.p90 * 100, y, "^", ms=4.5, mfc="white", mec=DARK, mew=0.8, zorder=3)
    if pd.notna(r.transfer): ax.plot(r.transfer * 100, y, "x", ms=4.5, color=GREY, mew=1, zorder=3)
    ax.plot(r.age * 100, y, "o", ms=4.5, mfc="white", mec=DARK, mew=0.9, zorder=4)
    ax.plot(r.idx * 100, y, "o", ms=4.8, color=ACC, zorder=5)
ax.axvline(100, color=DARK, lw=0.8); ax.text(100, -0.9, "oracle", ha="center", fontsize=7.5, color=DARK)
ax.set_yticks(ys); ax.set_yticklabels([f"$B$ = {b}" for b in R.B]); ax.invert_yaxis()
ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
nA = int((R.line == "A").sum()); nB = int((R.line == "B").sum())
for L, y in [("Line A", (nA - 1) / 2), ("Line B", nA + 0.6 + (nB - 1) / 2)]:
    ax.text(-0.13, y, L, transform=ax.get_yaxis_transform(), rotation=90, va="center", ha="center", fontsize=8, fontweight="bold")
ax.set_xlim(0, 104); ax.set_xlabel("Hold hours recovered (% of oracle)")
h = [plt.Line2D([], [], marker="^", ls="", mfc="white", mec=DARK, label="static p90 rule"),
     plt.Line2D([], [], marker="x", ls="", color=GREY, label="model from the other line"),
     plt.Line2D([], [], marker="o", ls="", mfc="white", mec=DARK, label="oldest-first"),
     plt.Line2D([], [], marker="o", ls="-", color=ACC, lw=2.2, label="index (gain over oldest-first)")]
ax.legend(handles=h, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17), handletextpad=0.3, columnspacing=1.2)
save(fig, "figure3_policies_vs_oracle")

# ------------------------------------------------------------------ Figure 4: benchmark heat map (muted)
SEL = [("LightGBM landmark (ext. features)", "LightGBM landmark"), ("LightGBM gain regression (ext.)", "LightGBM gain"),
       ("LightGBM LambdaRank (ext.)", "LightGBM LambdaRank"), ("MLP survival (ext.)", "MLP survival"),
       ("FT-Transformer survival (ext.)", "FT-Transformer survival"), ("LotSeq survival (ext. + history)", "LotSeq survival"),
       ("LotSeq decision-focused ranker", "LotSeq ranker"), ("Ensemble new", "Hybrid (deep + LightGBM)")]
nb = len(B); V = np.full((len(SEL), 2 * nb), np.nan); S = np.zeros_like(V, dtype=bool)
for c0, d in [(0, BA), (nb, BB)]:
    P = pd.read_csv(f"{d}/tables/bench_policy.csv"); P = P[~P.policy.str.contains(r"\[mean score\]")]
    for i, (key, _) in enumerate(SEL):
        g = P[P.policy.str.startswith(key)]
        for j, b in enumerate(B):
            if not len(g[g.budget == b]) or pd.isna(g[g.budget == b].rel_vs_paper_ens.iloc[0]): continue
            r = g[g.budget == b].iloc[0]
            V[i, c0 + j] = 100 * r.rel_vs_paper_ens; S[i, c0 + j] = (r.lo_ens > 0) or (r.hi_ens < 0)
cmap = LinearSegmentedColormap.from_list("muted", ["#A0574A", "#E8D3CE", "#F7F7F7", "#CBD8E6", ACC])
fig, ax = plt.subplots(figsize=(W, 2.65))
im = ax.imshow(np.clip(V, -10, 10), cmap=cmap, norm=TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10), aspect="auto")
for i in range(V.shape[0]):
    for j in range(V.shape[1]):
        v = V[i, j]
        if np.isnan(v): continue
        t = ("0.0" if abs(v) < 0.05 else f"{v:+.1f}".replace("-", "−")) + ("*" if S[i, j] else "")
        ax.text(j, i, t, ha="center", va="center", fontsize=7, color="white" if abs(v) > 7 else DARK)
ax.set_xticks(range(2 * nb)); ax.set_xticklabels([str(b) for b in B] * 2); ax.xaxis.tick_top()
ax.set_yticks(range(len(SEL))); ax.set_yticklabels([n for _, n in SEL]); ax.tick_params(length=0)
for s_ in ax.spines.values(): s_.set_visible(False)
ax.set_xticks(np.arange(-0.5, 2 * nb, 1), minor=True); ax.set_yticks(np.arange(-0.5, len(SEL), 1), minor=True)
ax.grid(which="minor", color="white", lw=0.8); ax.tick_params(which="minor", length=0)
ax.axvline(nb - 0.5, color="white", lw=4)
for y in [2.5, 6.5]: ax.axhline(y, color="white", lw=2.5)
ax.text((nb - 1) / 2, -1.3, "Line A,  $B$", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.text(nb + (nb - 1) / 2, -1.3, "Line B,  $B$", ha="center", va="bottom", fontsize=8, fontweight="bold")
for y, t in [(1.0, "GBM"), (4.5, "deep"), (7.0, "hybrid")]:
    ax.text(-0.36, y, t, transform=ax.get_yaxis_transform(), rotation=90, va="center", ha="center", fontsize=7, color=GREY, style="italic")
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015, ticks=[-10, 0, 10])
cb.ax.set_yticklabels(["≤ −10", "0", "≥ +10"]); cb.ax.tick_params(labelsize=7, length=2, width=0.5); cb.outline.set_linewidth(0.4)
cb.set_label("gain over index (%)", fontsize=7.5)
save(fig, "figure4_benchmark")
print("ok")
