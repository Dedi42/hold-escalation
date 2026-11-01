"""Rebuild every table and figure of the paper from the result folders of run_hypotheses.py and run_benchmark.py.

    python scripts/make_paper.py --hyp paper_results/hypotheses \
        --bench A=paper_results/benchmark_A --bench B=paper_results/benchmark_B --out paper_output

Inputs are aggregated result files only (no hold-level data), so the paper's tables and figures can be checked
from the archived results in paper_results/ or from a fresh run on your own data.

Writes to --out:
    tables/table2_lines.{csv,md}        Table 2  Data of the two production lines
    tables/table4_prediction.{csv,md}   Table 4  Discrimination among open holds
    tables/table5_policies.{csv,md}     Table 5  Hold hours recovered per day (test period)
    tables/table6_robustness.{csv,md}   Table 6  Robustness, transfer, delivery, flexible budget
    tables/tableA1_context.{csv,md}     Table A1 Effect of decision-time context in the benchmark
    tables/key_numbers.md               every number quoted in the text of Section 5, with its source file
    figures/figure2_concentration_residual_life.{png,pdf}
    figures/figure3_policies_vs_oracle.{png,pdf}
    figures/figure4_benchmark.{png,pdf}
Tables 1 and 3 (literature positioning, hypothesis definitions) are text tables in paper/; Figure 1 is an editable
PowerPoint drawing stored as figures/figure1_framework.{pptx,png}.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

BUDGETS = [6, 12, 24, 48]
COL = {"A": "#1f4e9a", "B": "#c0661c", "pooled": "#222222"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False})
W = 6.3  # inches, text width of an A4 page with 2.5 cm margins


# ------------------------------------------------------------------------------------------------ helpers
def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r.values) + " |")
    return "\n".join(out) + "\n"


def save_table(df, path, title, note=""):
    df.to_csv(path + ".csv", index=False)
    with open(path + ".md", "w", encoding="utf-8") as f:
        f.write(f"**{title}**\n\n{md_table(df)}\n{note}\n")


def pct(x, d=1, sign=True):
    if sign and abs(100 * x) < 0.05:
        return "0.0"
    return ("{:+." + str(d) + "f}").format(100 * x) if sign else ("{:." + str(d) + "f}").format(100 * x)


def savefig(fig, path):
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def mins(x):
    return ("{:+.1f}".format(x)).replace("-", "−")


# ------------------------------------------------------------------------------------------------ tables
def table2(H, lines, T):
    rows = {}
    for L in lines:
        f = f"{H}/tables/{L}_line_summary.csv"
        if not os.path.exists(f):
            print(f"  (no {f}; Table 2 column for {L} skipped)"); continue
        s = pd.read_csv(f).iloc[0]
        h10 = T[(T.hypothesis == "H10") & (T.line == L)]
        late = int(h10.late_lots.iloc[0]) if len(h10) and h10.late_lots.notna().any() else None
        rows[f"Line {L}"] = {
            "Package families": int(s.package_families),
            "Observation period (months)": int(s.months),
            "Holds / lots": f"{int(s.holds):,} / {int(s.lots):,}",
            "Training / validation / test holds": f"{int(s.train):,} / {int(s.val):,} / {int(s.test):,}",
            "Test days (weeks)": f"{s.test_days:g} ({int(s.test_weeks)})",
            "Median hold duration (h)": f"{s.median_h:.2f}",
            "Holds longer than 8 h (%)": f"{100 * s.share_gt8h:.1f}",
            "Departments; share of largest (%)": f"{int(s.departments)}; {100 * s.largest_department_share:.1f}",
            "Lots with customer SOD; late lots in test period": ("all" if s.lots_with_sod >= s.lots else ("none" if s.lots_with_sod == 0 else f"{int(s.lots_with_sod):,}")) + "; " + (f"{late:,}" if late else "–"),
        }
    df = pd.DataFrame(rows).reset_index().rename(columns={"index": ""})
    return df


def table4(H, lines):
    order = ["Static p90", "Kaplan–Meier", "LightGBM onset", "Landmark GBM", "Ensemble"]
    label = {"Static p90": "Static p90 rule", "Kaplan–Meier": "Kaplan–Meier (department × reason)",
             "Ensemble": "Ensemble (Equation 4)"}
    out = pd.DataFrame({"Model": [label.get(m, m) for m in order]})
    for L in lines:
        P = pd.read_csv(f"{H}/tables/{L}_prediction_ci.csv").set_index("model").reindex(order)
        out[f"C-index {L}"] = [f"{r.C:.3f} [{r.C_lo:.3f}, {r.C_hi:.3f}]" for r in P.itertuples()]
        out[f"AUC(> 8 h) {L}"] = [f"{r.AUC:.3f} [{r.AUC_lo:.3f}, {r.AUC_hi:.3f}]" for r in P.itertuples()]
    return out


def table5(H, lines, T, F, min_weeks):
    order = ["Age (oldest first)", "Static p90 rule", "Kaplan–Meier (D×R)", "LightGBM onset", "Landmark GBM",
             "Ensemble", "Ensemble (gain)", "Oracle"]
    label = {"Age (oldest first)": "Oldest-first", "Kaplan–Meier (D×R)": "Kaplan–Meier (department × reason)",
             "Ensemble": "Ensemble (Equation 4)", "Ensemble (gain)": "Ensemble, gain-based"}
    out = pd.DataFrame({"Policy": [label.get(p, p) for p in order] + ["Ensemble vs oldest-first, test period (%)",
                        "Ensemble vs oldest-first, rolling origins (%)", "Pooled over both lines, rolling origins (%)"]})
    for li, L in enumerate(lines):
        P = pd.read_csv(f"{H}/tables/{L}_policies.csv").pivot(index="policy", columns="budget", values="h_per_day")
        for B in BUDGETS:
            if B not in P.columns: continue
            col = [f"{P.loc[p, B]:.1f}" if p in P.index else "" for p in order]
            f = F[(F.line == L) & (F.kind == "test period") & (F.budget == B)]
            h4 = T[(T.hypothesis == "H4") & (T.line == L) & (T.budget == B)]
            mark = ""
            if len(f):
                if f.weeks.iloc[0] < min_weeks: mark = "†"
                elif len(h4) and h4.p_holm.iloc[0] < 0.05: mark = "*"
                col.append(pct(f.rel.iloc[0]) + mark)
            else:
                col.append("")
            ro = F[(F.line == L) & (F.kind == "rolling origins") & (F.budget == B)]
            h4ro = T[(T.hypothesis == "H4-RO") & (T.line == L) & (T.budget == B)]
            col.append(pct(ro.rel.iloc[0]) + ("*" if len(h4ro) and h4ro.p_holm.iloc[0] < 0.05 else "") if len(ro) else "")
            po = F[(F.line == "pooled") & (F.kind == "rolling origins") & (F.budget == B)]
            h9 = T[(T.hypothesis == "H9") & (T.budget == B)]
            col.append(pct(po.rel.iloc[0]) + ("*" if len(h9) and h9.p_holm.iloc[0] < 0.05 else "") if (li == 0 and len(po)) else "")
            out[f"{L}, B = {B}"] = col
    return out


def decision_text(sub):
    ok = sub[sub.decision.str.startswith(("supported", "equivalent"), na=False)].budget.astype(int).tolist()
    if len(ok) == len(sub): return "Supported" if not sub.decision.str.startswith("equivalent").any() else "Equivalent at all"
    if not ok: return "Not supported"
    word = "Equivalent" if sub.decision.str.startswith("equivalent").any() else "Supported"
    return f"{word} at B = " + ", ".join(map(str, ok))


def table6(T, lines, benches):
    rows = []
    def add(label, line, vals, dec):
        rows.append({"Hypothesis": label, "Line": line, **{f"B = {B}": v for B, v in zip(BUDGETS, vals)}, "Decision": dec})
    for L in lines:
        s = T[(T.hypothesis == "H6") & (T.line == L)].set_index("budget")
        if len(s):
            add("H6 Heterogeneous α_g, δ_g: mean gain (share of draws > 0)", L,
                [f"{pct(s.estimate.get(B, np.nan))} ({s.share_draws_positive.get(B, np.nan):.2f})" if B in s.index else "" for B in BUDGETS],
                decision_text(T[(T.hypothesis == "H6") & (T.line == L)]))
    for L in lines:
        s = T[(T.hypothesis == "H7") & (T.line == L)].set_index("budget")
        if len(s):
            add("H7 Gain-based vs mean-based index (TOST ±3%)", L,
                [pct(s.estimate[B]) + ("*" if str(s.decision[B]).startswith("equivalent") else "") if B in s.index else "" for B in BUDGETS],
                decision_text(T[(T.hypothesis == "H7") & (T.line == L)]))
    for hyp, lab in [("H8a", "H8a Transferred model vs oldest-first"), ("H8b", "H8b Transferred vs in-line model (non-inferiority 3%)")]:
        for ln in T[T.hypothesis == hyp].line.unique():
            s = T[(T.hypothesis == hyp) & (T.line == ln)].set_index("budget")
            add(lab, ln.replace("→", " → "), [pct(s.estimate[B]) if B in s.index else "" for B in BUDGETS],
                decision_text(T[(T.hypothesis == hyp) & (T.line == ln)]))
    for L in lines:
        s = T[(T.hypothesis == "H10") & (T.line == L)].set_index("budget")
        if len(s):
            add("H10 Lateness removed, ensemble + urgency vs oldest-first", L,
                [pct(s.estimate[B]) + ("*" if s.p_holm[B] < 0.05 else "") if B in s.index else "" for B in BUDGETS],
                decision_text(T[(T.hypothesis == "H10") & (T.line == L)]))
    for L, d in benches.items():
        f = f"{d}/tables/bench_flex_budget.csv"
        if not os.path.exists(f): continue
        X = pd.read_csv(f)
        vals, better, worse = [], False, False
        for B in BUDGETS:
            x = X[X.budget == B]
            vals.append(f"{pct(x.rel_flex_vs_fixed.min())} to {pct(x.rel_flex_vs_fixed.max())}" if len(x) else "")
            better |= bool((x.lo > 0).any()); worse |= bool((x.hi < 0).any())
        add("Flexible daily budget vs fixed per review (range over learners)", L, vals,
            "Not supported" if worse else ("Not significant" if not better else "Mixed"))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ figures
def figure2(H, lines, path):
    E = [0, 0.5, 1, 2, 4, 8, 16, 24, 48, 96]
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.25), gridspec_kw=dict(wspace=0.3))
    for L in lines:
        r = pd.read_csv(f"{H}/tables/{L}_residual_life.csv")
        q = r[r.elapsed_h > 0]
        axs[0].plot(q.elapsed_h, q.median_remaining_h, "-o", ms=3, lw=1.1, color=COL.get(L), label=f"Line {L}")
        r = r.set_index("elapsed_h").reindex(E)
        axs[1].plot(range(len(E)), 100 * r.p_remaining_gt_8h.values, "-o", ms=3, lw=1.1, color=COL.get(L), label=f"Line {L}")
    x = np.array([0.4, 130]); axs[0].plot(x, x, ":", color="#888", lw=0.9, label="remaining = elapsed")
    axs[0].set_xscale("log"); axs[0].set_yscale("log")
    for ax_ in (axs[0].xaxis, axs[0].yaxis): ax_.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    axs[0].set_xlabel("Elapsed time of the open hold, $e$ (h)"); axs[0].set_ylabel("Median remaining time (h)")
    axs[1].set_xticks(range(len(E))); axs[1].set_xticklabels(["%g" % e for e in E])
    axs[1].set_xlabel("Elapsed time of the open hold, $e$ (h)"); axs[1].set_ylabel("P(remaining > 8 h) (%)"); axs[1].set_ylim(0, 100)
    for a, t in zip(axs, ["(a)", "(b)"]):
        a.legend(frameon=False, loc="upper left", handlelength=1.6, borderaxespad=0.2)
        a.text(0.0, 1.03, t, transform=a.transAxes, fontsize=8.5, fontweight="bold", va="bottom")
    savefig(fig, path)


def figure3(F, lines, path):
    fig, ax = plt.subplots(figsize=(W, 2.2))
    series = [(L, f"Line {L} ({int(F[(F.line == L) & (F.kind == 'rolling origins')].weeks.max())} weeks)", m, o)
              for L, m, o in zip(lines, ["o", "s"], [-0.22, 0.0])] + [("pooled", "Pooled over lines", "D", 0.22)]
    for L, lab, mk, off in series:
        g = F[(F.line == L) & (F.kind == "rolling origins")].set_index("budget").reindex(BUDGETS)
        if g.rel.isna().all(): continue
        xs = np.arange(len(BUDGETS)) + off
        ax.errorbar(xs, 100 * g.rel, yerr=[100 * (g.rel - g.rel_lo), 100 * (g.rel_hi - g.rel)], fmt=mk, ms=4.2,
                    color=COL.get(L, "#555"), mfc="white" if L != "pooled" else COL["pooled"], mew=1, elinewidth=1, capsize=2, label=lab)
        if L == "pooled":
            for xx, v in zip(xs, g.rel):
                ax.text(xx + 0.07, 100 * v, f"+{100 * v:.1f}%", fontsize=7, va="center", ha="left")
    ax.axhline(0, color="#888", lw=0.7, ls="--")
    ax.set_xticks(range(len(BUDGETS))); ax.set_xticklabels([f"$B$ = {b}" for b in BUDGETS]); ax.set_xlim(-0.5, 3.75)
    ax.set_xlabel("Daily escalation budget"); ax.set_ylabel("Gain over oldest-first (%)")
    ax.legend(frameon=False, ncol=3, loc="upper right", handletextpad=0.3, columnspacing=1.2, borderaxespad=0.1)
    savefig(fig, path)


SEL = [("LightGBM landmark (ext. features)", "LightGBM landmark + context"),
       ("LightGBM gain regression (ext.)", "LightGBM gain regression + context"),
       ("LightGBM LambdaRank (ext.)", "LightGBM LambdaRank + context"),
       ("MLP survival (ext.)", "MLP survival + context"),
       ("FT-Transformer survival (ext.)", "FT-Transformer survival + context"),
       ("LotSeq survival (ext. + history)", "LotSeq survival + history"),
       ("LotSeq decision-focused ranker", "LotSeq decision-focused ranker"),
       ("Ensemble new", "Hybrid ensemble (deep + LightGBM)")]


def figure4(benches, path):
    lines = list(benches)
    V = np.full((len(SEL), 4 * len(lines)), np.nan); S = np.zeros_like(V, dtype=bool)
    for c, L in enumerate(lines):
        P = pd.read_csv(f"{benches[L]}/tables/bench_policy.csv")
        P = P[~P.policy.str.contains(r"\[mean score\]")]
        for i, (key, _) in enumerate(SEL):
            g = P[P.policy.str.startswith(key)]
            for j, b in enumerate(BUDGETS):
                r = g[g.budget == b]
                if len(r) and pd.notna(r.rel_vs_paper_ens.iloc[0]):
                    r = r.iloc[0]
                    V[i, 4 * c + j] = 100 * r.rel_vs_paper_ens; S[i, 4 * c + j] = (r.lo_ens > 0) or (r.hi_ens < 0)
    keep = ~np.isnan(V).all(1)
    V, S, names = V[keep], S[keep], [n for (k, n), kk in zip(SEL, keep) if kk]
    fig, ax = plt.subplots(figsize=(W, 0.32 * len(names) + 0.2))
    im = ax.imshow(np.clip(V, -10, 10), cmap="RdBu", norm=TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10), aspect="auto")
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            if np.isnan(V[i, j]): continue
            t = ("0.0" if abs(V[i, j]) < 0.05 else mins(V[i, j])) + ("*" if S[i, j] else "")
            ax.text(j, i, t, ha="center", va="center", fontsize=7, color="white" if abs(V[i, j]) > 6.5 else "#111")
    ax.set_xticks(range(V.shape[1])); ax.set_xticklabels([str(b) for b in BUDGETS] * len(lines))
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.tick_params(length=0); ax.xaxis.tick_top()
    for s in ax.spines.values(): s.set_visible(False)
    for c in range(1, len(lines)): ax.axvline(4 * c - 0.5, color="white", lw=3)
    ax.set_xticks(np.arange(-0.5, V.shape[1], 1), minor=True); ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6); ax.tick_params(which="minor", length=0)
    for c, L in enumerate(lines):
        ax.text(4 * c + 1.5, -1.25, f"Line {L}: daily budget $B$", ha="center", va="bottom", fontsize=8, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015, ticks=[-10, -5, 0, 5, 10])
    cb.ax.set_yticklabels(["≤ −10", "−5", "0", "+5", "≥ +10"]); cb.ax.tick_params(labelsize=7, length=2, width=0.5)
    cb.outline.set_linewidth(0.4); cb.set_label("Gain over ensemble index (%)", fontsize=7.5)
    savefig(fig, path)


# ------------------------------------------------------------------------------------------------ key numbers
def tableA1(benches):
    """Table A1: without -> with context features (AUC of P(remaining > 8 h); hold hours recovered per day)."""
    pairs = [("LightGBM landmark", "LightGBM landmark (paper features)", "LightGBM landmark (ext. features)"),
             ("MLP survival", "MLP survival (paper features)", "MLP survival (ext.)")]
    rows = []
    for L, d in benches.items():
        P = pd.read_csv(f"{d}/tables/bench_policy.csv"); P = P[~P.policy.str.contains(r"\[mean score\]")]
        Q = pd.read_csv(f"{d}/tables/bench_prediction_pooled.csv")
        for name, a, b in pairs:
            qa, qb = Q[Q.model == a], Q[Q.model == b]
            if not len(qa) or not len(qb): continue
            r = {"Line, model": f"{L}, {name}", "AUC(> 8 h)": f"{qa.AUC8_mean.iloc[0]:.3f} → {qb.AUC8_mean.iloc[0]:.3f}"}
            for bud in BUDGETS:
                va = P[(P.policy == a) & (P.budget == bud)].saved_h_per_day
                vb = P[(P.policy == b) & (P.budget == bud)].saved_h_per_day
                if len(va) and len(vb):
                    fmt = "{:.1f}" if max(va.iloc[0], vb.iloc[0]) < 1000 and L == "A" else "{:.0f}"
                    r[f"B = {bud}"] = (fmt + " → " + fmt).format(va.iloc[0], vb.iloc[0])
            rows.append(r)
    return pd.DataFrame(rows)


def key_numbers(H, T, F, lines, benches, path):
    L = []
    def row(text, val, src): L.append(f"| {text} | {val} | `{src}` |")
    for l in lines:
        for hyp, txt, f in [("H1", "share of hold hours in longest 10% of holds", "{:.3f}"), ("H2a", "Weibull shape", "{:.3f}"),
                            ("H2b", "median remaining after 8 h / median at onset", "{:.2f}")]:
            r = T[(T.hypothesis == hyp) & (T.line == l)]
            if len(r):
                r = r.iloc[0]; row(f"Line {l}: {txt} ({hyp})", (f + " [" + f + ", " + f + "]").format(r.estimate, r.ci_lo, r.ci_hi), "hypotheses_summary.csv")
    for l in lines:
        for hyp in ["H3a", "H3b", "H3c", "H3d"]:
            r = T[(T.hypothesis == hyp) & (T.line == l)]
            if len(r): r = r.iloc[0]; row(f"Line {l}: {r.test} ({hyp})", f"ΔAUC {r.estimate:+.3f}, p_Holm " + ("< 0.001" if r.p_holm < 0.001 else f"= {r.p_holm:.3f}"), "hypotheses_summary.csv")
    for kind in ["rolling origins"]:
        for l in lines + ["pooled"]:
            g = F[(F.line == l) & (F.kind == kind)].set_index("budget")
            for B in BUDGETS:
                if B in g.index:
                    r = g.loc[B]; row(f"{'Pooled' if l == 'pooled' else 'Line ' + l}: ensemble vs oldest-first, {kind}, B = {B}",
                                     f"{pct(r.rel)}% [{pct(r.rel_lo)}, {pct(r.rel_hi)}]", "tables/forest_gain_vs_age.csv")
    r = T[T.hypothesis == "H9"]
    for x in r.itertuples():
        row(f"H9 heterogeneity, B = {int(x.budget)}", f"I² = {x.I2:.2f}, Q p = {x.Q_p:.2f}", "hypotheses_summary.csv")
    for l in lines:
        s = T[(T.hypothesis == "H5") & (T.line == l)]
        if len(s): row(f"Line {l}: static p90 vs oldest-first (H5), B = 6..48", ", ".join(pct(v) for v in s.sort_values("budget").estimate), "hypotheses_summary.csv")
    for l, d in benches.items():
        P = pd.read_csv(f"{d}/tables/bench_prediction_pooled.csv")
        for m in ["MLP survival (paper features)", "MLP survival (ext.)", "LightGBM landmark (paper features)",
                  "LightGBM landmark (ext. features)", "LotSeq decision-focused ranker"]:
            x = P[P.model == m]
            if len(x): row(f"Line {l}: benchmark AUC, {m}", f"{x.AUC8_mean.iloc[0]:.3f}", f"benchmark_{l}/tables/bench_prediction_pooled.csv")
        R = pd.read_csv(f"{d}/tables/bench_runtime.csv")
        row(f"Line {l}: seconds per fit", "; ".join(f"{a}: {b:.1f}" for a, b in zip(R.model, R.sec_per_fit)), f"benchmark_{l}/tables/bench_runtime.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Numbers quoted in the text\n\n| Quantity | Value | Source |\n|---|---|---|\n" + "\n".join(L) + "\n")


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hyp", required=True, help="output folder of run_hypotheses.py")
    ap.add_argument("--bench", action="append", default=[], help="LINE=output folder of run_benchmark.py (repeat)")
    ap.add_argument("--out", default="paper_output")
    ap.add_argument("--min-weeks", dest="min_weeks", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(f"{a.out}/tables", exist_ok=True); os.makedirs(f"{a.out}/figures", exist_ok=True)
    T = pd.read_csv(f"{a.hyp}/hypotheses_summary.csv")
    F = pd.read_csv(f"{a.hyp}/tables/forest_gain_vs_age.csv")
    lines = [l for l in T.line.unique() if "→" not in str(l) and l != "pooled"]
    benches = dict(s.split("=", 1) for s in a.bench)
    t = f"{a.out}/tables"
    save_table(table2(a.hyp, lines, T), f"{t}/table2_lines", "Table 2. Data of the production lines.",
               "Note: codes of departments, reasons, and steps are anonymised; the test period is the last 20% of holds by onset time.")
    save_table(table4(a.hyp, lines), f"{t}/table4_prediction", "Table 4. Discrimination among open holds on the test period (landmarks 0.5–48 h).",
               "Note: 95% lot cluster bootstrap intervals in brackets.")
    save_table(table5(a.hyp, lines, T, F, a.min_weeks), f"{t}/table5_policies", "Table 5. Hold hours recovered per day on the test period (α = 0.5, δ = 2 h).",
               "Note: * p_Holm < 0.05 (one-sided, weekly block bootstrap). † fewer than "
               f"{a.min_weeks} test weeks; inference relies on rolling origins. Pooled row: inverse-variance meta-analysis, listed under the first line.")
    save_table(table6(T, lines, benches), f"{t}/table6_robustness", "Table 6. Robustness, transfer, and delivery: gain over oldest-first (%) unless stated.",
               "Note: * significant after Holm correction (equivalence test for H7, one-sided test otherwise).")
    if benches:
        save_table(tableA1(benches), f"{t}/tableA1_context",
                   "Table A1. Effect of decision-time context in the benchmark: without → with context features (test period, mean over five seeds).",
                   "Note: columns B = 6 to 48 give hold hours recovered per day; the benchmark divides by elapsed review days.")
    f = f"{a.out}/figures"
    if set(benches) >= {"A", "B"} and set(lines) >= {"A", "B"}:
        import subprocess
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "paper_figures.py"), a.hyp,
                        benches["A"], benches["B"], f], check=True)
    else:  # other numbers of lines: generic versions
        figure2(a.hyp, lines, f"{f}/figure2_residual_life")
        figure3(F, lines, f"{f}/figure3_gain_over_oldest_first")
        if benches:
            figure4(benches, f"{f}/figure4_benchmark")
    key_numbers(a.hyp, T, F, lines, benches, f"{t}/key_numbers.md")
    print(f"tables and figures written to {a.out}/")


if __name__ == "__main__":
    main()
