"""Paper figures (static, print, light mode) from results/*.json. One claim per figure:
fig1 the hidden state and the inputs that reach it; fig2 wiring explains it; fig3 the screen; fig4 prediction and
correction; figS1 (supplement) the LB3 subset phase diagram. Palette: blue #2a78d6, orange #eb6834, aqua #1baf7a,
sequential blue ramp, ink #0b0b0b / #52514e, hairline grid #e1e0d9.  python figs.py [fig1 ...]  (cwd = paper/)
The previous per-experiment layout is kept in figs_v1.py."""
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

R = os.path.join("..", "results")
OUT = "figs"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, PALE = "#0b0b0b", "#52514e", "#e1e0d9", "#f0efec"
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
                     "legend.frameon": False, "pdf.fonttype": 42, "axes.titlesize": 8.5})


def J(name):
    return json.load(open(os.path.join(R, name)))


def title(ax, s):
    ax.set_title(s, loc="left", fontsize=8.5)


def save(fig, name):
    kw = dict(bbox_inches="tight", pad_inches=0.03)
    fig.savefig(os.path.join(OUT, name + ".pdf"), **kw); fig.savefig(os.path.join(OUT, name + ".png"), dpi=200, **kw)


def ignition_table():
    """Rows (label, modality, {rate_hz: (ignited, runs)}) from E1, E8, E9 and E13 (original model)."""
    agg = defaultdict(lambda: [0, 0])

    def add(key, hz, ign):
        a = agg[(key, hz)]; a[0] += int(ign); a[1] += 1

    for r in J("e1_ignition_map.json")["rows"]:
        part, name, hz = r["tag"][0], r["tag"][1], r["tag"][2]
        if part == "a":
            add({"SHIU20": "Shiu et al. protocol (20 sugar GRNs)", "SUGAR_ALL": "all sugar/water GRNs (129)"}[name], hz,
                r["ignited"])
        elif name == "LB3":
            add("sugar GRN type LB3 (122)", hz, r["ignited"])
        else:
            add("each other gustatory type (11 types)", hz, r["ignited"])
    lab9 = {"ALL_BITTER": "all bitter GRNs (65)", "ALL_LOWSALT": "all low-salt GRNs (19)",
            "ALL_GUSTATORY": "all gustatory neurons (334)", "JO_CE": "Johnston's organ C/E (221)",
            "JO_F": "Johnston's organ F (106)"}
    for r in J("e9_other_stimuli.json")["rows"]:
        add(lab9[r["set"]], r["hz"], r["ignited"])
    for r in J("e8_visual.json")["rows"]:
        add({"LC4": "visual LC4 (104)", "LPLC2": "visual LPLC2 (210)"}[r["type"]], r["hz"], r["ignited"])
    for r in J("e13_olfactory.json")["models"]["original"]:
        s = r["stim"]
        hz = int(s.split()[-2])
        if s.startswith("single"):
            add("single ORN (5 cells)", hz, r["ignited"])
        elif s.startswith("type"):
            add("one glomerulus (5 glomeruli)", hz, r["ignited"])
        else:
            add("all ORNs (2,279)", hz, r["ignited"])
    order = [("Taste", ["Shiu et al. protocol (20 sugar GRNs)", "sugar GRN type LB3 (122)", "all sugar/water GRNs (129)",
                        "each other gustatory type (11 types)", "all bitter GRNs (65)", "all low-salt GRNs (19)",
                        "all gustatory neurons (334)"]),
             ("Smell", ["single ORN (5 cells)", "one glomerulus (5 glomeruli)", "all ORNs (2,279)"]),
             ("Mechanosensation, vision", ["Johnston's organ C/E (221)", "Johnston's organ F (106)", "visual LC4 (104)",
                                "visual LPLC2 (210)"])]
    rows = []
    for mod, labs in order:
        for lab in labs:
            rows.append((lab, mod, {hz: tuple(v) for (k, hz), v in agg.items() if k == lab}))
    return rows


def fig1():
    fig = plt.figure(figsize=(6.5, 3.6))
    # (a) persistence over 10 s
    ax = fig.add_axes([0.09, 0.62, 0.27, 0.31])
    e2 = J("e2_persistence.json")
    t = np.arange(1, 11) - 0.5                      # 1-s windows, plotted at their centres
    for a in e2["active"]:
        ax.plot(t, a[2:], color=ORANGE, lw=1.1, alpha=0.85)
    g3 = [r for r in J("g3_recovery.json")["persistence"]["rows"] if r["s1"] == 0.8]
    t3 = np.arange(8) * 0.25 + 0.125                # 250-ms windows
    for r in g3:
        stays = r["active"][9] > 1000
        ax.plot(t3, r["active"][2:10], color=ORANGE if stays else BLUE, lw=1.1, ls="--" if stays else "-")
    ax.text(10, 7300, "LB3 at 120 Hz (5 seeds)", ha="right", va="top", fontsize=6.6, color=ORANGE)
    ax.text(2.3, 1300, "LB3 at 80 Hz: 3 seeds stay\n(dashed), 2 fall silent", ha="left", va="bottom", fontsize=6.6,
            color=BLUE)
    ax.set_ylim(-400, 9800); ax.set_xlim(0, 10.2)
    ax.set_xlabel("time after the input ends (s)")
    ax.set_ylabel("neurons > 1 Hz")
    title(ax, "a  activity without input")
    # (b) feeding output after the pulse
    ax = fig.add_axes([0.09, 0.13, 0.27, 0.31])
    rows = [r for r in J("g0_g2.json")["rows"] if r["tag"][0] == "G2"]
    s1s = sorted({r["tag"][1] for r in rows})
    rng = np.random.default_rng(0)
    for s in s1s:
        y = [r["mn9_hz"]["p2"] for r in rows if r["tag"][1] == s]
        ax.scatter(s * 100 + rng.uniform(-3.5, 3.5, len(y)), y, s=8, color=INK2, alpha=0.5, linewidths=0)
    ax.plot(np.array(s1s) * 100, [np.mean([r["mn9_hz"]["p2"] for r in rows if r["tag"][1] == s]) for s in s1s],
            color=INK, lw=1.4)
    ax.set_xlabel("rate of the first LB3 pulse (Hz)")
    ax.set_ylabel("MN9 rate, later\nsugar probe (Hz)")
    title(ax, "b  feeding output after the pulse")
    # (c) which inputs reach the state
    ax = fig.add_axes([0.665, 0.13, 0.33, 0.80])
    rows = ignition_table()
    rates = [20, 30, 50, 100, 150, 200]
    ys = np.arange(len(rows))[::-1].astype(float)
    gap, cur, mods = 0.0, None, []
    for i, (lab, mod, d) in enumerate(rows):
        if mod != cur:
            if cur is not None:
                gap += 0.6
            cur = mod
            mods.append((mod, i))
        ys[i] -= gap
    for (lab, mod, d), y in zip(rows, ys):
        for j, hz in enumerate(rates):
            if hz not in d:
                continue
            k, n = d[hz]
            f = k / n
            ax.add_patch(plt.Rectangle((j - 0.46, y - 0.42), 0.92, 0.84, facecolor=SEQ(0.08 + 0.92 * f),
                                       edgecolor="#ffffff", linewidth=0.8))
            ax.text(j, y, "%d/%d" % (k, n), ha="center", va="center", fontsize=6.0, color="#ffffff" if f > 0.55 else INK)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=6.6)
    ax.set_xticks(range(len(rates))); ax.set_xticklabels(["%d" % h for h in rates])
    ax.set_xlabel("stimulation rate (Hz)")
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(-0.6, len(rates) - 0.4)
    ax.set_ylim(ys.min() - 0.7, ys.max() + 0.7)
    for mod, i in mods:
        ax.annotate(mod, xy=(0, ys[i] + 0.55), xycoords=("axes fraction", "data"), xytext=(-124, 0),
                    textcoords="offset points", fontsize=6.8, color=INK2, ha="left", va="bottom", style="italic",
                    annotation_clip=False)
    ax.annotate("c  inputs that reach the state", xy=(0, 1), xycoords="axes fraction", xytext=(-124, 6),
                textcoords="offset points", fontsize=8.5, ha="left", va="bottom", annotation_clip=False)
    save(fig, "fig1")


def fig2():
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.65), gridspec_kw=dict(width_ratios=[1.25, 1.2, 0.95]))
    # (a) membership predicted from wiring vs simulated
    ax = axs[0]
    names = {"Kenyon_Cell": "Kenyon cells", "olfactory": "olfactory receptor", "optic_lobe_intrinsic": "optic lobe",
             "ALPN": "AL projection", "LHLN": "lateral-horn local", "ALLN": "AL local", "DAN": "dopaminergic",
             "CX": "central complex", "descending": "descending", "MBON": "MB output"}
    rows = [r for r in J("e4b_composition.json")["rows"] if r["cls"] in names]
    y = np.arange(len(rows))[::-1]
    h = 0.38
    ax.barh(y + h / 2, [r["simulated"] for r in rows], height=h, color=INK2, label="simulation")
    ax.barh(y - h / 2, [r["predicted"] for r in rows], height=h, color=ORANGE, label="mean field (wiring)")
    ax.set_yticks(y); ax.set_yticklabels([names[r["cls"]] for r in rows], fontsize=6.6)
    ax.set_xlabel("neurons in the persistent state")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=6.3, handlelength=1.0)
    title(ax, "a  membership (Jaccard 0.70)")
    # (b) class ablations
    ax = axs[1]
    lab3 = {"minus Kenyon_Cell": "Kenyon cells", "minus ALPN": "AL projection", "minus olfactory": "olfactory receptor",
            "minus LHLN": "lateral-horn local", "minus DAN": "dopaminergic", "minus MBON": "MB output",
            "minus APL": "APL", "no input to sensory": "all input to sensory cells", "minus ALLN": "AL local (429)"}
    lab3b = {"ALLN ACH": "AL local, ACh", "ALLN GABA": "AL local, GABA", "ALLN GLUT": "AL local, Glu",
             "ALLN other/unknown nt": "AL local, no call (176)"}
    e3 = {r["cond"]: r["ignited"] for r in J("e3_ablation.json")["rows"]}
    e3b = {r["cond"]: r["ignited"] for r in J("e3b_alln.json")["rows"]}
    items = [(lab3[k], e3[k]) for k in lab3] + [(v, e3b[k]) for k, v in lab3b.items()]
    yy = np.arange(len(items))[::-1].astype(float)
    yy[len(lab3):] -= 0.6
    vals = [v for _, v in items]
    ax.barh(yy, vals, height=0.62, color=[ORANGE if v == 0 else "#cfcdc6" for v in vals])
    for yi, (lab, v) in zip(yy, items):
        if v == 0:
            ax.text(0.12, yi, "0/5", va="center", fontsize=6.4, color=ORANGE)
    ax.set_yticks(yy); ax.set_yticklabels([lab for lab, _ in items], fontsize=6.4)
    ax.set_xlim(0, 5.3); ax.set_xticks(range(6))
    ax.set_xlabel("runs still igniting (of 5)")
    ax.grid(axis="y", visible=False)
    title(ax, "b  outputs removed")
    # (c) leading eigenvector mass
    ax = axs[2]
    ev = J("e11b_eigvec.json")
    mb = ev["mass_by_class"]
    cls = [("ALLN", "AL local"), ("ALPN", "AL projection"), ("Kenyon_Cell", "Kenyon cells"), ("central", "central"),
           ("LHLN", "lateral-horn local")]
    other = 1 - sum(mb.get(c, 0) for c, _ in cls)
    vals = [mb.get(c, 0) for c, _ in cls] + [other]
    labs = [l for _, l in cls] + ["other"]
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, height=0.62, color="#cfcdc6")
    ax.barh(y[0], ev["core_mass"], height=0.62, color=ORANGE)
    ax.text(ev["core_mass"] / 2, y[0], "core", fontsize=6.2, color="#ffffff", ha="center", va="center")
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=6.6)
    ax.set_xlabel("share of the leading mode")
    ax.set_xlim(0, 0.48)
    ax.grid(axis="y", visible=False)
    title(ax, "c  leading mode of the wiring")
    fig.subplots_adjust(left=0.125, right=0.985, bottom=0.17, top=0.9, wspace=0.95)
    save(fig, "fig2")


def fig3():
    rk = J("e14_rank.json")
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.4), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    drop = -np.array(rk["pred_drop"])
    is_core = np.array(J("e14_core_flags.json")["is_core"])
    n = 150
    x = np.arange(1, n + 1)
    ax[0].scatter(x[is_core[:n]], drop[:n][is_core[:n]], s=7, color=ORANGE, linewidths=0, label="core cell")
    ax[0].scatter(x[~is_core[:n]], drop[:n][~is_core[:n]], s=7, color=BLUE, linewidths=0, label="other candidate")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("rank among 376 candidates")
    ax[0].set_ylabel(r"predicted drop of $\lambda$")
    ax[0].legend(loc="upper right", fontsize=6.3, handletextpad=0.2)
    title(ax[0], "a  ranking from wiring")
    ks = [23, 45, 90]
    base = J("e14_baselines.json")["out-weight within P"]
    rnd = J("e14_random_eig.json")
    ax[1].plot(ks, [rk["top%d_exact_lam" % k] for k in ks], "-o", ms=3.5, color=ORANGE, label="top k by screen")
    ax[1].plot(ks, [base[str(k)][1] for k in ks], "--o", ms=3, color=AQUA, label="top k by output weight")
    for k in ks:
        ax[1].scatter([k] * 3, rnd[str(k)], s=9, color=BLUE, linewidths=0, zorder=3)
    ax[1].scatter([], [], s=9, color=BLUE, label="k random candidates")
    ax[1].axhline(rk["lam"], color=INK2, lw=0.8, ls=":")
    ax[1].text(92, rk["lam"] + 0.05, "unchanged wiring", ha="right", va="bottom", fontsize=6.3, color=INK2)
    ax[1].set_ylim(0, 3.4); ax[1].set_xticks(ks)
    ax[1].set_xlabel("cells re-signed (k)")
    ax[1].set_ylabel(r"$\lambda$ after re-signing")
    ax[1].legend(loc="center right", bbox_to_anchor=(1.02, 0.43), fontsize=6.0, handlelength=1.2)
    title(ax[1], "b  exact eigenvalue")
    sim = {c["cond"]: c for c in J("e14_simulate.json")["conds"]}
    e12 = {c["cond"]: c for c in J("e12_partial_resign.json")["conds"]}
    ax[2].plot(ks, [sim["top-%d" % k]["ignited"] / 6 for k in ks], "-o", ms=3.5, color=ORANGE, label="top k by screen")
    for k in ks:
        v = [sim["random-%d #%d" % (k, i)]["ignited"] / 6 for i in (1, 2, 3)]
        ax[2].scatter(np.array([k] * 3) + np.array([-2, 0, 2]), v, s=9, color=BLUE, linewidths=0, zorder=3)
    ax[2].scatter([], [], s=9, color=BLUE, label="k random candidates")
    for k in (23, 45):
        v = [e12["random k=%d #%d" % (k, i)]["ignited"] / 6 for i in (1, 2, 3)]
        ax[2].scatter(np.array([k] * 3) + np.array([-2, 0, 2]), v, s=12, facecolors="none", edgecolors=AQUA,
                      linewidths=0.8)
    ax[2].scatter([], [], s=12, facecolors="none", edgecolors=AQUA, label="k random core cells")
    ax[2].set_ylim(-0.08, 1.08); ax[2].set_xticks(ks)
    ax[2].set_xlabel("cells re-signed (k)")
    ax[2].set_ylabel("runs ending in the state")
    ax[2].legend(loc="center right", bbox_to_anchor=(1.02, 0.62), fontsize=6.0, handlelength=1.2)
    title(ax[2], "c  simulation")
    fig.tight_layout()
    save(fig, "fig3")


def fig4():
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.5), gridspec_kw=dict(width_ratios=[0.8, 1.1, 1]))
    # (a) phase diagram on which the score was designed
    ax = axs[0]
    rows = J("e1c_phase.json")["rows"]
    ks = sorted({r["k"] for r in rows}); hz = sorted({r["hz"] for r in rows})
    M = np.array([[np.mean([r["ignited"] for r in rows if r["k"] == k and r["hz"] == h]) for h in hz] for k in ks])
    ax.imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect="auto", origin="lower")
    for i, k in enumerate(ks):
        for j, h in enumerate(hz):
            n = sum(r["ignited"] for r in rows if r["k"] == k and r["hz"] == h)
            ax.text(j, i, "%d/3" % n, ha="center", va="center", fontsize=6.3, color="#ffffff" if M[i, j] > 0.5 else INK)
    ax.set_xticks(range(len(hz))); ax.set_xticklabels(["%d" % h for h in hz])
    ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks)
    ax.set_xlabel("rate (Hz)")
    ax.set_ylabel("LB3 cells (random subset)")
    ax.grid(False)
    title(ax, "a  design runs")
    # (b) prospective test
    ax = axs[1]
    P = J("e6_predictions.json"); res = J("e6_result.json")
    th = P["threshold_core"]
    ign = np.array(res["ignited"]); sc = np.array([p["core_score"] for p in P["predictions"]])
    k = np.array([p["k"] for p in P["predictions"]])
    for val, colr, lab in ((True, ORANGE, "ignited"), (False, BLUE, "did not")):
        m = ign == val
        ax.scatter(sc[m], k[m] + np.where(val, 1.6, -1.6), s=13, color=colr, label=lab, linewidths=0)
    wrong = (sc > th) != ign
    ax.scatter(sc[wrong], k[wrong] + np.where(ign[wrong], 1.6, -1.6), s=48, facecolors="none", edgecolors=INK, lw=0.7,
               label="wrong call")
    ax.axvline(th, color=INK2, ls="--", lw=0.9)
    ax.set_xscale("log"); ax.minorticks_off()
    ax.set_xticks([1e5, 3e5, 1e6, 3e6]); ax.set_xticklabels(["1e5", "3e5", "1e6", "3e6"])
    ax.set_ylim(24, 104)
    ax.set_xlabel("drive onto the core x rate")
    ax.set_ylabel("LB3 cells stimulated")
    ax.legend(loc="upper left", fontsize=6.0, handletextpad=0.2, borderaxespad=0.1)
    title(ax, "b  frozen prediction: 37/42")
    # (c) odour responses before and after the correction
    ax = axs[2]
    e13 = J("e13_olfactory.json")
    types = e13["pick_types"]
    conds = [("type %s" % t, 100) for t in types]
    def find(model, prefix, hz):
        return [r for r in e13["models"][model] if r["stim"].startswith(prefix) and r["stim"].endswith(" %d Hz" % hz)]
    xs = np.arange(len(types) + 1)
    labels = [t.replace("ORN_", "") for t in types] + ["all\nORNs"]
    for i, t in enumerate(types):
        for model, c, dx in (("original", ORANGE, -0.13), ("corrected", BLUE, 0.13)):
            v = [r["active_stim"] for r in find(model, "type %s" % t, 100)]
            ax.scatter([i + dx] * len(v), v, s=12, color=c, linewidths=0, zorder=3)
    for model, c, dx in (("original", ORANGE, -0.13), ("corrected", BLUE, 0.13)):
        v = [r["active_stim"] for r in e13["models"][model] if r["stim"].startswith("all")]
        ax.scatter([len(types) + dx] * len(v), v, s=12, color=c, linewidths=0, zorder=3)
    ax.scatter([], [], s=12, color=ORANGE, label="original")
    ax.scatter([], [], s=12, color=BLUE, label="corrected")
    ax.set_yscale("log"); ax.set_ylim(20, 30000)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=6.3)
    ax.set_xlabel("glomerulus driven at 100 Hz")
    ax.set_ylabel("neurons active during odour")
    ax.legend(loc="upper left", fontsize=6.2, handletextpad=0.2, ncol=2, columnspacing=0.8)
    title(ax, "c  odour responses")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig4")


def fig0():
    """Overview: what the screen takes, what it computes, what it returns and how it is checked."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.2, 2.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 28); ax.axis("off")

    def box(x, y, w, h, head, body, fc="#ffffff", ec=INK2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0", fc=fc, ec=ec, lw=0.8))
        ax.text(x + 0.9, y + h - 1.0, head, fontsize=7.2, color=INK, va="top", ha="left", weight="bold")
        ax.text(x + 0.9, y + h - 4.4, body, fontsize=6.4, color=INK2, va="top", ha="left", linespacing=1.3)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, shrinkA=0, shrinkB=0))

    top, h = 15.5, 11.5
    xs = [0.5, 20.5, 40.5, 60.5, 80.5]
    w = 17.5
    box(xs[0], top, w, h, "Input", "signed wiring $W$;\n19,042 cells signed\nby convention", fc=PALE)
    box(xs[1], top, w, h, "1  Mean field", "clamp, then release\nthe stimulus; cells\nthat persist form $P$")
    box(xs[2], top, w, h, "2  Candidates", "excitatory cells\nof $P$ without a\ncall (376)")
    box(xs[3], top, w, h, "3  Rank", r"by $\Delta\lambda_j$, first-order" "\n" r"change of the top" "\n" r"eigenvalue of $\tau W_{PP}$")
    box(xs[4], top, w + 1.5, h, "Output", "annotations to\ncheck, in order\n(top 66: core)", fc="#fbe3d9", ec=ORANGE)
    for a in range(4):
        arrow(xs[a] + w + 0.5, top + h / 2, xs[a + 1] - 0.5, top + h / 2)
    bh = 10.5
    box(xs[1], 1, w, bh, "Membership", "$P$ vs. simulation:\nJaccard 0.70\n(random sets 0.06)")
    box(xs[2], 1, w, bh, "Triggers", "stimulus drive onto\nthe core predicts\nignition (37/42)")
    box(xs[3], 1, 2 * w + 4.5, bh, "Check by simulation",
        "re-sign top 45: 0/6 runs ignite, MN9 unchanged\nre-sign 45 random candidates: 18/18 ignite\n21 s on one CPU; same top 66 in 10 variants")
    arrow(xs[1] + w / 2, top - 0.4, xs[1] + w / 2, 1 + bh + 0.4)
    arrow(xs[2] + w / 2, top - 0.4, xs[2] + w / 2, 1 + bh + 0.4)
    arrow(xs[4] + w / 2, top - 0.4, xs[4] + w / 2, 1 + bh + 0.4)
    save(fig, "fig0")


def figS1():
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    rows = J("e1c_phase.json")["rows"]
    ks = sorted({r["k"] for r in rows}); hz = sorted({r["hz"] for r in rows})
    M = np.array([[np.mean([r["ignited"] for r in rows if r["k"] == k and r["hz"] == h]) for h in hz] for k in ks])
    ax.imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect="auto", origin="lower")
    for i, k in enumerate(ks):
        for j, h in enumerate(hz):
            n = sum(r["ignited"] for r in rows if r["k"] == k and r["hz"] == h)
            ax.text(j, i, "%d/3" % n, ha="center", va="center", fontsize=7, color="#ffffff" if M[i, j] > 0.5 else INK)
    ax.set_xticks(range(len(hz))); ax.set_xticklabels(["%d Hz" % h for h in hz])
    ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks)
    ax.set_ylabel("LB3 cells (random subset)")
    ax.grid(False)
    fig.tight_layout()
    save(fig, "figS1")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "figS1"]):
        globals()[f]()
    print("ok")
