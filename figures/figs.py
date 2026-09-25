"""Paper figures (static, print, light mode) from results/*.json. Palette: validated reference slots 1-3
(blue #2a78d6, orange #eb6834, aqua #1baf7a -- all-pairs pass; aqua needs direct labels), sequential blue ramp,
text ink #0b0b0b / #52514e, hairline grid #e1e0d9.  python figs.py  (cwd = paper/)"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

R = os.path.join("..", "results")
OUT = "."
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
                     "legend.frameon": False, "pdf.fonttype": 42})


def J(name):
    return json.load(open(os.path.join(R, name)))


def fig1():
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    # (a) dose-response of the probe response
    rows = [r for r in J("g0_g2.json")["rows"] if r["tag"][0] == "G2"]
    s1s = sorted({r["tag"][1] for r in rows})
    rng = np.random.default_rng(0)
    for s in s1s:
        y = [r["mn9_hz"]["p2"] for r in rows if r["tag"][1] == s]
        ax[0].scatter(s + rng.uniform(-0.035, 0.035, len(y)), y, s=9, color=BLUE, alpha=0.55, linewidths=0)
    means = [np.mean([r["mn9_hz"]["p2"] for r in rows if r["tag"][1] == s]) for s in s1s]
    ax[0].plot(s1s, means, color=BLUE, lw=2)
    ax[0].set_xlabel("first-pulse drive $s$ (x100 Hz, LB3)")
    ax[0].set_ylabel("MN9 rate in later probe (Hz)")
    ax[0].set_title("a  probe response vs prior drive", loc="left", fontsize=8.5)
    ax[0].annotate("bimodal", xy=(0.8, 36), xytext=(1.05, 45), fontsize=7, color=INK2,
                   arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    # (b) persistence over 10 s with no input
    e2 = J("e2_persistence.json")
    t = np.arange(1, 11)
    for a in e2["active"]:
        ax[1].plot(t, a[2:], color=BLUE, lw=1.2, alpha=0.8)
    ax[1].plot(t, np.zeros(10), color=INK2, lw=1.2, ls="--")
    ax[1].text(10, 400, "no / weak prior input: 0", ha="right", va="bottom", fontsize=7, color=INK2)
    ax[1].text(10, 8600, "after strong input (5 seeds)", ha="right", va="bottom", fontsize=7, color=BLUE)
    ax[1].set_ylim(-300, 10000)
    ax[1].set_xlabel("time after stimulus offset (s)")
    ax[1].set_ylabel("neurons firing > 1 Hz")
    ax[1].set_title("b  self-sustained for >= 10 s", loc="left", fontsize=8.5)
    # (c) ablations
    e3 = J("e3_ablation.json")["rows"]
    e3b = [r for r in J("e3b_alln.json")["rows"] if r["cond"] != "intact"][:4]
    labels, vals = [], []
    for r in e3:
        labels.append(r["cond"].replace("minus ", "- ").replace("no input to sensory", "no input to sensory cells"))
        vals.append(r["ignited"])
    for r in e3b:
        labels.append("- " + r["cond"].replace("ALLN other/unknown nt", "ALLN, nt unknown/5HT/OA").replace("ALLN ", "ALLN, "))
        vals.append(r["ignited"])
    y = np.arange(len(labels))[::-1]
    cols = [ORANGE if v <= 1 else BLUE for v in vals]
    ax[2].barh(y, vals, color=cols, height=0.62)
    ax[2].set_yticks(y)
    ax[2].set_yticklabels(labels, fontsize=6.3)
    ax[2].set_xlim(0, 5.4)
    ax[2].set_xticks([0, 1, 2, 3, 4, 5])
    ax[2].set_xlabel("ignited runs (of 5)")
    ax[2].set_title("c  necessary cells", loc="left", fontsize=8.5)
    ax[2].grid(axis="y", visible=False)
    for yi, v in zip(y, vals):
        if v <= 1:
            ax[2].text(v + 0.1, yi, "%d/5" % v, va="center", fontsize=6.5, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1.pdf")); fig.savefig(os.path.join(OUT, "fig1.png"), dpi=200)


def fig2():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1, 1.35]))
    rows = J("e1c_phase.json")["rows"]
    ks = sorted({r["k"] for r in rows}); hz = sorted({r["hz"] for r in rows})
    M = np.array([[np.mean([r["ignited"] for r in rows if r["k"] == k and r["hz"] == h]) for h in hz] for k in ks])
    ax[0].imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect="auto", origin="lower")
    for i, k in enumerate(ks):
        for j, h in enumerate(hz):
            n = sum(r["ignited"] for r in rows if r["k"] == k and r["hz"] == h)
            ax[0].text(j, i, "%d/3" % n, ha="center", va="center", fontsize=7, color="#ffffff" if M[i, j] > 0.5 else INK)
    ax[0].set_xticks(range(len(hz))); ax[0].set_xticklabels(["%d Hz" % h for h in hz])
    ax[0].set_yticks(range(len(ks))); ax[0].set_yticklabels(ks)
    ax[0].set_ylabel("LB3 cells stimulated (random subset)")
    ax[0].grid(False)
    ax[0].set_title("a  ignition vs number and rate", loc="left", fontsize=8.5)
    P = J("e6_predictions.json"); res = J("e6_result.json")
    th = P["threshold_core"]
    ign = np.array(res["ignited"]); sc = np.array([p["core_score"] for p in P["predictions"]])
    k = np.array([p["k"] for p in P["predictions"]])
    pred = sc > th
    for val, colr, lab in ((True, ORANGE, "ignited"), (False, BLUE, "not ignited")):
        m = ign == val
        ax[1].scatter(sc[m], k[m] + np.where(val, 1.6, -1.6), s=16, color=colr, label=lab, linewidths=0)
    wrong = pred != ign
    ax[1].scatter(sc[wrong], k[wrong] + np.where(ign[wrong], 1.6, -1.6), s=60, facecolors="none", edgecolors=INK, lw=0.8,
                  label="misclassified")
    ax[1].axvline(th, color=INK2, ls="--", lw=1)
    ax[1].text(th * 0.95, 99, "frozen threshold", fontsize=6.5, color=INK2, va="bottom", ha="right")
    ax[1].set_xscale("log")
    ax[1].set_xticks([1e5, 3e5, 1e6, 3e6])
    ax[1].set_xticklabels(["1e5", "3e5", "1e6", "3e6"])
    ax[1].minorticks_off()
    ax[1].set_ylim(24, 104)
    ax[1].set_xlabel("2-hop excitatory drive onto the core x rate (wiring only)")
    ax[1].set_ylabel("LB3 cells stimulated")
    ax[1].set_title("b  prospective test: 37/42 correct (count rule: 28/42)", loc="left", fontsize=8.5)
    ax[1].legend(loc="upper left", fontsize=6.5, handletextpad=0.2, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2.pdf")); fig.savefig(os.path.join(OUT, "fig2.png"), dpi=200)


def fig3():
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    pts = [("original model", 0.0, 1.0, INK2)]
    for r in J("e5c_sign_fix.json")["rows"]:
        dev = max(abs(r["shiu20_mn9_ratio"][str(h)] - 1) for h in (100, 150, 200)) * 100
        ig = (r["ignited"]["ign_LB3_120"] + r["ignited"]["ign_SUGAR_ALL_200"]) / 10
        lab = {"c1": "176 ALLN re-signed", "c2": "150 ALLN re-signed", "c3": "all 19,042 unknown-nt silenced"}[r["cond"][:2]]
        pts.append((lab, dev, ig, ORANGE if r["cond"].startswith(("c1", "c2")) else BLUE))
    for r in J("e7_global_fixes.json")["rows"]:
        dev = max(abs(r["shiu20_mn9_ratio"][str(h)] - 1) for h in (100, 150, 200)) * 100
        ig = (r["ignited"]["ign_LB3_120"] + r["ignited"]["ign_SUGAR_ALL_200"]) / 10
        pts.append((r["cond"][:2], dev, ig, BLUE))
    for r in J("e5_fix.json")["rows"]:
        dev = max(abs(r["shiu20_mn9_ratio"][str(h)] - 1) for h in (100, 150, 200)) * 100
        ig = (r["ignited"]["ign_LB3_120"] + r["ignited"]["ign_SUGAR_ALL_200"]) / 10
        pts.append(("adaptation %.2g mV" % r["d_a"], dev, ig, AQUA))
    ax.axvspan(-2, 20, color="#f0efec", zorder=0)
    ax.text(9, 0.42, "validated\nbehaviour kept\n(<= 20%\nchange)", ha="center", fontsize=6.3, color=INK2)
    for lab, x, y, c in pts:
        ax.scatter(x, y, s=28, color=c, zorder=3, linewidths=0)
    ax.text(-0.5, 1.06, "original", fontsize=6.2, color=INK, ha="right", va="bottom")
    sfa = [(x, y) for lab, x, y, c in pts if lab.startswith("adaptation")]
    for (x, y), d in zip(sfa, ("0.25", "0.5", "1", "2")):
        ax.text(x, y - 0.1, d, fontsize=6.2, color=INK, ha="center")
    ax.text(np.mean([x for x, _ in sfa]) + 8, 0.7, "spike-frequency adaptation\n(increment, mV)", fontsize=6.2,
            color=INK, ha="center")
    ax.text(1.5, 0.07, "150 or 176 AL local neurons\nre-signed inhibitory", fontsize=6.2, color=INK)
    glob = {lab: (x, y) for lab, x, y, c in pts if lab in ("g1", "g2", "g3") or lab.startswith("all 19")}
    x, y = glob["all 19,042 unknown-nt silenced"]
    ax.text(x, y - 0.1, "unknown-nt\nsilenced", fontsize=6.0, color=INK, ha="center", va="top")
    x, y = glob["g3"]; ax.text(x, y - 0.1, "synapse cap\n100", fontsize=6.0, color=INK, ha="center", va="top")
    x, y = glob["g1"]; ax.text(x - 1.5, y - 0.1, "w_syn\n0.15 mV", fontsize=6.0, color=INK, ha="right", va="top")
    x, y = glob["g2"]; ax.text(x, y + 0.06, "neuromodulator output removed", fontsize=6.0, color=INK, ha="left", va="bottom")
    ax.set_xlim(-2, 105); ax.set_ylim(-0.35, 1.22)
    ax.set_xlabel("max change of validated MN9 responses (%)")
    ax.set_ylabel("fraction of strong stimuli that ignite")
    ax.set_title("Targeted correction vs generic fixes", loc="left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3.pdf")); fig.savefig(os.path.join(OUT, "fig3.png"), dpi=200)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig1(); fig2(); fig3()
    print("ok")
