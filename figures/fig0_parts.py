"""Data panels for the overview figure (Figure 1). Each panel is saved as a 600-dpi PNG sized to its slot in
fig0_overview.pptx, which adds arrows, labels and panel letters.  python fig0_parts.py  (cwd = paper/)"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from figs import J, BLUE, ORANGE, AQUA, INK, INK2, GRID

R = os.path.join("..", "results")
OUT = os.path.join("figs", "parts")
plt.rcParams.update({"font.size": 6.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
DIV = LinearSegmentedColormap.from_list("div", [BLUE, "#ffffff", ORANGE])


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=600, transparent=True)
    plt.close(fig)


def votes():
    v = json.load(open(os.path.join(R, "fig_overview_votes.json")))["lLN1_bc_votes"]
    parts = [("DA", v["da"], "#f2a07b"), ("5-HT", v["ser"], "#f6c3a8"), ("ACh", v["ach"], ORANGE),
             ("GABA", v["gaba"], BLUE), ("Glu", v["glut"], "#86b6ef")]
    fig = plt.figure(figsize=(1.30, 0.95))
    ax = fig.add_axes([0.03, 0.55, 0.94, 0.38])
    x = 0
    for lab, w, c in parts:
        ax.barh(0, w, left=x, color=c, height=1.0, edgecolor="#ffffff", lw=0.6)
        if w > 0.1:
            ax.text(x + w / 2, 0, lab, ha="center", va="center", fontsize=5.8, color=INK)
        x += w
    ax.set_xlim(0, 1); ax.axis("off")
    ax.annotate("", xy=(0.0, -0.95), xytext=(0.735, -0.95), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=2.2))
    ax.text(0.37, -1.2, "counted as excitatory", fontsize=5.8, color=ORANGE, transform=ax.transAxes, ha="center",
            va="top")
    save(fig, "votes")


def traces():
    fig = plt.figure(figsize=(1.40, 1.55))
    ax = fig.add_axes([0.22, 0.24, 0.74, 0.70])
    e2 = J("e2_persistence.json")
    t = np.arange(1, 11) - 0.5
    for a in e2["active"]:
        ax.plot(t, np.array(a[2:]) / 1000, color=ORANGE, lw=1.0)
    g3 = [r for r in J("g3_recovery.json")["persistence"]["rows"] if r["s1"] == 0.8]
    t3 = np.arange(8) * 0.25 + 0.125
    for r in g3:
        stays = r["active"][9] > 1000
        ax.plot(t3, np.array(r["active"][2:10]) / 1000, color=ORANGE if stays else BLUE, lw=1.0,
                ls="--" if stays else "-")
    ax.text(9.8, 7.3, "persists", color=ORANGE, fontsize=6, ha="right", va="top")
    ax.text(9.8, 0.9, "falls silent", color=BLUE, fontsize=6, ha="right", va="bottom")
    ax.set_ylim(-0.4, 9.5); ax.set_xlim(0, 10)
    ax.set_xticks([0, 5, 10]); ax.set_yticks([0, 4, 8])
    ax.set_xlabel("s after input", fontsize=6, labelpad=1)
    ax.set_ylabel("active (x1000)", fontsize=6, labelpad=1)
    ax.tick_params(length=2, pad=1, labelsize=5.6)
    save(fig, "traces")


def matrix():
    d = np.load(os.path.join(R, "fig_overview_data.npz"))
    sub, v, core = d["sub"], d["v_sub"], d["is_core"]
    lim = np.percentile(np.abs(sub[sub != 0]), 95)
    fig = plt.figure(figsize=(1.15, 1.15))
    ax = fig.add_axes([0.0, 0.0, 0.84, 1.0])
    ax.imshow(np.clip(sub, -lim, lim), cmap=DIV, vmin=-lim, vmax=lim, interpolation="nearest", aspect="equal")
    for b in (50, 90):
        ax.axhline(b - 0.5, color=INK2, lw=0.4); ax.axvline(b - 0.5, color=INK2, lw=0.4)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(INK2); s.set_linewidth(0.5)
    ax2 = fig.add_axes([0.88, 0.0, 0.12, 1.0])
    ax2.barh(np.arange(len(v)), v / v.max(), height=1.0,
             color=[ORANGE if c else "#b9b7b0" for c in core])
    ax2.set_ylim(len(v) - 0.5, -0.5); ax2.axis("off")
    save(fig, "matrix")


def ranking():
    n = 90
    drop = -np.array(J("e14_rank.json")["pred_drop"][:n])
    flags = np.array(J("e14_core_flags.json")["is_core"][:n])
    fig = plt.figure(figsize=(1.05, 1.15))
    ax = fig.add_axes([0.03, 0.05, 0.94, 0.9])
    y = np.arange(len(drop))
    ax.barh(y, np.log10(drop / drop.min()) + 0.2, color=[ORANGE if f else BLUE for f in flags], height=1.0)
    ax.set_ylim(len(drop) - 0.4, -0.6); ax.axis("off")
    save(fig, "ranking")


def validation():
    sim = {c["cond"]: c for c in J("e14_simulate.json")["conds"]}
    top = sim["top-45"]["ignited"]
    rnd = sum(sim["random-45 #%d" % i]["ignited"] for i in (1, 2, 3))
    fig = plt.figure(figsize=(1.0, 1.40))
    ax = fig.add_axes([0.32, 0.30, 0.64, 0.62])
    ax.bar([0, 1], [top / 6, rnd / 18], color=[ORANGE, BLUE], width=0.62)
    ax.text(0, top / 6 + 0.04, "%d/6" % top, ha="center", fontsize=6.2, color=ORANGE)
    ax.text(1, rnd / 18 + 0.04, "%d/18" % rnd, ha="center", fontsize=6.2, color=BLUE)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["top 45", "45\nrandom"], fontsize=5.8)
    ax.set_ylim(0, 1.22); ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("runs in the state", fontsize=6, labelpad=1)
    ax.tick_params(length=2, pad=1, labelsize=5.6)
    save(fig, "validation")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    votes(); traces(); matrix(); ranking(); validation()
    print("ok")
