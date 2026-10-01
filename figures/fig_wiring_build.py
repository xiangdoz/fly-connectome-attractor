"""Figure 3 (wiring explains the state): data panels from matplotlib plus a circuit diagram of the antennal-lobe loop,
laid out in PowerPoint (fig_wiring.pptx) and exported to figs/fig2.pdf.  python fig_wiring_build.py  (cwd = paper/)
Node areas follow the leading-mode shares (results/e11b_eigvec.json); edge widths follow summed tau*w between the
classes inside the mean-field set (results/fig_loop_weights.json)."""
import json
import math
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

from figs import J, ORANGE as O_HEX, BLUE as B_HEX, INK as I_HEX, INK2 as I2_HEX

PARTS = os.path.join("figs", "parts")
INK, INK2 = RGBColor.from_string(I_HEX[1:]), RGBColor.from_string(I2_HEX[1:])
ORANGE, BLUE, GREY = RGBColor.from_string(O_HEX[1:]), RGBColor.from_string(B_HEX[1:]), RGBColor(0xCF, 0xCD, 0xC6)
FONT = "Arial"
W, H = 7.0, 2.55


def parts():
    names = {"Kenyon_Cell": "Kenyon cells", "olfactory": "olfactory receptor", "optic_lobe_intrinsic": "optic lobe",
             "ALPN": "AL projection", "LHLN": "lateral-horn local", "ALLN": "AL local", "DAN": "dopaminergic",
             "CX": "central complex", "descending": "descending", "MBON": "MB output"}
    rows = [r for r in J("e4b_composition.json")["rows"] if r["cls"] in names]
    fig = plt.figure(figsize=(2.15, 2.15))
    ax = fig.add_axes([0.42, 0.20, 0.55, 0.78])
    y = np.arange(len(rows))[::-1]
    h = 0.38
    ax.barh(y + h / 2, [r["simulated"] for r in rows], height=h, color=I2_HEX, label="simulation")
    ax.barh(y - h / 2, [r["predicted"] for r in rows], height=h, color=O_HEX, label="mean field")
    ax.set_yticks(y); ax.set_yticklabels([names[r["cls"]] for r in rows], fontsize=6.4)
    ax.set_xticks([0, 2000, 4000]); ax.set_xticklabels(["0", "2k", "4k"])
    ax.set_xlabel("neurons in the state", fontsize=6.8)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=6.0, handlelength=1.0)
    ax.tick_params(labelsize=6.4)
    fig.savefig(os.path.join(PARTS, "membership.png"), dpi=600, transparent=True); plt.close(fig)

    lab3 = {"minus Kenyon_Cell": "Kenyon cells", "minus ALPN": "AL projection", "minus olfactory": "olfactory receptor",
            "minus LHLN": "lateral-horn local", "minus DAN": "dopaminergic", "minus MBON": "MB output",
            "minus APL": "APL", "no input to sensory": "input to sensory cells", "minus ALLN": "AL local (429)"}
    lab3b = {"ALLN ACH": "AL local, ACh", "ALLN GABA": "AL local, GABA", "ALLN GLUT": "AL local, Glu",
             "ALLN other/unknown nt": "AL local, no call (176)"}
    e3 = {r["cond"]: r["ignited"] for r in J("e3_ablation.json")["rows"]}
    e3b = {r["cond"]: r["ignited"] for r in J("e3b_alln.json")["rows"]}
    items = [(lab3[k], e3[k]) for k in lab3] + [(v, e3b[k]) for k, v in lab3b.items()]
    fig = plt.figure(figsize=(2.15, 2.15))
    ax = fig.add_axes([0.50, 0.20, 0.47, 0.78])
    yy = np.arange(len(items))[::-1].astype(float)
    yy[len(lab3):] -= 0.6
    vals = [v for _, v in items]
    ax.barh(yy, vals, height=0.62, color=[O_HEX if v == 0 else "#cfcdc6" for v in vals])
    for yi, v in zip(yy, vals):
        if v == 0:
            ax.text(0.15, yi, "0/5", va="center", fontsize=6.2, color=O_HEX)
    ax.set_yticks(yy); ax.set_yticklabels([lab for lab, _ in items], fontsize=6.2)
    ax.set_xlim(0, 5.3); ax.set_xticks([0, 5])
    ax.set_xlabel("runs still igniting (of 5)", fontsize=6.8)
    ax.grid(axis="y", visible=False)
    ax.tick_params(labelsize=6.2)
    fig.savefig(os.path.join(PARTS, "ablation.png"), dpi=600, transparent=True); plt.close(fig)


def text(slide, x, y, w, h, s, size=7, color=INK2, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for k, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run(); r.text = line
        r.font.size = Pt(size); r.font.name = FONT; r.font.bold = bold; r.font.color.rgb = color
    return tb


def arrow(slide, x0, y0, x1, y1, color, width):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x0), Inches(y0), Inches(x1), Inches(y1))
    c.line.color.rgb = color
    c.line.width = Pt(width)
    tail = etree.SubElement(c.line._get_or_add_ln(), qn("a:tailEnd"))
    tail.set("type", "triangle"); tail.set("w", "sm"); tail.set("len", "sm")


def node(slide, cx, cy, r, fill, label, label_color=INK, above=False):
    sh = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - r), Inches(cy - r), Inches(2 * r), Inches(2 * r))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = RGBColor(0xFF, 0xFF, 0xFF); sh.line.width = Pt(0.75)
    sh.shadow.inherit = False
    ty = cy - r - 0.16 if above else cy + r + 0.02
    text(slide, cx - 0.4, ty, 0.8, 0.15, label, 6.2, label_color, align=PP_ALIGN.CENTER)


def circuit(slide, x0, y0, resigned, mode, wts):
    width = lambda w: 0.5 + 3.2 * math.sqrt(w / 424.0)
    rad = lambda share: max(0.045, 0.30 * math.sqrt(share))
    pos = {"ORN": (x0 + 0.13, y0 + 0.30), "PN": (x0 + 0.55, y0 + 0.30), "KC": (x0 + 0.98, y0 + 0.30),
           "LN": (x0 + 0.55, y0 + 1.02)}
    r = {"ORN": rad(mode.get("olfactory", 0)), "PN": rad(mode["ALPN"]), "KC": rad(mode["Kenyon_Cell"]),
         "LN": rad(mode["core"])}

    def edge(a, b, w, color, dx=0.0):
        (xa, ya), (xb, yb) = pos[a], pos[b]
        d = math.hypot(xb - xa, yb - ya)
        ux, uy = (xb - xa) / d, (yb - ya) / d
        arrow(slide, xa + ux * (r[a] + 0.02) + dx, ya + uy * (r[a] + 0.02), xb - ux * (r[b] + 0.03) + dx,
              yb - uy * (r[b] + 0.03), color, width(w))

    edge("ORN", "PN", wts["ORN->PN"][0], ORANGE)
    edge("PN", "KC", wts["PN->KC"][0], ORANGE)
    edge("PN", "LN", wts["PN->core LN"][0], ORANGE, dx=-0.07)
    if resigned:
        edge("LN", "PN", wts["core LN->PN"][0] + wts["core LN->PN"][1], BLUE, dx=0.07)
    else:
        edge("LN", "PN", wts["core LN->PN"][0], ORANGE, dx=0.07)
    node(slide, *pos["ORN"], r["ORN"], GREY, "ORN")
    node(slide, *pos["PN"], r["PN"], GREY, "PN", above=True)
    node(slide, *pos["KC"], r["KC"], GREY, "KC")
    node(slide, *pos["LN"], r["LN"], ORANGE if not resigned else BLUE, "core LN")


def main():
    parts()
    ev = J("e11b_eigvec.json")
    mode = dict(ev["mass_by_class"]); mode["core"] = ev["core_mass"]
    wts = J("fig_loop_weights.json")
    e11 = J("e11_spectrum.json")
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    for x, letter, ttl in ((0.02, "a", "Membership from the mean field"), (2.33, "b", "Outputs removed"),
                           (4.62, "c", "The loop that sustains it")):
        text(s, x, 0.02, 0.2, 0.2, letter, 9, INK, bold=True)
        text(s, x + 0.18, 0.03, 2.2, 0.2, ttl, 7.5, INK)
    s.shapes.add_picture(os.path.join(PARTS, "membership.png"), Inches(0.02), Inches(0.30), Inches(2.15), Inches(2.15))
    s.shapes.add_picture(os.path.join(PARTS, "ablation.png"), Inches(2.25), Inches(0.30), Inches(2.15), Inches(2.15))
    text(s, 1.15, 1.05, 1.0, 0.3, "Jaccard 0.70\n(random sets 0.06)", 6.2, INK2)
    circuit(s, 4.62, 0.55, False, mode, wts)
    circuit(s, 5.86, 0.55, True, mode, wts)
    text(s, 4.62, 1.86, 1.15, 0.3, "model signs\nλ = %.2f" % e11["original"], 6.4, INK, align=PP_ALIGN.CENTER)
    text(s, 5.86, 1.86, 1.15, 0.3, "core re-signed\nλ = %.2f" % e11["core_resigned"], 6.4, INK,
         align=PP_ALIGN.CENTER)
    text(s, 4.62, 2.26, 2.36, 0.4, "orange excitatory, blue inhibitory; node area: share of the leading mode; "
         "edge width: summed weight", 5.8, INK2)
    prs.save("fig_wiring.pptx")
    pptx, pdf = os.path.abspath("fig_wiring.pptx"), os.path.abspath(os.path.join("figs", "fig2.pdf"))
    ps = ("$pp = New-Object -ComObject PowerPoint.Application; "
          "$p = $pp.Presentations.Open('%s', $true, $false, $false); $p.SaveAs('%s', 32); $p.Close(); $pp.Quit()"
          % (pptx, pdf))
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    print("wrote", pptx, pdf)


if __name__ == "__main__":
    main()
