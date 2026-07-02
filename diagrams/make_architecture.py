"""
Generate two clean, slide-ready (16:9) architecture diagrams as PNGs:

  1. architecture_pipeline.png  — the full 4-stage SHRE pipeline (horizontal,
     JD feeding in, CTAE fallback strip), laid out to fit a widescreen slide.
  2. architecture_fallback.png  — the LTR -> Ensemble -> CTAE reliability chain.

Pure matplotlib, no extra deps. Run:  python diagrams/make_architecture.py

Design notes: the axes use equal units/inch on both axes (100 x 56.25 data
units in a 13.333 x 7.5 in figure), so geometry is predictable and circles
stay round. Text is kept short and hand-wrapped so nothing overflows a box.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# palette — restrained, professional
# ---------------------------------------------------------------------------
INK      = '#0f172a'   # near-black slate
MUTED    = '#475569'
FAINT    = '#94a3b8'
LINE     = '#d5dbe3'

INDIGO   = '#4f46e5'; INDIGO_BG = '#eef0fe'
TEAL     = '#0d9488'; TEAL_BG   = '#e6f5f3'
AMBER    = '#d97706'; AMBER_BG  = '#fdf1e0'
BLUE     = '#2563eb'; BLUE_BG   = '#e7effe'
GREEN    = '#059669'; GREEN_BG  = '#e4f6ee'
SLATE_BG = '#f1f5f9'
PURPLE   = '#7c3aed'; PURPLE_BG = '#f4edfe'

plt.rcParams['font.family'] = 'DejaVu Sans'


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def rbox(ax, x, y, w, h, fc='#ffffff', ec=LINE, lw=1.4, rounding=1.1, z=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z, mutation_aspect=1.0))


def txt(ax, x, y, s, size=11, color=INK, weight='normal', ha='center',
        va='center', z=5, style='normal'):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=z, style=style, linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, color=MUTED, lw=2.4, dashed=False, rad=0.0,
          z=3, ms=15):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=ms, linewidth=lw,
        color=color, connectionstyle=f"arc3,rad={rad}",
        linestyle=(0, (4, 2)) if dashed else '-', zorder=z, shrinkA=1, shrinkB=1))


def stage_lane(ax, x, y, w, h, num, title_lines, accent, accent_bg, bullets):
    """A stage lane: rounded card, accent header banner (stacked 2-line title),
    a corner number badge, and short bullets that stay inside the card body."""
    rbox(ax, x, y, w, h, fc='#ffffff', ec=LINE, lw=1.6, rounding=1.1, z=2)

    hh = 8.4                                   # header height
    hy = y + h - hh
    # header banner: rounded box + square lower half => rounded-top banner
    ax.add_patch(FancyBboxPatch((x, hy), w, hh,
                 boxstyle="round,pad=0,rounding_size=1.1", linewidth=0,
                 facecolor=accent_bg, zorder=3))
    ax.add_patch(FancyBboxPatch((x, hy), w, hh * 0.55, boxstyle="square,pad=0",
                 linewidth=0, facecolor=accent_bg, zorder=3))
    ax.add_line(Line2D([x + 0.6, x + w - 0.6], [hy, hy], color=accent,
                lw=1.4, alpha=0.6, zorder=4))

    # centered 2-line title
    cy = hy + hh / 2
    txt(ax, x + w / 2, cy + 1.7, title_lines[0], size=11, color=INK, weight='bold')
    txt(ax, x + w / 2, cy - 1.7, title_lines[1], size=11, color=INK, weight='bold')

    # number badge, top-left corner
    bd = 3.3
    bx, by_ = x + 0.9, y + h - 0.9 - bd
    rbox(ax, bx, by_, bd, bd, fc=accent, ec=accent, lw=0, rounding=0.85, z=4)
    txt(ax, bx + bd / 2, by_ + bd / 2, str(num), size=12.5, color='#ffffff',
        weight='bold', z=5)

    # bullets (lines starting with a space are wrap-continuations: no dot)
    by = hy - 3.1
    for b in bullets:
        cont = b.startswith(' ')
        if not cont:
            txt(ax, x + 1.8, by, "\u2022", size=11, color=accent, ha='left', weight='bold')
        txt(ax, x + 3.1, by, b.lstrip(), size=9.3, color=MUTED, ha='left')
        by -= 3.05 if cont else 3.35


# ===========================================================================
# DIAGRAM 1 — full pipeline (16:9)
# ===========================================================================
def diagram_pipeline():
    W, H = 100.0, 56.25
    fig, ax = plt.subplots(figsize=(13.333, 7.5), dpi=240)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    fig.patch.set_facecolor('#ffffff')

    # ---- title
    txt(ax, 3.0, 54.0, "Staged Hybrid Ranking Engine (SHRE)", size=17,
        color=INK, weight='bold', ha='left')
    txt(ax, 3.0, 50.6,
        "4-stage hybrid pipeline:  anomaly filter  \u2192  93 features  \u2192  "
        "ensemble + learning-to-rank  \u2192  grounded reasoning",
        size=10, color=MUTED, ha='left')

    # ---- geometry (equal units/inch, so spacing is exact)
    lane_y, lane_h, lane_w = 9.5, 32.0, 18.0
    xs = [12.0, 32.2, 52.4, 72.6]
    in_x, in_w = 0.5, 10.5
    out_x, out_w = 91.6, 8.0
    midy = lane_y + lane_h / 2

    # ---- input
    ih = 12.0; iy = midy - ih / 2
    rbox(ax, in_x, iy, in_w, ih, fc=INK, ec=INK, lw=0, rounding=1.1, z=2)
    txt(ax, in_x + in_w / 2, iy + ih / 2 + 2.4, "candidates", size=11,
        color='#ffffff', weight='bold')
    txt(ax, in_x + in_w / 2, iy + ih / 2 - 0.6, ".jsonl", size=10.5,
        color='#ffffff', weight='bold')
    txt(ax, in_x + in_w / 2, iy + ih / 2 - 3.7, "100k+ profiles", size=8.5,
        color='#cbd5e1')

    # ---- stage lanes
    stage_lane(ax, xs[0], lane_y, lane_w, lane_h, 1, ["Anomaly", "Pre-Filter"],
               AMBER, AMBER_BG,
               ["Timeline & skill", "  synthetic checks",
                "Gate: JD exp band", "  + \u22652 skill pillars",
                "Drops honeypots"])
    stage_lane(ax, xs[1], lane_y, lane_w, lane_h, 2, ["Feature", "Engineering"],
               BLUE, BLUE_BG,
               ["78 base features", "  career / domain / co.",
                "+5 anomaly, +5", "  behavioral",
                "+5 multi-vector", "  semantic (vs JD)"])
    stage_lane(ax, xs[2], lane_y, lane_w, lane_h, 3, ["Ensemble", "+ LTR"],
               INDIGO, INDIGO_BG,
               ["XGB / LGBM / Cat", "  soft-vote ensemble",
                "LambdaMART LTR", "  (XGBRanker ndcg)",
                "Fuse 0.6 ens", "  + 0.4 LTR"])
    stage_lane(ax, xs[3], lane_y, lane_w, lane_h, 4, ["Ranker &", "Reasoning"],
               GREEN, GREEN_BG,
               ["Sort \u2192 Top 100", "Grounded, non-",
                "  hallucinated why", "Outputs: CSV /",
                "  XLSX + detailed"])

    # ---- output
    oh = 18.0; oy = midy - oh / 2
    rbox(ax, out_x, oy, out_w, oh, fc=GREEN, ec=GREEN, lw=0, rounding=1.1, z=2)
    txt(ax, out_x + out_w / 2, oy + oh / 2 + 4.2, "Top", size=12,
        color='#ffffff', weight='bold')
    txt(ax, out_x + out_w / 2, oy + oh / 2 + 0.3, "100", size=21,
        color='#ffffff', weight='bold')
    txt(ax, out_x + out_w / 2, oy + oh / 2 - 4.4, "ranked +\nreasoning",
        size=8.2, color='#d6f3e5')

    # ---- JD banner (above stage 1 & 2) — stacked, no overlap
    jd_x = xs[0]; jd_w = (xs[1] + lane_w) - xs[0]
    jd_y, jd_h = 43.4, 5.6
    rbox(ax, jd_x, jd_y, jd_w, jd_h, fc=PURPLE_BG, ec=PURPLE, lw=1.6, rounding=1.0, z=2)
    jcx = jd_x + jd_w / 2
    txt(ax, jcx, jd_y + jd_h / 2 + 1.3, "Job Description  ( --jd  \u00b7  text or file )",
        size=10, color=PURPLE, weight='bold')
    txt(ax, jcx, jd_y + jd_h / 2 - 1.5,
        "parsed \u2192 3 semantic facets  +  experience band",
        size=9, color=MUTED)

    # ---- main flow arrows
    arrow(ax, in_x + in_w, midy, xs[0], midy, color=INK, lw=2.6)
    for i in range(3):
        arrow(ax, xs[i] + lane_w, midy, xs[i + 1], midy, color=MUTED, lw=2.6)
    arrow(ax, xs[3] + lane_w, midy, out_x, midy, color=GREEN, lw=2.6)

    # ---- JD dashed feeds -> stage 1 gate & stage 2 features
    arrow(ax, xs[0] + lane_w / 2, jd_y, xs[0] + lane_w / 2, lane_y + lane_h,
          color=PURPLE, lw=1.7, dashed=True)
    arrow(ax, xs[1] + lane_w / 2, jd_y, xs[1] + lane_w / 2, lane_y + lane_h,
          color=PURPLE, lw=1.7, dashed=True)

    # ---- fallback strip (bottom)
    fb_x, fb_y, fb_w, fb_h = 12.0, 2.6, 78.6, 5.4
    rbox(ax, fb_x, fb_y, fb_w, fb_h, fc=SLATE_BG, ec=LINE, lw=1.4, rounding=0.9, z=1)
    fcy = fb_y + fb_h / 2
    txt(ax, fb_x + 2.2, fcy, "Reliability fallback", size=9.5, color=INK,
        weight='bold', ha='left')
    txt(ax, fb_x + fb_w / 2 + 5, fcy,
        "LambdaMART   \u2192   Validated Ensemble   \u2192   Pure-Python CTAE",
        size=9.5, color=MUTED)
    txt(ax, fb_x + fb_w - 2.0, fcy, "never fails", size=9, color=GREEN,
        weight='bold', ha='right', style='italic')
    arrow(ax, fb_x + fb_w, fcy, out_x + out_w / 2, oy, color=FAINT, lw=1.6,
          dashed=True, rad=-0.25)

    fig.savefig(os.path.join(OUT, 'architecture_pipeline.png'),
                bbox_inches='tight', pad_inches=0.18, facecolor='#ffffff')
    plt.close(fig)
    print("wrote architecture_pipeline.png")


# ===========================================================================
# DIAGRAM 2 — fallback reliability chain (16:9)
# ===========================================================================
def diagram_fallback():
    W, H = 100.0, 56.25
    fig, ax = plt.subplots(figsize=(13.333, 7.5), dpi=240)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    fig.patch.set_facecolor('#ffffff')

    txt(ax, 50, 50.5, "Fallback Reliability Chain", size=18, color=INK, weight='bold')
    txt(ax, 50, 45.6,
        "Every layer degrades gracefully \u2014 the engine always returns a Top-100 shortlist",
        size=11, color=MUTED)

    boxes = [("LambdaMART", "LTR head", INDIGO, INDIGO_BG),
             ("Validated", "Voting Ensemble", BLUE, BLUE_BG),
             ("Pure-Python", "CTAE ranker", AMBER, AMBER_BG)]
    bw, bh, by = 21.0, 13.0, 20.0
    xs = [6.0, 33.0, 60.0]

    for i, (t1, t2, ac, bg) in enumerate(boxes):
        x = xs[i]
        rbox(ax, x, by, bw, bh, fc=bg, ec=ac, lw=2.2, rounding=1.1, z=2)
        ax.add_patch(FancyBboxPatch((x, by + bh - 2.6), bw, 2.6,
                     boxstyle="round,pad=0,rounding_size=1.1", linewidth=0,
                     facecolor=ac, zorder=3))
        ax.add_patch(FancyBboxPatch((x, by + bh - 2.6), bw, 1.4,
                     boxstyle="square,pad=0", linewidth=0, facecolor=ac, zorder=3))
        txt(ax, x + bw / 2, by + bh - 1.3, f"Layer {i + 1}", size=9,
            color='#ffffff', weight='bold', z=4)
        txt(ax, x + bw / 2, by + bh / 2 - 0.6, t1, size=13, color=INK, weight='bold')
        txt(ax, x + bw / 2, by + bh / 2 - 4.1, t2, size=11, color=MUTED)

    for i in range(2):
        x1, x2 = xs[i] + bw, xs[i + 1]
        arrow(ax, x1, by + bh / 2, x2, by + bh / 2, color=MUTED, lw=2.6)
        txt(ax, (x1 + x2) / 2, by + bh / 2 + 2.6, "on failure", size=8.5,
            color=FAINT, weight='bold', style='italic')

    ox, ow, oh = 84.0, 12.5, 17.0
    oy = by - 2.0
    rbox(ax, ox, oy, ow, oh, fc=GREEN, ec=GREEN, lw=0, rounding=1.1, z=2)
    txt(ax, ox + ow / 2, oy + oh / 2 + 3.2, "Top-100", size=13, color='#ffffff', weight='bold')
    txt(ax, ox + ow / 2, oy + oh / 2 - 0.6, "shortlist", size=11, color='#ffffff')
    txt(ax, ox + ow / 2, oy + oh / 2 - 4.2, "(identical\nCSV schema)", size=8.2,
        color='#d6f3e5')

    styles = [(INDIGO, False, 0.0, 2.6), (BLUE, True, -0.18, 1.7), (AMBER, True, -0.30, 1.7)]
    for i, (col, dsh, rad, lw) in enumerate(styles):
        arrow(ax, xs[i] + bw / 2, by, ox, oy + oh / 2, color=col, lw=lw,
              dashed=dsh, rad=rad)
    txt(ax, ox - 5, oy + oh + 1.4, "preferred", size=8.5, color=INDIGO,
        weight='bold', style='italic', ha='center')

    txt(ax, 50, 8.0,
        "No GPU  \u00b7  no network during ranking  \u00b7  $0 API cost  \u00b7  "
        "CTAE is standard-library only",
        size=9.5, color=MUTED)

    fig.savefig(os.path.join(OUT, 'architecture_fallback.png'),
                bbox_inches='tight', pad_inches=0.18, facecolor='#ffffff')
    plt.close(fig)
    print("wrote architecture_fallback.png")


if __name__ == '__main__':
    diagram_pipeline()
    diagram_fallback()
    print("Done ->", OUT)
