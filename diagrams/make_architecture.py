"""
Generate two clean, slide-ready (16:9) architecture diagrams as PNGs:

  1. architecture_pipeline.png  — the full 4-stage SHRE pipeline (horizontal,
     JD feeding in, CTAE fallback strip), laid out to fit a widescreen slide.
  2. architecture_fallback.png  — the LTR -> Ensemble -> CTAE reliability chain.

Pure matplotlib, no extra deps. Run:  python diagrams/make_architecture.py
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
LINE     = '#cbd5e1'

INDIGO   = '#4f46e5'; INDIGO_BG = '#eef0fe'
TEAL     = '#0d9488'; TEAL_BG   = '#e6f5f3'
AMBER    = '#d97706'; AMBER_BG  = '#fdf3e3'
BLUE     = '#2563eb'; BLUE_BG   = '#e8f0fe'
GREEN    = '#059669'; GREEN_BG  = '#e7f7ef'
SLATE_BG = '#f1f5f9'
PURPLE   = '#7c3aed'; PURPLE_BG = '#f3ecfe'

plt.rcParams['font.family'] = 'DejaVu Sans'


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def rbox(ax, x, y, w, h, fc='#ffffff', ec=LINE, lw=1.4, rounding=1.1, z=2):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z,
        mutation_aspect=1.0)
    ax.add_patch(p)
    return p


def txt(ax, x, y, s, size=11, color=INK, weight='normal', ha='center',
        va='center', z=5, style='normal'):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=z, style=style, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=MUTED, lw=2.2, style='-|>', dashed=False,
          rad=0.0, z=3, ms=14):
    ap = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
        linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}",
        linestyle='--' if dashed else '-', zorder=z,
        shrinkA=0, shrinkB=0)
    ax.add_patch(ap)


def stage_lane(ax, x, y, w, h, num, title, accent, accent_bg, bullets):
    """A stage lane: colored header with number+title, white body w/ bullets."""
    rbox(ax, x, y, w, h, fc='#ffffff', ec=accent, lw=1.8, rounding=1.0, z=2)
    # header band
    hh = 6.2
    header = FancyBboxPatch(
        (x, y + h - hh), w, hh,
        boxstyle="round,pad=0,rounding_size=1.0",
        linewidth=0, facecolor=accent_bg, zorder=3)
    ax.add_patch(header)
    # square off bottom of header so it reads like a banner
    ax.add_patch(FancyBboxPatch((x, y + h - hh), w, hh * 0.55,
                 boxstyle="square,pad=0", linewidth=0,
                 facecolor=accent_bg, zorder=3))
    # stage number chip
    chip = 3.0
    rbox(ax, x + 1.4, y + h - hh + (hh - chip) / 2, chip, chip,
         fc=accent, ec=accent, lw=0, rounding=0.7, z=4)
    txt(ax, x + 1.4 + chip / 2, y + h - hh + hh / 2, str(num),
        size=13, color='#ffffff', weight='bold', z=5)
    # title
    txt(ax, x + 1.4 + chip + 1.0, y + h - hh / 2, title, size=11.3,
        color=INK, weight='bold', ha='left', z=5)
    # bullets — lines starting with whitespace are wrap-continuations (no dot)
    by = y + h - hh - 2.9
    for b in bullets:
        is_cont = b.startswith(' ')
        if not is_cont:
            txt(ax, x + 1.6, by, "\u2022", size=11, color=accent, ha='left', weight='bold')
        txt(ax, x + 3.0, by, b.strip() if not is_cont else b.lstrip(),
            size=9.5, color=MUTED, ha='left')
        by -= 3.3 if is_cont else 3.6


# ===========================================================================
# DIAGRAM 1 — full pipeline (16:9)
# ===========================================================================
def diagram_pipeline():
    W, H = 100.0, 56.25
    fig, ax = plt.subplots(figsize=(13.333, 7.5), dpi=240)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    fig.patch.set_facecolor('#ffffff')

    # ---- title
    txt(ax, 3.2, 53.2, "Staged Hybrid Ranking Engine (SHRE)", size=17,
        color=INK, weight='bold', ha='left')
    txt(ax, 3.2, 49.6, "4-stage hybrid pipeline  ·  anomaly filter → 93 features → ensemble + LTR → grounded reasoning",
        size=10.5, color=MUTED, ha='left')

    # ---- geometry
    lane_y, lane_h = 12.0, 27.0
    lane_w = 18.5
    xs = [11.5, 32.0, 52.5, 73.0]         # stage lane left edges
    in_x, in_w = 0.8, 9.0                  # input box
    out_x, out_w = 92.2, 7.2               # output box (narrow, tall)

    # ---- input (candidates)
    iy, ih = 20.0, 11.0
    rbox(ax, in_x, iy, in_w, ih, fc=INK, ec=INK, lw=0, rounding=1.0, z=2)
    txt(ax, in_x + in_w / 2, iy + ih / 2 + 1.7, "candidates", size=11,
        color='#ffffff', weight='bold')
    txt(ax, in_x + in_w / 2, iy + ih / 2 - 1.3, ".jsonl", size=10,
        color='#ffffff', weight='bold')
    txt(ax, in_x + in_w / 2, iy + ih / 2 - 4.2, "100k+ profiles", size=8.5,
        color='#cbd5e1')

    # ---- stage lanes
    stage_lane(ax, xs[0], lane_y, lane_w, lane_h, 1, "Anomaly Pre-Filter",
               AMBER, AMBER_BG,
               ["Timeline / skill /", "  synthetic checks",
                "Gate: JD exp band", "  + \u22652 skill pillars",
                "Drops honeypots"])
    stage_lane(ax, xs[1], lane_y, lane_w, lane_h, 2, "Feature Engineering",
               BLUE, BLUE_BG,
               ["78 base signals", "  (career/domain/co.)",
                "+5 anomaly +5", "  behavioral",
                "+5 multi-vector", "  semantic"])
    stage_lane(ax, xs[2], lane_y, lane_w, lane_h, 3, "Ensemble + LTR",
               INDIGO, INDIGO_BG,
               ["XGB + LGBM + Cat", "  (leakage-safe SMOTE)",
                "LambdaMART", "  (XGBRanker ndcg)",
                "Fuse 0.6 ens", "  + 0.4 LTR"])
    stage_lane(ax, xs[3], lane_y, lane_w, lane_h, 4, "Ranker + Reasoning",
               GREEN, GREEN_BG,
               ["Sort \u2192 Top 100", "Grounded, non-",
                "  hallucinated why", "submission .csv /",
                "  .xlsx / detailed"])

    # ---- output
    oy, oh = 16.0, 19.0
    rbox(ax, out_x, oy, out_w, oh, fc=GREEN, ec=GREEN, lw=0, rounding=1.0, z=2)
    txt(ax, out_x + out_w / 2, oy + oh / 2 + 4.0, "Top", size=12,
        color='#ffffff', weight='bold')
    txt(ax, out_x + out_w / 2, oy + oh / 2 + 0.3, "100", size=20,
        color='#ffffff', weight='bold')
    txt(ax, out_x + out_w / 2, oy + oh / 2 - 4.2, "ranked +\nreasoning",
        size=8.2, color='#dcfce7')

    # ---- JD banner (top), feeds stage 1 & 2
    jd_x, jd_y, jd_w, jd_h = 11.5, 42.5, 39.0, 5.6
    rbox(ax, jd_x, jd_y, jd_w, jd_h, fc=PURPLE_BG, ec=PURPLE, lw=1.6, rounding=0.9, z=2)
    txt(ax, jd_x + 2.0, jd_y + jd_h / 2, "Job Description  (--jd)", size=10,
        color=PURPLE, weight='bold', ha='left')
    txt(ax, jd_x + jd_w - 1.8, jd_y + jd_h / 2,
        "parsed \u2192 3 semantic facets  +  experience band",
        size=9, color=MUTED, ha='right')

    # ---- flow arrows (main spine)
    midy = lane_y + lane_h / 2
    arrow(ax, in_x + in_w, iy + ih / 2, xs[0], midy, color=INK, lw=2.6)
    for i in range(3):
        arrow(ax, xs[i] + lane_w, midy, xs[i + 1], midy, color=MUTED, lw=2.6)
    arrow(ax, xs[3] + lane_w, midy, out_x, oy + oh / 2, color=GREEN, lw=2.6)

    # ---- JD dashed feeds into stage 1 (gate) & stage 2 (semantic)
    arrow(ax, jd_x + 8, jd_y, xs[0] + lane_w / 2, lane_y + lane_h,
          color=PURPLE, lw=1.8, dashed=True, rad=-0.15)
    arrow(ax, jd_x + 30, jd_y, xs[1] + lane_w / 2, lane_y + lane_h,
          color=PURPLE, lw=1.8, dashed=True, rad=0.12)

    # ---- fallback strip (bottom)
    fb_y, fb_h = 2.2, 6.4
    fb_x, fb_w = 11.5, 79.7
    rbox(ax, fb_x, fb_y, fb_w, fb_h, fc=SLATE_BG, ec=LINE, lw=1.4, rounding=0.8, z=1)
    txt(ax, fb_x + 2.2, fb_y + fb_h / 2, "Reliability fallback",
        size=9.5, color=INK, weight='bold', ha='left')
    txt(ax, fb_x + fb_w / 2 + 6, fb_y + fb_h / 2,
        "LambdaMART   \u2192   Validated Ensemble   \u2192   Pure-Python CTAE",
        size=9.5, color=MUTED, ha='center')
    txt(ax, fb_x + fb_w - 2.0, fb_y + fb_h / 2, "never fails",
        size=8.5, color=GREEN, weight='bold', ha='right', style='italic')
    # small hook from fallback up to output
    arrow(ax, fb_x + fb_w, fb_y + fb_h / 2, out_x + out_w / 2, oy,
          color=FAINT, lw=1.6, dashed=True, rad=-0.3)

    fig.savefig(os.path.join(OUT, 'architecture_pipeline.png'),
                bbox_inches='tight', pad_inches=0.15, facecolor='#ffffff')
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

    txt(ax, 50, 50.5, "Fallback Reliability Chain", size=18, color=INK,
        weight='bold')
    txt(ax, 50, 45.6,
        "Every layer can degrade gracefully — the engine always returns a Top-100 shortlist",
        size=11, color=MUTED)

    boxes = [
        ("LambdaMART", "LTR head", INDIGO, INDIGO_BG, "preferred"),
        ("Validated", "Voting Ensemble", BLUE, BLUE_BG, "on failure"),
        ("Pure-Python", "CTAE ranker", AMBER, AMBER_BG, "on failure"),
    ]
    bw, bh = 21.0, 13.0
    by = 20.0
    xs = [6.0, 33.0, 60.0]

    for i, (t1, t2, ac, bg, _) in enumerate(boxes):
        x = xs[i]
        rbox(ax, x, by, bw, bh, fc=bg, ec=ac, lw=2.2, rounding=1.1, z=2)
        rbox(ax, x, by + bh - 2.6, bw, 2.6, fc=ac, ec=ac, lw=0, rounding=1.0, z=3)
        txt(ax, x + bw / 2, by + bh - 1.3, f"Layer {i + 1}", size=9,
            color='#ffffff', weight='bold', z=4)
        txt(ax, x + bw / 2, by + bh / 2 + 0.6, t1, size=13, color=INK, weight='bold')
        txt(ax, x + bw / 2, by + bh / 2 - 3.0, t2, size=11, color=MUTED)

    # arrows between layers (labeled "on failure")
    for i in range(2):
        x1 = xs[i] + bw
        x2 = xs[i + 1]
        arrow(ax, x1, by + bh / 2, x2, by + bh / 2, color=MUTED, lw=2.6)
        txt(ax, (x1 + x2) / 2, by + bh / 2 + 2.6, "on failure", size=8.5,
            color=FAINT, weight='bold', style='italic')

    # output box on right
    ox, ow, oh = 84.0, 12.5, 17.0
    oy = by - 2.0
    rbox(ax, ox, oy, ow, oh, fc=GREEN, ec=GREEN, lw=0, rounding=1.1, z=2)
    txt(ax, ox + ow / 2, oy + oh / 2 + 3.2, "Top-100", size=13,
        color='#ffffff', weight='bold')
    txt(ax, ox + ow / 2, oy + oh / 2 - 0.6, "shortlist", size=11, color='#ffffff')
    txt(ax, ox + ow / 2, oy + oh / 2 - 4.2, "(identical\nCSV schema)", size=8.2,
        color='#dcfce7')

    # each layer -> output
    styles = [(INDIGO, '-', 0.0, 2.6), (BLUE, '--', -0.18, 1.6), (AMBER, '--', -0.30, 1.6)]
    for i, (col, ls, rad, lw) in enumerate(styles):
        arrow(ax, xs[i] + bw / 2, by, ox, oy + oh / 2,
              color=col, lw=lw, dashed=(ls == '--'), rad=rad)
    txt(ax, ox - 6, oy + oh + 1.2, "preferred", size=8.5, color=INDIGO,
        weight='bold', style='italic', ha='center')

    # footnote
    txt(ax, 50, 8.0,
        "No GPU  ·  no network during ranking  ·  $0 API cost  ·  CTAE is standard-library only",
        size=9.5, color=MUTED)

    fig.savefig(os.path.join(OUT, 'architecture_fallback.png'),
                bbox_inches='tight', pad_inches=0.15, facecolor='#ffffff')
    plt.close(fig)
    print("wrote architecture_fallback.png")


if __name__ == '__main__':
    diagram_pipeline()
    diagram_fallback()
    print("Done ->", OUT)
