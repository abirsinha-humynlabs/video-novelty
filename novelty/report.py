"""Self-contained HTML report: what the index looks like, at a glance.

One file, no network, no build step, no JS libraries. Open it in a browser or
mail it to someone. Three views, in the order you should read them:

1. the similarity matrix (sequential blue heatmap) -- is anything actually
   similar to anything?
2. the coverage curve -- how fast do extra clips stop paying for themselves?
3. the marginal-gain bars -- which specific clips are doing the work?

Deliberately separate charts rather than one chart with two y-axes: coverage
fraction and marginal gain have unrelated scales, and overlaying them invents a
crossing point that means nothing.
"""
from __future__ import annotations

import html
import json
import os
from typing import List, Optional, Sequence

import numpy as np

# ---- design tokens (validated palette; see docs/07-tuning.md) ---------------
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--surface-2:#f4f3f0;--line:#e2e1dc;
--ink:#0b0b0b;--ink-2:#52514e;--ink-3:#84837d;--s1:#2a78d6;--s2:#eb6834;--grid:#eceae5;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--surface:#1a1a19;--surface-2:#232322;--line:#3a3a37;--ink:#fff;--ink-2:#c3c2b7;
--ink-3:#8f8e85;--s1:#3987e5;--s2:#d95926;--grid:#2c2c2a;}}
:root[data-theme="dark"]{color-scheme:dark;--surface:#1a1a19;--surface-2:#232322;
--line:#3a3a37;--ink:#fff;--ink-2:#c3c2b7;--ink-3:#8f8e85;--s1:#3987e5;--s2:#d95926;--grid:#2c2c2a;}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);
font:14px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.wrap{max-width:1000px;margin:0 auto;padding:40px 16px 80px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.02em}
h2{font-size:17px;margin:44px 0 2px;letter-spacing:-.01em}
p.sub{color:var(--ink-2);margin:0 0 18px;max-width:62ch}
p.note{color:var(--ink-3);font-size:12.5px;margin:6px 0 0;max-width:70ch}
.card{background:var(--surface-2);border:1px solid var(--line);border-radius:12px;padding:18px;margin-top:14px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}
th{text-align:left;color:var(--ink-2);font-weight:600;padding:6px 10px 6px 0;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:5px 10px 5px 0;border-bottom:1px solid var(--grid);white-space:nowrap}
.tag{display:inline-block;padding:1px 7px;border-radius:999px;font-size:11px;border:1px solid var(--line);color:var(--ink-2)}
.kpi{display:flex;gap:28px;flex-wrap:wrap;margin:18px 0 0}
.kpi div span{display:block}
.kpi .v{font-size:24px;font-weight:650;letter-spacing:-.02em}
.kpi .k{font-size:12px;color:var(--ink-2)}
.warn{border-left:3px solid var(--s2);padding:10px 14px;background:var(--surface-2);
border-radius:0 8px 8px 0;margin-top:14px;color:var(--ink-2);font-size:13px}
svg{display:block;max-width:100%;height:auto;overflow:visible}
svg text{fill:var(--ink-2);font:11px ui-sans-serif,-apple-system,sans-serif}
svg .ax{stroke:var(--line);stroke-width:1}
svg .gl{stroke:var(--grid);stroke-width:1}
svg .hit{fill:transparent;cursor:crosshair}
svg .hit:hover+ *{opacity:1}
#tip{position:fixed;pointer-events:none;opacity:0;transform:translate(-50%,-140%);
background:var(--ink);color:var(--surface);padding:6px 9px;border-radius:7px;
font-size:12px;white-space:pre;z-index:9;transition:opacity .08s}
"""

JS = """
const tip=document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el=>{
  el.addEventListener('mousemove',e=>{tip.textContent=el.dataset.tip;tip.style.opacity=1;
    tip.style.left=e.clientX+'px';tip.style.top=e.clientY+'px';});
  el.addEventListener('mouseleave',()=>{tip.style.opacity=0;});
});
"""


def _esc(s) -> str:
    return html.escape(str(s))


def _seq_color(v: float) -> str:
    v = 0.0 if not np.isfinite(v) else float(np.clip(v, 0, 1))
    return SEQ[int(round(v * (len(SEQ) - 1)))]


# --------------------------------------------------------------------------- charts
def heatmap_svg(S: np.ndarray, labels: Sequence[str], cell: int = 30) -> str:
    n = len(labels)
    pad_l, pad_t = 210, 14
    w, h = pad_l + n * cell + 16, pad_t + n * cell + 26
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="pairwise similarity matrix">']
    for i in range(n):
        y = pad_t + i * cell
        out.append(f'<text x="{pad_l - 8}" y="{y + cell/2 + 4}" text-anchor="end">{_esc(labels[i][:34])}</text>')
        for j in range(n):
            x = pad_l + j * cell
            v = float(S[i, j])
            tip = labels[i] + "\n" + labels[j] + f"\nsimilarity {v:.3f}"
            # 2px surface gap between cells keeps adjacent fills readable
            out.append(
                f'<rect data-tip="{_esc(tip)}" x="{x+1}" y="{y+1}" width="{cell-2}" '
                f'height="{cell-2}" rx="3" fill="{_seq_color(v)}"/>'
            )
    out.append(f'<text x="{pad_l}" y="{h - 8}">each cell = similarity between two signatures '
               f'(light = distinct, dark = near-identical)</text>')
    out.append("</svg>")
    return "".join(out)


def _axes(out, x0, y0, w, h, yticks, ylabel_fmt):
    out.append(f'<line class="ax" x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}"/>')
    for t in yticks:
        y = y0 + h - t * h
        out.append(f'<line class="gl" x1="{x0}" y1="{y:.1f}" x2="{x0+w}" y2="{y:.1f}"/>')
        out.append(f'<text x="{x0-8}" y="{y+4:.1f}" text-anchor="end">{ylabel_fmt(t)}</text>')


def coverage_svg(frac: Sequence[float], labels: Sequence[str], knee: int) -> str:
    n = len(frac)
    if n == 0:
        return "<p>nothing selected</p>"
    x0, y0, w, h = 52, 14, 620, 190
    out = [f'<svg viewBox="0 0 {x0+w+80} {y0+h+40}" role="img" aria-label="coverage curve">']
    _axes(out, x0, y0, w, h, [0, .25, .5, .75, 1.0], lambda t: f"{t*100:.0f}%")
    step = w / max(n - 1, 1)
    pts = [(x0 + i * step, y0 + h - float(frac[i]) * h) for i in range(n)]
    d = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
    out.append(f'<path d="{d}" fill="none" stroke="var(--s1)" stroke-width="2" '
               f'stroke-linejoin="round" stroke-linecap="round"/>')
    nl = "\n"
    for i, (x, y) in enumerate(pts):
        tip = _esc(f"#{i+1} {labels[i]}{nl}cumulative coverage {frac[i]*100:.1f}%")
        out.append(f'<circle data-tip="{tip}" cx="{x:.1f}" cy="{y:.1f}" r="4.5" '
                   f'fill="var(--s1)" stroke="var(--surface-2)" stroke-width="2"/>')
    if 0 < knee <= n:
        kx = x0 + (knee - 1) * step
        out.append(f'<line x1="{kx:.1f}" y1="{y0}" x2="{kx:.1f}" y2="{y0+h}" stroke="var(--s2)" '
                   f'stroke-width="2" stroke-dasharray="4 4"/>')
        out.append(f'<text x="{kx+6:.1f}" y="{y0+12}" style="fill:var(--s2)">knee: {knee} clips</text>')
    lx, ly = pts[-1]
    out.append(f'<text x="{lx+8:.1f}" y="{ly+4:.1f}" style="fill:var(--ink)">{frac[-1]*100:.1f}% covered</text>')
    out.append(f'<text x="{x0}" y="{y0+h+26}">clips selected, in greedy order &rarr;</text>')
    out.append("</svg>")
    return "".join(out)


def gains_svg(gains: Sequence[float], labels: Sequence[str]) -> str:
    n = len(gains)
    if n == 0:
        return ""
    x0, y0, w, h = 52, 14, 620, 150
    mx = max(max(gains), 1e-9)
    out = [f'<svg viewBox="0 0 {x0+w+80} {y0+h+40}" role="img" aria-label="marginal gain per pick">']
    _axes(out, x0, y0, w, h, [0, .5, 1.0], lambda t: f"{t*mx:.2f}")
    bw = min(w / max(n, 1) - 2, 34)
    nl = "\n"
    for i, g in enumerate(gains):
        bh = max(float(g) / mx * h, 1.5)
        x = x0 + i * (w / max(n, 1)) + 1
        tip = _esc(f"#{i+1} {labels[i]}{nl}marginal gain {g:.4f}")
        out.append(f'<rect data-tip="{tip}" x="{x:.1f}" y="{y0+h-bh:.1f}" '
                   f'width="{bw:.1f}" height="{bh:.1f}" rx="4" fill="var(--s1)"/>')
    out.append(f'<text x="{x0}" y="{y0+h+26}">marginal coverage added by each pick &rarr;</text>')
    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------- page
def write_report(index, out_path: str, *, kind: str = "fused", budget: Optional[int] = None) -> str:
    from .select import facility_location_greedy

    sigs = index.signatures()
    labels = [s.label for s in sigs]
    null = index.null()
    S = index.similarity_matrix(kind=kind, null=null)
    res = facility_location_greedy(S, budget=budget)
    frac = res.fraction_covered()
    order_labels = [labels[i] for i in res.order]
    knee = res.knee()

    off = S[~np.eye(len(S), dtype=bool)] if len(S) > 1 else np.zeros(1)
    body = [
        '<div class="wrap">',
        "<h1>novelty &mdash; index report</h1>",
        f'<p class="sub">{len(sigs)} signature(s) from '
        f'{len({s.video_id for s in sigs})} source file(s). Similarity kind: '
        f'<b>{_esc(kind)}</b>. '
        f'{"Calibrated against a corpus null model." if null else "<b>UNCALIBRATED</b> — run <code>novelty calibrate</code>."}</p>',
        '<div class="kpi">',
        f'<div><span class="v">{len(sigs)}</span><span class="k">signatures</span></div>',
        f'<div><span class="v">{float(off.mean()):.3f}</span><span class="k">mean pairwise similarity</span></div>',
        f'<div><span class="v">{float(off.max()):.3f}</span><span class="k">most-similar pair</span></div>',
        f'<div><span class="v">{knee}</span><span class="k">clips to the knee</span></div>',
        f'<div><span class="v">{(null.n_pairs if null else 0)}</span><span class="k">null pairs fitted</span></div>',
        "</div>",
    ]
    if null is None:
        body.append('<div class="warn">No null model. Every number on this page is a raw score '
                    'on an uncalibrated scale, which means the thresholds are arbitrary. '
                    'Run <code>novelty calibrate</code> and regenerate.</div>')
    elif len(sigs) < 50:
        body.append(f'<div class="warn">The null model and whitener were fitted on only '
                    f'{len(sigs)} signatures. Treat these percentiles as indicative; they '
                    f'need a few hundred signatures to mean anything.</div>')

    body += [
        "<h2>Pairwise similarity</h2>",
        '<p class="sub">Darker means more alike. Block structure along the diagonal is what '
        'redundancy looks like: a cluster of clips that are all the same to your model.</p>',
        f'<div class="card">{heatmap_svg(S, labels)}</div>',

        "<h2>How fast does new footage stop paying?</h2>",
        '<p class="sub">Greedy facility-location coverage. Each point adds the clip that best '
        'represents whatever is still poorly covered. A curve that flattens early means the '
        'corpus is repetitive &mdash; and tells you exactly how much of it you actually need.</p>',
        f'<div class="card">{coverage_svg(frac, order_labels, knee)}</div>',
        f'<p class="note">Knee at {knee} of {len(res.order)}: past this point each extra clip '
        f'adds under 1% of total coverage.</p>',

        "<h2>Which clips are doing the work</h2>",
        f'<div class="card">{gains_svg(res.gains, order_labels)}</div>',

        "<h2>Selection order</h2>",
        '<div class="card"><table><thead><tr><th>#</th><th>clip</th>'
        '<th>marginal gain</th><th>cumulative coverage</th><th>cycle</th></tr></thead><tbody>',
    ]
    for k, i in enumerate(res.order):
        s = sigs[i]
        cyc = f"{s.period_s:.1f}s @ {s.period_strength:.2f}" if s.period_s > 0 else "&mdash;"
        tag = ' <span class="tag">knee</span>' if k + 1 == knee else ""
        body.append(f"<tr><td>{k+1}</td><td>{_esc(s.label)}{tag}</td>"
                    f"<td>{res.gains[k]:.4f}</td><td>{frac[k]*100:.1f}%</td><td>{cyc}</td></tr>")
    body += ["</tbody></table></div>", "</div>", '<div id="tip"></div>']

    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>novelty index report</title>"
        f"<style>{CSS}</style></head><body>" + "".join(body) +
        f"<script>{JS}</script></body></html>"
    )
    with open(out_path, "w") as fh:
        fh.write(page)
    return os.path.abspath(out_path)
