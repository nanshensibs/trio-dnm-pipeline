"""Self-contained HTML report (no external assets)."""
from __future__ import annotations

import html
from typing import Dict, List

from .genome import SBS96

CSS = """
:root{--bg:#fff;--fg:#1d2433;--muted:#5b6475;--line:#d9dee7;--accent:#1f6f8b;--warn:#b54708;--ok:#2e7d32;--head:#e8f1f5;--sbs-cg:#010101}
@media (prefers-color-scheme:dark){:root{--bg:#12161c;--fg:#e6e9ef;--muted:#9aa3b2;--line:#2c3440;--accent:#5fb3d0;--warn:#f0a35e;--ok:#7bc67e;--head:#1c2630;--sbs-cg:#9aa3b2}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;margin:0 auto;max-width:1200px;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.muted{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.card{border:1px solid var(--line);border-radius:8px;padding:10px 12px}.card b{display:block;font-size:22px}
table{border-collapse:collapse;width:100%;font-size:12.5px}th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top}
th{background:var(--head);position:sticky;top:0}.scroll{overflow-x:auto;max-height:520px}
.warn{color:var(--warn)}.ok{color:var(--ok)}svg text{fill:var(--muted);font-size:9px}
svg .sbs1{fill:var(--sbs-cg)}
"""

SBS_COLORS = ["#03bcee", "#010101", "#e32926", "#cac9c9", "#a1ce63", "#ebc6c4"]


def _esc(x) -> str:
    return html.escape(str(x))


def _pct(v) -> str:
    return f"{v:.0%}" if isinstance(v, (int, float)) else str(v)


def spectrum_svg(counts: Dict[str, int]) -> str:
    w, h, pad = 960, 140, 20
    mx = max(counts.values()) or 1
    bw = (w - 2 * pad) / 96
    bars = []
    for i, ch in enumerate(SBS96):
        v = counts.get(ch, 0)
        bh = (h - 40) * v / mx
        color = SBS_COLORS[i // 16]
        # class sbsN lets CSS recolour the near-black C>G block in dark mode
        bars.append(f'<rect class="sbs{i // 16}" x="{pad + i * bw:.1f}" y="{h - 20 - bh:.1f}" width="{bw * 0.8:.1f}" '
                    f'height="{bh:.1f}" fill="{color}"><title>{_esc(ch)}: {_esc(v)}</title></rect>')
    labels = "".join(
        f'<text x="{pad + (k * 16 + 8) * bw:.0f}" y="{h - 6}" text-anchor="middle">{_esc(m)}</text>'
        for k, m in enumerate(["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"])
    )
    return f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="SBS-96 spectrum">{"".join(bars)}{labels}</svg>'


def write_html(path: str, result: Dict, rows: List[Dict]) -> None:
    cs = result.get("call_summary") or {}
    san = result.get("sanity") or {}
    wf = cs.get("waterfall") or {}
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Trio DNM Report</title><style>{CSS}</style></head><body>"]
    parts.append(f"<h1>Trio DNM report — {_esc(cs.get('proband', ''))}</h1>")
    parts.append(f"<div class='muted'>father {_esc(cs.get('father', ''))} · mother {_esc(cs.get('mother', ''))} · proband sex {_esc(cs.get('proband_sex', ''))}</div>")
    status = san.get("status", "")
    parts.append("<div class='cards' style='margin-top:16px'>")
    for label, val in [
        ("Germline SNV DNMs", san.get("n_germline_snv")), ("Germline indel DNMs", san.get("n_germline_indel")),
        ("Post-zygotic mosaic", san.get("n_mosaic")), ("Parental mosaic", san.get("n_parental_mosaic")),
        ("Ti/Tv", san.get("titv")), ("CpG transition fraction", san.get("cpg_transition_fraction")),
        ("Paternal fraction", san.get("paternal_fraction")), ("Sanity status", status),
    ]:
        parts.append(f"<div class='card'><span class='muted'>{_esc(label)}</span><b>{_esc('—' if val is None else val)}</b></div>")
    parts.append("</div>")
    if san.get("warnings"):
        parts.append("<h2>Sanity-check warnings</h2><ul>" + "".join(f"<li class='warn'>{_esc(w)}</li>" for w in san["warnings"]) + "</ul>")
    if wf:
        parts.append("<h2>Filter waterfall</h2><table><tr><th>Step</th><th>Remaining</th></tr>")
        for k, v in wf.items():
            parts.append(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>")
        parts.append("</table>")
        gate = cs.get("mendelian_error_gate")
        if gate:
            rate = cs.get("raw_mendelian_error_rate")
            cls = "warn" if gate == "FAIL" else "muted"
            parts.append(f"<p class='{cls}'>Raw Mendelian error gate: {_esc(gate)} (rate {_esc(rate)}, "
                         f"{_esc(cs.get('raw_mendelian_informative_sites'))} informative sites).</p>")
        if cs.get("population_af_missing"):
            parts.append("<p class='warn'>No gnomAD allele-frequency annotation was found in the input VCF: the population filter (Layer 4) was inactive.</p>")
        if cs.get("second_engine_only_candidates"):
            parts.append(f"<p class='muted'>Candidates found only by the second calling engine: {_esc(cs['second_engine_only_candidates'])}.</p>")
        lk = cs.get("parental_leakage") or {}
        if lk:
            parts.append(f"<p class='muted'>Parental ALT-read leakage at germline DNMs: father {_esc(lk.get('father_any_alt_frac', '—'))}, "
                         f"mother {_esc(lk.get('mother_any_alt_frac', '—'))} (n={_esc(lk.get('n'))}).</p>")
    if san.get("sbs96"):
        parts.append("<h2>Mutational spectrum (SBS-96, germline SNV DNMs)</h2>" + spectrum_svg(san["sbs96"]))
        sig = san.get("signatures")
        if sig:
            parts.append(f"<p class='muted'>Signature refit (cosine {_esc(sig.get('cosine'))}): "
                         + ", ".join(f"{_esc(k)} {_esc(_pct(v))}" for k, v in sig.get("exposures", {}).items()) + "</p>")
    parts.append("<h2>Prioritised variants</h2><div class='scroll'><table><tr>")
    cols = ["priority_rank", "priority_tier", "acmg_class", "acmg_codes", "gene", "hgvsp", "consequence", "track", "dnm_tier",
            "proband_vaf", "REVEL", "AlphaMissense", "spliceai", "LOEUF", "disease", "flags"]
    parts.append("".join(f"<th>{_esc(c)}</th>" for c in cols) + "</tr>")
    for r in rows[:500]:
        parts.append("<tr>" + "".join(f"<td>{_esc(r.get(c, ''))}</td>" for c in cols) + "</tr>")
    parts.append("</table></div>")
    parts.append("<p class='muted'>ACMG classes are provisional, automated from annotation only, and require expert curation before clinical reporting.</p>")
    parts.append("</body></html>")
    with open(path, "w") as fh:
        fh.write("".join(parts))
