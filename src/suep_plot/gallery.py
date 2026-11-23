"""The index.html thumbnail gallery of a figure directory."""

from __future__ import annotations

from pathlib import Path


def write_gallery(output_dir: str, formats=None, since: float | None = None, *,
                  ranked=None, title: str | None = None, subtitle: str = "",
                  score_label: str = "score") -> str | None:
    """Write an index.html thumbnail gallery of the PNGs under *output_dir*.

    Figures are grouped by folder (a region histogram ``csc/iso_muon/x`` sits
    under ``csc/iso_muon``), and a filter box narrows the page to the figures
    whose name contains every word typed.  With *since*, only figures written
    since then are shown -- a figure left from an earlier run of a histogram
    that no longer exists does not reappear.  A thumbnail links to its PDF
    only when this run wrote PDFs (*formats*), never to a stale one.

    *ranked* -- ``[(name, score)]``, as the comparison scripts produce -- shows
    exactly those figures, in that order, with their score in the caption.
    """
    import html as _html

    out = Path(output_dir)
    link_pdf = formats is None or "pdf" in formats
    if ranked is not None:
        scores = {str(name): score for name, score in ranked}
        groups = {"": [out / f"{name}.png" for name, _ in ranked
                       if (out / f"{name}.png").exists()]}
    else:
        scores = {}
        groups = {}
        for png in sorted(p for p in out.rglob("*.png")
                          if since is None or p.stat().st_mtime >= since - 1):
            groups.setdefault(png.parent.relative_to(out).as_posix(), []).append(png)
    pngs = [p for g in groups.values() for p in g]
    if not pngs:
        return None

    sections = []
    for folder in (groups if ranked is not None else sorted(groups)):
        cards = []
        for png in groups[folder]:
            rel = png.relative_to(out).as_posix()
            pdf = png.with_suffix(".pdf")
            fresh_pdf = link_pdf and pdf.exists() and \
                (since is None or pdf.stat().st_mtime >= since - 1)
            href = pdf.relative_to(out).as_posix() if fresh_pdf else rel
            key = rel[:-len(".png")]
            score = (f'<br><span class="score">{score_label} {scores[key]:.3f}</span>'
                     if key in scores else "")
            cards.append(
                f'<figure data-name="{_html.escape(rel.lower())}"><a href="{href}">'
                f'<img src="{_html.escape(rel)}" loading="lazy" alt="{png.stem}"></a>'
                f'<figcaption>{_html.escape(key if ranked is not None else png.stem)}'
                f'{score}</figcaption></figure>')
        heading = "" if ranked is not None else (
            f'<h2>{_html.escape("(top level)" if folder == "." else folder)} '
            f'<small>({len(cards)})</small></h2>\n')
        sections.append(f'<section>{heading}<div class="grid">\n'
                        + "\n".join(cards) + "\n</div></section>")

    cutflow_link = ('<p><a href="cutflow.txt">cutflow.txt</a> · '
                    '<a href="cutflow.csv">cutflow.csv</a></p>'
                    if (out / "cutflow.txt").exists() else "")
    if subtitle:
        cutflow_link += f"<p>{_html.escape(subtitle)}</p>"
    heading_title = _html.escape(title or out.name)

    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>suep-plot: {_html.escape(out.name)}</title>
<style>
 body {{ font-family: sans-serif; margin: 1.5em; }}
 #filter {{ font-size: 1.1em; padding: 4px 8px; width: min(40em, 90%); }}
 h2 {{ font-family: monospace; margin-top: 1.5em; }}
 .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1em; }}
 figure {{ margin: 0; border: 1px solid #ddd; border-radius: 6px; padding: 6px; }}
 img {{ width: 100%; height: auto; }}
 figcaption {{ text-align: center; font-size: 0.85em; padding-top: 4px;
               font-family: monospace; word-break: break-all; }}
 .score {{ color: #666; }}
</style></head><body>
<h1>{heading_title} ({len(pngs)} plots)</h1>
{cutflow_link}
<p><input id="filter" placeholder="filter: words that must all appear in the name, e.g. iso_muon size" autofocus>
 <span id="count"></span></p>
{chr(10).join(sections)}
<script>
const box = document.getElementById("filter"), count = document.getElementById("count");
box.addEventListener("input", () => {{
  const words = box.value.toLowerCase().split(/\\s+/).filter(Boolean);
  let shown = 0;
  document.querySelectorAll("section").forEach(sec => {{
    let n = 0;
    sec.querySelectorAll("figure").forEach(fig => {{
      const ok = words.every(w => fig.dataset.name.includes(w));
      fig.style.display = ok ? "" : "none";
      if (ok) n++;
    }});
    sec.style.display = n ? "" : "none";
    shown += n;
  }});
  count.textContent = words.length ? shown + " shown" : "";
}});
</script>
</body></html>
"""
    path = out / "index.html"
    path.write_text(page)
    return str(path)
