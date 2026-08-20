# ============================================================
# DASHBOARD — builds the homepage (dashboard.html) that links the
# Clinical Pipeline (pipeline_viz.py's pipeline_overview.html) and the
# Biomarker Dashboard (adni_viz.py's biomarker_dashboard.html), and
# builds the Biomarker Dashboard itself from the aggregate-only
# outputs/ files (see adni_viz.py / adni_viz_data.py).
#
# Does NOT re-run pipeline_viz.py — a full ClinicalTrials.gov refresh
# is expensive and stays run_pipeline.py's job. This script only
# combines whatever pipeline_overview.html already exists on disk; if
# it's missing, the homepage says so instead of linking to a 404.
#
# All three pages (dashboard.html, pipeline_overview.html,
# biomarker_dashboard.html) stay independently openable in a browser —
# each carries dashboard_nav.py's same top nav bar, so moving between
# them feels like tabs, without inlining pipeline_overview.html's
# multi-megabyte standalone document (its own embedded Plotly bundle,
# its own top-level JS globals) into a shared page where it would
# collide with anything else on it.
# ============================================================

import os

import adni_viz
from dashboard_nav import NAV_CSS, render_nav_bar

PIPELINE_HTML = "pipeline_overview.html"
BIOMARKER_HTML = adni_viz.OUTPUT_HTML
DASHBOARD_HTML = "dashboard.html"

ARIBIO_BLUE = "#2e5fa3"
ARIBIO_ACCENT = "#c2255c"
CARD_RADIUS = "12px"
CARD_SHADOW = "0 1px 3px rgba(20, 40, 70, 0.09)"

HOME_CSS = f"""
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: #f4f5f8; margin: 0; padding: 0; color: #1a1a1a;
  }}
  {NAV_CSS}
  .home-hero {{ max-width: 900px; margin: 0 auto; padding: 64px 24px 80px; text-align: center; }}
  .home-hero h1 {{ font-size: 28px; font-weight: 700; letter-spacing: -0.01em; margin: 0 0 40px; }}
  .home-cards {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; text-align: left; }}
  @media (max-width: 680px) {{ .home-cards {{ grid-template-columns: 1fr; }} }}
  .home-card {{
    background: white; border-radius: {CARD_RADIUS}; box-shadow: {CARD_SHADOW};
    padding: 25px 26px 24px; text-decoration: none; color: inherit; display: block;
    border-top: 3px solid {ARIBIO_BLUE}; transition: transform 0.15s ease, box-shadow 0.15s ease;
  }}
  .home-card:hover {{ transform: translateY(-2px); box-shadow: 0 6px 20px rgba(20, 40, 70, 0.16); }}
  .home-card-eyebrow {{ font-size: 11px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: {ARIBIO_BLUE}; margin-bottom: 8px; }}
  .home-card h2 {{ font-size: 18px; font-weight: 700; margin: 0 0 8px; }}
  .home-card p {{ font-size: 13.5px; color: #666; line-height: 1.55; margin: 0; }}
  .home-card-note {{ margin-top: 14px; font-size: 13px; font-weight: 600; }}
"""


def build_biomarker_dashboard():
    """
    Builds biomarker_dashboard.html from the aggregate-only files under
    ADNI_OUTPUTS_DIR (preprocessing/statistics/robustness stages must
    already have run). Reads nothing participant-level itself --
    adni_viz.build_dashboard_html() goes through adni_viz_data's
    governed loader, which refuses raw/interim/processed/parquet and
    any participant-identifier column.
    """
    html_out = adni_viz.build_dashboard_html()
    with open(BIOMARKER_HTML, "w", encoding="utf-8") as f:
        f.write(html_out)
    print(f"=== SAVED: {BIOMARKER_HTML} ===")
    return html_out


def _home_card(eyebrow, title, description, href, missing):
    note = (
        f'<div class="home-card-note" style="color:{ARIBIO_ACCENT}">Not generated yet — run its builder first</div>'
        if missing else
        f'<div class="home-card-note" style="color:{ARIBIO_BLUE}">Open &rarr;</div>'
    )
    return f"""
    <a class="home-card" href="{href}">
      <div class="home-card-eyebrow">{eyebrow}</div>
      <h2>{title}</h2>
      <p>{description}</p>
      {note}
    </a>"""


def build_dashboard_shell():
    """Writes dashboard.html — the homepage linking to both dashboards."""
    pipeline_missing = not os.path.exists(PIPELINE_HTML)
    biomarker_missing = not os.path.exists(BIOMARKER_HTML)

    cards = _home_card(
        "ClinicalTrials.gov", "Clinical Pipeline",
        "Every Alzheimer's disease drug candidate on ClinicalTrials.gov — phase, sponsor, "
        "target pathway, FDA status, and AR1001 competitive relevance.",
        PIPELINE_HTML, pipeline_missing,
    ) + _home_card(
        "ADNI", "Natural History",
        "Alzheimer's Disease Neuroimaging Initiative cohort data — MRI, PET, CSF, cognitive, "
        "and genetic biomarkers.",
        BIOMARKER_HTML, biomarker_missing,
    )

    html_out = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AriBio Alzheimer's Intelligence</title>
<style>{HOME_CSS}</style>
</head>
<body>
{render_nav_bar("home")}
<div class="home-hero">
  <h1>AriBio Alzheimer's Intelligence</h1>
  <div class="home-cards">{cards}</div>
</div>
</body>
</html>"""

    with open(DASHBOARD_HTML, "w", encoding="utf-8") as f:
        f.write(html_out)
    print(f"=== SAVED: {DASHBOARD_HTML} ===")
    if pipeline_missing:
        print(f"  NOTE: {PIPELINE_HTML} doesn't exist yet — run pipeline_viz.py (or run_pipeline.py) first.")


if __name__ == "__main__":
    build_biomarker_dashboard()
    build_dashboard_shell()
