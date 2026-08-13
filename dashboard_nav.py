# ============================================================
# Shared top navigation bar for the multi-page dashboard suite
# (dashboard.html, pipeline_overview.html, biomarker_dashboard.html).
#
# Deliberately tiny and dependency-free (no pandas, no imports from
# any other project module) so every page can pull it in cheaply —
# including pipeline_viz.py's own expensive top-level script, which
# must never import adni_viz.py/adni_analysis.py or anything else with
# real work at import time. This is pure navigation chrome, not ADNI
# (or any other) domain logic — it belongs to none of the dashboards
# it links between.
# ============================================================

# Same AriBio blue family used everywhere else in this dashboard suite
# (pipeline_viz.py's ARIBIO_BLUE/ARIBIO_ACCENT) — the nav bar background
# is the same darkened shade pipeline_viz.py already uses for its
# darkest blue-ramp values (e.g. Phase 3 / "FDA Approved"), not an
# unrelated navy, so the bar reads as "AriBio blue" rather than a
# generic dark app-shell color. ARIBIO_ACCENT marks the active tab —
# the one thing on this bar worth a second glance — matching how it's
# reserved for "the one value worth noticing" everywhere else on the
# dashboard (Manual review required, Needs Attention, the AR1001 star).
ARIBIO_BLUE = "#2e5fa3"
ARIBIO_ACCENT = "#c2255c"
NAV_BG = "#1d3d69"          # darken(ARIBIO_BLUE, 0.35)
NAV_LINK = "#a0b7d5"        # lighten(ARIBIO_BLUE, 0.55)

NAV_CSS = f"""
  .dash-nav {{
    display: flex; align-items: center; gap: 2px; background: {NAV_BG}; padding: 0 20px;
    height: 44px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }}
  .dash-nav-brand {{ color: white; font-weight: 700; font-size: 13px; margin-right: 18px; white-space: nowrap; }}
  .dash-nav a {{
    color: {NAV_LINK}; text-decoration: none; font-size: 13px; font-weight: 600;
    padding: 0 14px; height: 44px; display: flex; align-items: center;
    border-bottom: 2px solid transparent;
  }}
  .dash-nav a:hover {{ color: white; }}
  .dash-nav a.active {{ color: white; border-bottom-color: {ARIBIO_ACCENT}; }}
"""

# (key, label, href) — order is display order in the bar
_TABS = [
    ("home", "Home", "dashboard.html"),
    ("pipeline", "Clinical Pipeline", "pipeline_overview.html"),
    ("biomarker", "ADNI Natural History", "biomarker_dashboard.html"),
]


def render_nav_bar(active):
    """
    active: one of the keys in _TABS ("home"/"pipeline"/"biomarker"),
    or None if this page isn't one of the three tabs.

    Self-hides when the page is loaded inside dashboard.html's own tab
    shell (an <iframe>) — the shell already shows its own copy of this
    same bar, so without this a tabbed view would show two stacked nav
    bars. Opening any of these three files directly (not through the
    shell) still shows the bar normally.
    """
    links = "".join(
        f'<a href="{href}"{" class=\"active\"" if key == active else ""}>{label}</a>'
        for key, label, href in _TABS
    )
    return f"""<nav class="dash-nav">
      <span class="dash-nav-brand">AriBio Alzheimer's Intelligence</span>
      {links}
    </nav>
    <script>
      if (window.self !== window.top) {{
        document.currentScript.previousElementSibling.style.display = 'none';
      }}
    </script>"""
