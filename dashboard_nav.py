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
    display: flex; align-items: center; gap: 2px; padding: 0 22px; width: 100%;
    height: 48px; min-height: 48px; max-height: 48px; flex: 0 0 48px; box-sizing: border-box;
    background: linear-gradient(120deg, #16304f 0%, {NAV_BG} 55%, #24466f 100%);
    box-shadow: 0 2px 10px rgba(10, 20, 40, 0.18);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }}
  .dash-nav-brand {{
    display: flex; align-items: center; gap: 7px; color: white; font-weight: 700; font-size: 13.5px;
    letter-spacing: 0.01em; line-height: 1; margin-right: 22px; white-space: nowrap;
  }}
  /* Small accent dot standing in for a logo mark -- ARIBIO_ACCENT is
     already reserved page-wide for "the one thing worth a second
     glance"; here that's simply the brand itself. */
  .dash-nav-brand-mark {{
    width: 7px; height: 7px; border-radius: 50%; background: {ARIBIO_ACCENT}; flex-shrink: 0;
    box-shadow: 0 0 0 3px rgba(194, 37, 92, 0.25);
  }}
  .dash-nav a {{
    color: {NAV_LINK}; text-decoration: none; font-size: 13px; font-weight: 600;
    padding: 0 15px; height: 48px; min-height: 48px; line-height: 1; display: flex; align-items: center; position: relative;
    transition: color 0.15s ease;
  }}
  .dash-nav a:hover {{ color: white; }}
  .dash-nav a.active {{ color: white; }}
  /* Animated underline (scaled, not a static border) -- full color and
     width for the active tab, a faint preview on hover for anything
     else, so the active state still reads as clearly the "on" one. */
  .dash-nav a::after {{
    content: ""; position: absolute; left: 15px; right: 15px; bottom: 0; height: 3px;
    background: {ARIBIO_ACCENT}; border-radius: 3px 3px 0 0;
    transform: scaleX(0); transform-origin: center; transition: transform 0.2s ease, opacity 0.2s ease;
  }}
  .dash-nav a.active::after {{ transform: scaleX(1); }}
  .dash-nav a:not(.active):hover::after {{ transform: scaleX(1); opacity: 0.45; }}
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
    """
    links = "".join(
        f'<a href="{href}"{" class=\"active\"" if key == active else ""}>{label}</a>'
        for key, label, href in _TABS
    )
    return f"""<nav class="dash-nav">
      <span class="dash-nav-brand"><span class="dash-nav-brand-mark"></span>AriBio Alzheimer's Intelligence</span>
      {links}
    </nav>"""
