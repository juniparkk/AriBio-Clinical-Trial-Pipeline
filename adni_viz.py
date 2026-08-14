# ============================================================
# ADNI_VIZ -- builds the standalone Biomarker Dashboard page
# (biomarker_dashboard.html). VISUALIZATION ONLY.
#
# Every number on this page is read from ADNI_OUTPUTS_DIR (aggregate
# CSVs only) via adni_viz_data.load_all() -> adni_viz_data.load_aggregate_csv(),
# the single governed entry point that refuses raw/, interim/,
# processed/, any .parquet file, and any CSV carrying a participant-
# identifier column. This module never fits a model, never computes a
# new inferential statistic, and never touches processed/ or raw/ --
# see adni_viz_data.py's module docstring for the narrow, documented
# exceptions (a pure unit re-expression of an already-computed HC3 log
# estimate to percent scale, plus purely descriptive reshapes for the
# Absolute/Disease-Continuum views added in the Medical Affairs
# redesign). Row-wise min/max normalization for the Disease Continuum
# heatmap's COLOR only (never the displayed number) happens in this
# module's JS, for the same reason: it is a display encoding, not a
# statistic.
#
# HC3 robust inference is the only inference ever displayed for an
# "Adjusted analysis" / "Sensitivity concern" point or table row --
# conventional (non-robust) inference is shown only inside the
# row-expansion detail of the Statistical Results table, explicitly
# labeled "conventional", never as the headline number. The Absolute
# view (both cognitive and biomarker) is ALWAYS descriptive -- no
# ANCOVA was ever fit against an absolute-score/absolute-level outcome
# (only against change-from-baseline), so there is no "adjusted
# absolute value" to show; showing one would be inventing a result
# that was never validated.
#
# Visual language matches pipeline_overview.html (pipeline_viz.py):
# same ARIBIO_BLUE/ARIBIO_ACCENT/CARD_RADIUS/CARD_SHADOW constants,
# same full-width main/page-title treatment, same nav bar
# (dashboard_nav.py, unmodified). Plotly is bundled inline the same
# way pipeline_overview.html does it (pyo.get_plotlyjs() once, no CDN).
# ============================================================

import html
import json

import plotly.offline as pyo

import adni_viz_data as D
from adni_analysis import ADNI_OUTPUTS_DIR
from dashboard_nav import NAV_BG, NAV_CSS, render_nav_bar

OUTPUT_HTML = "biomarker_dashboard.html"

ARIBIO_BLUE = "#2e5fa3"
ARIBIO_ACCENT = "#c2255c"
CARD_RADIUS = "12px"
CARD_SHADOW = "0 1px 3px rgba(20, 40, 70, 0.09)"
SURFACE_TINT = "#f6f8fb"
SURFACE_BORDER = "#dfe7f1"
ACCENT_BG = "#faeff3"
ACCENT_BORDER = "#ecbdce"
WARN_AMBER = "#b8860b"
WARN_AMBER_BG = "#fff6e0"
WARN_AMBER_BORDER = "#f0dca0"

# Matches adni_stats.MIN_GROUP_N (the actual prespecified small-cell
# threshold used by the statistical-analysis stage) -- duplicated here
# as a plain int, not imported, so this visualization-only module
# never depends on the modeling stack (statsmodels/scipy/patsy), per
# adni_viz_data.py's own documented boundary. Used ONLY to decide
# whether to show an inline "n=" label on a chart point (display
# concern), never to make an inferential decision.
MIN_GROUP_N_FOR_DISPLAY = 10

COGNITIVE_ENDPOINTS = [
    {"key": "ADAS_COG13", "label": "ADAS-Cog13", "up_label": "Worse", "down_label": "Better"},
    {"key": "MMSE", "label": "MMSE", "up_label": "Better", "down_label": "Worse"},
]

BIOMARKER_SPECS = [
    {
        "key": "pTau181", "label": "pTau181", "platforms": [("Gothenburg_Simoa", "primary", "Primary")],
        "interpretation": "↑ More tau-associated pathology",
    },
    {
        "key": "pTau217", "label": "pTau217",
        "platforms": [("Fujirebio_Lumipulse", "primary", "Primary"), ("Fujirebio_Lumipulse", "sensitivity_incl_lot_bias", "Lot-bias sensitivity")],
        "interpretation": "↑ More tau-associated pathology",
    },
    {
        "key": "Abeta42_40_ratio", "label": "Aβ42/Aβ40", "platforms": [("Fujirebio_Lumipulse", "primary", "Primary")],
        "interpretation": "↓ Lower ratio is generally associated with greater amyloid pathology",
    },
    {
        "key": "GFAP", "label": "GFAP",
        "platforms": [("Quanterix", "primary", "Quanterix"), ("Fujirebio", "sensitivity_fujirebio", "Fujirebio")],
        "interpretation": "Marker of astroglial activation/neurodegeneration; not specific to amyloid or tau pathology.",
    },
    {
        "key": "NfL", "label": "NfL",
        "platforms": [("Quanterix", "primary", "Quanterix"), ("Fujirebio", "sensitivity_fujirebio", "Fujirebio")],
        "interpretation": "Marker of astroglial activation/neurodegeneration; not specific to amyloid or tau pathology.",
    },
]

# Direction/formatting metadata for the Disease Continuum heatmap --
# presentation-layer concerns (which way "worse" points, how many
# decimals to show), never a statistic. Lives here (not
# adni_viz_data.py) for the same reason COGNITIVE_ENDPOINTS'
# up_label/down_label and BIOMARKER_SPECS' "interpretation" do.
DISEASE_CONTINUUM_META = {
    "ADAS_COG13": {"higher_is_worse": True, "digits": 1},
    "MMSE": {"higher_is_worse": False, "digits": 1},
    "pTau181": {"higher_is_worse": True, "digits": 2},
    "pTau217": {"higher_is_worse": True, "digits": 3},
    "Abeta42_40_ratio": {"higher_is_worse": False, "digits": 3},
    "GFAP": {"higher_is_worse": True, "digits": 1},
    "NfL": {"higher_is_worse": True, "digits": 1},
}

PAGE_CSS = f"""
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: #f4f5f8; margin: 0; padding: 0 0 60px; color: #1a1a1a;
  }}
  {NAV_CSS}
  main {{ margin: 0; padding: 24px 96px 0 96px; }}
  @media (max-width: 1100px) {{ main {{ padding: 20px 20px 0; }} }}

  .page-title-block {{ margin-bottom: 6px; }}
  .page-title {{ font-size: 22px; font-weight: 700; letter-spacing: -0.01em; color: {NAV_BG}; }}
  .page-title-sub {{ font-size: 13px; color: #666; margin-top: 4px; line-height: 1.5; }}

  /* Header row: description/cohort-summary/legend/population-filter on
     the left, Disease Continuum beside it on the right, instead of
     stacked full-width -- wraps to a single column on narrow viewports.
     align-items: stretch (the flex default) so both columns' BOTTOMS
     line up -- but the actual white card in each column also has to
     grow to fill that stretched height itself (flex: 1 / height: 100%
     below), otherwise stretching just the outer column leaves a bare
     gray gap below a shorter card instead of the card's own bottom
     edge lining up with its neighbor. */
  .header-continuum-row {{ display: flex; flex-wrap: wrap; gap: 20px; align-items: stretch; margin-bottom: 20px; }}
  .header-left-col {{ flex: 1 1 500px; min-width: 440px; display: flex; flex-direction: column; }}
  /* header-info-card grows to fill the stretched column; the bottom
     (cohort) card grows within it, and the population section -- its
     own last always-present child -- absorbs the extra height, rather
     than leaving a blank gap after it. (The POLARIS cohort-construction
     detail nested inside that section only appears once selected, and
     stays collapsed until clicked open, so it never itself needs to
     absorb stretch height.) */
  .header-left-col .header-info-card {{ flex: 1; display: flex; flex-direction: column; }}
  .header-info-population {{ flex: 1; }}
  .header-info-cohort-card {{ flex: 1; display: flex; flex-direction: column; }}
  .header-right-col {{ flex: 1 1 500px; min-width: 440px; display: flex; }}
  /* Same idea for Disease Continuum: the panel grows, and the heatmap
     itself (not blank space below it) absorbs the extra height. */
  .header-right-col section.panel {{ margin-bottom: 0; flex: 1; display: flex; flex-direction: column; }}
  .header-right-col .continuum-card {{ flex: 1; }}
  @media (max-width: 1300px) {{ .header-continuum-row {{ flex-direction: column; }} }}

  /* .header-right-col (Disease Continuum) and .trajectories-row's right
     column (Plasma Biomarker Trajectories) share the IDENTICAL
     flex-basis/min-width and the IDENTICAL wrap breakpoint as their
     left-column siblings, so their edges line up vertically down the
     page and both rows switch between 1-column/2-column together,
     never independently. */

  /* Cognitive Trajectories + Plasma Biomarker Trajectories side by
     side on wide screens -- same bottom-alignment reasoning as the
     header row above: stretch the row, and let each panel's own card
     grow to fill it (flex: 1 1 500px already makes each panel a flex
     item that stretches to the row's full height under the default
     align-items: stretch). */
  .trajectories-row {{ display: flex; flex-wrap: wrap; gap: 20px; align-items: stretch; margin-bottom: 20px; }}
  .trajectories-row > section.panel {{ flex: 1 1 500px; min-width: 440px; margin-bottom: 0; display: flex; flex-direction: column; }}
  .trajectories-row > section.panel .chart-card {{ flex: 1; }}
  @media (max-width: 1300px) {{ .trajectories-row {{ flex-direction: column; }} }}

  /* Header info card: two cards with the SAME border/bg treatment
     (SURFACE_TINT + SURFACE_BORDER + CARD_RADIUS -- the same tokens
     every other panel on this page already uses, not a one-off tint
     invented for this section) so they read as a consistent pair, not
     one boxed and one blank. The bottom card (stats + filter + POLARIS
     panel -- "the cohort itself": who's in it, which version you're
     viewing, how that version was built -- the one you actually act
     on) keeps the extra box-shadow the top card (disclaimer + legend
     -- "how to read this page") doesn't get, so it still reads as the
     more important of the two without looking unrelated to it.
     Two-tier type scale across both cards -- 13px for body/description
     copy, 15px bold for labels/headlines -- consistent WITHIN each
     tier without forcing every line to the same size. */
  .header-info-card {{ display: flex; flex-direction: column; gap: 12px; margin-bottom: 14px; }}

  .header-info-top-card {{
    background: {SURFACE_TINT}; border: 1px solid {SURFACE_BORDER}; border-radius: {CARD_RADIUS};
    overflow: hidden;
  }}
  .header-info-disclaimer {{ font-size: 13px; color: #45566b; line-height: 1.55; padding: 14px 20px; }}

  /* Status legend -- symbols only, full definitions on hover (title
     attr) and in the Analysis Details disclosure. */
  .header-info-legend {{
    display: flex; flex-wrap: wrap; gap: 16px; align-items: center; font-size: 13px; color: #556;
    padding: 10px 20px 14px; border-top: 1px solid {SURFACE_BORDER};
  }}
  .header-info-legend .cl-item {{ cursor: help; border-bottom: 1px dotted #bbb; }}
  .header-info-legend .cl-info-btn {{
    background: none; border: 1px solid {SURFACE_BORDER}; border-radius: 50%; width: 20px; height: 20px;
    color: #888; cursor: pointer; font-size: 12px; line-height: 1; font-family: inherit;
  }}

  .header-info-cohort-card {{
    background: {SURFACE_TINT}; border: 1px solid {SURFACE_BORDER}; border-radius: {CARD_RADIUS};
    box-shadow: {CARD_SHADOW}; overflow: hidden;
  }}
  .header-info-stats {{ padding: 16px 20px; }}
  /* Big value, small label underneath -- a stat-tile look without the
     bordered/shadowed tile itself -- just the number/label pair,
     floating on the shared cohort card's own background. CN/MCI/
     Dementia values reuse GROUP_COLORS (the same blue/orange/red used
     for these groups on every chart on this page), so the same color
     means the same group everywhere, not just in the charts. */
  .header-info-stat-row {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 16px; }}
  @media (max-width: 620px) {{ .header-info-stat-row {{ grid-template-columns: repeat(3, 1fr); }} }}
  .header-info-stat-item {{ display: flex; flex-direction: column; }}
  .header-info-stat-value {{ font-size: 22px; font-weight: 700; color: {ARIBIO_BLUE}; letter-spacing: -0.01em; line-height: 1.15; }}
  .header-info-stat-label {{ font-size: 11.5px; color: #666; margin-top: 3px; white-space: nowrap; }}

  /* Population section -- the one control in this card, restructured
     as label -> toggle -> summary (the active population's name, n,
     and a one-line description) instead of a wordy "Filter by
     population:" sentence. Generous padding (the actual "blank space"
     doing the separating work) plus a hairline divider is what sets it
     apart from the stats above it. */
  .header-info-population {{
    border-top: 1px solid {SURFACE_BORDER}; padding: 16px 20px;
  }}
  /* label/toggle/summary on the left, the retention badge (POLARIS
     only) on the right -- keeps the row from reading as half-empty on
     wide cards once the wordy "Filter by population:" sentence was
     replaced by the shorter label -> toggle -> summary stack. */
  .population-header-row {{ display: flex; align-items: center; justify-content: space-between; gap: 16px; }}
  .population-main {{ flex: 1; min-width: 0; }}
  .population-label {{
    font-size: 11px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: #889;
    margin: 0 0 10px;
  }}
  .population-summary {{ margin-top: 12px; }}
  .population-summary-name {{ font-size: 15px; font-weight: 700; color: {ARIBIO_BLUE}; }}
  .population-summary-desc {{
    display: flex; flex-wrap: wrap; align-items: baseline; column-gap: 10px; row-gap: 2px;
    font-size: 13px; color: #445; line-height: 1.4; margin-top: 2px;
  }}
  /* Only relevant while POLARIS is selected -- fully removed from flow
     (not just visually hidden) the rest of the time, unlike .is-hidden
     elsewhere on this page, since there's no fixed-height neighbor here
     that needs its space reserved. */
  .view-cohort-def-link {{
    font-size: 12.5px; font-weight: 600; color: {ARIBIO_BLUE}; cursor: pointer; white-space: nowrap;
    text-decoration: none; border-bottom: 1px solid transparent;
  }}
  .view-cohort-def-link.is-hidden {{ display: none; }}
  .view-cohort-def-link:hover {{ border-bottom-color: {ARIBIO_BLUE}; }}

  /* Retention badge -- how much POLARIS eligibility filtering narrowed
     Overall ADNI, at a glance, without reading the funnel below.
     White + bordered (like .meta-chip) rather than SURFACE_TINT so it
     still stands out against the tinted cohort card behind it. Overall
     ADNI has nothing to compare against, so it's fully removed then,
     not just faded. */
  .population-retention {{
    flex-shrink: 0; text-align: center; background: white; border: 1px solid {SURFACE_BORDER};
    border-radius: 10px; padding: 8px 18px; box-shadow: {CARD_SHADOW};
  }}
  .population-retention.is-hidden {{ display: none; }}
  .population-retention-pct {{ font-size: 20px; font-weight: 700; color: {ARIBIO_BLUE}; line-height: 1.1; white-space: nowrap; }}
  .population-retention-detail {{ font-size: 10.5px; color: #667; margin-top: 3px; line-height: 1.35; white-space: nowrap; }}

  .meta-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }}
  .meta-chip {{
    background: white; border: 1px solid {SURFACE_BORDER}; border-radius: 999px; padding: 6px 14px;
    font-size: 12.5px; color: #444; box-shadow: {CARD_SHADOW}; animation: contentFadeIn 0.3s ease;
  }}
  .meta-chip b {{ color: {ARIBIO_BLUE}; }}

  section.panel {{
    background: white; border-radius: {CARD_RADIUS}; box-shadow: {CARD_SHADOW};
    padding: 20px 24px; margin-bottom: 20px;
  }}
  section.panel h2 {{ font-size: 16px; font-weight: 700; margin: 0 0 4px; color: {NAV_BG}; }}
  .panel-sub {{ font-size: 12.5px; color: #666; margin: 0 0 16px; line-height: 1.6; }}

  .toggle-group {{ display: inline-flex; border: 1px solid {SURFACE_BORDER}; border-radius: 8px; overflow: hidden; margin: 0 10px 10px 0; }}
  .toggle-btn {{
    background: white; border: none; border-right: 1px solid {SURFACE_BORDER}; padding: 7px 14px;
    font-size: 12.5px; font-weight: 600; color: #556; cursor: pointer; font-family: inherit;
    transition: background-color 0.2s ease, color 0.2s ease;
  }}
  .toggle-btn:last-child {{ border-right: none; }}
  .toggle-btn.active {{ background: {ARIBIO_BLUE}; color: white; }}
  .toggle-btn:hover:not(.active) {{ background: {SURFACE_TINT}; }}
  .toggle-row {{ display: flex; flex-wrap: wrap; align-items: center; }}

  /* Shared fade-in for freshly-inserted content (table rows, cards) --
     runs automatically on new DOM nodes (no JS retrigger needed), so
     rebuilding the Statistical Results table on a dropdown/toggle
     change reveals its rows smoothly instead of snapping into place. */
  @keyframes contentFadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}

  /* Compact, muted sensitivity indicator (routine descriptive-only
     count) vs. the genuinely prominent amber warning (reserved for an
     actual C-status sensitivity concern). */
  .compact-warn {{ font-size: 11.5px; color: #888; margin: -2px 0 10px; }}
  .compact-warn--concern {{
    color: {WARN_AMBER}; font-weight: 700; background: {WARN_AMBER_BG}; border: 1px solid {WARN_AMBER_BORDER};
    border-radius: 6px; padding: 5px 10px; display: inline-block;
  }}

  .interpretation-note {{ font-size: 12.5px; color: #444; background: {SURFACE_TINT}; border-radius: 8px; padding: 10px 14px; margin-top: 10px; line-height: 1.55; }}

  /* Key Pattern -- compact, deterministic summary directly under each
     main chart (never inside the collapsible detail sections). */
  .key-pattern {{
    background: {SURFACE_TINT}; border-left: 3px solid {ARIBIO_BLUE}; border-radius: 0 8px 8px 0;
    padding: 10px 14px; margin-top: 10px; font-size: 12.5px; color: #33475b; line-height: 1.55;
  }}
  .key-pattern b {{ color: {ARIBIO_BLUE}; }}

  .collapsible-toggle {{
    display: flex; align-items: center; justify-content: space-between; cursor: pointer;
    padding: 4px 0; user-select: none;
  }}
  .collapsible-toggle .chev {{ transition: transform 0.15s ease; color: {ARIBIO_BLUE}; font-size: 13px; }}
  .collapsible-toggle.open .chev {{ transform: rotate(90deg); }}
  /* Smooth open/close (max-height + opacity, not an instant display
     swap) -- max-height's own declared ceiling is generous enough that
     real content always finishes growing well within the transition
     duration, the standard pure-CSS accordion technique. */
  .collapsible-body {{
    max-height: 0; opacity: 0; overflow: hidden; margin-top: 0;
    transition: max-height 0.4s ease, opacity 0.3s ease, margin-top 0.4s ease;
  }}
  .collapsible-body.open {{ max-height: 6000px; opacity: 1; margin-top: 12px; }}

  table.results-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  table.results-table th {{
    text-align: left; font-weight: 700; color: white; font-size: 10.5px; letter-spacing: 0.02em;
    text-transform: uppercase; padding: 8px 8px; background: {NAV_BG}; white-space: nowrap;
  }}
  table.results-table td {{ padding: 7px 8px; border-bottom: 1px solid {SURFACE_BORDER}; color: #333; white-space: nowrap; }}
  table.results-table tr.result-row {{ cursor: pointer; animation: contentFadeIn 0.3s ease; }}
  table.results-table tr.result-row td {{ transition: background-color 0.15s ease; }}
  table.results-table tr.result-row:hover td {{ background: {SURFACE_TINT}; }}
  table.results-table tr.detail-row {{ display: none; }}
  /* display:table-row can't itself be transitioned smoothly, but a
     freshly-revealed row's content still fades in (animation runs on
     the row becoming part of the rendered tree, not a property
     transition) -- close enough to "smooth" for a single table row. */
  table.results-table tr.detail-row.open {{ display: table-row; animation: contentFadeIn 0.25s ease; }}
  table.results-table tr.detail-row td {{ background: {SURFACE_TINT}; padding: 12px 16px; white-space: normal; }}
  .status-pill {{
    display: inline-block; font-size: 10.5px; font-weight: 700; letter-spacing: 0.02em; border-radius: 999px;
    padding: 2px 9px;
  }}
  .status-pill.A {{ background: #e3edfb; color: {ARIBIO_BLUE}; }}
  .status-pill.B {{ background: #eef0f3; color: #667; }}
  .status-pill.C {{ background: {WARN_AMBER_BG}; color: {WARN_AMBER}; }}
  .status-pill.D {{ background: #f2f2f2; color: #999; }}
  .warning-badge {{ color: {WARN_AMBER}; margin-left: 4px; }}
  .multiplicity-note {{ font-size: 11px; color: #888; margin-top: 10px; font-style: italic; }}

  .methods-list li, .limitations-list li {{ font-size: 13px; line-height: 1.7; color: #444; }}
  .limitations-list li {{ margin-bottom: 4px; }}
  .legend-detail-list {{ font-size: 12.5px; color: #444; line-height: 1.8; margin: 0 0 14px; padding-left: 18px; }}

  /* Charts reduced ~28% from the original 460px min-height. */
  .chart-card {{ min-height: 330px; }}
  .continuum-card {{ min-height: 300px; }}

  /* Fade-in helper for content swapped inside a PERSISTENT container
     (innerHTML on an existing element doesn't retrigger that element's
     own CSS animation -- see fadeSpan() in the JS): wrap the fresh
     content in this class instead, which is a brand-new node every
     render and therefore always replays. */
  .fade-in-span {{ animation: contentFadeIn 0.3s ease; }}

  /* Generic "hidden but still reserves its layout space" utility --
     used everywhere a population-dependent piece of text has a
     same-population-independent alternate (or absence) that would
     otherwise change a line count / add-or-remove a line when toggled,
     which (via align-items: stretch on the two-column rows) would
     grow the WHOLE card -- and its stretch-aligned sibling -- taller
     and shift the page. display:none is never used for these. */
  .is-hidden {{ visibility: hidden; }}

  #biomarkerPlatformGroup {{ animation: contentFadeIn 0.3s ease; }}

  /* Compact per-section population label ("Population: Overall ADNI" /
     "Population: POLARIS AD-Aligned - n=620 baseline eligible") -- the
     explicit boundary marker required wherever a chart could otherwise
     be mistaken for having silently switched population. */
  .population-note {{
    font-size: 11.5px; font-weight: 700; color: {ARIBIO_BLUE}; background: {SURFACE_TINT};
    border-radius: 6px; padding: 4px 10px; display: inline-block; margin: 0 0 10px;
  }}

  /* Deterministic data-support status line -- distinct from
     .compact-warn (which flags a genuine sensitivity concern): this is
     routine, expected context about follow-up sparsity, never hidden
     but not alarm-styled either. */
  .data-support-note {{
    font-size: 12px; color: #556; background: {SURFACE_TINT}; border-left: 3px solid {ARIBIO_ACCENT};
    border-radius: 0 6px 6px 0; padding: 6px 12px; margin: 0 0 10px;
    transition: opacity 0.2s ease; animation: contentFadeIn 0.3s ease;
  }}

  /* POLARIS cohort-construction detail (funnel + population profile) --
     nested inside .header-info-population, directly under the summary
     line, with no gap and no background of its own. Its title uses
     ARIBIO_ACCENT (the same "worth a second glance" color as the AR1001
     star and manual-review flags elsewhere) so it still reads as its
     own distinct sub-topic without a colored fill. Starts collapsed
     (the standard .collapsible-body open/close animation) and only
     expands on an explicit "View cohort definition" click
     (togglePolarisCohortDefinition() in the JS) -- selecting Overall
     ADNI resets it back to collapsed rather than leaving it open. */
  #polarisCollapsibleBody h2 {{ color: {ARIBIO_ACCENT}; }}
  .polaris-context-box {{
    background: {SURFACE_TINT}; border-radius: 8px; padding: 12px 16px; font-size: 13px; color: #33475b;
    line-height: 1.6; margin-bottom: 14px;
  }}
  .polaris-disclaimer {{ display: inline-block; margin-top: 6px; color: {ARIBIO_ACCENT}; font-weight: 600; }}

  .polaris-funnel {{ display: flex; flex-direction: column; align-items: center; gap: 2px; margin: 10px 0 20px; }}
  .funnel-step {{
    background: {SURFACE_TINT}; border: 1px solid {SURFACE_BORDER}; border-radius: 8px; padding: 8px 18px;
    text-align: center; min-width: 300px; animation: contentFadeIn 0.35s ease;
  }}
  .funnel-step-label {{ font-size: 12px; color: #556; font-weight: 600; }}
  .funnel-step-n {{ font-size: 18px; font-weight: 700; color: {ARIBIO_BLUE}; }}
  .funnel-step-meta {{ font-size: 11px; color: #888; margin-top: 2px; }}
  .funnel-arrow {{ font-size: 14px; color: #999; line-height: 1.2; padding: 2px 0; }}

  .profile-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; margin-top: 4px; }}
  .profile-card {{ border: 1px solid {SURFACE_BORDER}; border-radius: 8px; padding: 10px 12px; animation: contentFadeIn 0.35s ease; }}
  .profile-var-label {{ font-size: 12px; font-weight: 700; color: #1a1a1a; margin-bottom: 6px; }}
  .profile-pop-row {{ display: flex; align-items: center; gap: 8px; font-size: 11.5px; color: #444; margin-bottom: 4px; }}
  .profile-pop-tag {{ display: inline-block; width: 78px; flex-shrink: 0; font-weight: 600; color: #667; }}
  .profile-numeric-value {{ color: #333; }}
  .profile-bar {{ flex: 1; height: 10px; border-radius: 5px; overflow: hidden; display: flex; background: #eee; }}
  .profile-bar-seg {{ height: 100%; }}
  .profile-note {{ font-size: 10.5px; color: #888; margin-top: 6px; line-height: 1.5; }}
"""


def _fmt_num(v, digits=2):
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def _fmt_ci(lo, hi, digits=2, suffix=""):
    if lo is None or hi is None:
        return "—"
    return f"[{lo:.{digits}f}{suffix}, {hi:.{digits}f}{suffix}]"


def _fmt_p(p):
    if p is None:
        return "—"
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"


# ------------------------------------------------------------------
# Header / compact cohort overview
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Header info card: description/disclaimer, cohort stats, status
# legend, and the population filter, unified into ONE card (consistent
# radius/shadow, thin dividers between sections) instead of four
# separately-boxed pieces.
#
# The filter itself is display-only wiring: selecting "POLARIS
# AD-Aligned" toggles which panels are shown (the cohort-construction
# funnel + population profile below) and shows a boundary note above
# the trajectory sections. It never re-fetches or re-filters any
# trajectory data -- Disease Continuum / Cognitive / Biomarker charts
# always render the same Overall-ADNI payload regardless of which
# population button is active, per the explicit "do not change
# trajectories yet" scope.
# ------------------------------------------------------------------


def render_header_section():
    return f"""
    <div class="header-info-card">
      <div class="header-info-top-card">
        <div class="header-info-section header-info-disclaimer">ADNI is an observational natural-history cohort, not a randomized clinical trial and not an external control arm. Results describe disease progression and should not be interpreted as treatment effects.</div>
        <div class="header-info-section header-info-legend" id="statusLegend">
          <span class="cl-item" title="Prespecified ANCOVA fit; HC3 robust inference displayed.">&#9679; Adjusted</span>
          <span class="cl-item" title="Insufficient subgroup sample size for inferential model; descriptive value shown instead (no p-value, no adjustment).">&#9675; Descriptive</span>
          <span class="cl-item" title="Result materially changed under HC3 and/or influence analysis.">&#9888; Sensitivity concern</span>
          <button type="button" class="cl-info-btn" onclick="toggleCollapsible(document.getElementById('analysisDetailsToggle'))" title="Full definitions in Analysis Details">&#9432;</button>
        </div>
      </div>
      <div class="header-info-cohort-card">
        <div class="header-info-section header-info-stats">
          <div class="header-info-stat-row">
            <div class="header-info-stat-item">
              <div class="header-info-stat-value">3,030</div>
              <div class="header-info-stat-label">participants</div>
            </div>
            <div class="header-info-stat-item">
              <div class="header-info-stat-value" style="color:{D.GROUP_COLORS['CN']}">1,215</div>
              <div class="header-info-stat-label">CN</div>
            </div>
            <div class="header-info-stat-item">
              <div class="header-info-stat-value" style="color:{D.GROUP_COLORS['MCI']}">1,338</div>
              <div class="header-info-stat-label">MCI</div>
            </div>
            <div class="header-info-stat-item">
              <div class="header-info-stat-value" style="color:{D.GROUP_COLORS['Dementia']}">477</div>
              <div class="header-info-stat-label">Dementia</div>
            </div>
            <div class="header-info-stat-item">
              <div class="header-info-stat-value">7</div>
              <div class="header-info-stat-label">endpoints</div>
            </div>
            <div class="header-info-stat-item">
              <div class="header-info-stat-value">48</div>
              <div class="header-info-stat-label">months (max)</div>
            </div>
          </div>
        </div>
        <div class="header-info-section header-info-population">
          <div class="population-header-row">
            <div class="population-main">
              <div class="population-label">Population</div>
              <div class="toggle-group" id="populationToggleGroup">
                <button class="toggle-btn active" data-population="overall" onclick="setPopulation('overall')">Overall ADNI</button>
                <button class="toggle-btn" data-population="polaris" onclick="setPopulation('polaris')">POLARIS AD&ndash;Aligned</button>
              </div>
              <div class="population-summary">
                <div class="population-summary-name" id="populationSummaryName">Overall ADNI &middot; n&nbsp;=&nbsp;{OVERALL_ADNI_N:,}</div>
                <div class="population-summary-desc">
                  <span id="populationSummaryDesc">Broad natural-history cohort.</span>
                  <a href="javascript:void(0)" class="view-cohort-def-link is-hidden" id="viewCohortDefLink" onclick="togglePolarisCohortDefinition()">View cohort definition</a>
                </div>
              </div>
            </div>
            <div class="population-retention is-hidden" id="populationRetention">
              <div class="population-retention-pct" id="populationRetentionPct"></div>
              <div class="population-retention-detail">of Overall ADNI<br><span id="populationRetentionN"></span></div>
            </div>
          </div>
          <div class="collapsible-body" id="polarisCollapsibleBody">
            <div class="polaris-context-box" id="polarisContextBox"></div>
            <div class="polaris-funnel" id="polarisFunnel"></div>
            <h2 style="font-size:14px;margin:4px 0 2px;">Overall ADNI vs. POLARIS AD&ndash;Aligned &mdash; Population Profile</h2>
            <p class="panel-sub">How eligibility filtering changed the population. Descriptive comparison only &mdash; not a statistical test of comparability.</p>
            <div class="profile-grid" id="polarisProfileGrid"></div>
          </div>
        </div>
      </div>
    </div>
    """


# ------------------------------------------------------------------
# Disease Continuum -- Overall ADNI only (population-aware trajectories
# do NOT extend here yet; a POLARIS-specific continuum is an explicit
# future decision, not silently computed now). #diseaseContinuumPolarisNote
# is shown only when POLARIS is the active population, via JS.
# ------------------------------------------------------------------


def render_disease_continuum_section():
    return f"""
    <section class="panel">
      <h2>Disease Continuum</h2>
      <p class="panel-sub"><b>Darker = greater disease-associated abnormality within each endpoint. Colors are not comparable across endpoints.</b> Row labels show which direction is worse for that endpoint. Baseline (month 0) values. Hover a cell for the exact value, n, and 95% descriptive CI.</p>
      <div class="population-note is-hidden" id="diseaseContinuumPolarisNote">Disease Continuum shown from Overall ADNI baseline reference.</div>
      <div id="diseaseContinuumChart" class="continuum-card"></div>
    </section>
    """


# ------------------------------------------------------------------
# Cognitive Trajectories
# ------------------------------------------------------------------


def render_cognitive_section():
    endpoint_btns = "".join(
        f'<button class="toggle-btn{" active" if i == 0 else ""}" data-endpoint="{e["key"]}" onclick="setCognitiveEndpoint(\'{e["key"]}\')">{html.escape(e["label"])}</button>'
        for i, e in enumerate(COGNITIVE_ENDPOINTS)
    )
    return f"""
    <section class="panel">
      <h2>Cognitive Trajectories</h2>
      <p class="panel-sub">Adjusted analyses use ANCOVA with HC3 robust inference, controlling for baseline score, age, and sex. Absolute view is always descriptive (no adjustment applies to a raw score).</p>
      <div class="population-note" id="cognitivePopulationLabel">Population: Overall ADNI</div>
      <div class="toggle-row">
        <div class="toggle-group">{endpoint_btns}</div>
        <div class="toggle-group">
          <button class="toggle-btn active" data-view="absolute" onclick="setCognitiveView('absolute')">Absolute</button>
          <button class="toggle-btn" data-view="change" onclick="setCognitiveView('change')">Change from baseline</button>
        </div>
      </div>
      <div id="cognitiveDataSupport" class="data-support-note" style="display:none;"></div>
      <div id="cognitiveWarn"></div>
      <div id="cognitiveChart" class="chart-card"></div>
      <div id="cognitiveKeyPattern" class="key-pattern"></div>
    </section>
    """


# ------------------------------------------------------------------
# Plasma Biomarker Trajectories
# ------------------------------------------------------------------


def render_biomarker_section():
    biomarker_btns = "".join(
        f'<button class="toggle-btn{" active" if i == 0 else ""}" data-biomarker="{b["key"]}" onclick="setBiomarker(\'{b["key"]}\')">{html.escape(b["label"])}</button>'
        for i, b in enumerate(BIOMARKER_SPECS)
    )
    return f"""
    <section class="panel">
      <h2>Plasma Biomarker Trajectories</h2>
      <p class="panel-sub"><b>Absolute</b> shows the actual concentration at each disease stage (geometric mean, always descriptive) -- the question "how do levels differ across CN/MCI/Dementia and at each stage over time." <b>% change from baseline</b> shows adjusted (ANCOVA + HC3 95% CI) or descriptive geometric mean percent change -- the question "how does this biomarker move over time." Never combines values across assays/platforms.</p>
      <div class="population-note" id="biomarkerPopulationLabel">Population: Overall ADNI</div>
      <div class="toggle-row">
        <div class="toggle-group">{biomarker_btns}</div>
        <div class="toggle-group">
          <button class="toggle-btn active" data-view="absolute" onclick="setBiomarkerView('absolute')">Absolute</button>
          <button class="toggle-btn" data-view="change" onclick="setBiomarkerView('change')">% change from baseline</button>
        </div>
        <div class="toggle-group" id="biomarkerPlatformGroup"></div>
      </div>
      <div class="meta-row" id="biomarkerMetaRow"></div>
      <div id="biomarkerDataSupport" class="data-support-note" style="display:none;"></div>
      <div id="biomarkerWarn"></div>
      <div id="biomarkerChart" class="chart-card"></div>
      <div id="biomarkerInterpretation" class="interpretation-note"></div>
      <div id="biomarkerKeyPattern" class="key-pattern"></div>
    </section>
    """


# ------------------------------------------------------------------
# Statistical Results (collapsible)
# ------------------------------------------------------------------


def render_results_table_section():
    endpoint_options = "".join(
        f'<option value="{e["key"]}|__cognitive__|primary">{html.escape(e["label"])} (primary)</option>'
        for e in COGNITIVE_ENDPOINTS
    )
    endpoint_options += '<option value="MMSE|__cognitive__|sensitivity_interval_excl">MMSE (screening-interval sensitivity)</option>'
    for b in BIOMARKER_SPECS:
        for platform, analysis_type, label in b["platforms"]:
            endpoint_options += f'<option value="{b["key"]}|{platform}|{analysis_type}">{html.escape(b["label"])} ({html.escape(label)})</option>'

    return f"""
    <section class="panel">
      <div class="collapsible-toggle" onclick="toggleCollapsible(this)">
        <h2 style="margin:0;">Statistical Results</h2>
        <span class="chev">&#9656;</span>
      </div>
      <div class="collapsible-body">
        <p class="panel-sub" style="margin-top:12px;">All p-values are exploratory and unadjusted for multiplicity. Click a row to expand HC3 / conventional / influence-sensitivity detail. Results shown here are for the Change-from-baseline (adjusted) analysis -- the Absolute view above is always descriptive and has no model to expand.</p>
        <div class="population-note" id="resultsPopulationLabel">Population: Overall ADNI</div>
        <select class="toggle-btn" style="width:100%;max-width:340px;padding:8px 12px;margin-bottom:12px;" id="resultsTableSelect" onchange="renderResultsTable()">{endpoint_options}</select>
        <div style="overflow-x:auto;">
          <table class="results-table" id="resultsTable">
            <thead>
              <tr>
                <th>Month</th><th>Group / Comparison</th><th>n</th><th>Estimate (HC3)</th><th>95% CI (HC3)</th>
                <th>Overall F</th><th>HC3 p</th><th title="Effect-size estimate from the conventional ANCOVA decomposition. HC3 robust covariance is used for displayed inferential p-values and confidence intervals.">Partial &eta;&sup2; (conventional ANCOVA) <span style="cursor:help;color:#999;">&#9432;</span></th><th>Analysis status</th><th>Robustness</th>
              </tr>
            </thead>
            <tbody id="resultsTableBody"></tbody>
          </table>
        </div>
        <div class="multiplicity-note">p-values: exploratory, unadjusted for multiplicity.</div>
      </div>
    </section>
    """


# ------------------------------------------------------------------
# Analysis Details (collapsible) -- technical metadata moved out of
# the always-visible header per the Medical Affairs redesign.
# ------------------------------------------------------------------


def render_analysis_details_section():
    meta_chips = [
        "Cognitive endpoints available: <b>2</b> (ADAS-Cog13, MMSE)",
        "Plasma biomarkers available: <b>5</b> (pTau181, pTau217, Aβ42/Aβ40, GFAP, NfL)",
        "Primary GFAP/NfL platform: <b>Quanterix</b>",
        "Analysis approach: <b>observed cases, no imputation</b>",
    ]
    meta_html = "".join(f'<div class="meta-chip">{c}</div>' for c in meta_chips)
    legend_items = [
        "<b>Adjusted analysis</b> -- Prespecified ANCOVA fit; HC3 robust inference displayed (Change-from-baseline view only).",
        "<b>Descriptive only</b> -- Insufficient subgroup sample size (n &lt; 10) for the prespecified inferential model; a real, unadjusted value is still shown, with no p-value.",
        "<b>Sensitivity concern</b> -- An adjusted result that changed materially under HC3 robust covariance and/or influence-observation exclusion; see Statistical Results for the comparison.",
        "<b>Not available</b> -- No usable data at this group/month cell (n = 0).",
        "<b>Absolute view</b> -- Always descriptive: no ANCOVA was ever fit against an absolute score/concentration, only against change-from-baseline, so there is no adjusted absolute value to show.",
    ]
    legend_html = "".join(f"<li>{item}</li>" for item in legend_items)
    return f"""
    <section class="panel">
      <div class="collapsible-toggle" id="analysisDetailsToggle" onclick="toggleCollapsible(this)">
        <h2 style="margin:0;">Analysis Details</h2>
        <span class="chev">&#9656;</span>
      </div>
      <div class="collapsible-body">
        <p class="panel-sub" style="margin-top:12px;">Status definitions</p>
        <ul class="legend-detail-list">{legend_html}</ul>
        <p class="panel-sub">Cohort and analysis metadata</p>
        <div class="meta-row">{meta_html}</div>
        <p class="panel-sub" style="margin:8px 0 0;">Cohort counts reflect the validated fixed baseline diagnosis group. Not every participant contributed data to every endpoint or timepoint.</p>
      </div>
    </section>
    """


# ------------------------------------------------------------------
# Methods & Limitations (collapsible)
# ------------------------------------------------------------------

METHODS_ITEMS = [
    "Fixed baseline diagnosis group (CN / MCI / Dementia) -- validated against ADNIMERGE2's official ADSL.DX derivation.",
    "Observed cases only.",
    "No imputation.",
    "Cognitive change = follow-up value − baseline value.",
    "Biomarker analysis performed on the natural-log scale.",
    "Biomarker adjusted results back-transformed via geometric percent change = (exp(adjusted log change) − 1) × 100.",
    "Adjustment for baseline value (or log-baseline), age, and sex in every ANCOVA model.",
    "HC3 heteroscedasticity-robust covariance used for all displayed inference.",
    "Minimum cell n = 10 per diagnosis group before any ANCOVA is fit.",
    "No multiplicity correction -- all p-values are exploratory.",
    "Primary GFAP/NfL assay platform = Quanterix (Fujirebio shown as sensitivity).",
    "pTau217: ADNI4 Batch 3 reagent-lot QC-drift records excluded from primary analysis; included in a labeled sensitivity analysis.",
    "Sensitivity analyses: MMSE screening-to-baseline interval, pTau217 lot-bias inclusion, GFAP/NfL platform, HC3 vs. conventional covariance, and influential-observation exclusion.",
    "Absolute-value views (Disease Continuum, Absolute trajectories) are purely descriptive summaries (n, mean/geometric mean, one-sample t-interval 95% CI) of the same observed-case data -- no model fit, no covariate adjustment.",
]

LIMITATIONS_ITEMS = [
    "ADNI is observational and non-randomized.",
    "Results are natural-history associations, not treatment effects.",
    "Sample size differs substantially across endpoints and timepoints.",
    "Many biomarker timepoints are descriptive only.",
    "GFAP/NfL do not support the planned month-specific ANCOVA under the prespecified sample-size rule.",
    "Different plasma biomarkers/platforms must not be directly equated.",
    "pTau217 contains a documented assay lot-bias sensitivity issue.",
    "Multiple comparisons are exploratory and unadjusted.",
    "Several fitted models show influential-observation sensitivity.",
    "Disease Continuum heatmap coloring is normalized independently within each endpoint row and must never be compared across rows/endpoints.",
    "Cross-study comparison with AR1001 requires caution due to cohort, assay, endpoint, and study-design differences.",
]


def render_methods_limitations_section():
    methods_html = "".join(f"<li>{html.escape(m)}</li>" for m in METHODS_ITEMS)
    limitations_html = "".join(f"<li>{html.escape(l)}</li>" for l in LIMITATIONS_ITEMS)
    return f"""
    <section class="panel">
      <div class="collapsible-toggle" onclick="toggleCollapsible(this)">
        <h2 style="margin:0;">Methods</h2>
        <span class="chev">&#9656;</span>
      </div>
      <div class="collapsible-body">
        <ul class="methods-list">{methods_html}</ul>
      </div>
    </section>
    <section class="panel">
      <div class="collapsible-toggle" onclick="toggleCollapsible(this)">
        <h2 style="margin:0;">Limitations</h2>
        <span class="chev">&#9656;</span>
      </div>
      <div class="collapsible-body">
        <ol class="limitations-list">{limitations_html}</ol>
      </div>
    </section>
    """


# ------------------------------------------------------------------
# Key Pattern -- deterministic, template-only summaries built
# EXCLUSIVELY from already-computed, already-displayed aggregate
# numbers on the corresponding chart. Never an LLM-generated or
# free-form interpretation; never a new calculation.
# ------------------------------------------------------------------


def _classify_group_separation(values_by_group, higher_is_worse):
    """Deterministic, rule-based classification of whether CN/MCI/
    Dementia values (already on their natural scale) show the
    expected disease-severity ordering, and if so, how large the
    separation is. Returns (ordered: bool or None, magnitude:
    "marked"|"modest"|None). `ordered=None` means not enough groups
    were available to judge at all.
    """
    if any(g not in values_by_group for g in D.GROUP_ORDER):
        return None, None
    cn, mci, dem = (values_by_group[g] for g in D.GROUP_ORDER)
    if higher_is_worse:
        ordered = cn <= mci <= dem
    else:
        ordered = cn >= mci >= dem
    if not ordered:
        return False, None
    best, worst = cn, dem
    denom = abs(best) if best != 0 else (abs(worst) if worst != 0 else 1.0)
    relative_separation = abs(worst - best) / denom
    magnitude = "marked" if relative_separation >= 0.5 else "modest"
    return True, magnitude


def _change_key_pattern(points, label, higher_is_worse):
    """Deterministic, rule-based interpretation of the HC3-adjusted
    change-from-baseline trajectory at its most recent fitted
    timepoint -- describes the PATTERN (ordered gradient, magnitude,
    sensitivity flag) rather than repeating the numbers themselves,
    which remain available via hover and Statistical Results. Never
    claims causality or a treatment effect; says so plainly whenever
    the data don't support a clean conclusion, rather than forcing one.
    """
    fitted = [p for p in points if p["classification"] in (D.CLASS_ADJUSTED, D.CLASS_SENSITIVITY_CONCERN) and p["estimate"] is not None]
    if not fitted:
        return f"<b>Key pattern —</b> No adjusted (HC3) timepoints are currently available for {label} under the prespecified small-cell rule; see the Absolute view for descriptive values."
    last_month = max(p["month"] for p in fitted)
    at_last = {p["group"]: p for p in fitted if p["month"] == last_month}
    if len(at_last) < 3:
        return f"<b>Key pattern —</b> At month {last_month}, HC3-adjusted results for {label} are not available for all three groups; see Statistical Results for what is available."

    values = {g: at_last[g]["estimate"] for g in D.GROUP_ORDER}
    ordered, magnitude = _classify_group_separation(values, higher_is_worse)
    concern = any(at_last[g]["classification"] == D.CLASS_SENSITIVITY_CONCERN for g in at_last)

    if not ordered:
        pattern = f"the HC3-adjusted change from baseline at month {last_month} does not show a consistent CN &rarr; MCI &rarr; Dementia gradient"
    else:
        magnitude_word = "markedly" if magnitude == "marked" else "modestly"
        pattern = f"the HC3-adjusted change from baseline is {magnitude_word} larger progressing from CN through Dementia by month {last_month}, consistent with the expected disease-severity gradient"

    sentence = f"<b>Key pattern —</b> For {label}, {pattern}."
    if concern:
        sentence += " At least one group at this timepoint is flagged as a sensitivity concern -- see Statistical Results."
    sentence += " This describes natural-history progression only, not a treatment effect. Exact estimates and CIs are available via hover and Statistical Results."
    return sentence


def _absolute_key_pattern(points, label, higher_is_worse):
    """Deterministic, rule-based interpretation of the Absolute
    (always-descriptive) trajectory -- baseline group separation, then
    whether that separation persists into a well-supported follow-up
    month, explicitly flagging when later timepoints are too sparse to
    trust. Never repeats the raw numbers (those stay in hover /
    Statistical Results); never claims causality or a treatment
    effect; falls back to a plain "no clear pattern" statement rather
    than forcing a conclusion the data don't support.
    """
    by_month = {}
    for p in points:
        if p["estimate"] is not None:
            by_month.setdefault(p["month"], {})[p["group"]] = p

    baseline = by_month.get(0, {})
    if len(baseline) < 3:
        return f"<b>Key pattern —</b> Baseline data for {label} are not available for all three groups, so a reliable CN/MCI/Dementia comparison cannot be made. See hover for what is available."

    baseline_values = {g: baseline[g]["estimate"] for g in D.GROUP_ORDER}
    ordered, magnitude = _classify_group_separation(baseline_values, higher_is_worse)
    if not ordered:
        return f"<b>Key pattern —</b> Baseline {label} does not show a consistent CN &rarr; MCI &rarr; Dementia separation in this cohort. Exact values are available via hover and Statistical Results."

    groups_present = list(D.GROUP_ORDER)
    magnitude_word = "markedly" if magnitude == "marked" else "modestly"
    sentence = f"<b>Key pattern —</b> {label} separates {magnitude_word} across CN &rarr; MCI &rarr; Dementia at baseline, consistent with the expected disease-severity gradient."

    # Prefer the latest month where EVERY group still has a reasonably
    # sized sample (>= the same threshold used elsewhere for "n small
    # enough to matter"). ADNI has well-documented differential
    # attrition by disease stage -- a late timepoint's mean can be
    # driven by a handful of surviving participants, which would
    # misrepresent the persistence of group separation if reported
    # without this check.
    well_supported = [
        m for m, groups in by_month.items()
        if m > 0 and all(g in groups and groups[g]["n"] is not None and groups[g]["n"] >= MIN_GROUP_N_FOR_DISPLAY for g in groups_present)
    ]
    if well_supported:
        last_month = max(well_supported)
        at_last_values = {g: by_month[last_month][g]["estimate"] for g in groups_present}
        ordered_last, _ = _classify_group_separation(at_last_values, higher_is_worse)
        if ordered_last:
            sentence += f" The groups remain separated through month {last_month}."
        else:
            sentence += f" Group separation is less consistent by month {last_month}."
    else:
        sentence += " Follow-up data are too sparse across all three groups to characterize a reliable later-timepoint pattern."

    # Flag a declining/sparse tail even when a well-supported month
    # exists earlier, whenever the LATEST available month itself falls
    # short of the threshold for one or more groups.
    last_available_month = max(by_month.keys())
    if last_available_month > 0:
        last_available = by_month[last_available_month]
        if any(g in last_available and last_available[g]["n"] is not None and last_available[g]["n"] < MIN_GROUP_N_FOR_DISPLAY for g in groups_present):
            sentence += " Later estimates should be interpreted cautiously as sample sizes decline."

    sentence += " Descriptive values only -- not adjusted for covariates, and not a treatment effect. Exact values are available via hover and Statistical Results."
    return sentence


# ------------------------------------------------------------------
# POLARIS AD-Aligned Key Pattern + data-support summaries -- SEPARATE
# functions from the Overall-ADNI ones above (never called for
# population="overall", never touched by that code path) so the
# already-validated Overall-ADNI Key Pattern wording is provably
# unchanged. Both read the identical point-list shape
# build_cognitive_chart_data()/build_biomarker_chart_data() already
# produce (now sourced from adni_viz_data.polaris_data_view()), so no
# new statistic is computed here either -- only different, explicitly
# population-labeled prose built from the same n/classification/
# estimate fields.
# ------------------------------------------------------------------

POLARIS_COHORT_LABEL = "POLARIS AD–Aligned"
POLARIS_KEY_PATTERN_FOOTER = (
    " This describes the 620 eligibility-filtered POLARIS AD-Aligned participants only -- "
    "not a treatment effect, and not compared statistically to Overall ADNI."
)


def _polaris_sparse_tail_note(points, group_levels=None, min_n=MIN_GROUP_N_FOR_DISPLAY):
    """Deterministic sparse-later-follow-up clause shared by both
    POLARIS Key Pattern variants: the latest month at which every
    group still has n >= min_n ("well supported"), plus which group(s)
    fall short of that threshold at the latest AVAILABLE month, if
    different. Returns (sentence, last_well_supported_month_or_None).
    """
    group_levels = group_levels or D.GROUP_ORDER
    by_month = {}
    for p in points:
        if p["n"] is not None:
            by_month.setdefault(p["month"], {})[p["group"]] = p["n"]
    non_baseline_months = sorted(m for m in by_month if m > 0)
    if not non_baseline_months:
        return "No POLARIS follow-up data are available beyond baseline.", None

    well_supported = [m for m in non_baseline_months if all(by_month[m].get(g, 0) >= min_n for g in group_levels)]
    last_available = max(non_baseline_months)
    sparse_groups = [g for g in group_levels if by_month[last_available].get(g, 0) < min_n]

    if well_supported:
        last_wm = max(well_supported)
        if sparse_groups and last_available > last_wm:
            note = (
                f"Follow-up is well supported through Month {int(last_wm)}, while later "
                f"{', '.join(sparse_groups)} observations (Month {int(last_available)}) are too sparse for reliable interpretation."
            )
        else:
            note = f"Follow-up is well supported through Month {int(last_wm)}."
        return note, last_wm
    return "Follow-up data are too sparse across all three groups at every timepoint beyond baseline to characterize a reliable pattern.", None


def _polaris_change_key_pattern(points, label, higher_is_worse):
    """POLARIS analogue of _change_key_pattern() -- prioritizes the
    latest WELL-SUPPORTED month (not simply the latest fitted one) when
    both exist, and always appends the sparse-tail caveat plus the
    POLARIS population footer."""
    sparse_note, last_wm = _polaris_sparse_tail_note(points)
    fitted = [p for p in points if p["classification"] in (D.CLASS_ADJUSTED, D.CLASS_SENSITIVITY_CONCERN) and p["estimate"] is not None]
    if not fitted:
        return (
            f"<b>Key pattern —</b> In the {POLARIS_COHORT_LABEL} cohort, no adjusted (HC3) timepoint is "
            f"currently available for {label} under the prespecified small-cell rule. {sparse_note}{POLARIS_KEY_PATTERN_FOOTER}"
        )

    fitted_months = {p["month"] for p in fitted}
    eval_month = last_wm if (last_wm is not None and last_wm in fitted_months) else max(fitted_months)
    at_eval = {p["group"]: p for p in fitted if p["month"] == eval_month}
    if len(at_eval) < 3:
        return (
            f"<b>Key pattern —</b> In the {POLARIS_COHORT_LABEL} cohort, HC3-adjusted results for {label} are "
            f"not available for all three diagnosis groups at month {int(eval_month)}. {sparse_note}{POLARIS_KEY_PATTERN_FOOTER}"
        )

    values = {g: at_eval[g]["estimate"] for g in D.GROUP_ORDER}
    ordered, magnitude = _classify_group_separation(values, higher_is_worse)
    concern = any(at_eval[g]["classification"] == D.CLASS_SENSITIVITY_CONCERN for g in at_eval)
    if not ordered:
        pattern = f"the HC3-adjusted change from baseline at month {int(eval_month)} does not show a consistent CN &rarr; MCI &rarr; Dementia gradient"
    else:
        magnitude_word = "markedly" if magnitude == "marked" else "modestly"
        pattern = (
            f"the HC3-adjusted change from baseline is {magnitude_word} larger progressing from CN through "
            f"Dementia by month {int(eval_month)}, consistent with the expected disease-severity gradient"
        )
    sentence = f"<b>Key pattern —</b> In the {POLARIS_COHORT_LABEL} cohort, {pattern}."
    if concern:
        sentence += " At least one group at this timepoint is flagged as a sensitivity concern -- see Statistical Results."
    sentence += f" {sparse_note}{POLARIS_KEY_PATTERN_FOOTER}"
    return sentence


def _polaris_absolute_key_pattern(points, label, higher_is_worse):
    """POLARIS analogue of _absolute_key_pattern() -- same baseline-
    separation-then-persistence structure, with the shared sparse-tail
    helper and the POLARIS population footer. Never turns a
    descriptive-only later value into a stronger claim than the
    baseline-separation finding itself."""
    by_month = {}
    for p in points:
        if p["estimate"] is not None:
            by_month.setdefault(p["month"], {})[p["group"]] = p

    baseline = by_month.get(0, {})
    if len(baseline) < 3:
        return (
            f"<b>Key pattern —</b> In the {POLARIS_COHORT_LABEL} cohort, baseline data for {label} are not "
            f"available for all three groups, so a reliable CN/MCI/Dementia comparison cannot be made.{POLARIS_KEY_PATTERN_FOOTER}"
        )

    baseline_values = {g: baseline[g]["estimate"] for g in D.GROUP_ORDER}
    ordered, magnitude = _classify_group_separation(baseline_values, higher_is_worse)
    sparse_note, _ = _polaris_sparse_tail_note(points)
    if not ordered:
        return (
            f"<b>Key pattern —</b> In the {POLARIS_COHORT_LABEL} cohort, baseline {label} does not show a "
            f"consistent CN &rarr; MCI &rarr; Dementia separation. {sparse_note}{POLARIS_KEY_PATTERN_FOOTER}"
        )

    magnitude_word = "markedly" if magnitude == "marked" else "modestly"
    sentence = (
        f"<b>Key pattern —</b> In the {POLARIS_COHORT_LABEL} cohort, baseline {label} remains {magnitude_word} "
        f"separated across CN, MCI and Dementia, consistent with the expected disease-severity gradient."
    )
    sentence += f" {sparse_note}{POLARIS_KEY_PATTERN_FOOTER}"
    return sentence


def build_cognitive_data_support_summary(points):
    """Compact, deterministic data-support line for the POLARIS
    cognitive chart -- reuses the identical sparse-tail logic behind
    _polaris_sparse_tail_note(), phrased for a status line rather than
    a Key Pattern sentence. No endpoint-specific text is hardcoded --
    everything is derived from the points' own n values."""
    note, last_wm = _polaris_sparse_tail_note(points)
    if last_wm is not None:
        by_month = {}
        for p in points:
            if p["n"] is not None:
                by_month.setdefault(p["month"], {})[p["group"]] = p["n"]
        last_available = max(m for m in by_month if m > 0)
        sparse_groups = [g for g in D.GROUP_ORDER if by_month[last_available].get(g, 0) < MIN_GROUP_N_FOR_DISPLAY]
        if sparse_groups and last_available > last_wm:
            return f"Data support: strong through Month {int(last_wm)}; later {', '.join(sparse_groups)} follow-up is sparse."
        return f"Data support: strong through Month {int(last_wm)}."
    return f"Data support: {note[0].lower()}{note[1:]}"


def build_biomarker_change_support_summary(points, label):
    """Compact, deterministic data-support line for the biomarker
    CHANGE-from-baseline view, derived purely from each month's
    already-computed classification (identical across the 3 groups for
    a given cell) -- never a hardcoded per-biomarker conclusion.
    Answers "how strong is the longitudinal evidence" -- distinct from
    build_biomarker_absolute_support_summary() below, which answers the
    different question "how many people have an actual level at this
    month" for the Absolute view. Population-agnostic (the population
    context is already shown via the "Population: X" label above the
    chart) -- used for both Overall ADNI and POLARIS AD-Aligned."""
    by_month_classification = {}
    for p in points:
        by_month_classification.setdefault(p["month"], p["classification"])
    followup_months = sorted(m for m in by_month_classification if m > 0)
    adjusted_months = [m for m in followup_months if by_month_classification[m] in (D.CLASS_ADJUSTED, D.CLASS_SENSITIVITY_CONCERN)]

    if adjusted_months:
        if len(adjusted_months) == 1:
            months_str = f"Month {int(adjusted_months[0])}"
        else:
            months_str = "Months " + ", ".join(str(int(m)) for m in adjusted_months[:-1]) + f" and {int(adjusted_months[-1])}"
        return f"{label}: adjusted follow-up available at {months_str}; other follow-up points are descriptive or unavailable."
    if not followup_months or all(by_month_classification[m] == D.CLASS_NOT_AVAILABLE for m in followup_months):
        return f"{label}: no follow-up data are available for this assay/platform."
    return f"{label}: follow-up is descriptive-only; no month meets the adjusted-analysis threshold."


def build_biomarker_absolute_support_summary(points, label, min_n=MIN_GROUP_N_FOR_DISPLAY):
    """Compact, deterministic data-support line for the biomarker
    ABSOLUTE view, derived purely from each (month, group) cell's
    cross-sectional n (the correct denominator for "what is the actual
    level at this month" -- see build_biomarker_absolute_chart_data()'s
    docstring) compared against the same MIN_GROUP_N_FOR_DISPLAY
    threshold used for the isolated/sparse-point rendering rule.
    Answers "how many people support the level shown at each stage" --
    distinct from build_biomarker_change_support_summary() above, which
    describes the change-from-baseline ANCOVA/HC3 classification.
    Population-agnostic; used for both Overall ADNI and POLARIS."""
    by_month = {}
    for p in points:
        if p["n"] is not None:
            by_month.setdefault(p["month"], {})[p["group"]] = p["n"]
    if not by_month:
        return f"{label}: no data are available for this assay/platform."

    non_baseline_months = sorted(m for m in by_month if m > 0)
    if not non_baseline_months:
        return f"{label}: baseline levels available; no follow-up data."

    well_supported = [m for m in non_baseline_months if all(by_month[m].get(g, 0) >= min_n for g in D.GROUP_ORDER)]
    last_available = max(non_baseline_months)
    sparse_groups = [g for g in D.GROUP_ORDER if by_month[last_available].get(g, 0) < min_n]
    if well_supported:
        last_wm = max(well_supported)
        if sparse_groups and last_available > last_wm:
            return f"{label}: absolute levels well-supported through Month {int(last_wm)}; later {', '.join(sparse_groups)} observations are sparse."
        return f"{label}: absolute levels well-supported through Month {int(last_wm)}."
    return f"{label}: absolute levels are descriptive with limited support (n<{min_n} per group) at every follow-up month."


def build_key_patterns(data, population="overall"):
    change_fn = _change_key_pattern if population == "overall" else _polaris_change_key_pattern
    absolute_fn = _absolute_key_pattern if population == "overall" else _polaris_absolute_key_pattern

    cognitive_change, cognitive_absolute = {}, {}
    for e in COGNITIVE_ENDPOINTS:
        higher_is_worse = DISEASE_CONTINUUM_META[e["key"]]["higher_is_worse"]
        change_pts = D.build_cognitive_chart_data(data, e["key"], "primary")
        absolute_pts = D.build_cognitive_absolute_chart_data(data, e["key"], "primary")
        cognitive_change[e["key"]] = change_fn(change_pts, e["label"], higher_is_worse)
        cognitive_absolute[e["key"]] = absolute_fn(absolute_pts, e["label"], higher_is_worse)

    biomarker_change, biomarker_absolute = {}, {}
    for b in BIOMARKER_SPECS:
        higher_is_worse = DISEASE_CONTINUUM_META[b["key"]]["higher_is_worse"]
        biomarker_change[b["key"]], biomarker_absolute[b["key"]] = {}, {}
        for platform, analysis_type, _label in b["platforms"]:
            change_pts = D.build_biomarker_chart_data(data, b["key"], platform, analysis_type)
            absolute_pts = D.build_biomarker_absolute_chart_data(data, b["key"], platform, analysis_type)
            biomarker_change[b["key"]].setdefault(platform, {})[analysis_type] = change_fn(change_pts, b["label"], higher_is_worse)
            biomarker_absolute[b["key"]].setdefault(platform, {})[analysis_type] = absolute_fn(absolute_pts, b["label"], higher_is_worse)

    return {
        "cognitive": {"change": cognitive_change, "absolute": cognitive_absolute},
        "biomarkers": {"change": biomarker_change, "absolute": biomarker_absolute},
    }


def build_polaris_cognitive_data_support(data):
    """Cognitive data-support summaries -- POLARIS only (Overall ADNI's
    cognitive section is unchanged by this redesign; see
    render_cognitive_section, out of scope for the biomarker redesign)."""
    cognitive = {}
    for e in COGNITIVE_ENDPOINTS:
        pts = D.build_cognitive_chart_data(data, e["key"], "primary")
        cognitive[e["key"]] = build_cognitive_data_support_summary(pts)
    return cognitive


def build_biomarker_data_support(data):
    """Biomarker data-support summaries, view-aware (change vs.
    absolute) and population-agnostic -- built identically for Overall
    ADNI and POLARIS AD-Aligned, since biomarker follow-up sparsity is
    a real, relevant concern for BOTH populations (Overall ADNI's own
    GFAP/NfL are entirely descriptive-only cohort-wide, which Medical
    Affairs should see regardless of which population is selected)."""
    biomarker = {}
    for b in BIOMARKER_SPECS:
        biomarker[b["key"]] = {}
        for platform, analysis_type, _label in b["platforms"]:
            change_pts = D.build_biomarker_chart_data(data, b["key"], platform, analysis_type)
            absolute_pts = D.build_biomarker_absolute_chart_data(data, b["key"], platform, analysis_type)
            biomarker[b["key"]].setdefault(platform, {})[analysis_type] = {
                "change": build_biomarker_change_support_summary(change_pts, b["label"]),
                "absolute": build_biomarker_absolute_support_summary(absolute_pts, b["label"]),
            }
    return biomarker


# ------------------------------------------------------------------
# JSON payload (aggregate-only)
# ------------------------------------------------------------------


OVERALL_ADNI_N = 3030


def build_population_payload(data):
    """The population-scoped chart/table payload -- built identically
    from either population's own "data" dict (Overall ADNI's
    adni_viz_data.load_all() output, or POLARIS's
    adni_viz_data.polaris_data_view() output). Every function called
    here already existed before population-awareness was added and is
    completely unchanged; only the "data" argument differs."""
    cognitive_change = {e["key"]: D.build_cognitive_chart_data(data, e["key"], "primary") for e in COGNITIVE_ENDPOINTS}
    cognitive_absolute = {e["key"]: D.build_cognitive_absolute_chart_data(data, e["key"], "primary") for e in COGNITIVE_ENDPOINTS}

    biomarkers_change, biomarkers_absolute = {}, {}
    for b in BIOMARKER_SPECS:
        biomarkers_change[b["key"]] = {}
        biomarkers_absolute[b["key"]] = {}
        for platform, analysis_type, _label in b["platforms"]:
            biomarkers_change[b["key"]].setdefault(platform, {})[analysis_type] = D.build_biomarker_chart_data(
                data, b["key"], platform, analysis_type
            )
            biomarkers_absolute[b["key"]].setdefault(platform, {})[analysis_type] = D.build_biomarker_absolute_chart_data(
                data, b["key"], platform, analysis_type
            )

    results_table = {}
    for e in COGNITIVE_ENDPOINTS:
        results_table[f'{e["key"]}|__cognitive__|primary'] = D.build_results_table_rows(
            data, e["key"], "", "primary", is_cognitive=True
        )
    results_table["MMSE|__cognitive__|sensitivity_interval_excl"] = D.build_results_table_rows(
        data, "MMSE", "", "sensitivity_interval_excl", is_cognitive=True
    )
    for b in BIOMARKER_SPECS:
        for platform, analysis_type, _label in b["platforms"]:
            results_table[f'{b["key"]}|{platform}|{analysis_type}'] = D.build_results_table_rows(
                data, b["key"], platform, analysis_type, is_cognitive=False
            )

    return {
        "cognitiveChange": cognitive_change,
        "cognitiveAbsolute": cognitive_absolute,
        "biomarkersChange": biomarkers_change,
        "biomarkersAbsolute": biomarkers_absolute,
        "resultsTable": results_table,
    }


def build_payload(data, polaris_data=None, polaris_traj_data=None):
    disease_continuum = []
    for row in D.build_disease_continuum_data(data):
        meta = DISEASE_CONTINUUM_META[row["key"]]
        disease_continuum.append({
            "key": row["key"], "label": row["label"], "cells": row["cells"],
            "higherIsWorse": meta["higher_is_worse"], "digits": meta["digits"],
        })

    overall_population = build_population_payload(data)
    overall_population.update({
        "label": "Overall ADNI",
        "populationNote": "",
        "n": OVERALL_ADNI_N,
        "keyPatterns": build_key_patterns(data, population="overall"),
        "cognitiveDataSupport": {},
        "biomarkerDataSupport": build_biomarker_data_support(data),
    })

    populations = {"overall": overall_population}

    payload = {
        "groupColors": D.GROUP_COLORS,
        "groupOrder": D.GROUP_ORDER,
        "targetMonths": D.TARGET_MONTHS,
        "minGroupN": MIN_GROUP_N_FOR_DISPLAY,
        "cognitiveEndpoints": COGNITIVE_ENDPOINTS,
        "biomarkerSpecs": BIOMARKER_SPECS,
        "diseaseContinuum": disease_continuum,
        "populations": populations,
    }
    if polaris_data is not None:
        funnel = D.build_polaris_funnel(polaris_data["attrition"])
        payload["polaris"] = {
            "funnel": funnel,
            "profile": D.build_polaris_profile(polaris_data["profile"]),
        }
        if polaris_traj_data is not None:
            polaris_n = funnel[-1]["remaining_n"] if funnel else None
            polaris_population = build_population_payload(polaris_traj_data)
            polaris_population.update({
                "label": POLARIS_COHORT_LABEL,
                "populationNote": f"n={polaris_n} baseline eligible" if polaris_n is not None else "",
                "n": polaris_n,
                "keyPatterns": build_key_patterns(polaris_traj_data, population="polaris"),
                "cognitiveDataSupport": build_polaris_cognitive_data_support(polaris_traj_data),
                "biomarkerDataSupport": build_biomarker_data_support(polaris_traj_data),
            })
            populations["polaris"] = polaris_population
    return payload


# ------------------------------------------------------------------
# JS (kept as a plain string -- not an f-string -- to avoid escaping
# every brace; the JSON payload is spliced in via a single placeholder)
# ------------------------------------------------------------------

DASHBOARD_JS = r"""
<script>
const DATA = __PAYLOAD_JSON__;
const CLASS_A = "A. Adjusted analysis";
const CLASS_B = "B. Descriptive only";
const CLASS_C = "C. Sensitivity concern";
const CLASS_D = "D. Not available";

// Spelled-out diagnosis-group labels -- CN/MCI abbreviations alone
// assume the reader already knows the jargon; used anywhere a group
// code is shown directly to the reader (e.g. Disease Continuum column
// headers), never in place of the short codes used internally for
// data keys/lookups.
const GROUP_FULL_LABELS = {
  CN: "CN (Cognitively Normal)",
  MCI: "MCI (Mild Cognitive Impairment)",
  Dementia: "Dementia",
};

// Compact multi-line form of the same labels, for placing directly ON
// the Disease Continuum heatmap's column headers (a single-line full
// label wouldn't fit 3-across) -- Plotly's ticktext supports <br>.
const GROUP_AXIS_LABELS = {
  CN: "CN<br>(Cognitively Normal)",
  MCI: "MCI<br>(Mild Cognitive<br>Impairment)",
  Dementia: "Dementia",
};

function statusLetter(cls) {
  if (!cls) return "D";
  return cls.charAt(0);
}

function fmtNum(v, d) {
  if (v === null || v === undefined) return "—";
  return Number(v).toFixed(d === undefined ? 2 : d);
}
function fmtCI(lo, hi, d) {
  if (lo === null || lo === undefined || hi === null || hi === undefined) return "—";
  return "[" + fmtNum(lo, d) + ", " + fmtNum(hi, d) + "]";
}
function fmtP(p) {
  if (p === null || p === undefined) return "—";
  return p < 0.0001 ? "<0.0001" : Number(p).toFixed(4);
}

// Wraps server-generated HTML in a freshly-created span so replacing a
// persistent container's innerHTML (which does NOT retrigger a CSS
// animation on the container itself, since that node was never
// removed/recreated) still gets a smooth fade-in: the wrapped span IS
// a brand-new DOM node every time, so its own `animation` CSS rule
// plays automatically on each update -- no manual reflow/class-toggle
// trick needed.
function fadeSpan(html) {
  return '<span class="fade-in-span">' + html + '</span>';
}

function buildTooltip(pt, yLabel, yUnit) {
  let lines = [];
  lines.push("<b>" + pt.group + "</b> · Month " + pt.month);
  const isAdjusted = pt.classification === CLASS_A || pt.classification === CLASS_C;
  if (pt.estimate !== null && pt.estimate !== undefined) {
    // Never "adjusted mean" wording for a descriptive (B) point -- and
    // never an HC3/p-value claim for one either (guarded below).
    const label = isAdjusted ? ("HC3-adjusted " + yLabel.toLowerCase()) : ("Descriptive " + yLabel.toLowerCase());
    lines.push(label + ": " + fmtNum(pt.estimate, 2) + (yUnit || ""));
  }
  if (hasDisplayableCI(pt)) {
    const ciLabel = isAdjusted ? "95% CI (HC3)" : "95% CI (descriptive)";
    lines.push(ciLabel + ": " + fmtCI(pt.ci_lower, pt.ci_upper, 2) + (yUnit || ""));
  }
  lines.push("n = " + (pt.n === null ? "—" : pt.n));
  lines.push("Analysis status: " + pt.classification);
  if (isAdjusted && pt.overall_p_hc3 !== null && pt.overall_p_hc3 !== undefined) {
    lines.push("HC3 overall group p: " + fmtP(pt.overall_p_hc3));
  }
  if (pt.classification === CLASS_C) {
    lines.push("<span style='color:#b8860b'>Sensitivity status: Concern — " + pt.reason + "</span>");
  } else if (pt.classification === CLASS_A) {
    lines.push("Sensitivity status: None flagged");
  } else if (pt.classification === CLASS_B) {
    lines.push("<b>Descriptive only</b> — no inferential model fit; no p-value.");
    if (pt.reason) lines.push(pt.reason);
  }
  return lines.join("<br>");
}

// A point gets a real (non-null) CI whenever one is present -- both
// the change-from-baseline view (HC3 for A/C, a genuine descriptive
// one-sample interval for B when derivable) and the always-
// descriptive Absolute view carry a real CI when n >= 2.
function hasDisplayableCI(p) {
  return p.ci_lower !== null && p.ci_lower !== undefined;
}

// Point classification for rendering purposes (distinct from the
// scientific A/B/C/D classification, which this never alters):
//   - solid:    Adjusted or Sensitivity-concern (A/C) -- always
//               n >= minGroupN by construction (ANCOVA requires it).
//               Drawn as a normal filled point, connected with a
//               SOLID line to an adjacent solid or non-isolated-
//               descriptive point.
//   - isolated: n < minGroupN, however classified -- shown as its own
//               hollow point with a "⚠ n=" label, but NEVER connected
//               to a neighbor, so a single extremely sparse
//               observation (e.g. dementia pTau181 n=1) never reads
//               as part of a reliable trajectory.
//   - descriptive (non-isolated): Descriptive-only (B) with
//               n >= minGroupN -- e.g. large-n Absolute-view points,
//               which are always classified descriptive regardless of
//               sample size. Shown as a hollow point, connected to an
//               adjacent solid/descriptive point with a DASHED
//               segment (never solid), since it's still real,
//               reasonably-supported data, just never model-adjusted.
//   - unavailable: D or no estimate -- a gap, exactly as before.
function buildGroupTraces(pointsByGroup, yLabel, yUnit, showErrorBars) {
  if (showErrorBars === undefined) showErrorBars = true;
  const traces = [];
  const warnX = [], warnY = [];

  const isPlottable = function (p) { return p && p.classification !== CLASS_D && p.estimate !== null; };
  const isIsolated = function (p) { return isPlottable(p) && p.n !== null && p.n < DATA.minGroupN; };
  const isConnectable = function (p) { return isPlottable(p) && !isIsolated(p); };

  Object.keys(pointsByGroup).forEach(function (group) {
    // Always index by the FULL month grid (not just the months that have a
    // value) so a missing/unavailable month becomes an explicit null in the
    // middle of the array -- Plotly's default connectgaps=false then breaks
    // the line there instead of visually bridging straight across a gap
    // month, which would otherwise imply continuous measurement.
    const byMonth = {};
    pointsByGroup[group].forEach(function (p) { byMonth[p.month] = p; });
    const months = DATA.targetMonths;
    const pts = months.map(function (m) { return byMonth[m] || null; });
    if (!pts.some(isPlottable)) return;
    const color = DATA.groupColors[group];

    const errorYFor = function (mask) {
      return {
        type: "data", symmetric: false, visible: showErrorBars,
        array: pts.map(function (p) { return (p && mask(p) && hasDisplayableCI(p)) ? (p.ci_upper - p.estimate) : 0; }),
        arrayminus: pts.map(function (p) { return (p && mask(p) && hasDisplayableCI(p)) ? (p.estimate - p.ci_lower) : 0; }),
        color: color, thickness: 1.5, width: 4,
      };
    };

    // 1) Solid trace: Adjusted/Sensitivity-concern points only, with a
    // solid connecting line between adjacent solid points. Deliberately
    // the BOLDEST element on the chart (thicker line, larger marker) --
    // this is the part of the trend that's actually trustworthy, and it
    // should read as the obvious focal point at a glance, not compete
    // visually with the sparse/isolated points below.
    const ySolid = pts.map(function (p) { return (isConnectable(p) && p.classification !== CLASS_B) ? p.estimate : null; });
    traces.push({
      x: months, y: ySolid, connectgaps: false,
      error_y: errorYFor(function (p) { return p.classification !== CLASS_B; }),
      mode: "lines+markers", name: group, legendgroup: group,
      line: { color: color, width: 3.5 },
      marker: { color: color, line: { color: color, width: 2 }, size: 12, symbol: "circle" },
      hovertext: pts.map(function (p) { return (isConnectable(p) && p.classification !== CLASS_B) ? buildTooltip(p, yLabel, yUnit) : ""; }),
      hovertemplate: "%{hovertext}<extra></extra>",
    });

    // 2) Descriptive, non-isolated points (real, reasonably-sized
    // sample, just never model-adjusted): hollow markers, no line of
    // their own -- connections are drawn as dashed segments below.
    // Still part of "the trend" (real, connected data), so kept at
    // near-full visual weight -- only the isolated/sparse points (4)
    // are the ones deliberately faded.
    const descX = [], descY = [], descHover = [], descErrHi = [], descErrLo = [];
    pts.forEach(function (p) {
      if (isConnectable(p) && p.classification === CLASS_B) {
        descX.push(p.month); descY.push(p.estimate); descHover.push(buildTooltip(p, yLabel, yUnit));
        descErrHi.push(hasDisplayableCI(p) ? (p.ci_upper - p.estimate) : 0);
        descErrLo.push(hasDisplayableCI(p) ? (p.estimate - p.ci_lower) : 0);
      }
    });
    if (descX.length) {
      traces.push({
        x: descX, y: descY, mode: "markers", legendgroup: group, showlegend: false,
        error_y: { type: "data", symmetric: false, visible: showErrorBars, array: descErrHi, arrayminus: descErrLo, color: color, thickness: 1.5, width: 4 },
        marker: { color: "white", line: { color: color, width: 2.5 }, size: 9, symbol: "circle" },
        hovertext: descHover, hovertemplate: "%{hovertext}<extra></extra>",
      });
    }

    // 3) Dashed connecting segments: any adjacent (in month order) pair
    // that are BOTH connectable (solid or non-isolated-descriptive) and
    // where AT LEAST ONE is descriptive -- a solid-solid pair is
    // already drawn by trace (1) and skipped here.
    for (let i = 0; i < months.length - 1; i++) {
      const a = pts[i], b = pts[i + 1];
      if (!isConnectable(a) || !isConnectable(b)) continue;
      if (a.classification !== CLASS_B && b.classification !== CLASS_B) continue;
      traces.push({
        x: [months[i], months[i + 1]], y: [a.estimate, b.estimate],
        mode: "lines", line: { color: color, width: 2, dash: "dot" },
        legendgroup: group, showlegend: false, hoverinfo: "skip",
      });
    }

    // 4) Isolated (extremely sparse, n < minGroupN) points: hollow,
    // never connected to any neighbor, always carrying a "n=" label
    // since an n this small is exactly the case worth flagging
    // directly on the chart rather than leaving to hover alone. This is
    // the "noise" -- deliberately faded (lower opacity, smaller marker,
    // thinner ring, muted label) so it never visually competes with the
    // bold, trustworthy trend in trace (1). Still fully visible and
    // fully in hover/tooltip, just clearly secondary at a glance.
    const isoX = [], isoY = [], isoHover = [], isoLabel = [], isoErrHi = [], isoErrLo = [];
    pts.forEach(function (p) {
      if (isIsolated(p)) {
        isoX.push(p.month); isoY.push(p.estimate); isoHover.push(buildTooltip(p, yLabel, yUnit));
        isoLabel.push("⚠ n=" + p.n);
        isoErrHi.push(hasDisplayableCI(p) ? (p.ci_upper - p.estimate) : 0);
        isoErrLo.push(hasDisplayableCI(p) ? (p.estimate - p.ci_lower) : 0);
      }
    });
    if (isoX.length) {
      traces.push({
        x: isoX, y: isoY, mode: "markers+text", legendgroup: group, showlegend: false, opacity: 0.45,
        error_y: { type: "data", symmetric: false, visible: showErrorBars, array: isoErrHi, arrayminus: isoErrLo, color: color, thickness: 1, width: 3 },
        marker: { color: "white", line: { color: color, width: 1.5 }, size: 6, symbol: "circle" },
        text: isoLabel, textposition: "bottom center", textfont: { size: 8, color: "#999" },
        hovertext: isoHover, hovertemplate: "%{hovertext}<extra></extra>",
      });
    }

    pts.filter(function (p) { return p && p.classification === CLASS_C && p.estimate !== null; }).forEach(function (p) {
      warnX.push(p.month); warnY.push(p.estimate);
    });
  });
  if (warnX.length) {
    traces.push({
      x: warnX, y: warnY, mode: "text", text: warnX.map(function () { return "⚠"; }),
      textposition: "top center", textfont: { size: 12, color: "#b8860b" },
      showlegend: false, hoverinfo: "skip",
    });
  }
  return traces;
}

// Both layout builders below include layout.transition -- Plotly.react()
// smoothly animates (position/size/color interpolation) between the
// previous and new trace data whenever this is set, instead of
// snapping instantly, so switching endpoint/biomarker/view/population
// redraws the SAME chart div with a smooth morph rather than a hard cut.
const CHART_TRANSITION = { duration: 400, easing: "cubic-in-out" };

function baseLayout(yTitle, upLabel, downLabel) {
  return {
    margin: { t: 20, r: 20, l: 55, b: 45 },
    xaxis: { title: "Time from baseline (months)", tickmode: "array", tickvals: [0, 6, 12, 18, 24, 36, 48] },
    yaxis: { title: yTitle + (upLabel ? "  (↑ " + upLabel + " / ↓ " + downLabel + ")" : "") },
    shapes: [{ type: "line", x0: 0, x1: 48, y0: 0, y1: 0, line: { dash: "dash", color: "#999", width: 1 } }],
    annotations: [{ x: 48, y: 0, xref: "x", yref: "y", text: "No change from baseline", showarrow: false, yshift: 10, font: { size: 10, color: "#999" }, xanchor: "right" }],
    legend: { orientation: "h", y: -0.28 },
    hovermode: "closest",
    transition: CHART_TRANSITION,
    font: { family: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif", size: 12 },
  };
}

// A layout WITHOUT the "no change from baseline" zero-line, used for
// the Absolute view -- an absolute score/concentration has no
// meaningful "zero change" reference line the way change-from-
// baseline does.
function absoluteLayout(yTitle) {
  return {
    margin: { t: 20, r: 20, l: 55, b: 45 },
    xaxis: { title: "Time from baseline (months)", tickmode: "array", tickvals: [0, 6, 12, 18, 24, 36, 48] },
    yaxis: { title: yTitle },
    legend: { orientation: "h", y: -0.28 },
    hovermode: "closest",
    transition: CHART_TRANSITION,
    font: { family: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif", size: 12 },
  };
}

// Biomarker charts can carry an extremely wide confidence interval on
// a single sparse (isolated, n < minGroupN) point -- e.g. a 2-3-person
// cell's CI can span hundreds or thousands of concentration units,
// vs. a real signal range of tens of units across every well-
// supported point. Left to Plotly's default autorange (which sizes
// the axis to fit every error bar), that one point silently drags the
// whole y-axis out, visually crushing every other, better-supported
// point into a thin band -- exactly the "clustered dots, can't see
// the pattern" symptom this exists to prevent. The fix ranges the
// axis on point ESTIMATES (every point, so isolated points' own
// position stays visible) plus the full CI only for well-supported
// (non-isolated) points -- an isolated point's outsized CI is still
// drawn, just clipped at the axis edge rather than dictating the
// scale. Returns null (falls back to Plotly's normal autorange) when
// there's nothing usable to range over.
function computeSensibleYRange(pointsByGroup) {
  const values = [];
  Object.keys(pointsByGroup).forEach(function (group) {
    pointsByGroup[group].forEach(function (p) {
      if (p.estimate === null || p.estimate === undefined) return;
      values.push(p.estimate);
      const isIsolated = p.n !== null && p.n !== undefined && p.n < DATA.minGroupN;
      if (!isIsolated && hasDisplayableCI(p)) {
        values.push(p.ci_lower, p.ci_upper);
      }
    });
  });
  if (!values.length) return null;
  const lo = Math.min.apply(null, values), hi = Math.max.apply(null, values);
  if (lo === hi) return null;
  const pad = (hi - lo) * 0.12;
  return [lo - pad, hi + pad];
}

// Compact, muted-by-default sensitivity indicator -- only becomes a
// visually prominent warning when a genuine C-status (sensitivity
// concern) is present; routine small-cell descriptiveness stays
// low-key, per the redesign's "reserve prominent warnings for
// findings that materially affect interpretation."
function computeCompactWarnHtml(seriesByGroup) {
  const all = Object.keys(seriesByGroup).reduce(function (acc, g) { return acc.concat(seriesByGroup[g]); }, []);
  const nonBaseline = all.filter(function (p) { return p.month !== 0 && p.classification !== CLASS_D; });
  if (!nonBaseline.length) return "";
  const concernCount = nonBaseline.filter(function (p) { return p.classification === CLASS_C; }).length;
  if (concernCount > 0) {
    return '<div class="compact-warn compact-warn--concern">&#9888; Sensitivity concern at ' + concernCount + ' of ' + nonBaseline.length + ' timepoints — see Statistical Results</div>';
  }
  const descriptiveCount = nonBaseline.filter(function (p) { return p.classification === CLASS_B; }).length;
  if (descriptiveCount > 0) {
    return '<div class="compact-warn">&#9888; ' + descriptiveCount + '/' + nonBaseline.length + ' timepoints descriptive-only</div>';
  }
  return "";
}

// ------------------------------------------------------------------
// Population selector + POLARIS AD-aligned cohort panel
//
// Purely a display toggle over already-built DATA.polaris (itself
// built server-side from the two governed POLARIS aggregate CSVs --
// see adni_viz_data.build_polaris_funnel / build_polaris_profile).
// Never touches DATA.cognitive*/biomarkers*/diseaseContinuum -- those
// stay exactly what they were regardless of which population button
// is active, per the "do not change trajectories yet" scope.
// ------------------------------------------------------------------

const CATEGORICAL_DISPLAY = {
  "Baseline diagnosis": {
    order: ["CN", "MCI", "Dementia"],
    colors: { CN: "#2196F3", MCI: "#FF9800", Dementia: "#F44336" },
    labels: {},
  },
  "Sex": {
    order: ["Female", "Male"],
    colors: { Female: "#c2255c", Male: "#2e5fa3" },
    labels: {},
  },
  "APOE4 carrier": {
    order: ["1.0", "0.0", "Missing"],
    colors: { "1.0": "#b8860b", "0.0": "#8fa8c7", "Missing": "#ccc" },
    labels: { "1.0": "Carrier", "0.0": "Non-carrier", "Missing": "Missing" },
  },
};

let currentPopulation = "overall";
let polarisRendered = false;

// The ONLY place population selection touches the trajectory/results
// charts: everything downstream (buildGroupTraces, key patterns,
// results table) reads DATA.populations[currentPopulation] -- there is
// no in-browser filtering, refitting, or participant-level access
// anywhere in this file; every number was already computed server-side
// (adni_viz_data.polaris_data_view() / run_adni_polaris_trajectories.py)
// and simply selected here.
function currentPop() {
  return DATA.populations[currentPopulation];
}

function populationLabelText(pop) {
  return "Population: " + pop.label + (pop.populationNote ? " · " + pop.populationNote : "");
}

const POPULATION_SUMMARY_DESC = {
  overall: "Broad natural-history cohort.",
  polaris: "Trial-aligned cohort based on POLARIS eligibility criteria.",
};

function setPopulation(pop) {
  currentPopulation = pop;
  document.querySelectorAll('[data-population]').forEach(function (b) { b.classList.toggle('active', b.dataset.population === pop); });

  const pd = currentPop();
  document.getElementById("populationSummaryName").textContent = pd.label + " · n = " + pd.n.toLocaleString();
  document.getElementById("populationSummaryDesc").textContent = POPULATION_SUMMARY_DESC[pop] || "";

  // Retention badge -- only meaningful for POLARIS (a fraction of
  // Overall ADNI); fully removed, not just faded, when Overall ADNI
  // itself is selected since there's nothing to compare it to.
  if (pop === "polaris") {
    const overallN = DATA.populations.overall.n;
    document.getElementById("populationRetentionPct").textContent = (pd.n / overallN * 100).toFixed(1) + "%";
    document.getElementById("populationRetentionN").textContent = pd.n.toLocaleString() + " / " + overallN.toLocaleString();
  }
  document.getElementById("populationRetention").classList.toggle("is-hidden", pop !== "polaris");

  // visibility, not display -- the note always reserves its own line
  // in Disease Continuum's card so switching population can't grow it
  // (and its stretch-aligned header-card sibling) taller.
  document.getElementById("diseaseContinuumPolarisNote").classList.toggle("is-hidden", pop !== "polaris");

  // The "View cohort definition" link only exists for POLARIS -- fully
  // removed (see .view-cohort-def-link.is-hidden), not just visually
  // disabled. Leaving POLARIS always resets its detail back to
  // collapsed, so re-selecting POLARIS later never shows it already
  // expanded.
  document.getElementById("viewCohortDefLink").classList.toggle("is-hidden", pop !== "polaris");
  document.getElementById("viewCohortDefLink").textContent = "View cohort definition";
  document.getElementById("polarisCollapsibleBody").classList.remove("open");

  if (pop === "polaris" && !polarisRendered) {
    renderPolarisPanel();
    polarisRendered = true;
  }

  const label = populationLabelText(currentPop());
  document.getElementById("cognitivePopulationLabel").innerHTML = fadeSpan(label);
  document.getElementById("biomarkerPopulationLabel").innerHTML = fadeSpan(label);
  document.getElementById("resultsPopulationLabel").innerHTML = fadeSpan(label);

  renderCognitiveChart();
  renderBiomarkerChart();
  renderResultsTable();
}

function togglePolarisCohortDefinition() {
  const open = document.getElementById("polarisCollapsibleBody").classList.toggle("open");
  document.getElementById("viewCohortDefLink").textContent = open ? "Hide cohort definition" : "View cohort definition";
}

function fmtNumericSummary(s) {
  if (!s || s.n === null || s.n === undefined || s.mean === null) return '<span class="profile-numeric-value">—</span>';
  const sdStr = s.sd === null || s.sd === undefined ? "" : " &plusmn; " + fmtNum(s.sd, 1);
  return '<span class="profile-numeric-value">' + fmtNum(s.mean, 1) + sdStr + " (n=" + s.n.toLocaleString() + ")</span>";
}

function buildProfileGridHtml(profile) {
  return profile.map(function (v) {
    if (v.kind === "numeric") {
      return '<div class="profile-card">' +
        '<div class="profile-var-label">' + v.variable + '</div>' +
        '<div class="profile-pop-row"><span class="profile-pop-tag">Overall ADNI</span>' + fmtNumericSummary(v.overall) + '</div>' +
        '<div class="profile-pop-row"><span class="profile-pop-tag">POLARIS</span>' + fmtNumericSummary(v.polaris) + '</div>' +
        (v.note ? '<div class="profile-note">' + v.note + '</div>' : '') +
        '</div>';
    }
    const disp = CATEGORICAL_DISPLAY[v.variable] || { order: v.levels.map(function (l) { return l.level; }), colors: {}, labels: {} };
    const presentLevels = disp.order.filter(function (lvl) { return v.levels.some(function (l) { return l.level === lvl; }); });
    const barHtml = function (popKey) {
      return presentLevels.map(function (lvl) {
        const entry = v.levels.find(function (l) { return l.level === lvl; });
        const pct = entry[popKey].percent;
        if (pct === null || pct === undefined) return "";
        const color = disp.colors[lvl] || "#999";
        const label = disp.labels[lvl] || lvl;
        return '<div class="profile-bar-seg" style="width:' + pct + '%;background:' + color + ';" title="' + label + " " + pct + '%"></div>';
      }).join("");
    };
    const legendHtml = presentLevels.map(function (lvl) {
      const color = disp.colors[lvl] || "#999";
      return '<span style="color:' + color + ';">&#9679;</span> ' + (disp.labels[lvl] || lvl);
    }).join(" &nbsp; ");
    const apoeNote = v.variable === "APOE4 carrier"
      ? '<div class="profile-note">APOE4-carrier prevalence differs between the overall and eligibility-filtered populations.</div>' : "";
    return '<div class="profile-card">' +
      '<div class="profile-var-label">' + v.variable + '</div>' +
      '<div class="profile-pop-row"><span class="profile-pop-tag">Overall ADNI</span><div class="profile-bar">' + barHtml("overall") + '</div></div>' +
      '<div class="profile-pop-row"><span class="profile-pop-tag">POLARIS</span><div class="profile-bar">' + barHtml("polaris") + '</div></div>' +
      '<div class="profile-note">' + legendHtml + '</div>' +
      apoeNote +
      '</div>';
  }).join("");
}

function renderPolarisPanel() {
  const funnel = DATA.polaris.funnel;
  const finalStep = funnel[funnel.length - 1];

  document.getElementById("polarisContextBox").innerHTML =
    "<b>" + finalStep.remaining_n.toLocaleString() + "</b> ADNI participants meet the validated eligibility definition: " +
    "baseline MMSE &ge;20 and QC-passed amyloid PET Centiloid &ge;30 within &plusmn;90 days of clinical baseline.<br>" +
    '<span class="polaris-disclaimer">This cohort has been eligibility-filtered only. It has <b>not</b> been propensity-score ' +
    "matched to an AR1001 trial cohort and should not be interpreted as an external control.</span>";

  let funnelHtml = "";
  funnel.forEach(function (s, i) {
    funnelHtml += '<div class="funnel-step"><div class="funnel-step-label">' + s.step + '</div>' +
      '<div class="funnel-step-n">' + s.remaining_n.toLocaleString() + "</div>" +
      (i > 0
        ? '<div class="funnel-step-meta">&minus;' + s.excluded_n.toLocaleString() + " excluded &middot; " +
          (s.percent_retained_of_previous === null ? "—" : s.percent_retained_of_previous + "%") + " retained from previous step</div>"
        : "") +
      "</div>";
    if (i < funnel.length - 1) funnelHtml += '<div class="funnel-arrow">&#8595;</div>';
  });
  document.getElementById("polarisFunnel").innerHTML = funnelHtml;

  document.getElementById("polarisProfileGrid").innerHTML = buildProfileGridHtml(DATA.polaris.profile);
}

// ------------------------------------------------------------------
// Disease Continuum heatmap
// ------------------------------------------------------------------

function renderDiseaseContinuum() {
  const rows = DATA.diseaseContinuum;
  const groups = DATA.groupOrder;
  const z = [], text = [], hover = [];
  rows.forEach(function (row) {
    const vals = groups.map(function (g) { return row.cells[g].value; });
    const validVals = vals.filter(function (v) { return v !== null; });
    if (!validVals.length) {
      z.push(groups.map(function () { return null; }));
      text.push(groups.map(function () { return "—"; }));
      hover.push(groups.map(function (g) { return "<b>" + (GROUP_FULL_LABELS[g] || g) + "</b> · " + row.label + "<br>No baseline data available."; }));
      return;
    }
    const lo = Math.min.apply(null, validVals), hi = Math.max.apply(null, validVals);
    const range = hi - lo;
    z.push(vals.map(function (v) {
      if (v === null) return null;
      let norm = range > 0 ? (v - lo) / range : 0.5;
      if (!row.higherIsWorse) norm = 1 - norm;
      return norm;
    }));
    text.push(vals.map(function (v) { return v === null ? "—" : fmtNum(v, row.digits); }));
    hover.push(groups.map(function (g) {
      const label = GROUP_FULL_LABELS[g] || g;
      const c = row.cells[g];
      if (c.value === null) return "<b>" + label + "</b> · " + row.label + "<br>No baseline data available.";
      return "<b>" + label + "</b> · " + row.label + "<br>Value: " + fmtNum(c.value, row.digits) +
        "<br>95% CI (descriptive): " + fmtCI(c.ci_lower, c.ci_upper, row.digits) + "<br>n = " + c.n;
    }));
  });

  // Row label carries an explicit direction arrow (up = higher value is
  // more abnormal, down = lower value is more abnormal) so directionality
  // never has to be inferred or remembered -- it's right on the axis.
  const trace = {
    type: "heatmap", z: z, x: groups,
    y: rows.map(function (r) { return r.label + (r.higherIsWorse ? " ↑" : " ↓"); }),
    text: text, texttemplate: "%{text}", textfont: { size: 12, color: "#1a1a1a" },
    customdata: hover, hovertemplate: "%{customdata}<extra></extra>",
    colorscale: [[0, "#eef2f8"], [1, "#2e5fa3"]], showscale: false, xgap: 4, ygap: 4, zmin: 0, zmax: 1,
  };
  Plotly.react("diseaseContinuumChart", [trace], {
    margin: { t: 56, r: 10, l: 110, b: 10 },
    xaxis: {
      side: "top", tickfont: { size: 11, color: "#333" },
      tickvals: groups, ticktext: groups.map(function (g) { return GROUP_AXIS_LABELS[g] || g; }),
    },
    yaxis: { autorange: "reversed", tickfont: { size: 12, color: "#333" } },
    font: { family: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif", size: 12 },
  }, { displayModeBar: false, responsive: true });
}

// ------------------------------------------------------------------
// Cognitive Trajectories
// ------------------------------------------------------------------

let cognitiveEndpointKey = DATA.cognitiveEndpoints[0].key;
let cognitiveView = "absolute";

function setCognitiveEndpoint(key) {
  cognitiveEndpointKey = key;
  document.querySelectorAll('[data-endpoint]').forEach(function (b) { b.classList.toggle('active', b.dataset.endpoint === key); });
  renderCognitiveChart();
}

function setCognitiveView(view) {
  cognitiveView = view;
  document.querySelectorAll('[data-view]').forEach(function (b) {
    if (b.closest('section').querySelector('#cognitiveChart')) {
      b.classList.toggle('active', b.dataset.view === view);
    }
  });
  renderCognitiveChart();
}

function renderCognitiveChart() {
  const pop = currentPop();
  const spec = DATA.cognitiveEndpoints.find(function (e) { return e.key === cognitiveEndpointKey; });
  const seriesSource = cognitiveView === "absolute" ? pop.cognitiveAbsolute : pop.cognitiveChange;
  const series = seriesSource[cognitiveEndpointKey];
  const pointsByGroup = {};
  DATA.groupOrder.forEach(function (g) { pointsByGroup[g] = series.filter(function (p) { return p.group === g; }); });

  const supportDiv = document.getElementById("cognitiveDataSupport");
  const supportText = pop.cognitiveDataSupport && pop.cognitiveDataSupport[cognitiveEndpointKey];
  if (supportText) {
    supportDiv.innerHTML = fadeSpan(supportText);
    supportDiv.style.display = "";
  } else {
    supportDiv.style.display = "none";
  }

  const warnDiv = document.getElementById("cognitiveWarn");
  warnDiv.innerHTML = cognitiveView === "change" ? computeCompactWarnHtml(pointsByGroup) : "";

  // Same rule as the biomarker chart: the y-axis range is sized off
  // point estimates plus the CI of non-isolated (n >= minGroupN) points
  // only -- an isolated, faded-out point's own huge error bar (still
  // drawn on the chart, still visible on hover) must not stretch the
  // axis and flatten every well-supported point's real signal.
  const yRange = computeSensibleYRange(pointsByGroup);

  if (cognitiveView === "absolute") {
    const traces = buildGroupTraces(pointsByGroup, spec.label + " score", "");
    const layout = absoluteLayout(spec.label + " score (↑ " + spec.up_label + " / ↓ " + spec.down_label + ")");
    if (yRange) layout.yaxis.range = yRange;
    Plotly.react("cognitiveChart", traces, layout, { displayModeBar: false, responsive: true });
  } else {
    const traces = buildGroupTraces(pointsByGroup, "Change from baseline", "");
    const layout = baseLayout("Change from baseline", spec.up_label, spec.down_label);
    if (yRange) layout.yaxis.range = yRange;
    Plotly.react("cognitiveChart", traces, layout, { displayModeBar: false, responsive: true });
  }

  document.getElementById("cognitiveKeyPattern").innerHTML = fadeSpan(pop.keyPatterns.cognitive[cognitiveView][cognitiveEndpointKey]);
}

// ------------------------------------------------------------------
// Plasma Biomarker Trajectories
// ------------------------------------------------------------------

let currentBiomarker = DATA.biomarkerSpecs[0].key;
let currentPlatform = null;
let currentAnalysisType = null;
let biomarkerView = "absolute";

function setBiomarker(key) {
  const spec = DATA.biomarkerSpecs.find(function (b) { return b.key === key; });
  currentBiomarker = key;
  currentPlatform = spec.platforms[0][0];
  currentAnalysisType = spec.platforms[0][1];
  document.querySelectorAll('[data-biomarker]').forEach(function (b) { b.classList.toggle('active', b.dataset.biomarker === key); });

  const platformGroup = document.getElementById("biomarkerPlatformGroup");
  if (spec.platforms.length > 1) {
    platformGroup.style.display = "inline-flex";
    platformGroup.innerHTML = spec.platforms.map(function (p, i) {
      return '<button class="toggle-btn' + (i === 0 ? ' active' : '') + '" data-platform="' + p[0] + '" data-analysis="' + p[1] + '" onclick="setBiomarkerToggle(\'' + p[0] + '\',\'' + p[1] + '\')">' + p[2] + '</button>';
    }).join("");
  } else {
    platformGroup.style.display = "none";
    platformGroup.innerHTML = "";
  }
  renderBiomarkerChart();
}

function setBiomarkerToggle(platform, analysisType) {
  currentPlatform = platform;
  currentAnalysisType = analysisType;
  document.querySelectorAll('#biomarkerPlatformGroup .toggle-btn').forEach(function (b) {
    b.classList.toggle('active', b.dataset.platform === platform && b.dataset.analysis === analysisType);
  });
  renderBiomarkerChart();
}

function setBiomarkerView(view) {
  biomarkerView = view;
  document.querySelectorAll('[data-view]').forEach(function (b) {
    if (b.closest('section').querySelector('#biomarkerChart')) {
      b.classList.toggle('active', b.dataset.view === view);
    }
  });
  renderBiomarkerChart();
}

function renderBiomarkerChart() {
  const pop = currentPop();
  const spec = DATA.biomarkerSpecs.find(function (b) { return b.key === currentBiomarker; });
  const seriesSource = biomarkerView === "absolute" ? pop.biomarkersAbsolute : pop.biomarkersChange;
  const seriesByGroupList = seriesSource[currentBiomarker][currentPlatform][currentAnalysisType];

  const metaRow = document.getElementById("biomarkerMetaRow");
  const availableMonths = [...new Set(seriesByGroupList.filter(function (p) { return p.classification !== CLASS_D; }).map(function (p) { return p.month; }))].sort(function (a, b) { return a - b; });
  metaRow.innerHTML =
    '<div class="meta-chip">Assay/platform: <b>' + currentPlatform.replace(/_/g, " ") + '</b></div>' +
    '<div class="meta-chip">Analysis type: <b>' + currentAnalysisType.replace(/_/g, " ") + '</b></div>' +
    '<div class="meta-chip">Available timepoints: <b>' + (availableMonths.length ? availableMonths.join(", ") : "none") + '</b></div>';

  const supportDiv = document.getElementById("biomarkerDataSupport");
  const supportEntry = pop.biomarkerDataSupport && pop.biomarkerDataSupport[currentBiomarker] &&
    pop.biomarkerDataSupport[currentBiomarker][currentPlatform] && pop.biomarkerDataSupport[currentBiomarker][currentPlatform][currentAnalysisType];
  const supportText = supportEntry && supportEntry[biomarkerView];
  if (supportText) {
    supportDiv.innerHTML = fadeSpan(supportText);
    supportDiv.style.display = "";
  } else {
    supportDiv.style.display = "none";
  }

  const pointsByGroup = {};
  DATA.groupOrder.forEach(function (g) { pointsByGroup[g] = seriesByGroupList.filter(function (p) { return p.group === g; }); });

  const warnDiv = document.getElementById("biomarkerWarn");
  warnDiv.innerHTML = biomarkerView === "change" ? computeCompactWarnHtml(pointsByGroup) : "";

  const yRange = computeSensibleYRange(pointsByGroup);
  // Error bars hidden on-chart for biomarkers -- CI is still exact and
  // available via hover; the isolated/dashed/solid marker styling
  // already carries the confidence signal without the visual weight
  // of a permanently-drawn bar on every point (see buildTooltip()).
  if (biomarkerView === "absolute") {
    const traces = buildGroupTraces(pointsByGroup, spec.label + " concentration", "", false);
    const layout = absoluteLayout(spec.label + " geometric mean concentration");
    if (yRange) layout.yaxis.range = yRange;
    Plotly.react("biomarkerChart", traces, layout, { displayModeBar: false, responsive: true });
  } else {
    const traces = buildGroupTraces(pointsByGroup, "Geometric mean % change", "%", false);
    const layout = baseLayout("Geometric mean % change from baseline");
    if (yRange) layout.yaxis.range = yRange;
    Plotly.react("biomarkerChart", traces, layout, { displayModeBar: false, responsive: true });
  }

  document.getElementById("biomarkerInterpretation").innerHTML = fadeSpan(spec.interpretation);
  document.getElementById("biomarkerKeyPattern").innerHTML = fadeSpan(pop.keyPatterns.biomarkers[biomarkerView][currentBiomarker][currentPlatform][currentAnalysisType]);
}

function toggleCollapsible(headerEl) {
  headerEl.classList.toggle("open");
  headerEl.nextElementSibling.classList.toggle("open");
}

function statusPill(cls) {
  const letter = statusLetter(cls);
  return '<span class="status-pill ' + letter + '">' + letter + '</span> ' + cls.replace(/^[A-D]\.\s*/, "");
}

function renderResultsTable() {
  const key = document.getElementById("resultsTableSelect").value;
  const rows = currentPop().resultsTable[key] || [];
  const tbody = document.getElementById("resultsTableBody");
  let html = "";
  rows.forEach(function (r, i) {
    const rowId = "detail-" + i;
    const warn = r.classification === CLASS_C ? '<span class="warning-badge">⚠</span>' : "";
    html += '<tr class="result-row" onclick="toggleDetailRow(\'' + rowId + '\')">' +
      '<td>' + r.month + '</td>' +
      '<td>' + r.group_or_comparison + '</td>' +
      '<td>' + (r.n === null ? "—" : r.n) + '</td>' +
      '<td>' + fmtNum(r.estimate) + '</td>' +
      '<td>' + fmtCI(r.ci_lower, r.ci_upper) + '</td>' +
      '<td>' + fmtNum(r.overall_F, 2) + '</td>' +
      '<td>' + fmtP(r.hc3_p) + '</td>' +
      '<td>' + fmtNum(r.partial_eta_squared, 3) + '</td>' +
      '<td>' + statusPill(r.classification) + warn + '</td>' +
      '<td>' + (r.row_type === "pairwise" ? (r.multiplicity_adjustment || "Exploratory, unadjusted for multiplicity") : "—") + '</td>' +
      '</tr>';
    html += '<tr class="detail-row" id="' + rowId + '"><td colspan="10">' + renderDetail(r) + '</td></tr>';
  });
  tbody.innerHTML = html;
}

function renderDetail(r) {
  let parts = [];
  parts.push("<b>Reason:</b> " + (r.reason || "—"));
  if (r.hc3_detail) {
    parts.push("<b>HC3 result:</b> estimate=" + fmtNum(r.hc3_detail.estimate) + ", SE=" + fmtNum(r.hc3_detail.se) + ", 95% CI=" + fmtCI(r.hc3_detail.ci_lower, r.hc3_detail.ci_upper) + (r.hc3_detail.p !== null ? ", p=" + fmtP(r.hc3_detail.p) : ""));
  }
  if (r.conventional) {
    parts.push("<b>Conventional result:</b> estimate=" + fmtNum(r.conventional.estimate) + ", SE=" + fmtNum(r.conventional.se) + ", 95% CI=" + fmtCI(r.conventional.ci_lower, r.conventional.ci_upper) + (r.conventional.p !== null ? ", p=" + fmtP(r.conventional.p) : ""));
  }
  if (r.influence_detail) {
    parts.push("<b>Influence-sensitivity result:</b> estimate=" + fmtNum(r.influence_detail.estimate) + ", SE=" + fmtNum(r.influence_detail.se) + ", 95% CI=" + fmtCI(r.influence_detail.ci_lower, r.influence_detail.ci_upper));
  }
  return parts.join("<br>");
}

function toggleDetailRow(id) {
  document.getElementById(id).classList.toggle("open");
}

document.addEventListener("DOMContentLoaded", function () {
  renderDiseaseContinuum();
  renderCognitiveChart();
  setBiomarker(DATA.biomarkerSpecs[0].key);
  renderResultsTable();
});
</script>
"""


# ------------------------------------------------------------------
# Page assembly
# ------------------------------------------------------------------


def render_page(data, polaris_data=None, polaris_traj_data=None):
    payload = build_payload(data, polaris_data, polaris_traj_data)
    payload_json = json.dumps(payload)
    plotlyjs_lib = pyo.get_plotlyjs()

    js = DASHBOARD_JS.replace("__PAYLOAD_JSON__", payload_json)

    header_continuum_row = f"""
    <div class="header-continuum-row">
      <div class="header-left-col">
        <div class="page-title-block">
          <div class="page-title">ADNI Natural History Dashboard</div>
          <div class="page-title-sub">Cognitive and biomarker progression across the Alzheimer's disease continuum &middot; Source: ADNI</div>
        </div>
        {render_header_section()}
      </div>
      <div class="header-right-col">
        {render_disease_continuum_section()}
      </div>
    </div>
    """

    trajectories_row = f"""
    <div class="trajectories-row">
      {render_cognitive_section()}
      {render_biomarker_section()}
    </div>
    """

    body = (
        trajectories_row
        + render_results_table_section()
        + render_analysis_details_section()
        + render_methods_limitations_section()
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ADNI Natural History</title>
<style>{PAGE_CSS}</style>
</head>
<body>
{render_nav_bar('biomarker')}
<main>
  {header_continuum_row}
  {body}
</main>
<script>{plotlyjs_lib}</script>
{js}
</body>
</html>"""


def build_dashboard_html(outputs_dir=ADNI_OUTPUTS_DIR):
    """
    Single entry point: loads every aggregate file through
    adni_viz_data's governed loader, asserts nothing forbidden was
    touched (the loader itself raises on any violation), and returns
    the full HTML string. Raises adni_viz_data.DataGovernanceError if
    governance is violated -- never falls back to a partial page.
    """
    data = D.load_all(outputs_dir)
    polaris_data = D.load_polaris_data(outputs_dir)
    polaris_traj_data = D.polaris_data_view(outputs_dir)
    return render_page(data, polaris_data, polaris_traj_data)


if __name__ == "__main__":
    html_out = build_dashboard_html()
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_out)
    print(f"=== SAVED: {OUTPUT_HTML} ===")
