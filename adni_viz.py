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
  main {{ margin: 0 auto; max-width: 1360px; padding: 20px 32px 0; }}
  @media (max-width: 1100px) {{ main {{ padding: 16px 16px 0; }} }}

  .page-title-block {{ margin-bottom: 8px; }}
  .page-title {{ font-size: 21px; font-weight: 700; letter-spacing: -0.01em; color: {NAV_BG}; }}
  .page-title-sub {{ font-size: 12.5px; color: #666; margin-top: 3px; line-height: 1.5; }}
  .page-context-note {{
    display: flex; align-items: center; gap: 7px; margin-top: 5px; color: #536176;
    font-size: 11.5px; line-height: 1.4;
  }}
  .page-context-note-icon {{ flex-shrink: 0; color: {ARIBIO_BLUE}; font-size: 12px; }}

  /* Sticky step-flow navigator (A-G) -- lets the reader always see
     where they are in the linear Define -> Eligibility -> Profile ->
     Cognitive -> Biomarker -> Stats -> Methods narrative and jump
     straight to any step, instead of the old 2-column header/trajectories
     layout that put no explicit flow structure on screen at all.
     Active-state highlighting is scroll-driven (IntersectionObserver,
     see initFlowNav() in the JS), not just a click state. */
  .flow-nav {{
    position: sticky; top: 0; z-index: 40; display: flex; align-items: center; gap: 2px;
    background: white; border: 1px solid {SURFACE_BORDER}; border-radius: {CARD_RADIUS};
    box-shadow: {CARD_SHADOW}; padding: 6px 10px; margin: 0 0 16px; overflow-x: auto;
    width: max-content; max-width: 100%;
  }}
  .flow-nav-item {{
    display: flex; align-items: center; gap: 6px; text-decoration: none; color: #556;
    padding: 5px 9px; border-radius: 7px; white-space: nowrap; font-size: 12px; font-weight: 600;
    transition: background-color 0.15s ease, color 0.15s ease; flex-shrink: 0;
  }}
  .flow-nav-item:hover {{ background: {SURFACE_TINT}; }}
  .flow-nav-item.active {{ background: {ARIBIO_BLUE}; color: white; }}
  .flow-nav-item.active .flow-nav-badge {{ background: white; color: {ARIBIO_BLUE}; }}
  .flow-nav-badge {{
    flex-shrink: 0; width: 19px; height: 19px; border-radius: 50%; background: {SURFACE_TINT}; color: {ARIBIO_BLUE};
    font-size: 10.5px; font-weight: 700; display: flex; align-items: center; justify-content: center;
  }}
  .flow-nav-arrow {{ color: #d5d9e0; font-size: 12px; flex-shrink: 0; }}
  @media (max-width: 900px) {{ .flow-nav-label {{ display: none; }} }}

  /* Section header: a colored step badge (A-G) beside the title, so
     the page's own linear order is visible at a glance on every panel,
     not just in the flow-nav above. */
  .step-header {{ display: flex; align-items: center; gap: 9px; margin-bottom: 4px; }}
  .step-header h2 {{ margin: 0; }}
  .step-badge {{
    flex-shrink: 0; width: 25px; height: 25px; border-radius: 50%; background: {ARIBIO_BLUE}; color: white;
    font-size: 12px; font-weight: 700; display: flex; align-items: center; justify-content: center;
  }}
  .step-badge--sub {{ background: {SURFACE_TINT}; color: {ARIBIO_BLUE}; border: 1px solid {SURFACE_BORDER}; }}

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
  .header-info-card {{
    display: block; margin: 8px 0 12px;
  }}

  /* Line-style swatches (solid/dotted/gap) for buildGroupTraces()'s three
     connector styles -- same hover-title-for-full-definition pattern as
     the point-status glyphs above, just a drawn line instead of a symbol
     since none of these three are single Unicode-representable marks. */
  .cl-item .cl-swatch {{ display: inline-block; width: 18px; vertical-align: middle; margin-right: 3px; }}
  .cl-swatch-solid {{ height: 0; border-top: 3px solid #556; }}
  .cl-swatch-dotted {{ height: 0; border-top: 3px dotted #556; }}
  .cl-swatch-gap {{ display: inline-flex; justify-content: space-between; }}
  .cl-swatch-gap span {{ width: 6px; height: 0; border-top: 3px solid #556; align-self: center; }}
  /* In-card line-style key for Cognitive Trajectories. Plasma Biomarker
     Trajectories renders the identical row invisibly (visibility:
     hidden, see render_biomarker_section()) purely to reserve the same
     height, so both cards' population-note/toggle-row/chart still
     start at the same Y position even though only one of them shows
     a real key. */
  /* NOTE: this previously carried a hardcoded `margin-top: -54px` left
     over from an earlier header layout's exact spacing -- once the
     content above it changed (step-badge headers, tighter panel
     padding, the new compare-mode/diagnosis-group toggle rows), that
     fixed offset pulled this legend row up into the panel-sub text
     above it. Normal (non-negative) flow only, below. */
  .panel-key-row {{ display: flex; flex-direction: column; gap: 4px; align-items: stretch; font-size: 12px; color: #556; margin: 2px 0 10px; }}
  .panel-key-row .cl-item {{ display: grid; grid-template-columns: 22px 76px minmax(0, 1fr); align-items: center; cursor: help; }}
  .panel-key-row .cl-label {{ font-weight: 600; color: #3f4a5a; }}
  .panel-key-row .cl-description {{ color: #6b7280; line-height: 1.25; }}

  .header-info-stats {{
    background: transparent; padding: 0; overflow: visible; display: flex; align-items: stretch;
  }}
  /* Big value, small label underneath, each tile separated by a
     hairline divider (not its own bordered/shadowed box) so the row
     reads as one connected strip of numbers rather than a grid of
     separate cards. Every tile gets a thin colored top accent -- the
     three diagnosis groups reuse GROUP_COLORS (the same blue/orange/
     red used everywhere else on the page, so the same color means the
     same group in every chart), the rest share the neutral ARIBIO_BLUE
     already used for their value text. */
  .header-info-stat-row {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); width: 100%; gap: 10px; }}
  .header-info-stat-item {{
    display: flex; flex-direction: column; justify-content: center; padding: 13px 15px 11px;
    background: white; border: 1px solid {SURFACE_BORDER}; border-radius: 10px;
    box-shadow: {CARD_SHADOW}; position: relative; min-width: 0; overflow: hidden;
  }}
  .header-info-stat-item::before {{
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--stat-accent, {ARIBIO_BLUE});
  }}
  .header-info-stat-value {{ font-size: 28px; font-weight: 750; color: {ARIBIO_BLUE}; letter-spacing: -0.025em; line-height: 1.05; }}
  .header-info-stat-label {{
    font-size: 10.5px; font-weight: 600; letter-spacing: 0.02em; color: #778; margin-top: 3px; white-space: nowrap;
  }}
  .acronym-key {{
    display: flex; flex-wrap: wrap; gap: 6px 18px; align-items: center;
    margin-top: 7px; padding: 0 2px; color: #6b7280; font-size: 10.5px; line-height: 1.4;
  }}
  .acronym-key b {{ color: {ARIBIO_BLUE}; font-weight: 700; }}
  @media (max-width: 980px) {{
    .header-info-stat-row {{ grid-template-columns: repeat(3, 1fr); }}
  }}
  @media (max-width: 620px) {{
    .header-info-stat-row {{ grid-template-columns: repeat(2, 1fr); }}
  }}

  .meta-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }}
  .meta-chip {{
    background: white; border: 1px solid {SURFACE_BORDER}; border-radius: 999px; padding: 5px 12px;
    font-size: 12px; color: #444; box-shadow: {CARD_SHADOW}; animation: contentFadeIn 0.3s ease;
  }}
  .meta-chip b {{ color: {ARIBIO_BLUE}; }}

  /* D + E side by side on wide screens, with "corresponding elements
     aligned" -- both cards stretch to the row's full height
     (align-items: stretch), and everything ABOVE the chart (subtitle,
     toggle rows, meta-row, notes-zone) gets a matched min-height
     between the two cards, so both charts start at the identical Y
     position even though the two cards' actual content differs (E's
     toggle row has one more control group -- the platform switcher --
     than D's, and both cards' toggle rows can independently show/hide
     a 2nd "by diagnosis group" row). The reserved heights below are
     sized for the WORST case (2-line-wrapped toggle row + the 2nd row
     visible) so switching either card's controls never shifts its
     neighbor. Falls back to a plain single column below 1300px, where
     there's no side-by-side alignment left to preserve. */
  .trajectories-row {{ display: flex; flex-wrap: wrap; gap: 14px; align-items: stretch; margin-bottom: 14px; }}
  .trajectories-row > section.panel {{ flex: 1 1 480px; min-width: 420px; margin-bottom: 0; display: flex; flex-direction: column; }}
  .trajectories-row > section.panel > .panel-sub {{ min-height: 50px; }}
  .trajectories-row .toggle-row-block {{ min-height: 106px; }}
  .trajectories-row .meta-row {{ min-height: 30px; }}
  .trajectories-row .notes-zone {{ min-height: 26px; }}
  .trajectories-row > section.panel .chart-card {{ flex: 1 1 auto; }}
  @media (max-width: 1300px) {{
    .trajectories-row {{ flex-direction: column; }}
    .trajectories-row > section.panel > .panel-sub,
    .trajectories-row .toggle-row-block,
    .trajectories-row .meta-row,
    .trajectories-row .notes-zone {{ min-height: 0; }}
  }}

  section.panel {{
    background: white; border-radius: {CARD_RADIUS}; box-shadow: {CARD_SHADOW};
    padding: 16px 20px; margin-bottom: 14px; scroll-margin-top: 56px;
  }}
  section.panel h2 {{ font-size: 15.5px; font-weight: 700; margin: 0; color: {NAV_BG}; }}
  .panel-sub {{ font-size: 12px; color: #666; margin: 0 0 12px; line-height: 1.55; max-width: 900px; }}

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
     "Population: Demo - n=620 baseline eligible") -- the explicit
     boundary marker required wherever a chart could otherwise be
     mistaken for having silently switched population. */
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

  /* Horizontal flow (row, wrapping), not the old vertical stack -- more
     compact and reads left-to-right like the rest of the page's flow
     metaphor (flow-nav above, step badges on every section). */
  .polaris-funnel {{ display: flex; flex-wrap: wrap; align-items: stretch; gap: 3px; margin: 6px 0 2px; }}
  .funnel-step {{
    background: {SURFACE_TINT}; border: 1px solid {SURFACE_BORDER}; border-radius: 8px; padding: 7px 12px;
    text-align: center; flex: 1 1 130px; min-width: 120px; animation: contentFadeIn 0.35s ease;
  }}
  .funnel-step-label {{ font-size: 10.5px; color: #556; font-weight: 600; line-height: 1.25; }}
  .funnel-step-n {{ font-size: 17px; font-weight: 700; color: {ARIBIO_BLUE}; }}
  .funnel-step-meta {{ font-size: 9.5px; color: #888; margin-top: 1px; line-height: 1.3; }}
  .funnel-arrow {{ font-size: 13px; color: #ccc; flex-shrink: 0; align-self: center; padding: 0 1px; }}

  .profile-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; margin-top: 4px; }}
  .profile-card {{ border: 1px solid {SURFACE_BORDER}; border-radius: 8px; padding: 10px 12px; animation: contentFadeIn 0.35s ease; }}
  .profile-var-label {{ font-size: 12px; font-weight: 700; color: #1a1a1a; margin-bottom: 6px; }}
  .profile-pop-row {{ display: flex; align-items: center; gap: 8px; font-size: 11.5px; color: #444; margin-bottom: 4px; }}
  .profile-pop-tag {{ display: inline-block; width: 78px; flex-shrink: 0; font-weight: 600; color: #667; }}
  .profile-numeric-value {{ color: #333; }}
  .profile-bar {{ flex: 1; height: 10px; border-radius: 5px; overflow: hidden; display: flex; background: #eee; }}
  .profile-bar-seg {{ height: 100%; }}
  .profile-note {{ font-size: 10.5px; color: #888; margin-top: 6px; line-height: 1.5; }}

  /* A. Define Target Population -- preset picker cards. Forced to
     exactly one row (repeat(6, 1fr), not auto-fit) so all 6 presets
     are visible and comparable at a glance without scrolling past
     them -- cards shrink to fit rather than wrapping to a 2nd row. */
  .preset-card-grid {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 8px; margin-top: 10px; }}
  @media (max-width: 900px) {{ .preset-card-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }} }}
  @media (max-width: 560px) {{ .preset-card-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
  .preset-card {{
    text-align: left; border: 1.5px solid {SURFACE_BORDER}; border-radius: 9px; padding: 9px 10px;
    background: white; cursor: pointer; font-family: inherit; min-width: 0;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
  }}
  .preset-card:hover {{ border-color: {ARIBIO_BLUE}; }}
  .preset-card.active {{ border-color: {ARIBIO_ACCENT}; box-shadow: 0 0 0 1px {ARIBIO_ACCENT}; background: {ACCENT_BG}; }}
  .preset-card-label {{ font-size: 12px; font-weight: 700; color: #1a1a1a; margin-bottom: 3px; line-height: 1.3; }}
  .preset-card-desc {{ font-size: 10.5px; color: #667; line-height: 1.35; margin-bottom: 6px; }}
  .preset-card-n {{ font-size: 11.5px; font-weight: 700; color: {ARIBIO_BLUE}; }}
  .preset-summary-row {{ font-size: 13px; color: #333; margin-top: 12px; padding-top: 10px; border-top: 1px dashed {SURFACE_BORDER}; }}
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
      <div class="header-info-stats">
        <div class="header-info-stat-row">
          <div class="header-info-stat-item">
            <div class="header-info-stat-value">3,030</div>
            <div class="header-info-stat-label">Overall ADNI participants</div>
          </div>
          <div class="header-info-stat-item" style="--stat-accent:{D.GROUP_COLORS['CN']}">
            <div class="header-info-stat-value" style="color:{D.GROUP_COLORS['CN']}">1,215</div>
            <div class="header-info-stat-label">CN</div>
          </div>
          <div class="header-info-stat-item" style="--stat-accent:{D.GROUP_COLORS['MCI']}">
            <div class="header-info-stat-value" style="color:{D.GROUP_COLORS['MCI']}">1,338</div>
            <div class="header-info-stat-label">MCI</div>
          </div>
          <div class="header-info-stat-item" style="--stat-accent:{D.GROUP_COLORS['Dementia']}">
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
      <div class="acronym-key" aria-label="Acronym key">
        <span><b>ADNI</b> &mdash; Alzheimer&rsquo;s Disease Neuroimaging Initiative</span>
        <span><b>CN</b> &mdash; Cognitively Normal</span>
        <span><b>MCI</b> &mdash; Mild Cognitive Impairment</span>
      </div>
    </div>
    """


# ------------------------------------------------------------------
# Sticky step-flow navigator -- the A-G anchors every section header's
# own step-badge points back to. Active-state highlighting is scroll-
# driven (see initFlowNav() in the JS), not just a click state.
# ------------------------------------------------------------------

FLOW_STEPS = [
    ("step-a", "A", "Define"),
    ("step-b", "B", "Eligibility"),
    ("step-c", "C", "Profile"),
    ("step-d", "D", "Cognitive"),
    ("step-e", "E", "Biomarker"),
    ("step-f", "F", "Stats"),
    ("step-g", "G", "Methods"),
]


def render_flow_nav():
    items = []
    for i, (anchor, letter, label) in enumerate(FLOW_STEPS):
        if i > 0:
            items.append('<span class="flow-nav-arrow">&#8594;</span>')
        items.append(
            f'<a href="#{anchor}" class="flow-nav-item" data-step="{anchor}">'
            f'<span class="flow-nav-badge">{letter}</span><span class="flow-nav-label">{html.escape(label)}</span></a>'
        )
    return f'<nav class="flow-nav" id="flowNav">{"".join(items)}</nav>'


def _step_header(letter, title):
    return f'<div class="step-header"><span class="step-badge">{letter}</span><h2>{html.escape(title)}</h2></div>'


# ------------------------------------------------------------------
# A. Define Target Population -- a compact, honestly-labeled preset
# picker (NOT a live filter builder -- see the module docstring and
# adni_eligibility.PRESET_LIBRARY). POLARIS-like eligibility is simply
# the first card, not a structurally distinct "the only alternative"
# population anymore.
# ------------------------------------------------------------------


def render_define_population_section(preset_catalog):
    cards = []
    for p in preset_catalog:
        cards.append(f"""
          <button type="button" class="preset-card" data-preset="{html.escape(p['id'])}" onclick="selectPreset('{p['id']}')">
            <div class="preset-card-label">{html.escape(p['label'])}</div>
            <div class="preset-card-desc">{html.escape(p['description'])}</div>
            <div class="preset-card-n">n = {p['n']:,}</div>
          </button>""")
    return f"""
    <section class="panel" id="step-a">
      {_step_header('A', 'Define Target Population')}
      <p class="panel-sub">Choose an eligibility preset to build a natural-history / placebo-reference population. Presets are precomputed, not a live filter &mdash; see Methods for why.</p>
      <div class="preset-card-grid">{"".join(cards)}</div>
      <div class="preset-summary-row" id="targetPopulationSummary"></div>
    </section>
    """


# ------------------------------------------------------------------
# B. Eligibility / Cohort Flow -- the funnel + eligible/ineligible/
# not-evaluable breakdown, answering "how did 3,030 ADNI participants
# become this target cohort?"
# ------------------------------------------------------------------


def render_eligibility_funnel_section():
    return f"""
    <section class="panel" id="step-b">
      {_step_header('B', 'Eligibility / Cohort Flow')}
      <p class="panel-sub">Each step narrows the cohort by one condition. &ldquo;Available&rdquo; steps flag missing data; other steps flag a threshold not met.</p>
      <div id="eligibilitySummary"></div>
      <div class="polaris-funnel" id="eligibilityFunnel"></div>
    </section>
    """


# ------------------------------------------------------------------
# C. Target Population Profile vs Overall ADNI -- compact cards by
# default (age, MMSE, diagnosis, Centiloid), full baseline-
# characteristics detail (sex, APOE4, biomarker availability, ADAS-
# Cog13) available on expansion. Purely descriptive, same n/mean/sd/
# median/percent shape the POLARIS profile already used -- no p-value.
# ------------------------------------------------------------------


def render_population_profile_section():
    return f"""
    <section class="panel" id="step-c">
      {_step_header('C', 'Target Population Profile vs Overall ADNI')}
      <p class="panel-sub">Baseline characteristics, descriptive only &mdash; Target is a subset of Overall ADNI, so no statistical test applies.</p>
      <div class="profile-grid" id="populationProfileGrid"></div>
      <div class="collapsible-toggle" onclick="toggleCollapsible(this)">
        <span style="font-size:13px;font-weight:600;color:#2e5fa3;">More baseline characteristics (sex, APOE4, biomarker availability, ADAS-Cog13)</span>
        <span class="chev">&#9656;</span>
      </div>
      <div class="collapsible-body">
        <div class="profile-grid" id="populationProfileGridDetail" style="margin-top:12px;"></div>
      </div>
    </section>
    """


# ------------------------------------------------------------------
# Disease Continuum -- Overall ADNI only, moved to Section G (no
# longer interrupts the primary A-through-E workflow). Population-
# aware trajectories do NOT extend here (an explicit future decision,
# not silently computed now).
# ------------------------------------------------------------------


def render_disease_continuum_section():
    return f"""
    <section class="panel">
      <h2>Disease Continuum (Overall ADNI)</h2>
      <p class="panel-sub"><b>Darker = greater abnormality (not comparable across endpoints).</b> Baseline values, Overall ADNI only.</p>
      <div id="diseaseContinuumChart" class="continuum-card"></div>
    </section>
    """


# ------------------------------------------------------------------
# Cognitive Trajectories
# ------------------------------------------------------------------


# Line-style key row, reusing the page header's .cl-item/.cl-swatch
# markup. It lives above the population note on
# the Cognitive card (where the chart it explains actually is) --
# Plasma Biomarker Trajectories carries the SAME markup, character for
# character, but invisible (visibility:hidden, not display:none, so it
# still reserves its layout height) purely so both cards' population-
# note/toggle-row/meta-row/chart start at the identical Y position
# again once Cognitive gains this extra row. See CHART_LINE_STYLE_KEY_HTML.
CHART_LINE_STYLE_KEY_HTML = """
        <span class="cl-item" title="Connects two Adjusted or Sensitivity-concern points -- the model-adjusted, trustworthy trend."><span class="cl-swatch cl-swatch-solid"></span><span class="cl-label">Solid line</span><span class="cl-description">Model-adjusted trend between supported points.</span></span>
        <span class="cl-item" title="Connects a pair of points where at least one is Descriptive only -- real, connected data, just never model-adjusted."><span class="cl-swatch cl-swatch-dotted"></span><span class="cl-label">Dotted line</span><span class="cl-description">Descriptive trend; at least one point is unadjusted.</span></span>
        <span class="cl-item" title="No connecting line -- no usable data for that month, or the neighboring point is too sparse (isolated) to connect to."><span class="cl-swatch cl-swatch-gap"><span></span><span></span></span><span class="cl-label">Gap</span><span class="cl-description">Missing or too-sparse neighboring data.</span></span>
"""


def render_cognitive_progression_section():
    endpoint_btns = "".join(
        f'<button class="toggle-btn{" active" if i == 0 else ""}" data-endpoint="{e["key"]}" onclick="setCognitiveEndpoint(\'{e["key"]}\')">{html.escape(e["label"])}</button>'
        for i, e in enumerate(COGNITIVE_ENDPOINTS)
    )
    dx_btns = "".join(
        f'<button class="toggle-btn{" active" if g == "MCI" else ""}" data-cognitive-dxgroup="{g}" onclick="setCognitiveCompareGroup(\'{g}\')">{g}</button>'
        for g in ["CN", "MCI", "Dementia"]
    )
    return f"""
    <section class="panel" id="step-d">
      {_step_header('D', 'Cognitive Progression Comparison')}
      <p class="panel-sub">Overall ADNI and Target Population plotted together &mdash; descriptive, never tested against each other. Pooled view is always descriptive; switch to By diagnosis group for adjusted (HC3) estimates.</p>
      <div class="panel-key-row">{CHART_LINE_STYLE_KEY_HTML}</div>
      <div class="population-note" id="cognitivePopulationLabel">Target: none selected</div>
      <div class="toggle-row-block">
        <div class="toggle-row">
          <div class="toggle-group">{endpoint_btns}</div>
          <div class="toggle-group" id="cognitiveCompareModeGroup">
            <button class="toggle-btn active" data-mode="pooled" onclick="setCognitiveCompareMode('pooled')">Pooled (all diagnoses)</button>
            <button class="toggle-btn" data-mode="byGroup" onclick="setCognitiveCompareMode('byGroup')">By diagnosis group</button>
          </div>
          <div class="toggle-group">
            <button class="toggle-btn" data-view="absolute" onclick="setCognitiveView('absolute')">Absolute</button>
            <button class="toggle-btn active" data-view="change" onclick="setCognitiveView('change')">Change from baseline</button>
          </div>
        </div>
        <div class="toggle-row" id="cognitiveCompareGroupRow" style="display:none;">
          <div class="toggle-group">{dx_btns}</div>
        </div>
      </div>
      <div class="meta-row" id="cognitiveMetaRow"></div>
      <div class="notes-zone">
        <div id="cognitiveDataSupport" class="data-support-note" style="display:none;"></div>
        <div id="cognitiveWarn"></div>
      </div>
      <div id="cognitiveChart" class="chart-card"></div>
      <div class="interpretation-note" style="visibility:hidden;" aria-hidden="true">placeholder</div>
      <div id="cognitiveKeyPattern" class="key-pattern"></div>
    </section>
    """


# ------------------------------------------------------------------
# E. Biomarker Progression Comparison
# ------------------------------------------------------------------


def render_biomarker_progression_section():
    biomarker_btns = "".join(
        f'<button class="toggle-btn{" active" if i == 0 else ""}" data-biomarker="{b["key"]}" onclick="setBiomarker(\'{b["key"]}\')">{html.escape(b["label"])}</button>'
        for i, b in enumerate(BIOMARKER_SPECS)
    )
    dx_btns = "".join(
        f'<button class="toggle-btn{" active" if g == "MCI" else ""}" data-biomarker-dxgroup="{g}" onclick="setBiomarkerCompareGroup(\'{g}\')">{g}</button>'
        for g in ["CN", "MCI", "Dementia"]
    )
    return f"""
    <section class="panel" id="step-e">
      {_step_header('E', 'Biomarker Progression Comparison')}
      <p class="panel-sub">Overall ADNI and Target Population plotted together. Pooled view uses each biomarker's primary assay; never combines platforms.</p>
      <div class="panel-key-row" style="visibility:hidden;" aria-hidden="true">{CHART_LINE_STYLE_KEY_HTML}</div>
      <div class="population-note" id="biomarkerPopulationLabel">Target: none selected</div>
      <div class="toggle-row-block">
        <div class="toggle-row">
          <div class="toggle-group">{biomarker_btns}</div>
          <div class="toggle-group" id="biomarkerCompareModeGroup">
            <button class="toggle-btn active" data-mode="pooled" onclick="setBiomarkerCompareMode('pooled')">Pooled (all diagnoses)</button>
            <button class="toggle-btn" data-mode="byGroup" onclick="setBiomarkerCompareMode('byGroup')">By diagnosis group</button>
          </div>
          <div class="toggle-group">
            <button class="toggle-btn" data-view="absolute" onclick="setBiomarkerView('absolute')">Absolute</button>
            <button class="toggle-btn active" data-view="change" onclick="setBiomarkerView('change')">% change from baseline</button>
          </div>
          <div class="toggle-group" id="biomarkerPlatformGroup"></div>
        </div>
        <div class="toggle-row" id="biomarkerCompareGroupRow" style="display:none;">
          <div class="toggle-group">{dx_btns}</div>
        </div>
      </div>
      <div class="meta-row" id="biomarkerMetaRow"></div>
      <div class="notes-zone">
        <div id="biomarkerDataSupport" class="data-support-note" style="display:none;"></div>
        <div id="biomarkerWarn"></div>
      </div>
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
    <section class="panel" id="step-f">
      <div class="collapsible-toggle" onclick="toggleCollapsible(this)">
        {_step_header('F', 'Statistical Results')}
        <span class="chev">&#9656;</span>
      </div>
      <div class="collapsible-body">
        <p class="panel-sub" style="margin-top:12px;">Exploratory p-values, unadjusted for multiplicity. Click a row for HC3/influence detail. One population at a time &mdash; not a Target-vs-Overall test.</p>
        <div class="population-note" id="resultsPopulationLabel">Population: Overall ADNI</div>
        <div class="toggle-row"><div class="toggle-group" id="resultsPopulationGroup"></div></div>
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
        "<b>Solid line</b> -- Connects two adjacent Adjusted or Sensitivity-concern points. The boldest element on each chart, since it's the part of the trend backed by the prespecified model.",
        "<b>Dotted line</b> -- Connects a pair of adjacent points where at least one is Descriptive only. Still real, connected data -- just never model-adjusted, so it's drawn lighter than a Solid line.",
        "<b>Gap (no line)</b> -- No connecting line is drawn where a month has no usable data, or where a point is too sparse (isolated, n below the reporting threshold) to connect to its neighbor.",
    ]
    legend_html = "".join(f"<li>{item}</li>" for item in legend_items)
    return f"""
    <section class="panel">
      <div class="collapsible-toggle" id="analysisDetailsToggle" onclick="toggleCollapsible(this)">
        <div class="step-header"><span class="step-badge step-badge--sub">F</span><h2>Analysis Details</h2></div>
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
    "Target Population presets (adni_eligibility.PRESET_LIBRARY) are precomputed offline, not a live filter: the dashboard is a single static HTML file with no backend, and the governed visualization loader (adni_viz_data.py) structurally refuses any participant-level data or .parquet file -- so an arbitrary, freely-typed eligibility query can never be evaluated in the browser. Each preset is instead computed once by run_adni_target_populations.py against the locked processed/ tables, producing governed aggregate CSVs the dashboard only ever reads. Requesting a new population means adding a preset to that pipeline's config and re-running it, not typing a new query here.",
    "Overall ADNI vs. Target Population comparisons (population profile, pooled trajectory) are purely descriptive -- no p-value or test statistic is computed anywhere for them. A Target Population is always a SUBSET of Overall ADNI (nested, non-independent samples), so no independent-samples test in this pipeline (all built for mutually-exclusive CN/MCI/Dementia comparison) is methodologically valid between them -- the same reasoning already applied to POLARIS vs. Overall ADNI before this feature existed.",
    "Pooled (non-diagnosis-stratified) trajectories reuse the identical descriptive one-sample mean/CI primitive (adni_stats.descriptive_mean_ci()) used elsewhere in this pipeline, called on the whole population instead of split by diagnosis group -- always classified Descriptive only or Not available, never Adjusted, since there is no ANCOVA group term once there is no group split.",
]

LIMITATIONS_ITEMS = [
    "ADNI is observational and non-randomized.",
    "Results are natural-history associations, not treatment effects.",
    "A Target Population is an eligibility-filtered subset of ADNI, not a propensity-score-matched cohort, and must not be interpreted as an external control arm for any specific trial.",
    "Sample size differs substantially across endpoints and timepoints, and is smaller yet again within any Target Population preset -- some presets (e.g. the biomarker-availability preset) are expected to hit small-cell suppression earlier than Overall ADNI does.",
    "Many biomarker timepoints are descriptive only.",
    "GFAP/NfL do not support the planned month-specific ANCOVA under the prespecified sample-size rule.",
    "Different plasma biomarkers/platforms must not be directly equated.",
    "pTau217 contains a documented assay lot-bias sensitivity issue.",
    "Multiple comparisons are exploratory and unadjusted.",
    "Several fitted models show influential-observation sensitivity.",
    "Disease Continuum heatmap coloring is normalized independently within each endpoint row and must never be compared across rows/endpoints; it reflects Overall ADNI only, not the selected Target Population.",
    "The MMSE bands used by some Target Population presets (mild-to-moderate/mild-dementia/prodromal-MCI) are general placeholders for commonly-used trial ranges, not any specific trial's actual protocol -- pending clinical review.",
    "Cross-study comparison with any specific trial requires caution due to cohort, assay, endpoint, and study-design differences, even when a Target Population preset was chosen to approximate that trial's eligibility.",
]


def render_methods_limitations_section():
    methods_html = "".join(f"<li>{html.escape(m)}</li>" for m in METHODS_ITEMS)
    limitations_html = "".join(f"<li>{html.escape(l)}</li>" for l in LIMITATIONS_ITEMS)
    return f"""
    {render_disease_continuum_section()}
    <section class="panel" id="step-g">
      <div class="collapsible-toggle" onclick="toggleCollapsible(this)">
        {_step_header('G', 'Methods')}
        <span class="chev">&#9656;</span>
      </div>
      <div class="collapsible-body">
        <ul class="methods-list">{methods_html}</ul>
      </div>
    </section>
    <section class="panel">
      <div class="collapsible-toggle" onclick="toggleCollapsible(this)">
        <div class="step-header"><span class="step-badge step-badge--sub">G</span><h2>Limitations</h2></div>
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

# Generic replacement for the old POLARIS-only POLARIS_COHORT_LABEL/
# POLARIS_KEY_PATTERN_FOOTER constants: every non-Overall-ADNI
# population is now one of adni_eligibility.PRESET_LIBRARY's N presets
# (POLARIS-like included, as just the first one), each with its own
# label and n -- so this footer/label must be a function of the
# specific preset being rendered, not a single hardcoded pair.
def _target_population_key_pattern_footer(cohort_label, cohort_n):
    n_text = f"{cohort_n} " if cohort_n is not None else ""
    return (
        f" This describes the {n_text}eligibility-filtered {cohort_label} participants only -- "
        "not a treatment effect, and not compared statistically to Overall ADNI."
    )


def _target_population_sparse_tail_note(points, cohort_label, group_levels=None, min_n=MIN_GROUP_N_FOR_DISPLAY):
    """Deterministic sparse-later-follow-up clause shared by both
    target-population Key Pattern variants: the latest month at which
    every group still has n >= min_n ("well supported"), plus which
    group(s) fall short of that threshold at the latest AVAILABLE
    month, if different. Returns (sentence, last_well_supported_month_or_None).
    """
    group_levels = group_levels or D.GROUP_ORDER
    by_month = {}
    for p in points:
        if p["n"] is not None:
            by_month.setdefault(p["month"], {})[p["group"]] = p["n"]
    non_baseline_months = sorted(m for m in by_month if m > 0)
    if not non_baseline_months:
        return f"No {cohort_label} follow-up data are available beyond baseline.", None

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


def _target_population_change_key_pattern(points, label, higher_is_worse, cohort_label, cohort_n):
    """Target-population analogue of _change_key_pattern() -- prioritizes
    the latest WELL-SUPPORTED month (not simply the latest fitted one)
    when both exist, and always appends the sparse-tail caveat plus the
    population footer, both parameterized by which preset this is."""
    footer = _target_population_key_pattern_footer(cohort_label, cohort_n)
    sparse_note, last_wm = _target_population_sparse_tail_note(points, cohort_label)
    fitted = [p for p in points if p["classification"] in (D.CLASS_ADJUSTED, D.CLASS_SENSITIVITY_CONCERN) and p["estimate"] is not None]
    if not fitted:
        return (
            f"<b>Key pattern —</b> In the {cohort_label} cohort, no adjusted (HC3) timepoint is "
            f"currently available for {label} under the prespecified small-cell rule. {sparse_note}{footer}"
        )

    fitted_months = {p["month"] for p in fitted}
    eval_month = last_wm if (last_wm is not None and last_wm in fitted_months) else max(fitted_months)
    at_eval = {p["group"]: p for p in fitted if p["month"] == eval_month}
    if len(at_eval) < 3:
        return (
            f"<b>Key pattern —</b> In the {cohort_label} cohort, HC3-adjusted results for {label} are "
            f"not available for all three diagnosis groups at month {int(eval_month)}. {sparse_note}{footer}"
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
    sentence = f"<b>Key pattern —</b> In the {cohort_label} cohort, {pattern}."
    if concern:
        sentence += " At least one group at this timepoint is flagged as a sensitivity concern -- see Statistical Results."
    sentence += f" {sparse_note}{footer}"
    return sentence


def _target_population_absolute_key_pattern(points, label, higher_is_worse, cohort_label, cohort_n):
    """Target-population analogue of _absolute_key_pattern() -- same
    baseline-separation-then-persistence structure, with the shared
    sparse-tail helper and the population footer, both parameterized by
    which preset this is. Never turns a descriptive-only later value
    into a stronger claim than the baseline-separation finding itself."""
    footer = _target_population_key_pattern_footer(cohort_label, cohort_n)
    by_month = {}
    for p in points:
        if p["estimate"] is not None:
            by_month.setdefault(p["month"], {})[p["group"]] = p

    baseline = by_month.get(0, {})
    if len(baseline) < 3:
        return (
            f"<b>Key pattern —</b> In the {cohort_label} cohort, baseline data for {label} are not "
            f"available for all three groups, so a reliable CN/MCI/Dementia comparison cannot be made.{footer}"
        )

    baseline_values = {g: baseline[g]["estimate"] for g in D.GROUP_ORDER}
    ordered, magnitude = _classify_group_separation(baseline_values, higher_is_worse)
    sparse_note, _ = _target_population_sparse_tail_note(points, cohort_label)
    if not ordered:
        return (
            f"<b>Key pattern —</b> In the {cohort_label} cohort, baseline {label} does not show a "
            f"consistent CN &rarr; MCI &rarr; Dementia separation. {sparse_note}{footer}"
        )

    magnitude_word = "markedly" if magnitude == "marked" else "modestly"
    sentence = (
        f"<b>Key pattern —</b> In the {cohort_label} cohort, baseline {label} remains {magnitude_word} "
        f"separated across CN, MCI and Dementia, consistent with the expected disease-severity gradient."
    )
    sentence += f" {sparse_note}{footer}"
    return sentence


def build_cognitive_data_support_summary(points, cohort_label="this population"):
    """Compact, deterministic data-support line for a target-population
    cognitive chart -- reuses the identical sparse-tail logic behind
    _target_population_sparse_tail_note(), phrased for a status line
    rather than a Key Pattern sentence. No endpoint-specific text is
    hardcoded -- everything is derived from the points' own n values."""
    note, last_wm = _target_population_sparse_tail_note(points, cohort_label)
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


def build_key_patterns(data, population="overall", cohort_label=None, cohort_n=None):
    if population == "overall":
        change_fn = lambda pts, label, hiw: _change_key_pattern(pts, label, hiw)
        absolute_fn = lambda pts, label, hiw: _absolute_key_pattern(pts, label, hiw)
    else:
        change_fn = lambda pts, label, hiw: _target_population_change_key_pattern(pts, label, hiw, cohort_label, cohort_n)
        absolute_fn = lambda pts, label, hiw: _target_population_absolute_key_pattern(pts, label, hiw, cohort_label, cohort_n)

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


def build_target_population_cognitive_data_support(data, cohort_label):
    """Cognitive data-support summaries for one target-population preset
    (Overall ADNI's cognitive section has no data-support line; see
    render_cognitive_section)."""
    cognitive = {}
    for e in COGNITIVE_ENDPOINTS:
        pts = D.build_cognitive_chart_data(data, e["key"], "primary")
        cognitive[e["key"]] = build_cognitive_data_support_summary(pts, cohort_label)
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


def build_payload(data, target_population_data=None):
    """`target_population_data`: adni_viz_data.load_target_population_data()'s
    output (the 7 governed adni_target_population_*.csv tables), or None
    for a standalone `python3 adni_viz.py` run with no target-population
    stage output yet -- degrades to Overall-ADNI-only, same honest-empty-
    state convention used throughout this dashboard suite. Every preset
    in adni_eligibility.PRESET_LIBRARY (POLARIS-like included, as simply
    the first one) becomes one populations["target_<id>"] entry, built
    identically via build_population_payload() -- there is no longer a
    separate hardcoded "polaris" population key or POLARIS-specific
    payload path."""
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
        "groupColors": {**D.GROUP_COLORS, "Overall ADNI": "#6b7280", "Target Population": ARIBIO_ACCENT},
        "groupOrder": D.GROUP_ORDER,
        "targetMonths": D.TARGET_MONTHS,
        "minGroupN": MIN_GROUP_N_FOR_DISPLAY,
        "cognitiveEndpoints": COGNITIVE_ENDPOINTS,
        "biomarkerSpecs": BIOMARKER_SPECS,
        "diseaseContinuum": disease_continuum,
        "populations": populations,
        "presetCatalog": [],
        "targetPopulations": {},
    }

    if target_population_data is not None:
        preset_catalog = D.build_preset_catalog(target_population_data["presets"])
        payload["presetCatalog"] = preset_catalog

        primary_platform_by_biomarker = {b["key"]: b["platforms"][0][0] for b in BIOMARKER_SPECS}

        for preset in preset_catalog:
            preset_id = preset["id"]
            pop_key = f"target_{preset_id}"

            attrition_sub = target_population_data["attrition"]
            attrition_sub = attrition_sub[attrition_sub["preset_id"] == preset_id].drop(columns=["preset_id"])
            profile_sub = target_population_data["profile"]
            profile_sub = profile_sub[profile_sub["preset_id"] == preset_id].drop(columns=["preset_id"])
            funnel = D.build_target_population_funnel(attrition_sub)
            profile = D.build_target_population_profile(profile_sub)

            preset_view = D.preset_data_view(target_population_data, preset_id)
            preset_population = build_population_payload(preset_view)
            preset_population.update({
                "label": preset["label"],
                "populationNote": f"n={preset['n']} eligible",
                "n": preset["n"],
                "keyPatterns": build_key_patterns(preset_view, population="target", cohort_label=preset["label"], cohort_n=preset["n"]),
                "cognitiveDataSupport": build_target_population_cognitive_data_support(preset_view, preset["label"]),
                "biomarkerDataSupport": build_biomarker_data_support(preset_view),
            })
            populations[pop_key] = preset_population

            pooled = {}
            for e in COGNITIVE_ENDPOINTS:
                pooled[e["key"]] = D.build_pooled_trajectory_chart_data(target_population_data["pooled"], preset_id, e["key"])
            for b in BIOMARKER_SPECS:
                pooled[b["key"]] = D.build_pooled_trajectory_chart_data(
                    target_population_data["pooled"], preset_id, b["key"], primary_platform_by_biomarker[b["key"]]
                )

            payload["targetPopulations"][preset_id] = {
                "id": preset_id, "label": preset["label"], "description": preset["description"],
                "n": preset["n"], "isPolarisEquivalent": preset["isPolarisEquivalent"],
                "populationKey": pop_key,
                "funnel": funnel, "profile": profile, "pooled": pooled,
            }
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

// ------------------------------------------------------------------
// Target Population selection (Section A: Define Target Population).
// Every criterion/funnel/profile/trajectory number was already
// computed server-side by run_adni_target_populations.py -- there is
// no in-browser filtering, refitting, or participant-level access
// anywhere in this file; selecting a preset just re-indexes into the
// already-baked DATA.targetPopulations[id] / DATA.populations[...]
// payload, exactly the same "no client-side computation" contract the
// old Overall/POLARIS toggle already followed.
// ------------------------------------------------------------------

let selectedPresetId = (DATA.presetCatalog[0] && DATA.presetCatalog[0].id) || null;
let resultsPopulationKey = "overall";

function currentTargetEntry() {
  return selectedPresetId ? DATA.targetPopulations[selectedPresetId] : null;
}

function selectPreset(presetId) {
  selectedPresetId = presetId;
  document.querySelectorAll('[data-preset]').forEach(function (b) { b.classList.toggle('active', b.dataset.preset === presetId); });

  const entry = currentTargetEntry();
  const overallN = DATA.populations.overall.n;
  if (entry) {
    document.getElementById("targetPopulationSummary").innerHTML =
      "<b>" + entry.label + "</b> &middot; n = " + entry.n.toLocaleString() +
      " (" + (entry.n / overallN * 100).toFixed(1) + "% of Overall ADNI)";
  }

  renderEligibilityFunnel();
  renderPopulationProfile();

  const label = "Target: " + (entry ? entry.label : "none selected");
  document.getElementById("cognitivePopulationLabel").innerHTML = fadeSpan(label);
  document.getElementById("biomarkerPopulationLabel").innerHTML = fadeSpan(label);

  renderCognitiveChart();
  renderBiomarkerChart();
  renderResultsTable();
}

function setResultsPopulation(popKey) {
  resultsPopulationKey = popKey;
  document.querySelectorAll('[data-results-population]').forEach(function (b) { b.classList.toggle('active', b.dataset.resultsPopulation === popKey); });
  const pd = DATA.populations[popKey];
  document.getElementById("resultsPopulationLabel").innerHTML = fadeSpan("Population: " + pd.label + (pd.populationNote ? " · " + pd.populationNote : ""));
  renderResultsTable();
}

function resultsPopulationOptions() {
  const entry = currentTargetEntry();
  const opts = [{ key: "overall", label: "Overall ADNI" }];
  if (entry) opts.push({ key: entry.populationKey, label: entry.label });
  return opts;
}

function renderResultsPopulationToggle() {
  const opts = resultsPopulationOptions();
  if (!opts.some(function (o) { return o.key === resultsPopulationKey; })) resultsPopulationKey = "overall";
  document.getElementById("resultsPopulationGroup").innerHTML = opts.map(function (o) {
    return '<button class="toggle-btn' + (o.key === resultsPopulationKey ? ' active' : '') + '" data-results-population="' + o.key +
      '" onclick="setResultsPopulation(\'' + o.key + '\')">' + o.label + '</button>';
  }).join("");
}

function fmtNumericSummary(s) {
  if (!s || s.n === null || s.n === undefined || s.mean === null) return '<span class="profile-numeric-value">—</span>';
  const sdStr = s.sd === null || s.sd === undefined ? "" : " &plusmn; " + fmtNum(s.sd, 1);
  return '<span class="profile-numeric-value">' + fmtNum(s.mean, 1) + sdStr + " (n=" + s.n.toLocaleString() + ")</span>";
}

function buildProfileGridHtml(profile, targetLabel) {
  return profile.map(function (v) {
    if (v.kind === "numeric") {
      return '<div class="profile-card">' +
        '<div class="profile-var-label">' + v.variable + '</div>' +
        '<div class="profile-pop-row"><span class="profile-pop-tag">Overall ADNI</span>' + fmtNumericSummary(v.overall) + '</div>' +
        '<div class="profile-pop-row"><span class="profile-pop-tag">' + targetLabel + '</span>' + fmtNumericSummary(v.polaris) + '</div>' +
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
    if (!presentLevels.length) {
      // Availability-flag variables (pTau181 available, etc.) aren't in
      // CATEGORICAL_DISPLAY's curated order/color map -- fall back to
      // whatever levels are actually present (usually True/False).
      const fallbackLevels = v.levels.map(function (l) { return l.level; });
      const fallbackBar = function (popKey) {
        return fallbackLevels.map(function (lvl) {
          const entry = v.levels.find(function (l) { return l.level === lvl; });
          const pct = entry[popKey].percent;
          if (pct === null || pct === undefined) return "";
          const color = lvl === "True" ? "#2e5fa3" : "#c8ccd4";
          return '<div class="profile-bar-seg" style="width:' + pct + '%;background:' + color + ';" title="' + lvl + " " + pct + '%"></div>';
        }).join("");
      };
      const fallbackLegend = fallbackLevels.map(function (lvl) {
        const color = lvl === "True" ? "#2e5fa3" : "#c8ccd4";
        return '<span style="color:' + color + ';">&#9679;</span> ' + lvl;
      }).join(" &nbsp; ");
      return '<div class="profile-card">' +
        '<div class="profile-var-label">' + v.variable + '</div>' +
        '<div class="profile-pop-row"><span class="profile-pop-tag">Overall ADNI</span><div class="profile-bar">' + fallbackBar("overall") + '</div></div>' +
        '<div class="profile-pop-row"><span class="profile-pop-tag">' + targetLabel + '</span><div class="profile-bar">' + fallbackBar("polaris") + '</div></div>' +
        '<div class="profile-note">' + fallbackLegend + '</div>' +
        '</div>';
    }
    return '<div class="profile-card">' +
      '<div class="profile-var-label">' + v.variable + '</div>' +
      '<div class="profile-pop-row"><span class="profile-pop-tag">Overall ADNI</span><div class="profile-bar">' + barHtml("overall") + '</div></div>' +
      '<div class="profile-pop-row"><span class="profile-pop-tag">' + targetLabel + '</span><div class="profile-bar">' + barHtml("polaris") + '</div></div>' +
      '<div class="profile-note">' + legendHtml + '</div>' +
      apoeNote +
      '</div>';
  }).join("");
}

function renderEligibilityFunnel() {
  const entry = currentTargetEntry();
  const container = document.getElementById("eligibilityFunnel");
  const summaryEl = document.getElementById("eligibilitySummary");
  if (!entry) { container.innerHTML = ""; summaryEl.innerHTML = ""; return; }

  let funnelHtml = "";
  entry.funnel.forEach(function (s, i) {
    funnelHtml += '<div class="funnel-step"><div class="funnel-step-label">' + s.step + '</div>' +
      '<div class="funnel-step-n">' + s.remaining_n.toLocaleString() + "</div>" +
      (i > 0
        ? '<div class="funnel-step-meta">&minus;' + s.excluded_n.toLocaleString() + " excluded &middot; " +
          (s.percent_retained_of_previous === null ? "—" : s.percent_retained_of_previous + "%") + " retained from previous step</div>"
        : "") +
      "</div>";
    if (i < entry.funnel.length - 1) funnelHtml += '<div class="funnel-arrow">&#8594;</div>';
  });
  container.innerHTML = funnelHtml;

  const overallN = DATA.populations.overall.n;
  const dxProfile = entry.profile.find(function (p) { return p.variable === "Baseline diagnosis"; });
  const dxHtml = dxProfile ? dxProfile.levels.map(function (l) {
    const s = l.polaris;
    return '<div class="meta-chip">' + l.level + ': <b>' + (s.n === null ? "—" : s.n.toLocaleString()) + '</b>' +
      (s.percent !== null ? " (" + s.percent + "%)" : "") + '</div>';
  }).join("") : "";
  summaryEl.innerHTML =
    '<p class="panel-sub" style="margin:0 0 8px;"><b>' + entry.n.toLocaleString() + '</b> of ' + overallN.toLocaleString() +
    ' (' + (entry.n / overallN * 100).toFixed(1) + '%) meet the criteria.</p>' +
    '<div class="meta-row">' + dxHtml + '</div>';
}

function renderPopulationProfile() {
  const entry = currentTargetEntry();
  const container = document.getElementById("populationProfileGrid");
  const detailContainer = document.getElementById("populationProfileGridDetail");
  if (!entry) { container.innerHTML = ""; if (detailContainer) detailContainer.innerHTML = ""; return; }
  const compactVars = ["Baseline age (years)", "Baseline MMSE", "Baseline diagnosis", "Baseline Centiloid"];
  const compact = entry.profile.filter(function (p) { return compactVars.indexOf(p.variable) !== -1; });
  const rest = entry.profile.filter(function (p) { return compactVars.indexOf(p.variable) === -1; });
  container.innerHTML = buildProfileGridHtml(compact, entry.label);
  if (detailContainer) detailContainer.innerHTML = buildProfileGridHtml(rest, entry.label);
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
let cognitiveView = "change";
let cognitiveCompareMode = "pooled";
let cognitiveCompareGroup = "MCI";

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
  // Absolute has no pooled counterpart (see setCognitiveCompareMode's
  // comment) -- selecting it implies "by diagnosis group" automatically
  // rather than silently doing nothing in pooled mode.
  if (view === 'absolute' && cognitiveCompareMode === 'pooled') { setCognitiveCompareMode('byGroup'); return; }
  renderCognitiveChart();
}

// Pooled trajectories (run_adni_target_populations.py's
// compute_pooled_trajectory_rows()) are computed ONLY on the change-
// from-baseline scale -- pooling raw Absolute scores across very
// different baseline severities is a materially different, less
// meaningful question than pooling their CHANGE, so it was never
// computed; the Absolute view is only ever available per-diagnosis-
// group (byGroup mode), same as before this feature.
function setCognitiveCompareMode(mode) {
  cognitiveCompareMode = mode;
  document.querySelectorAll('#cognitiveCompareModeGroup .toggle-btn').forEach(function (b) { b.classList.toggle('active', b.dataset.mode === mode); });
  document.getElementById('cognitiveCompareGroupRow').style.display = mode === 'byGroup' ? '' : 'none';
  if (mode === 'pooled' && cognitiveView === 'absolute') { setCognitiveView('change'); return; }
  renderCognitiveChart();
}

function setCognitiveCompareGroup(g) {
  cognitiveCompareGroup = g;
  document.querySelectorAll('[data-cognitive-dxgroup]').forEach(function (b) { b.classList.toggle('active', b.dataset.cognitiveDxgroup === g); });
  renderCognitiveChart();
}

// Reshapes a flat points array (each carrying its own p.group) into
// Plotly-ready {groupLabel: [points]} buckets -- shared by both the
// pooled default view (points already labeled "Overall ADNI"/"Target
// Population" by build_pooled_trajectory_chart_data()) and the
// by-diagnosis-group drill-down (points relabeled here from their
// original DX code to a population label, since buildGroupTraces()
// colors/names/legends purely off the OUTER dict key, never a point's
// own .group field).
function pooledPointsByPopulation(pooledPoints) {
  const out = { "Overall ADNI": [], "Target Population": [] };
  (pooledPoints || []).forEach(function (p) { if (out[p.group]) out[p.group].push(p); });
  return out;
}

function byGroupPointsByPopulation(overallSeries, targetSeries, dxGroup) {
  const relabel = function (points, label) {
    return (points || []).filter(function (p) { return p.group === dxGroup; }).map(function (p) {
      const copy = Object.assign({}, p); copy.group = label; return copy;
    });
  };
  return { "Overall ADNI": relabel(overallSeries, "Overall ADNI"), "Target Population": relabel(targetSeries, "Target Population") };
}

function renderCognitiveChart() {
  const entry = currentTargetEntry();
  const spec = DATA.cognitiveEndpoints.find(function (e) { return e.key === cognitiveEndpointKey; });
  const chartEl = document.getElementById("cognitiveChart");
  if (!entry) {
    Plotly.purge(chartEl);
    document.getElementById("cognitiveMetaRow").innerHTML = "";
    document.getElementById("cognitiveKeyPattern").innerHTML = fadeSpan("Select a Target Population preset above to compare its progression against Overall ADNI.");
    return;
  }

  let pointsByGroup, targetPop;
  if (cognitiveCompareMode === "pooled") {
    pointsByGroup = pooledPointsByPopulation(entry.pooled[cognitiveEndpointKey]);
    targetPop = DATA.populations[entry.populationKey];
  } else {
    const seriesSource = cognitiveView === "absolute" ? "cognitiveAbsolute" : "cognitiveChange";
    const overallSeries = DATA.populations.overall[seriesSource][cognitiveEndpointKey];
    targetPop = DATA.populations[entry.populationKey];
    const targetSeries = targetPop[seriesSource][cognitiveEndpointKey];
    pointsByGroup = byGroupPointsByPopulation(overallSeries, targetSeries, cognitiveCompareGroup);
  }

  const allPoints = pointsByGroup["Overall ADNI"].concat(pointsByGroup["Target Population"]);
  const metaRow = document.getElementById("cognitiveMetaRow");
  const availableMonths = [...new Set(allPoints.filter(function (p) { return p.classification !== CLASS_D; }).map(function (p) { return p.month; }))].sort(function (a, b) { return a - b; });
  metaRow.innerHTML =
    '<div class="meta-chip">Endpoint: <b>' + spec.label + '</b></div>' +
    '<div class="meta-chip">Comparison: <b>' + (cognitiveCompareMode === "pooled" ? "Pooled (all diagnoses)" : "Diagnosis: " + cognitiveCompareGroup) + '</b></div>' +
    '<div class="meta-chip">Available timepoints: <b>' + (availableMonths.length ? availableMonths.join(", ") : "none") + '</b></div>';

  const warnDiv = document.getElementById("cognitiveWarn");
  warnDiv.innerHTML = (cognitiveCompareMode === "byGroup" && cognitiveView === "change") ? computeCompactWarnHtml(pointsByGroup) : "";

  const supportDiv = document.getElementById("cognitiveDataSupport");
  const supportText = cognitiveCompareMode === "byGroup" ? (targetPop.cognitiveDataSupport && targetPop.cognitiveDataSupport[cognitiveEndpointKey]) : null;
  if (supportText) { supportDiv.innerHTML = fadeSpan(supportText); supportDiv.style.display = ""; } else { supportDiv.style.display = "none"; }

  const yRange = computeSensibleYRange(pointsByGroup);
  const showAbsolute = cognitiveCompareMode === "byGroup" && cognitiveView === "absolute";
  const traces = buildGroupTraces(pointsByGroup, showAbsolute ? (spec.label + " score") : "Change from baseline", "");
  const layout = showAbsolute
    ? absoluteLayout(spec.label + " score (↑ " + spec.up_label + " / ↓ " + spec.down_label + ")")
    : baseLayout("Change from baseline", spec.up_label, spec.down_label);
  if (yRange) layout.yaxis.range = yRange;
  Plotly.react("cognitiveChart", traces, layout, { displayModeBar: false, responsive: true });

  document.getElementById("cognitiveKeyPattern").innerHTML = fadeSpan(
    targetPop.keyPatterns.cognitive[showAbsolute ? "absolute" : "change"][cognitiveEndpointKey]
  );
}

// ------------------------------------------------------------------
// Plasma Biomarker Trajectories
// ------------------------------------------------------------------

let currentBiomarker = DATA.biomarkerSpecs[0].key;
let currentPlatform = null;
let currentAnalysisType = null;
let biomarkerView = "change";
let biomarkerCompareMode = "pooled";
let biomarkerCompareGroup = "MCI";

function setBiomarker(key) {
  const spec = DATA.biomarkerSpecs.find(function (b) { return b.key === key; });
  currentBiomarker = key;
  currentPlatform = spec.platforms[0][0];
  currentAnalysisType = spec.platforms[0][1];
  document.querySelectorAll('[data-biomarker]').forEach(function (b) { b.classList.toggle('active', b.dataset.biomarker === key); });

  const platformGroup = document.getElementById("biomarkerPlatformGroup");
  if (spec.platforms.length > 1) {
    platformGroup.style.display = biomarkerCompareMode === 'byGroup' ? "inline-flex" : "none";
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
  if (view === 'absolute' && biomarkerCompareMode === 'pooled') { setBiomarkerCompareMode('byGroup'); return; }
  renderBiomarkerChart();
}

// Pooled biomarker trajectories are computed only for each biomarker's
// PRIMARY assay/platform (see run_adni_target_populations.py's
// POOLED_BIOMARKER_SPECS) -- the pooled default view always shows that
// primary platform; switching platform/sensitivity-analysis type is
// only available in byGroup mode, same scoping as the Absolute-view
// restriction on the cognitive chart above.
function setBiomarkerCompareMode(mode) {
  biomarkerCompareMode = mode;
  document.querySelectorAll('#biomarkerCompareModeGroup .toggle-btn').forEach(function (b) { b.classList.toggle('active', b.dataset.mode === mode); });
  document.getElementById('biomarkerCompareGroupRow').style.display = mode === 'byGroup' ? '' : 'none';
  // Platform/sensitivity-type switching only has an effect in byGroup
  // mode -- pooled always uses each biomarker's primary platform (see
  // POOLED_BIOMARKER_SPECS in run_adni_target_populations.py) -- so the
  // control is hidden rather than shown-but-inert in pooled mode.
  const spec = DATA.biomarkerSpecs.find(function (b) { return b.key === currentBiomarker; });
  const platformGroup = document.getElementById("biomarkerPlatformGroup");
  platformGroup.style.display = (mode === 'byGroup' && spec.platforms.length > 1) ? "inline-flex" : "none";
  if (mode === 'pooled' && biomarkerView === 'absolute') { setBiomarkerView('change'); return; }
  renderBiomarkerChart();
}

function setBiomarkerCompareGroup(g) {
  biomarkerCompareGroup = g;
  document.querySelectorAll('[data-biomarker-dxgroup]').forEach(function (b) { b.classList.toggle('active', b.dataset.biomarkerDxgroup === g); });
  renderBiomarkerChart();
}

function renderBiomarkerChart() {
  const entry = currentTargetEntry();
  const spec = DATA.biomarkerSpecs.find(function (b) { return b.key === currentBiomarker; });
  const chartEl = document.getElementById("biomarkerChart");
  if (!entry) {
    Plotly.purge(chartEl);
    document.getElementById("biomarkerMetaRow").innerHTML = "";
    document.getElementById("biomarkerKeyPattern").innerHTML = fadeSpan("Select a Target Population preset above to compare its progression against Overall ADNI.");
    return;
  }

  let pointsByGroup, targetPop;
  if (biomarkerCompareMode === "pooled") {
    pointsByGroup = pooledPointsByPopulation(entry.pooled[currentBiomarker]);
    targetPop = DATA.populations[entry.populationKey];
  } else {
    const seriesSource = biomarkerView === "absolute" ? "biomarkersAbsolute" : "biomarkersChange";
    const overallSeries = DATA.populations.overall[seriesSource][currentBiomarker][currentPlatform][currentAnalysisType];
    targetPop = DATA.populations[entry.populationKey];
    const targetSeries = targetPop[seriesSource][currentBiomarker][currentPlatform][currentAnalysisType];
    pointsByGroup = byGroupPointsByPopulation(overallSeries, targetSeries, biomarkerCompareGroup);
  }

  const allPoints = pointsByGroup["Overall ADNI"].concat(pointsByGroup["Target Population"]);
  const metaRow = document.getElementById("biomarkerMetaRow");
  const availableMonths = [...new Set(allPoints.filter(function (p) { return p.classification !== CLASS_D; }).map(function (p) { return p.month; }))].sort(function (a, b) { return a - b; });
  const platformLabel = biomarkerCompareMode === "pooled" ? spec.platforms[0][2] : currentPlatform.replace(/_/g, " ");
  metaRow.innerHTML =
    '<div class="meta-chip">Assay/platform: <b>' + platformLabel + '</b></div>' +
    '<div class="meta-chip">Comparison: <b>' + (biomarkerCompareMode === "pooled" ? "Pooled (all diagnoses)" : "Diagnosis: " + biomarkerCompareGroup) + '</b></div>' +
    '<div class="meta-chip">Available timepoints: <b>' + (availableMonths.length ? availableMonths.join(", ") : "none") + '</b></div>';

  const supportDiv = document.getElementById("biomarkerDataSupport");
  const supportEntry = biomarkerCompareMode === "byGroup" && targetPop.biomarkerDataSupport && targetPop.biomarkerDataSupport[currentBiomarker] &&
    targetPop.biomarkerDataSupport[currentBiomarker][currentPlatform] && targetPop.biomarkerDataSupport[currentBiomarker][currentPlatform][currentAnalysisType];
  const supportText = supportEntry && supportEntry[biomarkerView];
  if (supportText) { supportDiv.innerHTML = fadeSpan(supportText); supportDiv.style.display = ""; } else { supportDiv.style.display = "none"; }

  const warnDiv = document.getElementById("biomarkerWarn");
  warnDiv.innerHTML = (biomarkerCompareMode === "byGroup" && biomarkerView === "change") ? computeCompactWarnHtml(pointsByGroup) : "";

  const yRange = computeSensibleYRange(pointsByGroup);
  const showAbsolute = biomarkerCompareMode === "byGroup" && biomarkerView === "absolute";
  // Error bars hidden on-chart for biomarkers -- CI is still exact and
  // available via hover; the isolated/dashed/solid marker styling
  // already carries the confidence signal without the visual weight
  // of a permanently-drawn bar on every point (see buildTooltip()).
  const traces = buildGroupTraces(pointsByGroup, showAbsolute ? (spec.label + " concentration") : "Geometric mean % change", showAbsolute ? "" : "%", false);
  const layout = showAbsolute
    ? absoluteLayout(spec.label + " geometric mean concentration")
    : baseLayout("Geometric mean % change from baseline");
  if (yRange) layout.yaxis.range = yRange;
  Plotly.react("biomarkerChart", traces, layout, { displayModeBar: false, responsive: true });

  document.getElementById("biomarkerInterpretation").innerHTML = fadeSpan(spec.interpretation);
  document.getElementById("biomarkerKeyPattern").innerHTML = fadeSpan(
    targetPop.keyPatterns.biomarkers[showAbsolute ? "absolute" : "change"][currentBiomarker][currentPlatform][currentAnalysisType]
  );
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
  renderResultsPopulationToggle();
  const key = document.getElementById("resultsTableSelect").value;
  const rows = (DATA.populations[resultsPopulationKey].resultsTable || {})[key] || [];
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

// Highlights the flow-nav item for whichever A-G section is currently
// most in view -- purely a display aid (click-to-jump already works
// via the anchors' native href behavior without this); degrades to no
// highlighting at all if IntersectionObserver isn't available, never
// breaks navigation itself.
function initFlowNav() {
  var items = Array.prototype.slice.call(document.querySelectorAll(".flow-nav-item"));
  if (!items.length || !window.IntersectionObserver) return;
  var sections = items.map(function (a) { return document.getElementById(a.dataset.step); }).filter(Boolean);
  var observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (!entry.isIntersecting) return;
      items.forEach(function (a) { a.classList.toggle("active", a.dataset.step === entry.target.id); });
    });
  }, { rootMargin: "-10% 0px -75% 0px", threshold: 0 });
  sections.forEach(function (s) { observer.observe(s); });
}

document.addEventListener("DOMContentLoaded", function () {
  initFlowNav();
  renderDiseaseContinuum();
  if (selectedPresetId) selectPreset(selectedPresetId);
  renderCognitiveChart();
  setBiomarker(DATA.biomarkerSpecs[0].key);
  renderResultsTable();
});
</script>
"""


# ------------------------------------------------------------------
# Page assembly
# ------------------------------------------------------------------


def render_page(data, target_population_data=None):
    payload = build_payload(data, target_population_data)
    payload_json = json.dumps(payload)
    plotlyjs_lib = pyo.get_plotlyjs()

    js = DASHBOARD_JS.replace("__PAYLOAD_JSON__", payload_json)

    preset_catalog = payload["presetCatalog"]

    header_row = f"""
    <div class="page-title-block">
      <div class="page-title">ADNI Natural History Dashboard</div>
      <div class="page-title-sub">Define a target population, see who is eligible, and compare its cognitive/biomarker progression against Overall ADNI &middot; Source: ADNI</div>
      <div class="page-context-note"><span class="page-context-note-icon">&#9432;</span><span>ADNI is observational, not a randomized trial or true external control arm. Results describe disease progression, never a treatment effect.</span></div>
    </div>
    {render_header_section()}
    {render_flow_nav()}
    """

    # A -> B -> C -> D -> E -> F -> G: the primary analytical narrative
    # (requirement: "Define Target Population" through "Cognitive/
    # Biomarker Progression Comparison") comes first; Statistical/
    # Robustness Details and Methods & Limitations (which now also
    # carries the deprioritized Disease Continuum heatmap) support that
    # workflow rather than interrupting it.
    trajectories_row = f"""
    <div class="trajectories-row">
      {render_cognitive_progression_section()}
      {render_biomarker_progression_section()}
    </div>
    """

    body = (
        render_define_population_section(preset_catalog)
        + render_eligibility_funnel_section()
        + render_population_profile_section()
        + trajectories_row
        + render_results_table_section()
        + render_analysis_details_section()
        + render_methods_limitations_section()
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ADNI Natural History</title>
<style>{PAGE_CSS}</style>
</head>
<body>
{render_nav_bar('biomarker')}
<main>
  {header_row}
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
    `target_population_data` degrades to None (Overall-ADNI-only) if
    run_adni_target_populations.py hasn't been run yet against this
    outputs_dir -- same honest-empty-state convention as the rest of
    this dashboard suite, not a hard failure.
    """
    data = D.load_all(outputs_dir)
    try:
        target_population_data = D.load_target_population_data(outputs_dir)
    except D.DataGovernanceError:
        target_population_data = None
    return render_page(data, target_population_data)


if __name__ == "__main__":
    html_out = build_dashboard_html()
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_out)
    print(f"=== SAVED: {OUTPUT_HTML} ===")
