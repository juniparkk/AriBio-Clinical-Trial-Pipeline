# ============================================================
# COMPETITIVE_ATTENTION_VIZ — "Needs Attention" + "Upcoming Competitive
# Milestones" dashboard section HTML
#
# Pure HTML-string rendering only — every number/word displayed here
# comes directly from competitive_attention.py's already-computed,
# deterministic scoring/bucketing output. This module never computes
# a score, never infers success/failure/FDA status, and never claims
# a result is "expected" — see _milestone_note() for the required
# careful phrasing.
#
# Matches pipeline_viz.py's existing visual language — the color/
# spacing constants below are an intentional, narrow duplication of
# pipeline_viz.py's own ARIBIO_BLUE/ARIBIO_ACCENT/CARD_RADIUS/
# CARD_SHADOW (same rationale as ctgov_changes.py's PHASE_LABELS
# duplication: this module must never import pipeline_viz.py itself,
# since that would execute its entire top-level pipeline as a side
# effect). Reuses ARIBIO_BLUE for Medium, a darker blue for High,
# ARIBIO_ACCENT (already the dashboard's "needs attention" color, used
# for Discontinued status and the AR1001 spotlight) for Critical, and
# the same neutral gray used for NA/Unknown/Other elsewhere for Low.
#
# INTEGRATION: pipeline_viz.py embeds PLACEHOLDER verbatim in its HTML
# output at the desired insertion point. run_pipeline.py — AFTER
# pipeline_viz.py has run and outputs/pipeline_changes.csv /
# outputs/competitive_attention.csv are freshly computed — does a
# plain string .replace(PLACEHOLDER, render_competitive_sections(...))
# on the already-written pipeline_overview.html. This avoids re-running
# the entire (expensive) pipeline_viz.py a second time just to pick up
# data that necessarily depends on pipeline_viz.py's own prior output.
# A standalone `python3 pipeline_viz.py` run (outside run_pipeline.py)
# simply leaves the placeholder comment in place, which renders as
# nothing — an honest empty state, not stale/wrong data.
# ============================================================

import html

import pandas as pd

PLACEHOLDER = "<!--COMPETITIVE_ATTENTION_SECTION-->"

PRIORITY_ORDER = ["Critical", "High", "Medium", "Low"]

ARIBIO_BLUE = "#2e5fa3"
ARIBIO_ACCENT = "#c2255c"
CARD_RADIUS = "12px"
CARD_SHADOW = "0 1px 3px rgba(20, 40, 70, 0.09)"


def _darken(hex_color, amount=0.15):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (max(0, int(c * (1 - amount))) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def _esc(value):
    return html.escape(str(value)) if value is not None else ""


_PRIORITY_COLORS = {
    "Critical": ARIBIO_ACCENT,
    "High": _darken(ARIBIO_BLUE, 0.25),
    "Medium": ARIBIO_BLUE,
    "Low": "#9e9e9e",
}

# Mirrors pipeline_viz.py's own relevanceColor() JS thresholds exactly
# (same blue-ramp convention, not a separate palette) — kept here as a
# narrow, intentional duplication for the same reason as every other
# mirrored constant in this file (see PLACEHOLDER/ARIBIO_BLUE above).
_RELEVANCE_HIGH_COLOR = _darken(ARIBIO_BLUE, 0.30)
_RELEVANCE_MID_COLOR = ARIBIO_BLUE
_RELEVANCE_LOW_COLOR = "#9e9e9e"


def _relevance_color(score):
    if score is None or pd.isna(score):
        return _RELEVANCE_LOW_COLOR
    if score >= 65:
        return _RELEVANCE_HIGH_COLOR
    if score >= 35:
        return _RELEVANCE_MID_COLOR
    return _RELEVANCE_LOW_COLOR


def _study_url(nct_id):
    nct_id = str(nct_id or "").strip()
    return f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else ""


def render_ar1001_relevance_section(ranking_df, top_n=10):
    """STATIC top-N drugs by similarity to AR1001 (competitive_
    intelligence.py's rule-based 0-100 score) — always populated as
    long as resolved_drugs_df has rows, unlike Needs Attention/Recent
    Changes beside it, which depend on something having changed.

    Rendered as a vertical stacked list (same .attention-card pattern
    as the other two panels) since all three now sit side by side in
    a single row — a narrow column has no room for a horizontal tile
    strip."""
    if ranking_df is None or ranking_df.empty:
        body = '<div class="attention-empty">No other drugs are currently resolved to compare against AR1001.</div>'
        count_note = "0 drugs"
    else:
        top = ranking_df.head(top_n)
        count_note = f"top {len(top)} of {len(ranking_df)} resolved drugs"
        rows_html = []
        for _, row in top.iterrows():
            score = pd.to_numeric(row.get("aribio_relevance_score"), errors="coerce")
            score_text = str(int(score)) if pd.notna(score) else "&mdash;"
            color = _relevance_color(score)
            name = _esc(row.get("display_name") or "")
            sponsor = _esc(row.get("sponsor") or "")
            phase = _esc(row.get("phase_reached") or "Phase not reported")
            reasons = _esc(row.get("aribio_relevance_reasons") or phase)
            rows_html.append(f"""
            <div class="attention-card">
              <div class="attention-badge" style="background:{color}">{score_text}</div>
              <div class="attention-main">
                <div class="attention-title">{name}<span class="attention-company">{sponsor}</span></div>
                <div class="attention-change">{reasons}</div>
              </div>
            </div>""")
        body = "".join(rows_html)

    return f"""
    <div class="attention-panel" style="border-radius:{CARD_RADIUS}; box-shadow:{CARD_SHADOW}">
      <div class="attention-panel-header">
        <div class="attention-panel-title">AR1001 Relevance</div>
        <div class="attention-panel-note">{count_note} &middot; ranked by deterministic similarity score (pathway/modality/phase), not AI judgment</div>
      </div>
      {body}
    </div>"""


def render_recent_changes_section(changes_df, top_n=15):
    """Raw, UNSCORED feed of every change pipeline_changes.csv detected
    in the most recent refresh — distinct from Needs Attention below
    it, which is the same changes filtered/ranked by competitive
    priority. This section shows everything that changed, regardless
    of how much competitive priority it scored."""
    if changes_df is None or changes_df.empty:
        body = '<div class="attention-empty">No pipeline changes were detected in the most recent refresh.</div>'
        count_note = "0 changes"
    else:
        top = changes_df.head(top_n)
        count_note = f"showing {len(top)} of {len(changes_df)} detected changes"
        rows_html = []
        for _, row in top.iterrows():
            drug = _esc(row.get("canonical_drug_name") or "(unresolved)")
            company = _esc(row.get("sponsor_or_company") or "")
            nct_id = row.get("nct_id") or ""
            url = _study_url(nct_id)
            link_html = (
                f'<a href="{_esc(url)}" target="_blank" rel="noopener" class="attention-link">{_esc(nct_id)}</a>'
                if url else '<span class="attention-link attention-link--none">no linked trial</span>'
            )
            change_type = _esc((row.get("change_type") or "").replace("_", " "))
            rows_html.append(f"""
            <div class="attention-card">
              <div class="attention-badge" style="background:#9e9e9e">{change_type}</div>
              <div class="attention-main">
                <div class="attention-title">{drug}<span class="attention-company">{company}</span></div>
                <div class="attention-change">{_esc(row.get('description') or '')}</div>
              </div>
              <div class="attention-side">{link_html}</div>
            </div>""")
        body = "".join(rows_html)

    return f"""
    <div class="attention-panel" style="border-radius:{CARD_RADIUS}; box-shadow:{CARD_SHADOW}">
      <div class="attention-panel-header">
        <div class="attention-panel-title">Recent Changes</div>
        <div class="attention-panel-note">{count_note} &middot; every detected change, unranked &mdash; see Needs Attention below for competitive priority</div>
      </div>
      {body}
    </div>"""


def render_needs_attention_section(attention_df, top_n=8):
    if attention_df is None or attention_df.empty:
        body = '<div class="attention-empty">No pipeline changes currently need attention.</div>'
        count_note = "0 items"
    else:
        top = attention_df.head(top_n)
        count_note = f"showing top {len(top)} of {len(attention_df)}"
        rows_html = []
        for _, row in top.iterrows():
            level = row["priority_level"]
            color = _PRIORITY_COLORS.get(level, "#9e9e9e")
            drug = _esc(row.get("canonical_drug_name") or "(unresolved)")
            company = _esc(row.get("company_or_sponsor") or "")
            nct_id = row.get("nct_id") or ""
            url = _study_url(nct_id)
            link_html = (
                f'<a href="{_esc(url)}" target="_blank" rel="noopener" class="attention-link">{_esc(nct_id)}</a>'
                if url else '<span class="attention-link attention-link--none">no linked trial</span>'
            )
            rows_html.append(f"""
            <div class="attention-card">
              <div class="attention-badge" style="background:{color}">{_esc(level)} &middot; {int(row['relevance_score'])}</div>
              <div class="attention-main">
                <div class="attention-title">{drug}<span class="attention-company">{company}</span></div>
                <div class="attention-change">{_esc(row.get('why_it_matters') or '')}</div>
                <div class="attention-factors">{_esc(row.get('relevance_factors') or '')}</div>
              </div>
              <div class="attention-side">{link_html}</div>
            </div>""")
        body = "".join(rows_html)

    return f"""
    <div class="attention-panel" style="border-radius:{CARD_RADIUS}; box-shadow:{CARD_SHADOW}">
      <div class="attention-panel-header">
        <div class="attention-panel-title">Needs Attention</div>
        <div class="attention-panel-note">{count_note} &middot; ranked by deterministic competitive-priority score, not AI judgment</div>
      </div>
      {body}
    </div>"""


def _milestone_note(bucket_key):
    # Deliberately careful, non-promissory language throughout — never
    # "results expected," never implies an outcome.
    notes = {
        "next_30_days": "Primary completion is approaching within 30 days; future disclosures may warrant monitoring.",
        "next_90_days": "Primary completion is approaching within 90 days; future disclosures may warrant monitoring.",
        "recently_completed": "Study completion date has recently passed on ClinicalTrials.gov; no outcome is implied by this alone.",
        "materially_delayed": "Completion date shifted materially later than previously listed; reason is not stated by ClinicalTrials.gov.",
    }
    return notes[bucket_key]


def _render_milestone_list(items, bucket_key, date_field):
    if not items:
        return '<div class="milestone-empty">None currently.</div>'
    rows = []
    for item in items[:10]:
        nct_id = item.get("nct_id", "")
        drug_name = item.get("drug_name") or nct_id
        url = _study_url(nct_id)
        sponsor = _esc(item.get("sponsor", ""))
        if bucket_key == "materially_delayed":
            detail = f"{_esc(item.get('old_value',''))} &rarr; {_esc(item.get('new_value',''))} ({item.get('days_delayed','')} days)"
        else:
            detail = _esc(item.get(date_field, ""))
        rows.append(
            f'<div class="milestone-row">'
            f'<a href="{_esc(url)}" target="_blank" rel="noopener" title="{_esc(nct_id)}">{_esc(drug_name)}</a>'
            f'<span class="milestone-sponsor">{sponsor}</span>'
            f'<span class="milestone-detail">{detail}</span>'
            f'</div>'
        )
    return "".join(rows)


def render_milestones_section(milestones):
    panels = [
        ("next_30_days", "Next 30 Days", "primary_completion_date"),
        ("next_90_days", "Next 90 Days", "primary_completion_date"),
        ("recently_completed", "Recently Completed", "completion_date"),
        ("materially_delayed", "Materially Delayed", None),
    ]
    panels_html = []
    for key, label, date_field in panels:
        items = milestones.get(key, [])
        panels_html.append(f"""
        <div class="milestone-col">
          <div class="milestone-col-title">{label} <span class="milestone-count">({len(items)})</span></div>
          <div class="milestone-col-note">{_milestone_note(key)}</div>
          {_render_milestone_list(items, key, date_field)}
        </div>""")

    return f"""
    <div class="attention-panel" style="border-radius:{CARD_RADIUS}; box-shadow:{CARD_SHADOW}; margin-top:16px;">
      <div class="attention-panel-header">
        <div class="attention-panel-title">Upcoming Competitive Milestones</div>
        <div class="attention-panel-note">Based on Primary Completion Date / Completion Date already on file with ClinicalTrials.gov</div>
      </div>
      <div class="milestone-grid">{"".join(panels_html)}</div>
    </div>"""


COMPETITIVE_ATTENTION_CSS = """
  .attention-panel { background: white; padding: 18px 20px; margin-bottom: 20px; }
  .attention-panel-header { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 12px; flex-wrap: wrap; gap: 6px; }
  .attention-panel-title { font-size: 16px; font-weight: 700; color: #1a1a1a; }
  .attention-panel-note { font-size: 12px; color: #888; }
  .attention-empty, .milestone-empty { font-size: 13px; color: #888; padding: 8px 0; }
  .attention-card { display: flex; align-items: flex-start; gap: 14px; padding: 12px 0; border-top: 1px solid #eee; }
  .attention-card:first-of-type { border-top: none; }
  .attention-badge { color: white; font-size: 11.5px; font-weight: 700; padding: 4px 9px; border-radius: 6px; white-space: nowrap; }
  .attention-main { flex: 1; min-width: 0; }
  .attention-title { font-size: 14px; font-weight: 700; color: #1a1a1a; }
  .attention-company { font-size: 12.5px; font-weight: 400; color: #666; margin-left: 8px; }
  .attention-change { font-size: 13px; color: #333; margin-top: 3px; }
  .attention-factors { font-size: 11.5px; color: #999; margin-top: 3px; }
  .attention-side { font-size: 12.5px; white-space: nowrap; }
  .attention-row { display: grid; grid-template-columns: repeat(3, 1fr); align-items: stretch; gap: 16px; margin-bottom: 20px; }
  .attention-row .attention-panel { margin-bottom: 0; }
  @media (max-width: 1100px) { .attention-row { grid-template-columns: 1fr; } }
  .attention-link { color: #2e5fa3; text-decoration: none; font-weight: 600; }
  .attention-link:hover { text-decoration: underline; }
  .attention-link--none { color: #aaa; font-weight: 400; }
  .milestone-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
  .milestone-col-title { font-size: 12.5px; font-weight: 700; color: #1a1a1a; }
  .milestone-count { font-weight: 400; color: #888; }
  .milestone-col-note { font-size: 10.5px; color: #999; margin: 3px 0 8px; line-height: 1.4; }
  .milestone-row { display: flex; flex-direction: column; gap: 1px; padding: 6px 0; border-top: 1px solid #f0f0f0; font-size: 12px; }
  .milestone-row:first-of-type { border-top: none; }
  .milestone-row a { color: #2e5fa3; text-decoration: none; font-weight: 600; }
  .milestone-row a:hover { text-decoration: underline; }
  .milestone-sponsor { color: #666; }
  .milestone-detail { color: #333; font-weight: 600; }
  @media (max-width: 1100px) { .milestone-grid { grid-template-columns: repeat(2, 1fr); } }
"""


def render_competitive_sections(ar1001_ranking_df, recent_changes_df, attention_df, milestones,
                                 ar1001_top_n=10, changes_top_n=15, attention_top_n=8):
    row_html = f"""
    <div class="attention-row">
      {render_ar1001_relevance_section(ar1001_ranking_df, ar1001_top_n)}
      {render_recent_changes_section(recent_changes_df, changes_top_n)}
      {render_needs_attention_section(attention_df, attention_top_n)}
    </div>"""
    return row_html + render_milestones_section(milestones)
