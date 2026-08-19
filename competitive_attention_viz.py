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
# INTEGRATION: pipeline_viz.py embeds two placeholders verbatim in its
# HTML output — PLACEHOLDER right after the KPI row (for Recent
# Changes / Needs Attention), and MILESTONES_PLACEHOLDER further down,
# beneath the "AR1001 Competitive Landscape" panel (so the milestones
# grid reads as a follow-on to that landscape view rather than
# competing with Needs Attention for top-of-page attention). run_pipeline.py
# — AFTER pipeline_viz.py has run and outputs/pipeline_changes.csv /
# outputs/competitive_attention.csv are freshly computed — does a
# plain string .replace(PLACEHOLDER, render_competitive_sections(...))
# and .replace(MILESTONES_PLACEHOLDER, render_milestones_section(...))
# on the already-written pipeline_overview.html. This avoids re-running
# the entire (expensive) pipeline_viz.py a second time just to pick up
# data that necessarily depends on pipeline_viz.py's own prior output.
# A standalone `python3 pipeline_viz.py` run (outside run_pipeline.py)
# simply leaves both placeholder comments in place, which render as
# nothing — an honest empty state, not stale/wrong data.
# ============================================================

import html

PLACEHOLDER = "<!--COMPETITIVE_ATTENTION_SECTION-->"
MILESTONES_PLACEHOLDER = "<!--COMPETITIVE_MILESTONES_SECTION-->"

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


def _clean(value):
    """Normalizes a missing cell to None -- covers Python None, blank
    strings, AND pandas' float NaN (which round-tripping a DataFrame
    through CSV produces for any blank field, e.g.
    update_changes_history()'s accumulated history file). NaN is
    truthy in Python, so a bare `value or default` fallback chain
    silently keeps NaN instead of falling through to the next
    fallback -- this guard is what prevents the literal text "nan"
    (or a broken ".../study/nan" link) from ever being rendered."""
    if value is None:
        return None
    if value != value:  # NaN is the only value that is never equal to itself
        return None
    value = str(value).strip()
    return value or None


_PRIORITY_COLORS = {
    "Critical": ARIBIO_ACCENT,
    "High": _darken(ARIBIO_BLUE, 0.25),
    "Medium": ARIBIO_BLUE,
    "Low": "#9e9e9e",
}

# Recent Changes badge text: the default is change_type with underscores
# swapped for spaces (e.g. "phase_change" -> "phase change"), which
# reads fine except for "completion_date_change" -- sitting next to a
# "primary completion date change" badge on the same trial (ct.gov
# tracks Primary Completion Date and Study/Overall Completion Date as
# two separate fields that can change independently or together), the
# generic "completion date change" reads as a duplicate of the primary
# one rather than the distinct field it actually is.
_CHANGE_TYPE_BADGE_LABELS = {
    "completion_date_change": "Study/Overall Completion Date",
}

def _study_url(nct_id):
    nct_id = _clean(nct_id)
    return f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else ""


def _drug_only(df):
    """Shared drug-only filter for both panels: drops any row whose
    drug couldn't be resolved to a name (see _clean())."""
    if df is None or df.empty:
        return df
    return df[df["canonical_drug_name"].apply(lambda v: _clean(v) is not None)]


def shown_recent_change_nct_ids(changes_df, top_n=15):
    """The NCT IDs that will actually render in Recent Changes' current
    top-`top_n` slice -- used by render_competitive_sections() to keep
    Needs Attention from repeating the same drug/change a second time
    further down the page (see de-duplication note on
    render_needs_attention_section's `exclude_nct_ids` param)."""
    filtered = _drug_only(changes_df)
    if filtered is None or filtered.empty:
        return set()
    top = filtered.head(top_n)
    return set(top["nct_id"].dropna()) - {""}


def _drug_profile_lookup(drugs_df):
    """lowercased canonical_drug_name -> pipeline_drugs.csv row dict,
    the same display_name-keyed pattern competitive_attention.
    build_drug_lookup() uses -- kept as a separate, narrower copy here
    (rather than an import from competitive_attention.py) since this
    module only ever needs a handful of display fields, not that
    module's full lookup contract. Keyed lowercase (queried the same
    way below) because a change row's canonical_drug_name isn't always
    cased identically to drugs_df's display_name -- e.g. a "new_drug"
    change can carry "buntanetap/posiphen" against a rollup row titled
    "Buntanetap/Posiphen"; an exact-case match would silently drop the
    profile (and therefore the expand toggle) for rows like that."""
    if drugs_df is None or drugs_df.empty:
        return {}
    return {str(r["display_name"]).lower(): r for _, r in drugs_df.iterrows()}


def render_recent_changes_section(changes_df, top_n=15, drugs_df=None):
    """Raw, UNSCORED feed of every change detected in the trailing
    30-day window (competitive_attention.prepare_recent_changes) —
    distinct from Needs Attention below it, which is the same changes
    filtered/ranked by competitive priority. This section shows every
    change tied to a resolved drug in the last 30 days, regardless of
    how much competitive priority it scored. Drug-level scope only:
    a change whose drug couldn't be resolved to a name is dropped here
    rather than shown as a bare NCT ID -- keeps this feed, like Needs
    Attention, "only about drugs."

    `drugs_df` (pipeline_drugs.csv's per-drug rollup, optional): powers
    the expand-for-detail panel's drug-profile fields (Sponsor, Start
    date, Primary completion, Modality, Drug type category, AR1001
    Relevance) -- without it, drug names render as plain text with no
    expand toggle, same as a drug that isn't found in the lookup.
    """
    changes_df = _drug_only(changes_df)
    drug_profiles = _drug_profile_lookup(drugs_df)

    if changes_df is None or changes_df.empty:
        body = '<div class="attention-empty">No drug-related pipeline changes were detected in the last 30 days.</div>'
        count_note = "0 changes in the last 30 days"
    else:
        top = changes_df.head(top_n)
        count_note = f"showing {len(top)} of {len(changes_df)} changes detected in the last 30 days"

        # Group by drug (lowercased, same key as drug_profiles below) so
        # a drug with multiple changes in the window -- e.g. a new-trial
        # registration AND a completion-date change on the same day --
        # renders as one card with multiple change lines instead of one
        # near-duplicate card per change. Plain dict, not a groupby: it
        # preserves first-seen order (top is already sorted by
        # detected_date desc), which a groupby would need an extra sort
        # step to guarantee.
        groups = {}
        group_order = []
        for _, row in top.iterrows():
            drug_name_key = _clean(row.get("canonical_drug_name"))
            group_key = (drug_name_key or "").lower()
            if group_key not in groups:
                groups[group_key] = {
                    "drug_name_key": drug_name_key,
                    "drug": _esc(drug_name_key),
                    "company": _esc(_clean(row.get("sponsor_or_company")) or ""),
                    "rows": [],
                }
                group_order.append(group_key)
            groups[group_key]["rows"].append(row)

        rows_html = []
        for group_key in group_order:
            group = groups[group_key]
            drug = group["drug"]
            company = group["company"]

            change_items_html = []
            for row in group["rows"]:
                nct_id = _clean(row.get("nct_id")) or ""
                url = _study_url(nct_id)
                link_html = (
                    f'<a href="{_esc(url)}" target="_blank" rel="noopener" class="attention-link">{_esc(nct_id)}</a>'
                    if url else '<span class="attention-link attention-link--none">no linked trial</span>'
                )
                change_type_key = _clean(row.get("change_type")) or ""
                change_type = _esc(_CHANGE_TYPE_BADGE_LABELS.get(change_type_key, change_type_key.replace("_", " ")))
                detected_date = _esc(_clean(row.get("detected_date")) or "")

                change_items_html.append(f"""
                  <div class="attention-change-item">
                    <div class="attention-change-item-main">
                      <div class="attention-change">{_esc(_clean(row.get('description')) or '')}</div>
                      <div class="attention-change-date">{detected_date}</div>
                    </div>
                    <div class="attention-change-item-side">
                      <div class="attention-badge" style="background:#9e9e9e">{change_type}</div>
                      {link_html}
                    </div>
                  </div>""")

            # Expand-for-detail: clicking the drug NAME reveals that
            # drug's pipeline_drugs.csv profile -- same name-is-the-toggle
            # pattern the main table uses for its detail row (see
            # .drug-name-toggle in pipeline_viz.py), deliberately without
            # that table's separate caret icon. Only rendered when the
            # drug actually resolves to a pipeline_drugs.csv row, so an
            # unresolved/legacy name stays plain text with nothing to expand.
            drug_name_key = group["drug_name_key"]
            profile = drug_profiles.get(drug_name_key.lower()) if drug_name_key else None

            detail_fields = []
            if profile is not None:
                sponsor = _esc(_clean(profile.get("sponsor")) or "")
                start_date = _esc(_clean(profile.get("start_date_display")) or "")
                primary_completion = _esc(_clean(profile.get("primary_completion_date_display")) or "")
                modality = _esc(_clean(profile.get("modality")) or "")
                drug_type = _esc(_clean(profile.get("drug_type")) or "")
                relevance_score = profile.get("aribio_relevance_score")

                if sponsor:
                    detail_fields.append(f'<div><strong>Sponsor</strong>{sponsor}</div>')
                if start_date:
                    detail_fields.append(f'<div><strong>Start date</strong>{start_date}</div>')
                if primary_completion:
                    detail_fields.append(f'<div><strong>Primary completion</strong>{primary_completion}</div>')
                if modality:
                    detail_fields.append(f'<div><strong>Modality</strong>{modality}</div>')
                if drug_type:
                    detail_fields.append(f'<div><strong>Drug type category</strong>{drug_type}</div>')
                if relevance_score is not None and relevance_score == relevance_score:  # excludes NaN
                    detail_fields.append(f'<div><strong>AR1001 Relevance</strong>{int(relevance_score)}/100</div>')

            if detail_fields:
                drug_html = f'<button type="button" class="drug-toggle" onclick="toggleDrugDetail(this)">{drug}</button>'
                # Two nested divs, not one: the outer's grid-template-rows
                # is what actually animates (0fr -> 1fr, see CSS below) --
                # a single div can't smoothly transition to/from
                # display:none, so this expands/collapses via a real
                # height transition instead of an instant hidden-attribute
                # snap. The inner div is what clips mid-transition.
                detail_html = (
                    f'<div class="attention-detail"><div class="attention-detail-inner">'
                    f'{"".join(detail_fields)}</div></div>'
                )
            else:
                drug_html = drug
                detail_html = ""

            rows_html.append(f"""
            <div class="attention-card js-drug-row">
              <div class="attention-main">
                <div class="attention-title">{drug_html}<span class="attention-company">{company}</span></div>
                <div class="attention-change-list">{"".join(change_items_html)}</div>
                {detail_html}
              </div>
            </div>""")
        body = f'<div class="attention-cards-grid">{"".join(rows_html)}</div>'

    return f"""
    <div class="attention-panel" style="border-radius:{CARD_RADIUS}; box-shadow:{CARD_SHADOW}">
      <div class="attention-panel-header">
        <div class="attention-panel-title">Recent Changes</div>
        <div class="attention-panel-note">{count_note} &middot; every drug-related change, unranked &mdash; see Needs Attention below for competitive priority</div>
      </div>
      {body}
    </div>"""


def render_needs_attention_section(attention_df, top_n=8, exclude_nct_ids=None):
    """`exclude_nct_ids`: NCT IDs already shown in Recent Changes' own
    top-N slice (see shown_recent_change_nct_ids()) -- Needs Attention
    is the same underlying change data filtered/ranked by competitive
    priority, so without this an item can legitimately qualify for
    both panels and appear twice on the same page. Dropped here,
    before the top_n slice and count, so the displayed count reflects
    what's actually new information on this panel.
    """
    attention_df = _drug_only(attention_df)
    had_items_before_dedup = attention_df is not None and not attention_df.empty
    if attention_df is not None and not attention_df.empty and exclude_nct_ids:
        attention_df = attention_df[~attention_df["nct_id"].isin(exclude_nct_ids)]

    if attention_df is None or attention_df.empty:
        body = (
            '<div class="attention-empty">Everything currently needing attention is already shown in Recent Changes above.</div>'
            if had_items_before_dedup and exclude_nct_ids
            else '<div class="attention-empty">No drug-related pipeline changes currently need attention.</div>'
        )
        count_note = "0 items"
    else:
        top = attention_df.head(top_n)
        # Named-competitor changes (see competitive_attention.py's
        # POINTS_NAMED_COMPETITOR / is_named_competitor) are shown
        # unconditionally, not just ranked higher -- a change from a
        # watchlist company (config/aribio_watchlist.yaml's
        # competitor_companies) must never silently fall off this
        # panel just because more than top_n other changes happened
        # the same day. Appended after the natural top_n (which the
        # scoring bonus already usually puts them inside anyway), kept
        # in their own relevance_score order.
        # .astype(bool) rather than a raw boolean & -- is_named_competitor
        # arrives as a real bool from compute_attention(), but a
        # hand-built test/empty-state DataFrame may default every
        # ATTENTION_COLUMNS field (this one included) to "", which a
        # bare `&` can't combine with a boolean mask. bool("") is
        # correctly False, so this treats "unset" the same as "no".
        is_named = attention_df["is_named_competitor"].astype(bool)
        forced = attention_df[is_named & ~attention_df.index.isin(top.index)]
        count_note = f"showing top {len(top)} of {len(attention_df)}"
        if not forced.empty:
            count_note += f" (+{len(forced)} watchlist-company change{'s' if len(forced) != 1 else ''} shown regardless of rank)"
        rows_html = []
        # Two separate passes (not a DataFrame concat) so this module
        # stays free of a pandas import -- attention_df is already a
        # DataFrame handed in by the caller; this file only ever reads
        # from it, never builds/merges DataFrames of its own.
        for _, row in list(top.iterrows()) + list(forced.iterrows()):
            level = row["priority_level"]
            color = _PRIORITY_COLORS.get(level, "#9e9e9e")
            nct_id = _clean(row.get("nct_id")) or ""
            drug = _esc(_clean(row.get("canonical_drug_name")))
            company = _esc(_clean(row.get("company_or_sponsor")) or "")
            url = _study_url(nct_id)
            link_html = (
                f'<a href="{_esc(url)}" target="_blank" rel="noopener" class="attention-link">{_esc(nct_id)}</a>'
                if url else '<span class="attention-link attention-link--none">no linked trial</span>'
            )
            rows_html.append(f"""
            <div class="attention-card">
              <div class="attention-main">
                <div class="attention-title">{drug}<span class="attention-company">{company}</span></div>
                <div class="attention-change">{_esc(_clean(row.get('why_it_matters')) or '')}</div>
                <div class="attention-factors">{_esc(_clean(row.get('relevance_factors')) or '')}</div>
              </div>
              <div class="attention-side">
                <div class="attention-badge" style="background:{color}">{_esc(level)} &middot; {int(row['relevance_score'])}</div>
                {link_html}
              </div>
            </div>""")
        body = "".join(rows_html)

    return f"""
    <div class="attention-panel" style="border-radius:{CARD_RADIUS}; box-shadow:{CARD_SHADOW}">
      <div class="attention-panel-header">
        <div class="attention-panel-title">Needs Attention
          <button type="button" class="attention-notes-edit-btn" id="attentionNotesEditBtn" onclick="openAttentionNotesModal()" title="Add meeting notes" aria-label="Add meeting notes">Edit &#9998;</button>
        </div>
        <div class="attention-panel-note">{count_note} &middot; ranked by deterministic competitive-priority score, not AI judgment</div>
      </div>
      {_render_attention_notes_widget()}
      {body}
    </div>"""


def _render_attention_notes_widget():
    """Meeting-notes widget for the Needs Attention panel: an "Edit ✎"
    button (in the header, see above) opens a popup with two sections
    -- an optional "Drug name" field and a "General notes" textarea --
    plus Save/Cancel. Saving APPENDS a new dated note to a running list
    (shown below the header) rather than replacing a single blob or a
    single field, so the modal can be reopened any number of times to
    keep adding more notes. Each entry shows its drug tag (if any), its
    save date, and reveals a delete button on hover.

    Persistence: this static HTML file is fully regenerated from
    scratch on every pipeline refresh (see run_pipeline.py), so notes
    baked into the page's own markup would be silently discarded the
    next time the page is rebuilt -- which is exactly what destroyed
    the original version of this feature. Instead, notes are saved as
    a JSON array in the browser's localStorage -- that storage lives
    in the browser, independent of the file's content, so it survives
    any number of future rebuilds of this same page.
    """
    return """
      <div class="attention-notes-list" id="attentionNotesList"></div>
      <div class="attention-notes-modal-overlay" id="attentionNotesModalOverlay">
        <div class="attention-notes-modal">
          <div class="attention-notes-modal-title">Add meeting notes</div>
          <div class="attention-notes-field">
            <label class="attention-notes-label" for="attentionNotesDrugInput">Drug name</label>
            <input type="text" id="attentionNotesDrugInput" class="attention-notes-input" placeholder="e.g. AR1001 (optional)">
          </div>
          <div class="attention-notes-field">
            <label class="attention-notes-label" for="attentionNotesTextarea">General notes</label>
            <textarea id="attentionNotesTextarea" class="attention-notes-textarea" placeholder="Add notes..."></textarea>
          </div>
          <div class="attention-notes-actions">
            <button type="button" class="attention-notes-cancel-btn" onclick="cancelAttentionNotesEdit()">Cancel</button>
            <button type="button" class="attention-notes-save-btn" onclick="saveAttentionNotes()">Save</button>
          </div>
        </div>
      </div>"""


ATTENTION_NOTES_SCRIPT = """
<script>
(function () {
  var NOTES_KEY = "aribio_needs_attention_notes";

  function getNotes() {
    try {
      var parsed = JSON.parse(localStorage.getItem(NOTES_KEY) || "[]");
      return Array.isArray(parsed) ? parsed : [];
    } catch (e) {
      return [];
    }
  }

  function setNotes(notes) {
    localStorage.setItem(NOTES_KEY, JSON.stringify(notes));
  }

  function escapeHtml(s) {
    var div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  function renderAttentionNotesList() {
    var list = document.getElementById("attentionNotesList");
    if (!list) return;
    var notes = getNotes();
    list.innerHTML = notes.slice().reverse().map(function (n) {
      var drugHtml = n.drug ? ('<div class="attention-note-drug">' + escapeHtml(n.drug) + '</div>') : '';
      return '<div class="attention-note-entry">' +
        drugHtml +
        '<div class="attention-note-text">' + escapeHtml(n.text) + '</div>' +
        '<div class="attention-note-date">' + escapeHtml(n.date) + '</div>' +
        '<button type="button" class="attention-note-delete-btn" onclick="deleteAttentionNote(\\'' + n.id + '\\')" title="Delete note" aria-label="Delete note">&times;</button>' +
        '</div>';
    }).join("");
  }

  window.openAttentionNotesModal = function () {
    var overlay = document.getElementById("attentionNotesModalOverlay");
    var drugInput = document.getElementById("attentionNotesDrugInput");
    var textarea = document.getElementById("attentionNotesTextarea");
    if (!overlay) return;
    drugInput.value = "";
    textarea.value = "";
    overlay.style.display = "flex";
    drugInput.focus();
  };

  window.cancelAttentionNotesEdit = function () {
    document.getElementById("attentionNotesModalOverlay").style.display = "none";
  };

  window.saveAttentionNotes = function () {
    var drugInput = document.getElementById("attentionNotesDrugInput");
    var textarea = document.getElementById("attentionNotesTextarea");
    var drug = drugInput.value.trim();
    var text = textarea.value.trim();
    if (text) {
      var notes = getNotes();
      notes.push({ id: String(Date.now()), drug: drug, text: text, date: new Date().toLocaleString() });
      setNotes(notes);
      renderAttentionNotesList();
    }
    document.getElementById("attentionNotesModalOverlay").style.display = "none";
  };

  window.deleteAttentionNote = function (id) {
    if (!window.confirm("Delete this note? This can't be undone.")) return;
    setNotes(getNotes().filter(function (n) { return n.id !== id; }));
    renderAttentionNotesList();
  };

  // Drug-name expand toggle for Recent Changes cards (see
  // render_recent_changes_section()) -- the row is marked ".js-drug-row"
  // so this function can find its ".attention-detail" panel and toggle
  // its "expanded" class (see the CSS grid-row transition on that
  // class -- a plain [hidden]/display:none toggle can't animate).
  // Reads DOM state off the class itself rather than tracking its own
  // open/closed flag, so it stays correct no matter how many rows are
  // on the page.
  window.toggleDrugDetail = function (btn) {
    var row = btn.closest(".js-drug-row");
    var detail = row && row.querySelector(".attention-detail");
    if (!detail) return;
    detail.classList.toggle("expanded");
  };

  document.addEventListener("DOMContentLoaded", renderAttentionNotesList);
})();
</script>
"""


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
        meta_parts = [_esc(p) for p in (_clean(item.get("phase")), _clean(item.get("drug_type"))) if p]
        meta_html = f'<span class="milestone-meta">{" &middot; ".join(meta_parts)}</span>' if meta_parts else ""
        if bucket_key == "materially_delayed":
            detail = f"{_esc(item.get('old_value',''))} &rarr; {_esc(item.get('new_value',''))} ({item.get('days_delayed','')} days)"
        else:
            detail = _esc(item.get(date_field, ""))
        rows.append(
            f'<div class="milestone-row">'
            f'<a href="{_esc(url)}" target="_blank" rel="noopener" title="{_esc(nct_id)}">{_esc(drug_name)}</a>'
            f'<span class="milestone-sponsor">{sponsor}</span>'
            f'{meta_html}'
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
          <div class="milestone-col-list">{_render_milestone_list(items, key, date_field)}</div>
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
  .attention-panel-title { font-size: 16px; font-weight: 700; color: #1d3d69; }  /* dashboard_nav.py's NAV_BG */
  .attention-notes-edit-btn {
    background: none; border: 1px solid #dde3ec; cursor: pointer; font-size: 12px; font-weight: 600;
    color: #556; margin-left: 8px; padding: 4px 10px; border-radius: 6px; vertical-align: middle;
    font-family: inherit;
  }
  .attention-notes-edit-btn:hover { color: #2e5fa3; background: #eef2f8; border-color: #2e5fa3; }
  .attention-notes-modal-overlay {
    display: none; position: fixed; inset: 0; background: rgba(20, 30, 50, 0.45);
    align-items: center; justify-content: center; z-index: 1000;
  }
  .attention-notes-modal {
    background: white; border-radius: 12px; padding: 20px 22px; width: 420px; max-width: 90vw;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
  }
  .attention-notes-modal-title { font-size: 15px; font-weight: 700; color: #1a1a1a; margin-bottom: 12px; }
  .attention-notes-field { margin-bottom: 12px; }
  .attention-notes-label {
    display: block; font-size: 11px; font-weight: 700; letter-spacing: 0.02em; text-transform: uppercase;
    color: #888; margin-bottom: 5px;
  }
  .attention-notes-input {
    width: 100%; font-family: inherit; font-size: 12.5px; padding: 8px 12px;
    border: 1px solid #dde3ec; border-radius: 8px; box-sizing: border-box;
  }
  .attention-notes-textarea {
    width: 100%; min-height: 90px; font-family: inherit; font-size: 12.5px; padding: 10px 12px;
    border: 1px solid #dde3ec; border-radius: 8px; resize: vertical; box-sizing: border-box;
  }
  .attention-notes-actions { margin-top: 12px; display: flex; justify-content: flex-end; gap: 8px; }
  .attention-notes-save-btn, .attention-notes-cancel-btn {
    font-size: 12px; font-weight: 600; padding: 6px 14px; border-radius: 6px; cursor: pointer;
    border: none; font-family: inherit;
  }
  .attention-notes-save-btn { background: #2e5fa3; color: white; }
  .attention-notes-cancel-btn { background: #eee; color: #555; }
  .attention-notes-list { margin: -4px 0 14px; display: flex; flex-direction: column; gap: 8px; }
  .attention-note-entry {
    position: relative; background: white; border: 1px solid #eee; border-radius: 8px;
    padding: 10px 34px 10px 14px; font-size: 12.5px; color: #444;
  }
  .attention-note-drug { font-size: 12.5px; font-weight: 700; color: #2e5fa3; margin-bottom: 2px; }
  .attention-note-text { white-space: pre-wrap; line-height: 1.5; }
  .attention-note-date { font-size: 10.5px; color: #999; margin-top: 4px; }
  .attention-note-delete-btn {
    display: none; position: absolute; top: 6px; right: 6px; background: none; border: none;
    cursor: pointer; font-size: 16px; color: #999; line-height: 1; padding: 3px 7px; border-radius: 4px;
  }
  .attention-note-entry:hover .attention-note-delete-btn { display: block; }
  .attention-note-delete-btn:hover { background: #eef2f8; color: #2e5fa3; }
  .attention-panel-note { font-size: 12px; color: #888; }
  .attention-empty, .milestone-empty { font-size: 13px; color: #888; padding: 8px 0; }
  .attention-card { display: flex; align-items: flex-start; gap: 14px; padding: 12px 0; border-top: 1px solid #eee; }
  .attention-card:first-of-type { border-top: none; }
  .attention-badge { color: white; font-size: 11.5px; font-weight: 700; padding: 4px 9px; border-radius: 6px; white-space: nowrap; }
  .attention-main { flex: 1; min-width: 0; }
  .attention-title { font-size: 14px; font-weight: 700; color: #1a1a1a; }
  .attention-company { font-size: 12.5px; font-weight: 400; color: #666; margin-left: 8px; }
  .attention-change { font-size: 13px; color: #333; margin-top: 3px; }
  .attention-change-date { font-size: 10.5px; color: #999; margin-top: 4px; }
  /* Expand-toggle button for a Recent Changes card's drug name --
     plain text by default, no icon, matching the main table's own
     .drug-name-toggle. */
  .drug-toggle { background: none; border: none; cursor: pointer; font: inherit; color: inherit; padding: 0; text-decoration: none; text-align: left; }
  .drug-toggle:hover { color: #2e5fa3; }
  /* Smooth expand/collapse: grid-template-rows 0fr -> 1fr is what
     actually animates. A plain height/max-height transition can't do
     this cleanly since the target height (the content's natural size)
     isn't known up front -- the 0fr/1fr grid track trick sidesteps
     that, and toggling a class here (not the [hidden] attribute) is
     what makes it transition at all, since display:none can't animate.
     The inner div is what actually gets clipped by overflow:hidden
     mid-transition -- grid-template-rows alone only sizes the track,
     it doesn't clip overflowing content on its own. Padding/border/gap
     live on the inner div unconditionally (not gated on .expanded) so
     they never have to snap on their own outside the transition. */
  .attention-detail {
    display: grid; grid-template-rows: 0fr; opacity: 0;
    transition: grid-template-rows 0.22s ease, opacity 0.18s ease;
  }
  .attention-detail.expanded { grid-template-rows: 1fr; opacity: 1; }
  .attention-detail-inner {
    overflow: hidden; min-height: 0; display: flex; flex-direction: column; gap: 4px;
    padding-top: 8px; margin-top: 8px; border-top: 1px dashed #e2e2e2;
  }
  .attention-detail div { font-size: 12px; color: #444; }
  .attention-detail strong { display: inline-block; min-width: 82px; font-size: 10px; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; color: #9aa0ab; }
  .attention-factors { font-size: 11.5px; color: #999; margin-top: 3px; }
  .attention-side { font-size: 12.5px; white-space: nowrap; display: flex; flex-direction: column; align-items: flex-end; gap: 6px; flex: none; }
  /* Recent Changes cards group every change for the same drug under one
     card (see render_recent_changes_section()) -- each change gets its
     own row with its own badge/date/link, rather than one .attention-side
     per card, since a single card can hold multiple changes with
     different types/dates/linked trials. */
  .attention-change-list { display: flex; flex-direction: column; gap: 8px; margin-top: 4px; }
  .attention-change-item { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; padding-top: 8px; border-top: 1px solid #f2f2f2; }
  .attention-change-item:first-child { padding-top: 0; border-top: none; }
  .attention-change-item-main { flex: 1; min-width: 0; }
  .attention-change-item-side { font-size: 12.5px; white-space: nowrap; display: flex; flex-direction: column; align-items: flex-end; gap: 6px; flex: none; }
  .attention-row { display: grid; grid-template-columns: 1fr; align-items: stretch; gap: 16px; margin-bottom: 20px; }
  .attention-row .attention-panel { margin-bottom: 0; }
  .attention-cards-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px 20px; }
  .attention-cards-grid .attention-card { border: 1px solid #eee; border-radius: 8px; padding: 12px 14px; }
  @media (max-width: 900px) { .attention-cards-grid { grid-template-columns: 1fr; } }
  .attention-link { color: #2e5fa3; text-decoration: none; font-weight: 600; }
  .attention-link:hover { text-decoration: underline; }
  .attention-link--none { color: #aaa; font-weight: 400; }
  .milestone-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
  .milestone-col-title { font-size: 12.5px; font-weight: 700; color: #1a1a1a; }
  .milestone-count { font-weight: 400; color: #888; }
  .milestone-col-note { font-size: 10.5px; color: #999; margin: 3px 0 8px; line-height: 1.4; }
  /* "Scroll shadow" trick: the two radial-gradient shadows are
     background-attachment:scroll (fixed to the box, not the content),
     the two white fades are background-attachment:local (scroll WITH
     the content) -- so the top shadow only shows once you've scrolled
     past the first row, and the bottom shadow only shows while there's
     more content below. That's the actual "this scrolls" affordance;
     a plain overflow:auto with an invisible/auto-hiding OS scrollbar
     (e.g. macOS) gives no such cue on its own. */
  .milestone-col-list {
    max-height: 320px; overflow-y: auto; padding-right: 6px;
    background:
      linear-gradient(white 30%, rgba(255,255,255,0)) center top,
      linear-gradient(rgba(255,255,255,0), white 70%) center bottom,
      radial-gradient(farthest-side at 50% 0, rgba(0,0,0,.18), rgba(0,0,0,0)) center top,
      radial-gradient(farthest-side at 50% 100%, rgba(0,0,0,.18), rgba(0,0,0,0)) center bottom;
    background-repeat: no-repeat;
    background-size: 100% 24px, 100% 24px, 100% 10px, 100% 10px;
    background-attachment: local, local, scroll, scroll;
    scrollbar-width: thin; scrollbar-color: #ccc transparent;
  }
  .milestone-col-list::-webkit-scrollbar { width: 7px; }
  .milestone-col-list::-webkit-scrollbar-thumb { background: #ccc; border-radius: 4px; }
  .milestone-col-list::-webkit-scrollbar-thumb:hover { background: #aaa; }
  .milestone-row { display: flex; flex-direction: column; gap: 1px; padding: 6px 0; border-top: 1px solid #f0f0f0; font-size: 12px; }
  .milestone-row:first-of-type { border-top: none; }
  .milestone-row a { color: #2e5fa3; text-decoration: none; font-weight: 600; }
  .milestone-row a:hover { text-decoration: underline; }
  .milestone-sponsor { color: #666; }
  .milestone-meta { color: #999; font-size: 11px; }
  .milestone-detail { color: #333; font-weight: 600; }
  @media (max-width: 1100px) { .milestone-grid { grid-template-columns: repeat(2, 1fr); } }
"""


def render_competitive_sections(recent_changes_df, attention_df,
                                 changes_top_n=15, attention_top_n=8, drugs_df=None):
    # Milestones is intentionally NOT part of this bundle — it renders
    # separately via render_milestones_section() into MILESTONES_PLACEHOLDER,
    # further down the page beneath "AR1001 Competitive Landscape".
    #
    # Needs Attention is the same underlying change data as Recent
    # Changes, just filtered/ranked by competitive priority -- so an
    # item can legitimately qualify for both. shown_recent_change_nct_ids()
    # tells Needs Attention which NCT IDs are already visible above it,
    # so the same drug/change is never rendered twice on one page.
    already_shown = shown_recent_change_nct_ids(recent_changes_df, changes_top_n)
    return f"""
    <div class="attention-row">
      {render_recent_changes_section(recent_changes_df, changes_top_n, drugs_df=drugs_df)}
      {render_needs_attention_section(attention_df, attention_top_n, exclude_nct_ids=already_shown)}
    </div>
    {ATTENTION_NOTES_SCRIPT}"""
