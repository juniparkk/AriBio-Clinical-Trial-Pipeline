# Migration Plan: AD Trials Dashboard → AriBio Competitive Intelligence Platform

**Status:** planning document only. No behavior changed. Git checkpoint: `1c93574` ("Checkpoint: current AD trials dashboard before AriBio competitive-intel migration").

This document answers the repository-inspection questions, maps the current dependency structure and data flow, lists concrete (data-verified, not theoretical) risks, and proposes a phased migration plan. Nothing in this document has been implemented yet.

---

## 1. File inventory — source vs. generated

| File | Type | Notes |
|---|---|---|
| `pipeline_viz.py` | **Source** | Orchestrator: loads data, runs both classification pipelines, builds `pipeline_overview.html`, writes all CSV outputs |
| `drug_classification.py` | **Source** | Pure, side-effect-free library: parsing, classification, matching, and rollup logic. No file I/O except `load_official_pipeline()`, which only reads |
| `data/official_pipeline.csv` | **Source (hand-curated reference data)** | Company → drug → synonym lookup used for "confirmed official match" classification. Currently 6 seed rows, all with blank `source_url` |
| `test_classification.py` | **Source (tests)** | 107 tests against `drug_classification.py`'s pure functions |
| `test_dashboard_table.py` | **Source (tests)** | 10 tests against the rendered HTML's embedded JSON data |
| `trials.csv` | **Source (manually-downloaded external data)** | Raw ClinicalTrials.gov export. Manually re-downloaded and dropped in by hand — no automated ingestion |
| `nih_data.csv` | **Source (unused)** | Present in the directory but not read anywhere in `pipeline_viz.py` or `drug_classification.py` — orphaned input, not part of the active pipeline |
| `pipeline_overview.html` | **Generated** | The dashboard itself |
| `pipeline_overview backup.html` | **Stray/manual** | Not produced by the script — appears to be a manually-saved duplicate (same size/timestamp pattern as the real output). Flagging rather than deleting; worth confirming with you whether it can be removed |
| `pipeline_annotated.csv` | **Generated** | One row per trial, both legacy and new classification columns side by side |
| `pipeline_drugs.csv` | **Generated** | One row per drug — the new pipeline's output only |
| `pipeline_interventions.csv` | **Generated** | One row per individual intervention (the full audit trail behind every `pipeline_drugs.csv` row) |
| `pipeline_unresolved_trials.csv` | **Generated** | Trials the new pipeline couldn't confidently resolve to a single drug — audit output, not silently dropped |
| `.venv/` | **Environment** | Local Python virtualenv — gitignored, not part of the checkpoint |

---

## 2. Answers to the inspection questions

### 1. Where ClinicalTrials.gov data is loaded
`pipeline_viz.py` STEP 1 (line 82): `df = pd.read_csv("trials.csv", low_memory=False)`. This is a **manual, one-time load** — there is no download automation, API integration, or scheduled refresh. Getting fresh data today means: go to clinicaltrials.gov, search "Alzheimer's Disease", export CSV, overwrite `trials.csv`, re-run the script.

STEP 2 (line 95) renames whichever of several known ClinicalTrials.gov column-header variants are present (`"NCT Number"`/`"nctId"` → `nct_id`, etc.) into a stable internal schema (`nct_id, title, status, phase, interventions, enrollment, sponsor, conditions, start_date`), then filters to Phase 1/2/3 only.

### 2. How intervention names are extracted
**Two independent extraction paths exist side by side**, which is itself a major structural risk (see §4):

- **Legacy path** — `primary_intervention_name()` (`pipeline_viz.py:389`): splits the raw `Interventions` cell on `|`, and returns the **first** entry typed `DRUG:` or `BIOLOGICAL:` that isn't literally "placebo". Everything else in that trial (other candidate drugs, comparators, diagnostics) is silently discarded at this step.
- **New path** — `parse_interventions()` (`drug_classification.py:81`): splits the same raw cell into **every** entry as a `{type, name}` pair, preserving all of them. Also strips a duplicated leading type prefix if present (e.g. `"DRUG: Drug: [18F]F-AraG (PET tracer)"` → name becomes `"[18F]F-AraG (PET tracer)"`, not `"Drug: [18F]F-AraG (PET tracer)"`).

### 3. How drug names are currently generated
- **Legacy**: `primary_intervention_name()` → `clean_drug_name()` (now an alias for `normalize_intervention_candidate_name()`, which strips dose/route/formulation text) → `canonical_drug_key()` (substring match against the `KNOWN_COMPOUNDS` dict to merge naming variants) → grouped by `summarize_drug()` into `legacy_drugs_df`.
- **New**: `classify_intervention()` (`drug_classification.py:751`) runs a full rules engine per intervention (placebo → non-therapeutic-control → diagnostic tracer → procedure → device → behavioral → official-pipeline match → approved-background-drug disambiguation using sibling interventions → development-code/known-compound pattern → sole-plausible-candidate rule → uncertain fallback), then `resolve_developed_drug()` aggregates all of one trial's classified interventions into a single trial-level `developed_drug`, then `build_resolved_drugs_dataframe()` aggregates across trials into one row per drug (`resolved_drugs_df`).

### 4. How drug type and target are classified
`guess_drug_type()` (line 185) and `guess_target()` (line 321) — **both legacy, and both still the only drug-type/target classification logic that exists.** Critically, **the new pipeline does not reclassify drug_type/target from the resolved `developed_drug`** — `resolved_drugs_df`'s `drug_type`/`target` columns are pulled straight from `df["drug_type"]`/`df["target"]`, which were computed by `guess_drug_type()`/`guess_target()` against the **raw, un-filtered, combined intervention text** of the whole trial (placebo arms and all). The new pipeline changed *which trials contribute a drug row at all*, but never touched *how that drug is categorized*. This is a real architectural gap for a competitive-intelligence use case, where "Amyloid vs. Tau vs. Neuroprotection" categorization needs to be trustworthy.

### 5. How trials are aggregated into drug-level rows
**Two parallel, independently-maintained rollups**:
- `legacy_drugs_df` — `groupby("drug_key")` where `drug_key` comes from the legacy substring-match `canonical_drug_key()`. Still drives the heatmap, the Phase 3 leaderboard, and the KPI tile counts.
- `resolved_drugs_df` — `build_resolved_drugs_dataframe()` (`drug_classification.py:1149`), `groupby("developed_drug_normalized")`, restricted to trials classified `sponsor_developed_therapeutic` or `investigational_therapeutic_unverified` with a resolved (non-ambiguous) `developed_drug`. Drives the visible table and `pipeline_drugs.csv`.

These two rollups **can and do disagree** (see §4, risk 1) — a genuine dependency-map hazard: the same page shows two different "how many drugs" answers depending on which panel you look at.

### 6. How `pipeline_overview.html` is generated
`pipeline_viz.py` STEP 4–6 (lines 616–1485): colors → 4 pie charts built from `df` directly (not either drug-level rollup) → heatmap + Phase 3 leaderboard built from `legacy_drugs_df` → the whole page (CSS + HTML + embedded JSON of `resolved_drugs_df` + hand-written JS for search/sort/filter) is assembled as one large Python f-string and written to disk with `open("pipeline_overview.html", "w")`. No templating engine, no build step — the Python script *is* the site generator.

### 7. Source files vs. generated outputs
See the table in §1.

### 8. Where current logic could incorrectly include non-drug interventions as drugs

This section is **evidence-based** — I queried the actual current `pipeline_interventions.csv` rather than reasoning abstractly. Results:

| Intervention type | Total rows | Rows leaking through as a drug candidate (`investigational_therapeutic_unverified`/`sponsor_developed_therapeutic`/`uncertain`) |
|---|---|---|
| BEHAVIORAL | 99 | **0** — cleanly excluded |
| DEVICE | 71 | **0** — cleanly excluded |
| PROCEDURE | 42 | **0** — cleanly excluded |
| RADIATION | 19 | **0** — cleanly excluded |
| **DIETARY_SUPPLEMENT** | 36 | **26** — leaking |
| **DIAGNOSTIC_TEST** | 18 | **10** — leaking |
| **COMBINATION_PRODUCT** | 7 | several — leaking |
| **GENETIC** | 3 | some legitimate (gene therapy candidates), some not |

Concrete examples currently sitting in `pipeline_drugs.csv`/`pipeline_annotated.csv` today:
- **Supplements treated as drug candidates**: `lutein/zeaxanthin`, `Resveratrol with Glucose, and Malate`, `Curcumin C3 Complex`, `tributyrin`, `Nicotinamide Riboside (NR)` — all typed `DIETARY_SUPPLEMENT`, none excluded, because `_passes_therapeutic_gate()` in `drug_classification.py` never checks for that type at all.
- **Diagnostics/procedures treated as drug candidates**: `Blood Test`, `cerebral RMI`, `Cerebrospinal fluid (CSF) Biomarkers`, `Urine test`, `PET/MR Imaging`, `Retinal fundus photography`, and two more PET tracers the curated list still misses (`[18F]-MFBG PET CT`, `[18F]-MFBG PET dosimetry scans`) — all typed `DIAGNOSTIC_TEST`, which (unlike `PROCEDURE`/`RADIATION`) is never checked by `_is_procedure()`.
- **Non-drug program/behavioral names treated as drug candidates**: `CBTi with Application` (Cognitive Behavioral Therapy for insomnia — clearly behavioral, but typed `COMBINATION_PRODUCT` so the behavioral-type check never runs), `Alianza Latina` (reads like a community outreach program name, not a therapeutic, typed `COMBINATION_PRODUCT`).

**Root cause, structurally**: `_passes_therapeutic_gate()` (`drug_classification.py:518`) explicitly checks for `DEVICE` type and the `PROCEDURE`/`RADIATION` types, but has no branch at all for `DIETARY_SUPPLEMENT`, `DIAGNOSTIC_TEST`, `COMBINATION_PRODUCT`, or `GENETIC` — those four types fall through entirely to name-keyword matching, which is necessarily incomplete (as every prior checkpoint's audits in this project's history have found — this is the same class of gap as the PET-tracer curated-list misses, just for an untouched type category rather than a name list).

---

## 3. Dependency map

```
trials.csv (manual download)
        │
        ▼
pipeline_viz.py STEP 1–2  (load, rename columns, filter to Phase 1/2/3)
        │
        ▼
df  ────────────────────────────────────────────────────────────────┐
        │                                                            │
        ▼ STEP 3 (legacy)                                            │
guess_drug_type() / guess_target()  ──► df["drug_type"], df["target"]│  (never re-derived downstream —
        │                                                            │   both rollups inherit these
        ▼ STEP 3.5 (legacy)                                          │   as-is from raw trial text)
primary_intervention_name() → clean_drug_name() → canonical_drug_key()
        │
        ▼
legacy_drugs_df  ──────────────► heatmap, Phase 3 leaderboard, KPI tiles, spotlight-adjacent counts

        (df, continued)
        │
        ▼ STEP 3.6 (new — drug_classification.py)
parse_interventions() → classify_intervention() → resolve_developed_drug()
        │
        ▼ STEP 3.7 (new)
build_resolved_drugs_dataframe()  ──► resolved_drugs_df ──► visible HTML table, pipeline_drugs.csv
        │
        └────────────────────────────► build_unresolved_trials_dataframe() ──► pipeline_unresolved_trials.csv

df (phase_clean/drug_type/target/status_clean, still legacy-derived)
        │
        ▼ STEP 5
4 pie charts (Trial Composition) — reads df directly, uses NEITHER rollup

interventions_df (from STEP 3.6) ──► pipeline_interventions.csv
df + both classification column sets ──► pipeline_annotated.csv

STEP 6: html_template (f-string) ──► pipeline_overview.html
```

**Key dependency hazard**: `legacy_drugs_df` and `resolved_drugs_df` are built from the same `df` but via completely different logic, and **four different parts of the same page** currently source from three different places (`legacy_drugs_df`, `resolved_drugs_df`, and `df` directly for the pies) with no single shared "what counts as a drug" definition.

---

## 4. Major risks

1. **Three sources of truth on one page.** The pies (from `df`), the heatmap/leaderboard/KPIs (from `legacy_drugs_df`), and the visible table (from `resolved_drugs_df`) can disagree about trial/drug counts, because they're built from different classification logic. For an internal descriptive dashboard this was a manageable, explicitly-flagged transitional state. For a competitive-intelligence product that AriBio stakeholders might use to make claims about competitor pipelines, this is not acceptable — every view needs to agree.

2. **drug_type/target are never re-derived from the verified developed_drug.** They're inherited from `guess_drug_type()`/`guess_target()` run against raw, unfiltered trial text — meaning a drug's "Amyloid vs. Tau" categorization in `pipeline_drugs.csv` isn't actually connected to the same classification logic that decided it was a real drug in the first place.

3. **Non-drug interventions are still becoming "drug" rows** — confirmed with real data above: supplements, diagnostic/lab tests, and at least one behavioral therapy and one program name. For competitive intelligence specifically, a false "competitor drug" is worse than a missing one — it risks a wrong claim being repeated internally.

4. **`official_pipeline.csv` is thin.** 6 rows, all with blank `source_url`. Every "confirmed official match" in the system today is really only "matched a company+drug pair with no citation" (`pipeline_record_match_without_source`) — meaningful for AriBio's own AR1001, but not yet a real competitive-intelligence-grade reference set for competitor sponsors.

5. **Data freshness is entirely manual.** No ingestion automation, no timestamp/versioning of `trials.csv` beyond file-modified-date, no diffing between refreshes to surface "what's new since last time" — which is close to the core value proposition of a competitive-intelligence tool (knowing when a competitor starts a new trial).

6. **No sponsor/competitor-centric view exists yet.** The entire data model is drug-centric (`resolved_drugs_df` groups by drug, with sponsor as a joined string field on each row). A competitive-intelligence platform's primary axis of analysis is usually the *sponsor* (who is doing what), not just the drug.

7. **`nih_data.csv` is an orphaned input** — present in the repo, unused by any code path. Worth clarifying whether it was meant to be incorporated (e.g., NIH grant funding data as a competitive signal) or is leftover from an earlier exploration.

---

## 5. Proposed migration plan

Phased, so each step is independently reviewable and testable before the next begins — consistent with how this project has been built so far.

**Phase 0 — Consolidate on one source of truth.**
Retire `primary_intervention_name()` / `canonical_drug_key()` / `legacy_drugs_df` and rewire the heatmap, Phase 3 leaderboard, KPI tiles, and pies onto `resolved_drugs_df`/`df`'s new classification columns. This removes risk #1 entirely. Must be done first — every later phase assumes one consistent drug definition across the whole page.

**Phase 1 — Close the classification gaps found in §2.8.**
Add type-based exclusion for `DIETARY_SUPPLEMENT` (→ `other`, non-treatment-control-style, not a candidate) and extend `_is_procedure()`/diagnostic checks to cover `DIAGNOSTIC_TEST` type. Audit `COMBINATION_PRODUCT`/`GENETIC` rows individually (these two types are legitimately mixed — real gene-therapy candidates exist alongside mislabeled ones) rather than blanket-excluding them.

**Phase 2 — Re-derive drug_type/target from the verified developed_drug, not raw trial text.**
Once a trial's `developed_drug` is resolved, `guess_drug_type()`/`guess_target()` (or their successors) should run against *that specific intervention's* type/name — not the whole trial's combined intervention string, placebo arms included. Closes risk #2.

**Phase 3 — Strengthen `official_pipeline.csv` into real competitive-intelligence reference data.**
Expand beyond the 6 seed rows to cover known competitor programs (Eisai/Biogen, Lilly, Roche, etc. already seeded; add others), and start populating real `source_url` citations so `confirmed_official_match` becomes meaningfully distinct from `pipeline_record_match_without_source`. This is manual curation work, same pattern as `KNOWN_COMPOUNDS` — no shortcut, but high leverage for trustworthiness.

**Phase 4 — Add a sponsor-centric data model and views.**
Introduce a sponsor-level rollup (parallel to the existing drug-level `resolved_drugs_df`) — one row per sponsor, listing their drug(s), highest phase, and trial activity. This is the natural home for "competitor leaderboard" / "who's moving in Amyloid this quarter" style views, and for benchmarking AriBio/AR1001 against named competitors specifically.

**Phase 5 — Data freshness and change-tracking.**
Not a code architecture change so much as a process one: version `trials.csv` snapshots (e.g. by date-stamped filename or a small ingestion log), and add a "what's new since last refresh" diff view — new trials, phase advancements, status changes for tracked competitors. This is likely the single highest-value feature for an actual competitive-intelligence use case, but depends on Phases 0–2 being solid first (no point diffing an unreliable classification).

**Phase 6 — Rebrand/reframe the UI copy and IA around competitive intelligence.**
Once the above is real, revisit page copy ("AD Pipeline Dashboard" → something AriBio-specific), add a competitor watchlist, and consider whether the AR1001 spotlight card should generalize into a "your drug vs. the field" comparison view.

Each phase should get its own checkpoint with tests and a before/after data audit, matching how every prior change in this project has been done — no phase should be started until the previous one is reviewed and approved.
