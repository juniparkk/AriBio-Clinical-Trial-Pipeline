# NIH Reference Dataset Integration Plan (Phase 1B audit)

**Status:** audit only. No dashboard behavior, classification logic, or output file changed. `git` checkpoint: `f315b9e` ("Phase 1A: close intervention-scope gaps in resolved_drugs_df").

This document profiles `nih_data.csv`, reconciles it against `resolved_drugs_df` (read from the already-generated `pipeline_drugs.csv`, not by re-running `pipeline_viz.py`), and proposes a phased integration approach. Nothing described here as "Phase 1C" or later has been implemented.

---

## 0. What `nih_data.csv` actually is

It is a curated, published "Alzheimer's Disease Drug Development Pipeline"-style report export — the kind produced annually from CADRO (Common Alzheimer's Disease Research Ontology, the NIA's standard mechanism/target classification system for AD drug development). It is:

- **A snapshot, not a live feed.** No version/date field exists anywhere in the file (see §1). It represents whatever the report's cutoff date was — unknown from the file alone.
- **Curated and selective, not exhaustive.** It does not claim to cover every ClinicalTrials.gov AD trial — only agents the report's authors chose to include (typically named, sponsor-attributed programs at Phase 1–3 with some public visibility).
- **Not a raw trial registry.** One row per *agent*, not per trial — a single agent can (and does) roll up several trials' NCT IDs/sponsors/dates into one row.

**Per the explicit instruction for this audit: a dashboard drug NOT found in NIH is not treated as an error, and NIH is never used to justify removing a historical/discontinued drug from the dashboard.**

---

## 1. Dataset profile

Full field-by-field profile: `outputs/nih_dataset_profile.csv` (11 fields, one row each). Summary:

| Property | Value |
|---|---|
| Total agent-entry rows | 165 |
| Rows by phase | Phase 2: 84 · Phase 1: 45 · Phase 3: 36 |
| Row granularity | One row per **Agent** (not per trial, not per NCT ID) |
| Duplicate canonical names (same drug appears in >1 row) | 7 — `Etalanetug (E2814) + Lecanemab (BAN2401)`, `levetiracetam`, `Semaglutide`, `Trontinemab`, `BIIB080`, `Cannabidiol`, `MIB-626` — same compound studied in more than one program/phase, each getting its own row |
| Rows bundling >1 trial (multiple NCT IDs in one cell) | 21 |
| Rows that are combination agents (`"+"`-joined) | 16 |
| A date/version field for the dataset itself | **None** — no field says when this report was generated or last updated |
| Missing values | Essentially none in the 8 raw columns for populated rows — the sparse fields are structural (see below), not blank cells |

**Format quirk worth flagging:** the file is not a normal single-header CSV. It's three stacked report sections (`Phase 3`, `Phase 2`, `Phase 1`), each with its own repeated header row and (for the Phase 2/1 boundary) blank spacer rows. `nih_reference.parse_nih_dataset()` handles this with Python's `csv` module (which correctly follows multi-line quoted cells — several fields pack one line per trial inside a single cell) rather than `pandas.read_csv()`, which would either choke on the repeated headers or misparse the embedded newlines.

### Which fields support which purpose

| Purpose | Field(s) | Reliability |
|---|---|---|
| Canonical name | `Agent` | High for single agents. Combination agents (16 rows) keep the full `"A + B"` text — Phase 1B does **not** invent a single canonical name for these, consistent with how `classify_pipeline_scope()` already treats ct.gov `COMBINATION_PRODUCT` entries |
| Alias / development code | Parenthetical codes inside `Agent`, e.g. `Zervimesine (CT1812)` → alias `CT1812` | High when present, but only ~15% of rows have one — most agents are listed only by their public/generic name |
| Modality (small molecule vs. biologic) | `Therapeutic purpose`, **DTT rows only** | High for DTT rows, **absent for STT rows** — STT's second segment is a symptomatic *category* (e.g. "cognition enhancer"), not a modality. Treating it as modality would be wrong; `nih_reference.py` returns `""` for STT rows rather than guessing |
| Mechanism (narrative) | `Mechanism of action` | Free text, not structured — reference/display only, not comparable programmatically |
| Target / pathway | `CADRO` | High, and **finer-grained than the dashboard's own 7-bucket `target` field** — CADRO has 13 distinct categories in this file (Amyloid-beta, Tau, Inflammation, Neurotransmitter receptors, Synaptic plasticity/neuroprotection, Metabolism and bioenergetics, Proteostasis/proteinopathies, Growth factors and hormones, Oxidative stress, Circadian rhythm, Vasculature, Gut-brain axis, Epigenetic regulators, Cell death, APOE/lipids/lipoprotein receptors, Multi-target, Undisclosed, Neurogenesis). Several have **no dashboard equivalent at all** — see §5 |
| Company / sponsor | `Lead sponsor` | Present for every row, but is the **trial's** lead sponsor (often an academic medical center running an investigator-initiated study), not necessarily the drug's originating/owning company — see §4 authority notes |
| Phase | Section membership (`Phase 3`/`2`/`1`) | A single current-report snapshot value per row, not a phase *history* — the same compound can (and does) appear in more than one section if it has programs at different phases |
| Trial linkage | `Clinical trial` (NCT IDs) | High — real NCT IDs, one or more per row |

---

## 2. Matching methodology

`nih_reference.match_nih_row_to_dashboard()` tries, per NIH row, every candidate name (canonical name, then its own parenthetical aliases, then — for combination agents — each component's name and alias) against `resolved_drugs_df["display_name"]`, in strict priority order:

1. **`exact_canonical`** — the raw `Agent` text is byte-identical to a dashboard `display_name`
2. **`exact_alias`** — a raw alias/component candidate is byte-identical to a dashboard `display_name`
3. **`normalized_exact`** — case/punctuation-insensitive (`normalize_text()`, the same function `drug_classification.py` uses everywhere else in this project) equality
4. **`fuzzy_suggestion`** — `difflib` similarity ≥ 0.85 on normalized text. **Never auto-accepted** — always surfaced for manual review, never merged automatically
5. **`unmatched`** — nothing close enough found

## 3. Match results

Full per-record detail: `outputs/nih_match_audit.csv` (830 rows: 165 NIH-side + 665 dashboard-side).

**NIH → dashboard (165 NIH agent rows):**

| Tier | Count | % |
|---|---|---|
| `exact_canonical` | 72 | 43.6% |
| `normalized_exact` | 30 | 18.2% |
| `exact_alias` | 11 | 6.7% |
| **Confident match subtotal** | **113** | **68.5%** |
| `fuzzy_suggestion` (needs human confirmation) | 4 | 2.4% |
| `unmatched` | 48 | 29.1% |

The 4 fuzzy suggestions are genuinely informative, not noise — e.g. dashboard drug **"Plasma exchenge"** (a real typo present in the raw ClinicalTrials.gov intervention text) fuzzy-matched to NIH's correctly-spelled **"Plasma exchange"**; **"AV-1959R"** (NIH) vs. dashboard's **"AV-1959D"** was correctly flagged as *not* auto-matched (same family, different suffix — genuinely ambiguous, deserves a human look, not an algorithmic guess).

**Dashboard → NIH (665 dashboard drugs with no confident NIH match), bucketed:**

| Bucket | Count | Meaning |
|---|---|---|
| `current_missing_from_nih` | 548 | Live/active-status therapeutic drugs NIH's curated list simply doesn't cover — **expected**, not an error (NIH is one curated report, not the whole of ClinicalTrials.gov) |
| `historical_or_discontinued` | 91 | `status_summary == "Discontinued"` — **not** flagged as a problem; NIH is a current-pipeline snapshot and isn't expected to carry discontinued programs |
| `non_therapeutic_or_ambiguous` | 23 | `pipeline_scope != "Therapeutic Drug"` (Phase 1A) — never expected to appear in a therapeutic-drug-only report to begin with |
| `unresolved_naming_alias_issue` | 3 | Only ever reached `fuzzy_suggestion` — a real candidate exists, needs a human to confirm |

---

## 4. What NIH is (and is not) authoritative for

Per the explicit instruction for this audit:

| Field | Authoritative / supporting? | Why |
|---|---|---|
| Modality (small molecule vs. biologic), DTT rows | **Supporting, strong** | See §5 — surfaced 25 real dashboard `drug_type` mislabels (monoclonal antibodies tagged "Small Molecule" by the legacy `guess_drug_type()` heuristic) |
| Target / pathway (CADRO) | **Supporting, strong** | See §5 — surfaced that the dashboard's legacy `guess_target()` over-uses "Other" where CADRO has a specific, defensible category |
| Canonical scientific name / alias | **Supporting** | Useful cross-reference, not a forced rename |
| Whether a historical/discontinued drug should stay in the dashboard | **Not authoritative** | NIH's absence of a drug proves nothing about whether it belongs — it's a curated current-pipeline snapshot, not a historical registry |
| Current FDA status | **Not authoritative** | Not even a field in this file |
| Current trial status (Recruiting/Completed/etc.) | **Not authoritative** | Not a field in this file either — only start/primary-completion *dates*, and only for the trials NIH happened to list |
| Company/sponsor ownership | **Not authoritative** | `Lead sponsor` is the *trial's* registered sponsor (frequently an academic site, not the drug's originating company) — see the Donanemab example in §5, sponsored in NIH's row by "Banner Health" (an academic/clinical site running an NIH-funded secondary-prevention trial), not Eli Lilly |

---

## 5. Conflict findings (matched pairs only)

Full per-pair detail: `outputs/nih_conflict_audit.csv` (113 rows — the confidently-matched pairs only; fuzzy suggestions are excluded, since there's no confirmed pair yet to compare).

| Conflict type | Count | % of matched pairs |
|---|---|---|
| `target_conflict` | 53 | 46.9% |
| `canonical_name_differs` (cosmetic only — case/punctuation, already resolved by normalization) | 30 | 26.5% |
| `drug_type_conflict` | 25 | 22.1% |
| `phase_conflict` | 25 | 22.1% |
| `company_conflict` | 19 | 16.8% |

**None of these assert NIH is "right" and the dashboard is "wrong."** They are disagreements surfaced for human review. That said, three of them point at a real, pre-existing, already-documented dashboard gap (`MIGRATION_PLAN.md` Phase 2 — `drug_type`/`target` re-derivation):

- **`drug_type_conflict` (25 cases)** — every single one is the dashboard calling something "Small Molecule" that NIH lists as `DTT; biologic` — e.g. **Remternetug, Semaglutide, BMS-986446, Sargramostim, SHR-1707, XPro1595, ALN-5288** are all real monoclonal antibodies / biologics mislabeled by the legacy `guess_drug_type()` heuristic. This is concrete, external, independently-sourced evidence for the Phase 2 gap already flagged in `MIGRATION_PLAN.md` §2.
- **`target_conflict` (53 cases)** — the dominant pattern is dashboard `target == "Other"` where NIH's CADRO clearly says `Inflammation`, `Amyloid-beta`, `Tau`, or `Metabolism and bioenergetics` (e.g. Aldesleukin, Baricitinib, Senicapoc, Contraloid acetate, RO7269162, Choline). This is the same pre-existing "Other = catch-all" gap `MIGRATION_PLAN.md` documented — again, now with concrete, externally-sourced examples. (Neuropsychiatric-category disagreements were deliberately excluded from this count — see `infer_nih_target()`, which prefers NIH's own `purpose_detail` "neuropsychiatric (...)" subcategory over the coarser CADRO→Symptomatic mapping specifically to avoid manufacturing a false conflict for drugs the dashboard already labels correctly, e.g. KarXT, ACP-204, Escitalopram.)
- **`phase_conflict` (25 cases)** — almost entirely the dashboard's `phase_reached` (highest phase across *all* CT.gov trials) exceeding the single NIH section a drug happens to be listed under (e.g. dashboard Phase 3 vs. NIH's Phase 2 section for Semaglutide, Trontinemab, KarXT). **Expected, not a data-quality problem** — NIH lists an agent once per program; the dashboard tracks every trial.
- **`company_conflict` (19 cases)** — mostly real multi-stakeholder situations (an academic site running a trial of a pharma-owned compound — e.g. dashboard's Donanemab sponsor list includes Eli Lilly *and* Paul S. Aisen; NIH's row for the same trial lists "Banner Health"), not a factual error either source should be corrected against.

---

## 6. Recommended integration approach (proposed — not implemented)

**Phase 1C (small, low-risk):** Wire `nih_reference.py`'s matching into a new, clearly-labeled **supplementary** column set on `resolved_drugs_df` — e.g. `nih_match_tier`, `nih_agent_name`, `nih_cadro`, `nih_modality` — surfaced in the row-detail panel only, never overwriting `drug_type`/`target`/`sponsor`. Zero risk to existing behavior; makes the cross-reference visible to a human reviewer without asserting authority.

**Phase 2 (the existing, already-planned `drug_type`/`target` re-derivation):** Use NIH's `CADRO` + DTT `purpose_detail` as one of the *inputs* (alongside the verified `developed_drug` name and `KNOWN_COMPOUNDS`) when re-deriving `drug_type`/`target` from the verified drug rather than raw trial text — the conflict audit above gives 25 + 53 concrete, pre-vetted correction candidates to seed that work, not a blind reclassification.

**Not recommended:** Using NIH match/non-match as a completeness or correctness signal for which dashboard drugs to keep, drop, or flag as "wrong" — per §4, it isn't built or intended for that.

**Curated overrides, if any NIH-driven correction is ever applied:** route through the existing `data/reference/intervention_scope_overrides.csv` mechanism (Phase 1A) rather than a new ad hoc file — same reviewer/source/verified_date audit trail, one place to look.

---

## 7. Files produced by this audit

| File | Rows | Purpose |
|---|---|---|
| `outputs/nih_dataset_profile.csv` | 11 | Field-by-field profile of `nih_data.csv` |
| `outputs/nih_match_audit.csv` | 830 | Two-sided reconciliation: 165 NIH rows + 665 unmatched dashboard rows |
| `outputs/nih_conflict_audit.csv` | 113 | Field-level conflict flags for every confidently-matched pair |
| `nih_reference.py` | — | Pure, side-effect-free module (parsing/profiling/matching/conflict logic) — same design pattern as `drug_classification.py` |
| `run_nih_audit.py` | — | Standalone runner — reads `nih_data.csv` + the already-generated `pipeline_drugs.csv`, writes the 3 CSVs above. Does **not** import or run `pipeline_viz.py` |
| `test_nih_reference.py` | — | Unit tests for every pure function above |
