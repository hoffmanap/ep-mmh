# El Paso Missing Middle Housing Analysis

Adapted from the methodology of Salazar-Miranda & Talen,
*"Mapping the Missing Middle: AI-Based Detection of Mid-Density Housing"*
(replication package: `github.com/ariannasm/missing-middle-housing`).

Live dashboard: `index.html` (open directly, or via GitHub Pages) at: 

## https://hoffmanap.github.io/ep-mmh/

## 1. The problem

Missing-middle housing — duplexes, triplexes, courtyard apartments, small
multi-unit buildings mixed into single-family blocks — almost certainly
exists in greater quantity in El Paso than official county records show,
because in most of the city it is **not legal to build**: single-family
zoning (`R3A`, `R3`, `R2`, etc.) covers the overwhelming majority of
residential land, so any informal second unit, converted garage, or
backyard casita has every incentive to stay off the books. A property
owner who adds a unit doesn't file a permit for something zoning
forbids. That means the central appraisal district's own classification
— built from permitted, assessed improvements — is a **floor**, not a
ceiling, on how much missing-middle housing actually exists. This
project tries to size that gap from several independent angles rather
than taking any single source's count at face value.

## 2. Baseline data — how EPCAD classifies housing, and where it can't see

El Paso Central Appraisal District (EPCAD) already publishes an
authoritative, deterministic classification per parcel — no model
needed for the baseline:

- **Primary classification** uses the GIS parcel layer's `STATE_CD`
  field (one row per parcel): `A1`=single-family, `A2`=mobile home,
  `A3`=condominium, `A4`=townhouse, `B1`=duplex, `B2`=apartment/
  multifamily, `B3`-`B6`=other multi-unit residential subtypes.
- **Secondary enrichment** sub-splits generic `B2` parcels into
  apartment-size bands (4-6 / 7-30 / 31-99 / 100+ units) using the
  `Improvements.ClassCD` field's `QS`/`QL`/`QN`/`QR` prefixes, joined
  via `Improvements.Property_dbId == Properties.PropertyId` directly
  (see "Known data bugs" below — the documented join key was wrong).
- **Zoning** comes from the GIS layer's own `ZONING` field.

Classified this way: **252,626** residential parcels (includes the
STATE_CD-gap fix below), of which **8,883** parcels are labeled
missing-middle by EPCAD's own code — **3.52%** of parcels. But that's a
parcel count, and parcels aren't units — one apartment building is a
single parcel holding 50+ units. Converted to a unit-weighted basis
(each parcel's typology's recorded/typical unit count, summed): EPCAD's
own records imply **67,050 of 323,627 residential units (20.72%)** are
missing-middle — a very different number from the 3.52% parcel-count
headline, and the first sign that parcel-level and unit-level views of
"how much missing middle exists" tell very different stories. See
section 5 for how this compares to the Census's independent count.

### Where missing middle exists outside EPCAD's own classification

Three signals live inside EPCAD's own data but outside its `STATE_CD`
label — direct evidence of missing-middle housing EPCAD's primary
classification doesn't catch:

- **Legal description (`LEGAL_DESC`) free-text scan**: 148 parcels
  classified `A1` (single-family) mention "CASITA" and 362 mention
  "CONDO" in their legal description — housing EPCAD's own structural
  code doesn't reflect, found independent of any model.
- **Homestead exemption (`EXEMPTIONS` contains `HS`)** — Texas's legal
  requirement that the exemption-holder occupy the property as their
  primary residence, a current, authoritative owner-occupancy signal.
  **75.9%** of single-family parcels claim it, down to **2.4-3.4%**
  for duplexes/multi-unit residential, and **0%** for apartments —
  exactly the pattern expected if non-owner-occupied rental housing
  (often informal, often missing-middle) clusters outside the
  single-family label.
- **Mailing-address mismatch** (owner's mailing city/state differs
  from the parcel's situs city, or isn't in Texas) — a weaker,
  secondary absentee-ownership signal (r = +0.028 with missing-middle
  status).

## 3. The model — learning what missing middle looks like structurally

A `HistGradientBoostingClassifier` was trained **only** on EPCAD's
confirmed missing-middle parcels and confirmed single-family parcels,
using structural features alone — parcel size, number of distinct main
dwelling structures, total square footage, year built, distance to
downtown — with `STATE_CD`/`ClassCD` (the fields that *define* the
label) explicitly excluded from the model's inputs. The point is to
learn what missing-middle housing looks like structurally, then use
that signature to flag single-family-labeled parcels EPCAD's own
classification might have missed — informal/undocumented missing
middle, which is exactly the population the problem statement above
says should exist.

- **Test AUC: 0.986** (trained on 179,493 parcels, held out 59,831)
- The strongest tells, in order: **distance to downtown core**,
  **parcel size (acres)**, **year built**, **number of main
  structures on the parcel**, and **total building square footage**
- Scored against every EPCAD-labeled single-family parcel, the model
  flags **7,798** candidates (after removing 111 parcels with
  implausible structural values) the county's own classification may
  have missed, of which **2,570** score at ≥80% confidence
- These appear on the dashboard map as hollow rust-colored markers,
  distinct from confirmed missing-middle parcels

**The descriptive statistics behind why this works** — the
correlations and non-linear patterns the model is implicitly learning:

Point-biserial/Pearson correlations between structural features and
`is_missing_middle`, computed on 218,009 residential parcels
(bedrooms/bathrooms/building sqft pulled from the First American join,
0.02-mile nearest-neighbor match):

| feature | r | direction |
|---|---|---|
| has_homestead | -0.274 | owner-occupied strongly predicts *not* missing-middle |
| land_acres | +0.197 | larger lots lean missing-middle (see U-shape below) |
| dist_to_core_mi | -0.151 | closer to downtown → more missing-middle |
| bedrooms | -0.073 | fewer bedrooms → more missing-middle |
| building_sqft | +0.068 | weak linear read — see U-shape below |
| year_built | -0.069 | older stock → more missing-middle |
| bathrooms | -0.039 | weak |
| mailing_mismatch | +0.028 | weak |

**The linear correlations understate the real pattern.** Binning lot
size and building size reveals a **U-shape**, not a straight line —
both very small *and* very large lots/buildings over-index on missing
middle, while "typical single-family" mid-range values under-index:

- Lot size: **29.3%** missing-middle share at 0-0.05 acres, falling to
  **1.7%** at 0.15-0.2 acres (the single-family sweet spot), then
  climbing back to **41.0%** at 2+ acres (large multi-family/apartment
  sites).
- Building size: **10.1%** share at 600-900 sqft (small/casita-sized),
  dropping to **2.3%** at 1500-1800 sqft, then rising to **16.7%** at
  3500+ sqft (large apartment buildings).

**Zoning concentration** (missing-middle share *within* each zoning
code, not raw counts, min n=100), using the **official City of El Paso
zoning layer** (`gis.elpasotexas.gov` Planning/Zoning FeatureServer,
spatially joined to parcel centroids — 214,558/400,274 parcels matched;
unmatched parcels are outside city limits/ETJ with no city zoning
record): `C-1` (50.1%), `C-3` (50.0%), `A-2` (33.5%), `S-D` (33.5%),
`A-O` (30.3%) lead. This replaces an earlier pass using EPCAD's raw
GIS `ZONING` field, which had inconsistent duplicate spellings for the
same district (`R-3`, `R3A`, `R 3A` all meaning similar things) —
the official layer's `ZONELABEL` field is clean. Full numbers:
`data/shape_and_signal_stats.json`. Reproduce with
`scripts/correlations_analysis.py` and
`scripts/join_tracts_and_compare.py` (requires the First American
extract — see Financing section below).

## 4. Validation, layer 1 — building footprints (external, geometric)

The original Salazar-Miranda & Talen paper detects housing typology
from a street-view CV model's read of building *shape*. El Paso has no
imagery pipeline, but real building footprint polygons (Microsoft's
[GlobalMLBuildingFootprints](https://github.com/microsoft/GlobalMLBuildingFootprints),
~384K buildings across El Paso County) let us compute actual shape
features and cross-check EPCAD's own structure counts against physical
reality — an independent, geometric check on whether more housing
exists on a parcel than EPCAD has on record.

`scripts/shape_features.py` spatially joins footprint centroids into
parcels and derives two signals distinguishing **a brand-new undetected
detached structure** from **an addition to an existing one**:

- **`new_detached_flag`**: the parcel has more dwelling-sized
  (≥300 sqft, to exclude sheds/carports) footprints than EPCAD has
  recorded main structures — a candidate undocumented ADU/casita.
- **`addition_flag`**: structure count matches the record, but total
  footprint area exceeds recorded square footage by more than a
  **1.5x** ratio — a candidate undocumented expansion. (This threshold
  was calibrated empirically against the population baseline — see
  commit history; an earlier 1.25x cutoff mostly caught ordinary
  garage/patio coverage, not real additions.)

Of 202,333 single-family (`A1`) parcels with footprint coverage,
**12,955** trip `new_detached_flag` and **4,564** trip `addition_flag`
(**17,519** total, ~8.7% of single-family stock) — a geometric
corroboration, independent of the structural classifier, that
undocumented housing on "single-family" parcels is a real and sizeable
phenomenon, though **not** a confirmed count on its own (footprint
polygons are noisy — see "Known caveats").

**Does the shape signal hold up across typology, not just
single-family?** Breaking the same footprint features out by typology
(`data/shape_by_typology.json`) — using medians, not means, since a
few parcels with implausibly large ratios skew the mean badly for the
smaller attached-housing categories:

| typology | n with footprint data | median footprint-to-recorded ratio | addition flag rate | new-detached flag rate |
|---|---|---|---|---|
| Single-Family Detached | 202,332 | 1.10 | 2.3% | 6.4% |
| Duplex | 3,271 | 1.18 | **25.0%** | 4.0% |
| Mobile/Manufactured Home | 2,472 | 1.01 | 5.9% | 20.5% |
| Multi-Unit Residential (Other) | 1,079 | 1.08 | 13.3% | 1.9% |
| Townhouse | 471 | 1.47 | 44.2% | 6.8% |
| Condominium | 49 | 3.26 | 49.0% | 16.3% |
| Apartment (7-30 units) | 735 | 0.47 | 1.4% | 6.3% |
| Apartment (31-99 units) | 215 | 0.34 | 0.5% | 21.9% |
| Apartment (100+ units) | 217 | 0.32 | 0.0% | **41.9%** |

Two real findings and one data-quality catch:

- **Duplexes show a genuinely elevated addition-flag rate (25% vs.
  2.3% for single-family) without the extreme outliers that distort
  the attached-housing rows below** — the 90th-percentile ratio for
  duplexes is a modest 2.36, not the 8-25x seen for condos/townhouses.
  That's a real structural signature: duplex parcels routinely carry
  more footprint area than what's recorded, independent of the
  structural classifier or the EPCAD typology code itself.
- **Large apartment complexes (31-99 and 100+ units) show high
  new-detached-flag rates (22%/42%)** — expected and not itself a
  "shadow housing" finding: big complexes legitimately have many
  separate buildings on one parcel, and EPCAD's "main structure" count
  per parcel doesn't always keep pace with how many buildings a large
  complex actually has.
- **The footprint-ratio signal breaks down for townhouses and condos**
  (median ratio 1.47/3.26, but 90th-percentile 4.4x/25.6x) — not
  because these are riddled with real additions, but because the
  footprint-centroid spatial join sometimes assigns a *shared*
  building's full footprint to an individual unit's small parcel
  (several attached units, one footprint polygon, each unit's own
  recorded square footage is a fraction of the whole building). The
  1.5x `addition_flag` threshold was calibrated against the
  single-family population baseline (see above) and was never meant
  to apply to attached/shared-structure typologies — their
  addition-flag rates (44%/49%) should be read as spatial-join noise,
  not confirmed undocumented expansions.

**Bottom line:** the footprint-shape validation layer is reliable for
single-family and mobile-home parcels (where it was calibrated) and
surfaces one genuine cross-typology signal — duplexes running hot on
footprint-vs-recorded ratio — but it is not a reliable check for
attached/shared-structure typologies (townhouse, condo) without
unit-splitting the shared footprints first, which this project doesn't
currently do.

## 5. Validation, layer 2 — ACS units-in-structure (external, survey-based)

The Census Bureau's American Community Survey asks households directly
how many units are in their building (table B25024), independent of any
local permitting or assessment process — a third benchmark alongside
EPCAD's parcel records and the footprint cross-check. This answers the
question directly: **whose count is closer to reality, ours (EPCAD-
based) or the Census's household-reported figure?**

Pulled via Census Reporter's open API (2020-2024 ACS 5-year, El Paso
County, all 188 tracts — `scripts/fetch_acs_units_in_structure.py`),
each parcel matched to its tract via a spatial join against TIGERweb
tract boundaries (`scripts/join_tracts_and_compare.py`, 400,183/400,274
parcels matched, 182/188 tracts had both EPCAD and ACS data to compare):

**Countywide:**

| source | missing-middle units | total units | share |
|---|---|---|---|
| EPCAD, parcel count | 8,883 parcels | 252,626 parcels | 3.52% (of *parcels*) |
| EPCAD, unit-weighted (exact) | 67,050 | 323,627 | **20.72%** |
| ACS B25024 (2+ units in structure), 182 tracts matched to EPCAD | 73,463 | 318,700 | **23.05%** |

Once EPCAD's own records are converted to the same unit basis ACS
uses, the headline 3.52%-vs-23% gap mostly disappears — it was largely
a unit-of-analysis artifact (parcels vs. units), not undercounting.
What's left is a real but modest **2.3-point gap**, with ACS running
*higher* than EPCAD — consistent with some degree of actual
undercounting (informal/undocumented units a self-reported survey
would catch that a parcel classification wouldn't), just far smaller
than the raw parcel-count headline implied.

**Tract-level tells a sharper story than the countywide average**:
mean absolute difference between EPCAD and ACS is **7.16 percentage
points per tract** — real local disagreement hiding behind a close
countywide match, because some tracts run EPCAD-high and others run
EPCAD-low, and they roughly cancel out countywide. The biggest gaps:
tract 48141001700 (EPCAD 53.0% vs. ACS 94.3%, a 41.3-point EPCAD
undercount), tract 48141003402 (EPCAD 17.8% vs. ACS 55.6%, 37.8
points), tract 48141002900 (EPCAD 32.6% vs. ACS 64.4%, 31.8 points).

**Adding the structural model's high-confidence candidates does not
close the gap** — a genuinely useful negative result. Treating each
≥80%-confidence candidate as one extra undocumented unit moves the
mean absolute difference from 7.16 to **7.28** (slightly worse), and
only improves the match in 57 of 182 tracts (worse than a coin flip).
**So whose count is more accurate?** For the countywide total, EPCAD's
unit-weighted figure (20.72%) and ACS (23.05%) are close enough that
neither should be treated as definitively right — they're measuring
related but not identical things (EPCAD: what's been assessed and
classified; ACS: what households self-report, survey-sampled, with its
own margins of error at the tract level). What the comparison does show
clearly is that the discrepancy is **not primarily "undocumented units
the structural model can find"** — if it were, model-adjusted EPCAD
numbers would track ACS more closely than raw EPCAD numbers, and they
don't. The gap is more likely driven by differences in what counts as
a "unit" per typology, EPCAD assessment lag, or ACS sampling noise at
the tract level (several of the biggest-gap tracts have small total
unit counts, e.g. 419-621 units, where ACS's margin of error is
largest).

Full tract-level numbers: `data/tract_comparison_epcad_acs.json`.

**A scope caveat that matters: "missing middle" above is broader than
the term usually means.** EPCAD's `is_missing_middle` flag — inherited
from the STATE_CD classification (B1–B6) — marks *any* non-single-family
residential structure, including large apartment complexes. Of the
67,050 unit-weighted "missing middle" units above, 39,695 (59%) are from
Apartment (31-99 units) and Apartment (100+ units) buildings — not
missing middle in the planning-literature sense (duplex through small
courtyard apartment, roughly 2-9 units), just large multifamily. That
broad scope happens to match how the ACS comparison above was scoped
(B25024 categories 004-009, i.e. any 2+-unit structure), so the
20.72%-vs-23.05% comparison is still apples-to-apples — it's just
answering "how much non-single-family housing is there," not "how much
*missing middle*, specifically, is there."

Re-running the same comparison scoped to 2-9 unit structures only
(`scripts/narrow_mm_comparison.py`, excluding the two apartment-tower
typologies) changes the picture:

| source | narrow (2-9 unit) missing-middle units | total units | share |
|---|---|---|---|
| EPCAD, unit-weighted | 29,601 | 323,575 | **9.15%** |
| EPCAD + model candidates | 32,222 | 326,196 | 9.88% |
| ACS B25024 (2-9 units in structure) | 39,525 | 318,700 | **12.40%** |

Under this narrower, more correct definition, EPCAD is further from
ACS, not closer: a **3.25-point countywide gap**, with EPCAD capturing
only about 75% of what ACS implies exists — a real undercount, not a
unit-of-analysis artifact. The tract-level mean absolute difference is
**8.77 points** (vs. 7.16 under the broad definition). And the same
negative result holds here too: adding the model's high-confidence
candidates barely moves the countywide share (9.15% → 9.88%, still well
short of ACS's 12.40%) and the mean absolute difference per tract gets
*slightly worse* (8.77 → 8.87), improving only 84 of 182 tracts (46%,
still under half). The model's candidate count (2,621 parcels
countywide) simply isn't large enough, relative to the ~10,000-unit gap
implied by ACS, to move the needle — whatever is driving the
EPCAD/ACS gap at the 2-9-unit scale, it isn't something a footprint-
shape anomaly detector of this size can close.

Full tract-level numbers: `data/narrow_mm_comparison_summary.json`
(checked in); per-tract detail CSV regenerates via
`scripts/narrow_mm_comparison.py`.

## Ownership and financing

El Paso County's First American property/ownership/lien extract
(`Prop48141.txt`, FIPS 48141, 397,150 records, snapshot 2020-04-09)
was spatially joined to the EPCAD parcel set via nearest-neighbor
matching on situs lat/lon. 86.8% of residential parcels matched within
100 feet.

- **Investor ownership is sharply higher for missing-middle types.**
  Duplexes are **73.9%** investor-owned (non-owner-occupant) vs.
  **16.4%** for single-family detached homes; 4-30-unit apartments
  run 92-98% investor-owned.
- Active-lien prevalence and median AVM (automated valuation) by
  typology are also broken out, along with a small (~11,700-record)
  fixed/adjustable/variable financing-type split.
- Caveat: this is a single 2020 snapshot, not current.

## Known data bugs found and fixed

Two bugs silently corrupted the first version of this analysis:

1. **Silent CSV-parsing failure.** The EPCAD flat files contain
   embedded NULL bytes and stray quote characters. Reading them with
   `engine="python", on_bad_lines="skip"` caused pandas to silently
   drop *every row* with no visible error. Fixed by switching to
   `engine="c", quoting=3` (`csv.QUOTE_NONE`).
2. **Wrong join key.** The documented schema implies
   `Improvements.Property_dbId == Properties.dbId`, requiring a
   two-hop join. Real data shows `Improvements.Property_dbId` equals
   `Properties.PropertyId` **directly** (99.95% overlap) — fixing this
   resolved the apartment-size sub-split almost completely and raised
   the candidate-detector's test AUC from ~0.93 to 0.986.
3. **Incomplete `STATE_CD` typology map.** Several codes with real
   volume (`A6`, `A51`, `A53`-`A55`, `A7`, `C10`, `D5`, `D6`) were
   missing from the typology dictionary, silently dropping ~10,640
   parcels from the residential denominator. Checked against EPCAD's
   official `2026_Codes for Export Tables_All.xlsx` data dictionary —
   none of these local extensions are defined there either, so their
   typology assignment remains inference from acreage/legal-
   description patterns, flagged as such in `build_dataset.py`.

## Setup

1. Download EPCAD's flat-file exports from `epcad.org/OpenGovernment`
   → "Property Search Dataset Downloads":
   `Properties2027Dump.zip`, `Improvements2027Dump.zip` → unzip into
   `data/raw/`
2. Download the EPCAD parcel GIS layer (same page, "GIS Data" section,
   `parcels2026.zip`) → unzip into `data/gis/` (it's a File
   Geodatabase — `Parcels2026/New File Geodatabase.gdb`)
3. Install dependencies:
   ```
   pip install pandas pyogrio numpy geopandas scipy scikit-learn
   ```

## Running the pipeline

```
python scripts/build_dataset.py              # classify parcels -> data/output/parcels_analysis.csv
python scripts/make_dashboard_data.py        # aggregate -> data/dashboard_stats.json, data/parcels_missing_middle.geojson
python scripts/train_mm_detector.py          # the model (section 3)
python scripts/correlations_analysis.py      # correlations/sweet-spots/zoning
python scripts/load_footprints.py            # building footprints (section 4)
python scripts/shape_features.py
python scripts/fetch_acs_units_in_structure.py  # ACS comparison (section 5)
python scripts/join_tracts_and_compare.py       # official zoning + tract-level EPCAD/model/ACS comparison (broad definition)
python scripts/narrow_mm_comparison.py          # same comparison re-scoped to 2-9 unit structures (section 5)
```

Then open `index.html` — it reads the generated JSON/GeoJSON files
directly.

## Structure

```
ep-mmh/
├── config.py                       # downtown anchor point, typology color tokens
├── index.html                      # Field Ledger-styled Leaflet + Chart.js dashboard
├── scripts/
│   ├── build_dataset.py             # load GIS + EPCAD flat files, classify typology, compute distance
│   ├── make_dashboard_data.py       # aggregate into dashboard JSON/GeoJSON
│   ├── train_mm_detector.py         # structural-feature-only classifier + candidate scoring (section 3)
│   ├── correlations_analysis.py     # correlations, size sweet-spots, zoning concentration
│   ├── load_footprints.py           # load Microsoft building-footprint tiles (section 4)
│   ├── shape_features.py            # spatial join footprints <-> parcels; shadow-structure signals
│   ├── fetch_acs_units_in_structure.py  # ACS B25024 comparison (section 5)
│   ├── join_tracts_and_compare.py   # official zoning join + tract-level EPCAD/model/ACS comparison
│   ├── join_first_american.py       # spatial join: First American mortgage data <-> EPCAD parcels
│   └── make_mortgage_dashboard_data.py  # aggregate financing stats for the dashboard
├── data/
│   ├── raw/                         # EPCAD flat-file exports (not checked in — see Setup)
│   ├── gis/                         # EPCAD parcel geodatabase + footprint tiles + zoning/tract boundaries (not checked in — see Setup)
│   ├── output/                      # generated intermediate CSVs (not checked in, large)
│   ├── dashboard_stats.json         # generated: aggregated stats for the dashboard (checked in)
│   ├── shape_and_signal_stats.json  # generated: correlations/sweet-spots/zoning/shape-feature stats (checked in)
│   ├── shape_by_typology.json       # generated: footprint-shape features (ratio, flags) broken out by typology (checked in)
│   ├── acs_units_in_structure.json  # generated: ACS B25024 comparison (checked in)
│   ├── tract_comparison_epcad_acs.json  # generated: tract-level EPCAD vs. model vs. ACS comparison, broad definition (checked in)
│   ├── narrow_mm_comparison_summary.json  # generated: same comparison re-scoped to 2-9 unit structures (checked in)
│   ├── tract_comparison_narrow_mm.csv   # generated: tract-level detail for the narrow-definition comparison (CSV, gitignored like other intermediates — regenerate via the script above)
│   ├── parcels_missing_middle.geojson   # generated: confirmed missing-middle parcel points (checked in)
│   ├── candidates_high_confidence.geojson  # generated: model-flagged candidate parcels (checked in)
│   ├── missing_middle_candidates.csv    # generated: full candidate list w/ probabilities (checked in)
│   ├── feature_importance.csv       # generated: classifier permutation importances (checked in)
│   ├── model_metrics.json           # generated: classifier test AUC etc. (checked in)
│   └── mortgage_stats.json          # generated: First American financing aggregates (checked in)
```

## Known caveats

- `B3`-`B6` and the inferred `STATE_CD` extensions above are EPCAD
  local codes with no documented meaning anywhere, including EPCAD's
  own official data dictionary. Their typology assignment is inference
  from acreage/legal-description patterns, flagged as such in code.
- The unit-weighted EPCAD share (20.72%) fills two small typologies
  with no documented typical-unit-count (`Multi-Unit Residential
  (Other)`, generic `Apartment/Multifamily` — 1,437 parcels combined)
  at a conservative 1 unit each, so the true figure is likely a touch
  higher than 20.72%, not lower.
- ACS figures are 5-year survey estimates with sampling error, largest
  in tracts with small total unit counts — several of the biggest
  EPCAD-vs-ACS tract gaps are in tracts under 1,000 units.
- The apartment-size sub-split only covers `B2` parcels whose
  `Improvements.ClassCD` happens to carry a `QS`/`QL`/`QN`/`QR` code.
- Microsoft's building-footprint polygons (section 4) are noisy — one
  roof is sometimes split into multiple polygons by the detection
  model. Read footprint-based flags as a screening signal, not a
  confirmed count.
- The footprint-to-recorded ratio (`addition_flag`) is only reliable
  for single-family and mobile-home parcels. For townhouses and
  condos, a shared building's footprint sometimes gets assigned to one
  attached unit's small individually-recorded parcel, producing
  implausible ratios (seen up to 25.6x at the 90th percentile) — their
  44-49% addition-flag rates in the typology breakdown (section 4) are
  mostly this join artifact, not real undocumented expansions.
- Distance is measured from a single downtown point (San Jacinto
  Plaza), straight-line, not drive-time.
- The headline "missing middle" figures (section 2-3, and the broad
  cut of section 5) count *any* non-single-family residential
  structure, including large (31-99 and 100+ unit) apartment
  buildings — not "missing middle" in the 2-9-unit sense the term
  usually carries in planning literature. Those large-building units
  are 59% of the 67,050 unit-weighted total. Section 5's narrower,
  2-9-unit-only recheck is the one to use if "missing middle"
  specifically (not "non-single-family generally") is what matters
  for your question — and it shows a real, larger undercount (9.15%
  vs. ACS's 12.40%) that the structural model's candidates don't close
  either.
- First American ownership/financing data (Ownership and financing
  section) is a single 2020 snapshot, not current.
