# El Paso Missing Middle Housing Analysis

Adapted from the methodology of Salazar-Miranda & Talen,
*"Mapping the Missing Middle: AI-Based Detection of Mid-Density Housing"*
(replication package: `github.com/ariannasm/missing-middle-housing`).

Live dashboard: `index.html` (open directly, or via GitHub Pages).

## Results (2026 EPCAD data)

- **242,694** residential parcels classified
- **8,883** missing-middle parcels (duplex, townhouse, condo,
  apartment/multifamily 2-99+ units) — **3.66%** of residential stock
- Strong distance-from-core gradient: **~30%** missing-middle share
  within 1 mile of downtown El Paso, falling to **under 1%** beyond
  12 miles
- Zoning cross-tab shows missing-middle stock concentrated in
  `R-5`/`R3A`-type multi-family zoning, as expected

## Candidate detector — what structural features give it away

A `HistGradientBoostingClassifier` was trained to predict missing-
middle status from **structural features alone** — parcel size,
number of distinct main dwelling structures, total square footage,
year built, distance to downtown — with `STATE_CD`/`ClassCD` (the
fields that *define* the label) explicitly excluded from the model's
inputs. This answers a different question than the primary
classification: not "what did EPCAD call this parcel" but "does this
parcel look like missing-middle housing even if nobody labeled it
that way."

- **Test AUC: 0.986** (trained on 179,493 parcels, held out 59,831)
- The strongest tells, in order: **distance to downtown core**,
  **parcel size (acres)**, **year built**, **number of main
  structures on the parcel**, and **total building square footage**
- Scored against every EPCAD-labeled single-family parcel, the model
  flags **7,798** candidates (after removing 111 parcels with
  implausible structural values — likely data-quality edge cases) the
  county's own classification may have missed, of which **2,570**
  score at ≥80% confidence
- These appear on the dashboard map as hollow rust-colored markers,
  distinct from confirmed missing-middle parcels

## Ownership and textual signals already in EPCAD's own data

Beyond `STATE_CD`, the EPCAD GIS parcel layer carries several fields
this project wasn't using:

- **Homestead exemption (`EXEMPTIONS` contains `HS`)** is Texas's
  legal requirement that the exemption-holder occupy the property as
  their primary residence — a current, authoritative owner-occupancy
  signal, cleaner than either a stale 2020 First American flag or a
  mailing-address mismatch. It shows a strong, clean gradient with
  typology: **75.9%** of single-family parcels claim it, down to
  **2.4-3.4%** for duplexes/multi-unit residential, and **0%** for
  apartments — exactly the pattern expected if non-owner-occupied
  rental housing is concentrated in missing-middle types.
- **Legal description (`LEGAL_DESC`) free-text scan**: 148 parcels
  classified `A1` (single-family) mention "CASITA" and 362 mention
  "CONDO" in their legal description — a direct textual signal of
  housing EPCAD's own structural code doesn't reflect, independent of
  any model.
- **Mailing-address mismatch** (owner's mailing city/state differs
  from the parcel's situs city, or isn't in Texas) is a weaker,
  secondary absentee-ownership signal (r = +0.028 with missing-middle
  status — real but far weaker than homestead status).

## Correlations, sweet spots, and zoning concentration

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
code, not raw counts, min n=100): `AO` (73.2%), `C3` (50.0%), `A3H`
(46.3%), `A2` (44.0%), `A-2` (41.7%) lead — note these zoning-code
labels (`A2`, `A3H`, etc.) are a *different* code space from EPCAD's
`STATE_CD` structural codes (`A1`/`A2`/etc. used for typology above)
and happen to share letters by coincidence, not meaning.

Full numbers: `data/shape_and_signal_stats.json`. Reproduce with
`scripts/correlations_analysis.py` (requires the First American
extract — see Financing section below).

## Building-footprint shape features (Salazar-Miranda & Talen adaptation)

The original paper detects housing typology from a street-view CV
model's read of building *shape*. El Paso has no imagery pipeline, but
real building footprint polygons (Microsoft's
[GlobalMLBuildingFootprints](https://github.com/microsoft/GlobalMLBuildingFootprints),
~384K buildings across El Paso County) let us compute actual shape
features and compare them against EPCAD's own structure counts —
phase one of a footprint-first, imagery-later plan.

`scripts/shape_features.py` spatially joins footprint centroids into
parcels and derives, per parcel: footprint count, total/largest
footprint area, compactness (4π·area/perimeter², lower = more
elongated/attached), and two signals distinguishing **a brand-new
undetected detached structure** from **an addition to an existing
one**:

- **`new_detached_flag`**: the parcel has more dwelling-sized
  (≥300 sqft, to exclude sheds/carports) footprints than EPCAD has
  recorded main structures — a candidate undocumented ADU/casita.
- **`addition_flag`**: structure count matches the record, but total
  footprint area exceeds recorded square footage by >25% — a
  candidate undocumented expansion.

Of 202,316 single-family (`A1`) parcels with footprint coverage,
**12,941** trip `new_detached_flag` and **29,229** trip
`addition_flag` (**42,170** total, ~21% of single-family stock).

**Caveat, stated plainly because it matters**: Microsoft's footprint
polygons are noisy — one roof is sometimes split into multiple
polygons by the detection model, and a large garage or carport
sometimes clears the 300 sqft dwelling-size threshold. A flag rate
this high is not credible as "42,000 confirmed undocumented housing
units." Read it as a **noisy screening signal** to prioritize manual
or imagery-based review, not as a finding on its own. Tightening
further (a higher size threshold, compactness filtering to exclude
elongated garage shapes, or cross-referencing against the
legal-description text flags above) would be the natural next step
before trusting individual flagged parcels.

## Financing — First American property/mortgage data

El Paso County's First American property/ownership/lien extract
(`Prop48141.txt`, FIPS 48141, 397,150 records, snapshot 2020-04-09)
was spatially joined to the EPCAD parcel set via nearest-neighbor
matching on situs lat/lon (its `APN` field does not share EPCAD's
`GEO_ID` numbering scheme, so geography — not an ID — is the join
key). 86.8% of residential parcels matched within 100 feet.

This adds an ownership/financing lens EPCAD's own data doesn't carry:

- **Investor ownership is sharply higher for missing-middle types.**
  Duplexes are **73.9%** investor-owned (non-owner-occupant) vs.
  **16.4%** for single-family detached homes; 4-30-unit apartments
  run 92-98% investor-owned, as expected for that housing type.
- Active-lien prevalence and median AVM (automated valuation) by
  typology are also broken out, along with a small (~11,700-record)
  fixed/adjustable/variable financing-type split for missing-middle
  vs. single-family mortgages.
- Caveat: this is a single 2020 snapshot, not current; it describes
  ownership structure at that point in time, not today's market.

## What's different from the original

The original paper detects housing typology with a computer-vision
model run on street-view imagery of Chicago buildings, producing
per-building typology *probabilities*. El Paso has no such imagery
pipeline, but El Paso Central Appraisal District (EPCAD) already
publishes an authoritative, deterministic classification per parcel —
no model needed:

- **Primary classification** uses the GIS parcel layer's `STATE_CD`
  field (one row per parcel): `A1`=single-family, `A2`=mobile home,
  `A3`=condominium, `A4`=townhouse, `B1`=duplex, `B2`=apartment/
  multifamily, `B3`-`B6`=other multi-unit residential subtypes.
- **Secondary enrichment** sub-splits generic `B2` parcels into
  apartment-size bands (4-6 / 7-30 / 31-99 / 100+ units) using the
  `Improvements.ClassCD` field's `QS`/`QL`/`QN`/`QR` prefixes, joined
  via `GIS.PROP_ID -> Properties.PropertyId -> Properties.dbId ->
  Improvements.Property_dbId`.
- **Zoning** comes directly from the GIS layer's own `ZONING` field
  (true zoning, not EPCAD's appraised-use code, which is unpopulated
  in this export).

This adaptation keeps the original paper's downstream analyses:
prevalence by typology, distance-from-core gradient, and
typology-by-zoning composition — just with deterministic ground truth
instead of CV probabilities.

## Known data bugs found and fixed

Two bugs silently corrupted the first version of this analysis:

1. **Silent CSV-parsing failure.** The EPCAD flat files contain
   embedded NULL bytes and stray quote characters. Reading them with
   `engine="python", on_bad_lines="skip"` caused pandas to silently
   drop *every row* with no visible error — the apartment-size
   sub-split and the structural-feature classifier were both
   unknowingly running against empty dataframes. Fixed by switching
   to `engine="c", quoting=3` (`csv.QUOTE_NONE`).
2. **Wrong join key.** The documented schema implies
   `Improvements.Property_dbId == Properties.dbId`, requiring a
   two-hop join through `Properties.PropertyId`. Real data shows
   `Improvements.Property_dbId` equals `Properties.PropertyId`
   **directly** (99.95% overlap) — the documented `dbId` hop was
   wrong and only matched ~11-16% of records. Fixing this resolved
   the apartment-size sub-split almost completely (14 unsplit
   parcels vs. thousands before) and raised the candidate-detector's
   test AUC from ~0.93 to 0.986.

## Setup

1. Download EPCAD's flat-file exports from `epcad.org/OpenGovernment`
   → "Property Search Dataset Downloads":
   `Properties2027Dump.zip`, `Improvements2027Dump.zip` → unzip into
   `data/raw/`
2. Download the EPCAD parcel GIS layer (same page, "GIS Data" section,
   `parcels2026.zip`) → unzip into `data/gis/` (it's a File
   Geodatabase, not a shapefile — `Parcels2026/New File Geodatabase.gdb`)
3. Install dependencies:
   ```
   pip install pandas pyogrio numpy
   ```

## Running the pipeline

```
python scripts/build_dataset.py       # classify parcels -> data/output/parcels_analysis.csv
python scripts/make_dashboard_data.py # aggregate -> data/dashboard_stats.json, data/parcels_missing_middle.geojson
```

Then open `index.html` — it reads those two generated files directly.

## Structure

```
ep-mmh/
├── config.py                       # downtown anchor point, typology color tokens
├── index.html                      # Field Ledger-styled Leaflet + Chart.js dashboard
├── scripts/
│   ├── build_dataset.py             # load GIS + EPCAD flat files, classify typology, compute distance
│   ├── make_dashboard_data.py       # aggregate into dashboard JSON/GeoJSON
│   ├── train_mm_detector.py         # structural-feature-only classifier + candidate scoring
│   ├── join_first_american.py       # spatial join: First American mortgage data <-> EPCAD parcels
│   ├── make_mortgage_dashboard_data.py  # aggregate financing stats for the dashboard
│   ├── correlations_analysis.py     # correlations, size sweet-spots, zoning concentration
│   ├── load_footprints.py           # load Microsoft building-footprint tiles for El Paso County
│   └── shape_features.py            # spatial join footprints <-> parcels; shape + shadow-structure signals
├── data/
│   ├── raw/                         # EPCAD flat-file exports (not checked in — see Setup)
│   ├── gis/                         # EPCAD parcel geodatabase + footprint tiles (not checked in — see Setup)
│   ├── output/                      # generated intermediate CSVs (not checked in, large)
│   ├── dashboard_stats.json         # generated: aggregated stats for the dashboard (checked in)
│   ├── shape_and_signal_stats.json  # generated: correlations/sweet-spots/zoning/shape-feature stats (checked in)
│   ├── parcels_missing_middle.geojson   # generated: confirmed missing-middle parcel points (checked in)
│   ├── candidates_high_confidence.geojson  # generated: model-flagged candidate parcels (checked in)
│   ├── missing_middle_candidates.csv    # generated: full candidate list w/ probabilities (checked in)
│   ├── feature_importance.csv       # generated: classifier permutation importances (checked in)
│   ├── model_metrics.json           # generated: classifier test AUC etc. (checked in)
│   └── mortgage_stats.json          # generated: First American financing aggregates (checked in)
```

## Known caveats

- `B3`-`B6` (lumped as "Multi-Unit Residential (Other)"), along with
  `A6`, `A51`, `A53`-`A55`, `A7`, `C10`, `D5`, `D6`, `O1`, `O2`, `J2`,
  `X23`, are EPCAD's own local extensions beyond the standard Texas
  Comptroller state codes. Checked against EPCAD's official
  `2026_Codes for Export Tables_All.xlsx` data dictionary (its
  `ptd_state_code` sheet is the real statewide Texas PTD/Comptroller
  table) — none of these local extensions are defined there either.
  Their typology assignments in `build_dataset.py` remain inference
  from acreage/legal-description patterns, not documented fact, and
  are commented as such in the code. (The dictionary did confirm two
  other things used elsewhere in the pipeline: the exemption-code
  meanings in `has_homestead`/`has_disabled_vet_exemption`/`tax_exempt`,
  and that `imprv_detail_type_codes` flags `MA`/`MA1`/`MA2`/`M+`/`MU`/`ME`
  all as `main_area=T`, validating `TypeCD=="MA"` as the right filter
  for counting distinct structures.)
- The apartment-size sub-split only covers `B2` parcels whose
  `Improvements.ClassCD` happens to carry a `QS`/`QL`/`QN`/`QR` code;
  some `B2` parcels have no matching Improvements record with that
  code and stay labeled generically as "Apartment/Multifamily".
- Distance is measured from a single downtown point (San Jacinto
  Plaza), straight-line, not drive-time.
