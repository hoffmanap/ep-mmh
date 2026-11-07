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
│   └── make_mortgage_dashboard_data.py  # aggregate financing stats for the dashboard
├── data/
│   ├── raw/                         # EPCAD flat-file exports (not checked in — see Setup)
│   ├── gis/                         # EPCAD parcel geodatabase (not checked in — see Setup)
│   ├── output/                      # generated intermediate CSVs (not checked in, large)
│   ├── dashboard_stats.json         # generated: aggregated stats for the dashboard (checked in)
│   ├── parcels_missing_middle.geojson   # generated: confirmed missing-middle parcel points (checked in)
│   ├── candidates_high_confidence.geojson  # generated: model-flagged candidate parcels (checked in)
│   ├── missing_middle_candidates.csv    # generated: full candidate list w/ probabilities (checked in)
│   ├── feature_importance.csv       # generated: classifier permutation importances (checked in)
│   ├── model_metrics.json           # generated: classifier test AUC etc. (checked in)
│   └── mortgage_stats.json          # generated: First American financing aggregates (checked in)
```

## Known caveats

- `B3`-`B6` (lumped as "Multi-Unit Residential (Other)") are EPCAD's
  own local extensions beyond the standard Texas Comptroller state
  codes; their exact unit-count semantics aren't documented and are
  inferred from context (they carry the same kind of improvement
  descriptions — MAIN AREA, OPEN PORCH, GARAGE — as other residential
  classes).
- The apartment-size sub-split only covers `B2` parcels whose
  `Improvements.ClassCD` happens to carry a `QS`/`QL`/`QN`/`QR` code;
  some `B2` parcels have no matching Improvements record with that
  code and stay labeled generically as "Apartment/Multifamily".
- Distance is measured from a single downtown point (San Jacinto
  Plaza), straight-line, not drive-time.
