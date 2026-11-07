# El Paso Missing Middle Housing Analysis

Adapted from the methodology of Salazar-Miranda & Talen,
*"Mapping the Missing Middle: AI-Based Detection of Mid-Density Housing"*
(replication package: `github.com/ariannasm/missing-middle-housing`).

Live dashboard: `index.html` (open directly, or via GitHub Pages).

## Results (2026 EPCAD data)

- **242,388** residential parcels classified
- **8,883** missing-middle parcels (duplex, townhouse, condo,
  apartment/multifamily 2-99 units) — **3.66%** of residential stock
- Strong distance-from-core gradient: **30.6%** missing-middle share
  within 1 mile of downtown El Paso, falling to **under 1%** beyond
  12 miles
- Zoning cross-tab shows missing-middle stock concentrated in
  `R-5`/`R3A`-type multi-family zoning, as expected

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
│   └── make_dashboard_data.py       # aggregate into dashboard JSON/GeoJSON
├── data/
│   ├── raw/                         # EPCAD flat-file exports (not checked in — see Setup)
│   ├── gis/                         # EPCAD parcel geodatabase (not checked in — see Setup)
│   ├── output/                      # generated: parcels_analysis.csv (not checked in, large)
│   ├── dashboard_stats.json         # generated: aggregated stats for the dashboard (checked in)
│   └── parcels_missing_middle.geojson  # generated: missing-middle parcel points (checked in)
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
