"""
Builds data/dashboard_stats.json and data/parcels_missing_middle.geojson --
the files the live dashboard's main stat block, distance-gradient chart,
decade-built chart, and zoning-typology chart actually read.

Was stale for a while this project: this script's RESIDENTIAL_TYPES list
had drifted from the one in join_tracts_and_compare.py (missing "Single-
Family Detached (Other Residential Subtype)", the STATE_CD-gap fix, and
wrongly including Farm/Ranch House/Mobile Home), so the dashboard kept
showing pre-fix numbers (242,694 parcels / 3.66%) after the underlying
data had already moved to 252,626 parcels / 20.72% unit-weighted. Fixed
by using the same canonical residential list everywhere.
"""
import pandas as pd
import numpy as np
import json
import os

OUT = os.environ.get("EP_MMH_OUT", "data/output")

# Canonical residential-typology list (must match join_tracts_and_compare.py).
# Excludes Farm/Ranch House and Farm/Ranch Mobile Home (rural acreage-style
# parcels EPCAD tracks separately from urban residential stock) and includes
# "Single-Family Detached (Other Residential Subtype)" -- the STATE_CD bucket
# that was originally dropped (the "STATE_CD gap" fixed earlier this project).
RESIDENTIAL_TYPES = [
    "Single-Family Detached", "Single-Family Detached (Other Residential Subtype)",
    "Mobile/Manufactured Home", "Duplex", "Townhouse", "Condominium", "Apartment/Multifamily",
    "Multi-Unit Residential (Other)", "Apartment (4-6 units)", "Apartment (7-30 units)",
    "Apartment (31-99 units)", "Apartment (100+ units)",
]

df = pd.read_csv(f"{OUT}/parcels_analysis.csv")
res = df[df["typology"].isin(RESIDENTIAL_TYPES)].copy()

# ---- citywide stats (parcel-count basis) ----
total = len(res)
mm = int(res["is_missing_middle"].sum())

# ---- unit-weighted stats (matches tract_comparison_epcad_acs.json methodology) ----
res["units_filled"] = res["approx_units"].fillna(1)
total_units = float(res["units_filled"].sum())
mm_units = float(res.loc[res["is_missing_middle"], "units_filled"].sum())

stats = {
    "total_residential_parcels": int(total),
    "missing_middle_parcels": mm,
    "missing_middle_share_pct": round(mm / total * 100, 2),
    "total_residential_units_weighted": int(round(total_units)),
    "missing_middle_units_weighted": int(round(mm_units)),
    "missing_middle_share_unit_weighted_pct": round(mm_units / total_units * 100, 2),
    "typology_counts": res["typology"].value_counts().to_dict(),
}

# ---- distance gradient (1-mile bins, 0-15mi) ----
res_d = res.dropna(subset=["dist_to_core_mi"]).copy()
bins = list(range(0, 16))
res_d["dist_bin"] = pd.cut(res_d["dist_to_core_mi"], bins=bins, right=False)
grouped = res_d.groupby("dist_bin", observed=True)["is_missing_middle"].agg(["sum", "count"])
grouped["share_pct"] = (grouped["sum"] / grouped["count"] * 100).round(2)
distance_gradient = [
    {"bin_start": interval.left, "bin_end": interval.right,
     "mm_share_pct": float(row["share_pct"]) if pd.notna(row["share_pct"]) else None,
     "n": int(row["count"])}
    for interval, row in grouped.iterrows()
]

# ---- typology by decade built ----
res_y = res.dropna(subset=["year_built"]).copy()
res_y = res_y[(res_y["year_built"] >= 1900) & (res_y["year_built"] <= 2026)]
res_y["decade"] = (res_y["year_built"] // 10 * 10).astype(int)
decade_typology = res_y.groupby(["decade", "typology"]).size().unstack(fill_value=0)
decade_data = {
    "decades": decade_typology.index.tolist(),
    "typologies": decade_typology.columns.tolist(),
    "matrix": decade_typology.values.tolist(),
}

# ---- typology by zoning (top 12 zoning codes) ----
res_z = res.dropna(subset=["ZONING"]).copy()
res_z["ZONING"] = res_z["ZONING"].astype(str).str.strip()
res_z = res_z[res_z["ZONING"] != "NULL"]
top_zones = res_z["ZONING"].value_counts().head(12).index.tolist()
res_z = res_z[res_z["ZONING"].isin(top_zones)]
zoning_typology = res_z.groupby(["ZONING", "typology"]).size().unstack(fill_value=0)
zoning_data = {
    "zones": zoning_typology.index.tolist(),
    "typologies": zoning_typology.columns.tolist(),
    "matrix": zoning_typology.values.tolist(),
}

with open(f"{OUT}/dashboard_stats.json", "w") as f:
    json.dump({
        "stats": stats,
        "distance_gradient": distance_gradient,
        "decade_typology": decade_data,
        "zoning_typology": zoning_data,
    }, f, indent=2, default=str)

# ---- GeoJSON of missing-middle parcels only (for map markers) ----
mm_df = res[res["is_missing_middle"] & res["lat"].notna() & res["lon"].notna()]
features = []
for _, r in mm_df.iterrows():
    features.append({
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [round(r["lon"], 6), round(r["lat"], 6)]},
        "properties": {
            "typology": r["typology"],
            "year_built": int(r["year_built"]) if pd.notna(r["year_built"]) else None,
            "dist_to_core_mi": round(float(r["dist_to_core_mi"]), 2) if pd.notna(r["dist_to_core_mi"]) else None,
            "zoning": r["ZONING"] if pd.notna(r["ZONING"]) else None,
        },
    })
with open(f"{OUT}/parcels_missing_middle.geojson", "w") as f:
    json.dump({"type": "FeatureCollection", "features": features}, f)

print(f"Residential parcels: {total:,}")
print(f"Missing-middle parcels: {mm:,} ({stats['missing_middle_share_pct']}%)")
print(f"Unit-weighted: {int(mm_units):,} / {int(total_units):,} units ({stats['missing_middle_share_unit_weighted_pct']}%)")
print(f"GeoJSON features (missing-middle only): {len(features):,}")
print("Saved dashboard_stats.json and parcels_missing_middle.geojson")
