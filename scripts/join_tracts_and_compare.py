"""
Ties together the three missing-middle estimates for El Paso County at
the census-tract level: EPCAD's own parcel classification (unit-
weighted), EPCAD plus the structural candidate detector's high-
confidence flags, and the Census ACS's self-reported units-in-structure
survey (table B25024). Answers directly: whose count is closer to the
Census's independent benchmark, and where do they disagree.

Also does the official-zoning spatial join (City of El Paso GIS,
Planning/Zoning FeatureServer) used to clean up the zoning-concentration
chart's labels.
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import json
import os

OUT = os.environ.get("EP_MMH_OUT", "data/output")
GIS = os.environ.get("EP_MMH_GIS", "data/gis")

RES = ["Single-Family Detached", "Single-Family Detached (Other Residential Subtype)",
       "Mobile/Manufactured Home", "Duplex", "Townhouse", "Condominium", "Apartment/Multifamily",
       "Multi-Unit Residential (Other)", "Apartment (4-6 units)", "Apartment (7-30 units)",
       "Apartment (31-99 units)", "Apartment (100+ units)"]

# ---------------------------------------------------------------------
# Official zoning: spatial join of parcel centroids against the City of
# El Paso's own zoning layer (clean ZONELABEL field), replacing the raw
# EPCAD GIS ZONING field's inconsistent label spellings.
# ---------------------------------------------------------------------
print("Official zoning spatial join...")
parcels = gpd.read_parquet(f"{OUT}/parcel_polygons.parquet")
zoning = gpd.read_file(f"{GIS}/zoning/el_paso_zoning.geojson").to_crs(parcels.crs)
parcels["centroid"] = parcels.geometry.centroid
pts = gpd.GeoDataFrame(parcels[["PROP_ID"]], geometry=parcels["centroid"], crs=parcels.crs)
zoning_joined = gpd.sjoin(pts, zoning[["ZONELABEL", "geometry"]], how="left", predicate="within")
zoning_joined = zoning_joined.drop_duplicates(subset="PROP_ID", keep="first")
print(f"  matched {zoning_joined['ZONELABEL'].notna().sum():,} / {len(zoning_joined):,} "
      f"(unmatched = outside city limits / ETJ, no city zoning record)")
zoning_joined[["PROP_ID", "ZONELABEL"]].rename(columns={"ZONELABEL": "official_zoning"}).to_csv(
    f"{OUT}/parcel_official_zoning.csv", index=False)

# ---------------------------------------------------------------------
# Tract join: parcel centroids against Census tract boundaries
# ---------------------------------------------------------------------
print("Census tract spatial join...")
tracts = gpd.read_file(f"{GIS}/tracts/el_paso_tracts.geojson").to_crs(parcels.crs)
tract_joined = gpd.sjoin(pts, tracts[["GEOID", "geometry"]], how="left", predicate="within")
tract_joined = tract_joined.drop_duplicates(subset="PROP_ID", keep="first")
print(f"  matched {tract_joined['GEOID'].notna().sum():,} / {len(tract_joined):,}")
tract_joined[["PROP_ID", "GEOID"]].rename(columns={"GEOID": "tract_fips"}).to_csv(
    f"{OUT}/parcel_tract.csv", index=False)

# ---------------------------------------------------------------------
# Unit-weighted EPCAD totals, with and without the model's high-
# confidence candidates, per tract
# ---------------------------------------------------------------------
print("Aggregating EPCAD unit-weighted totals per tract...")
parcels_df = pd.read_csv(f"{OUT}/parcels_analysis.csv")
tract_map = pd.read_csv(f"{OUT}/parcel_tract.csv", dtype={"tract_fips": str})
parcels_df["PROP_ID_num"] = pd.to_numeric(parcels_df["PROP_ID"], errors="coerce")
tract_map["PROP_ID_num"] = pd.to_numeric(tract_map["PROP_ID"], errors="coerce")
m = parcels_df.merge(tract_map[["PROP_ID_num", "tract_fips"]], on="PROP_ID_num", how="left")

res = m[m["typology"].isin(RES)].copy()
res["units_filled"] = res["approx_units"].fillna(1)
res["mm_units_row"] = np.where(res["is_missing_middle"], res["units_filled"], 0)

cand = pd.read_csv(f"{OUT}/missing_middle_candidates.csv")
cand_hi = cand[cand["mm_probability"] >= 0.8].copy()
cand_hi["PROP_ID_num"] = pd.to_numeric(cand_hi["PROP_ID"], errors="coerce")
cand_tract = cand_hi.merge(tract_map[["PROP_ID_num", "tract_fips"]], on="PROP_ID_num", how="left")

g = res.groupby("tract_fips").agg(total_units_epcad=("units_filled", "sum"),
                                   mm_units_epcad=("mm_units_row", "sum")).reset_index()
g["mm_share_epcad_pct"] = (100 * g["mm_units_epcad"] / g["total_units_epcad"]).round(1)

cand_per_tract = cand_tract.groupby("tract_fips").size().rename("n_model_candidates_hi").reset_index()
g = g.merge(cand_per_tract, on="tract_fips", how="left")
g["n_model_candidates_hi"] = g["n_model_candidates_hi"].fillna(0)
g["mm_units_epcad_plus_model"] = g["mm_units_epcad"] + g["n_model_candidates_hi"]
g["total_units_epcad_plus_model"] = g["total_units_epcad"] + g["n_model_candidates_hi"]
g["mm_share_epcad_plus_model_pct"] = (100 * g["mm_units_epcad_plus_model"] / g["total_units_epcad_plus_model"]).round(1)

# ---------------------------------------------------------------------
# ACS comparison (B25024, Census Reporter API)
# ---------------------------------------------------------------------
print("Loading ACS tract data...")
acs_raw = json.load(open(f"{OUT}/acs_b25024_raw.json"))
acs_rows = []
for geoid, tables in acs_raw["data"].items():
    tract_fips = geoid.split("US")[1]
    est = tables["B25024"]["estimate"]
    total = est.get("B25024001", 0) or 0
    mm = sum((est.get(f"B25024{i:03d}", 0) or 0) for i in [4, 5, 6, 7, 8, 9])
    acs_rows.append({"tract_fips": tract_fips, "total_units_acs": total, "mm_units_acs": mm})
acs_df = pd.DataFrame(acs_rows)
acs_df["mm_share_acs_pct"] = (100 * acs_df["mm_units_acs"] / acs_df["total_units_acs"].replace(0, np.nan)).round(1)

full = g.merge(acs_df, on="tract_fips", how="inner")
full["diff_epcad_vs_acs"] = full["mm_share_epcad_pct"] - full["mm_share_acs_pct"]
full["diff_epcad_plus_model_vs_acs"] = full["mm_share_epcad_plus_model_pct"] - full["mm_share_acs_pct"]

print(f"\nN tracts compared: {len(full)}")
print(f"Mean |EPCAD - ACS|: {full['diff_epcad_vs_acs'].abs().mean():.2f} pct points")
print(f"Mean |EPCAD+model - ACS|: {full['diff_epcad_plus_model_vs_acs'].abs().mean():.2f} pct points")
print(f"Countywide EPCAD: {int(full['mm_units_epcad'].sum())} / {int(full['total_units_epcad'].sum())} units "
      f"({100*full['mm_units_epcad'].sum()/full['total_units_epcad'].sum():.2f}%)")
print(f"Countywide ACS:   {int(full['mm_units_acs'].sum())} / {int(full['total_units_acs'].sum())} units "
      f"({100*full['mm_units_acs'].sum()/full['total_units_acs'].sum():.2f}%)")

summary = {
    "n_tracts_compared": int(len(full)),
    "mean_abs_diff_epcad_vs_acs_pct_points": round(float(full["diff_epcad_vs_acs"].abs().mean()), 2),
    "mean_abs_diff_epcad_plus_model_vs_acs_pct_points": round(float(full["diff_epcad_plus_model_vs_acs"].abs().mean()), 2),
    "tracts_improved_by_model": int((full["diff_epcad_plus_model_vs_acs"].abs() < full["diff_epcad_vs_acs"].abs()).sum()),
    "countywide_total_units_epcad": int(full["total_units_epcad"].sum()),
    "countywide_mm_units_epcad": int(full["mm_units_epcad"].sum()),
    "countywide_mm_share_epcad_pct": round(100 * full["mm_units_epcad"].sum() / full["total_units_epcad"].sum(), 2),
    "countywide_total_units_acs": int(full["total_units_acs"].sum()),
    "countywide_mm_units_acs": int(full["mm_units_acs"].sum()),
    "countywide_mm_share_acs_pct": round(100 * full["mm_units_acs"].sum() / full["total_units_acs"].sum(), 2),
    "tract_level": full[["tract_fips", "mm_share_epcad_pct", "mm_share_epcad_plus_model_pct",
                          "mm_share_acs_pct", "total_units_acs"]].to_dict(orient="records"),
}
json.dump(summary, open("data/tract_comparison_epcad_acs.json", "w"), indent=2)
print("\nSaved data/tract_comparison_epcad_acs.json")
