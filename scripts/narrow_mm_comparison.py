"""
Re-runs the EPCAD-vs-ACS tract comparison under the conventional,
literature-standard "missing middle" definition: 2-9 unit structures
(duplex, triplex/fourplex, townhouse, small courtyard apartments), NOT
large apartment complexes.

Why this exists: join_tracts_and_compare.py's is_missing_middle flag
(inherited from EPCAD's own STATE_CD classification, B1-B6) marks ANY
multifamily structure as "missing middle" -- including Apartment
(31-99 units) and Apartment (100+ units) buildings, which most planning
definitions of "missing middle housing" explicitly exclude (it's a scale
category, not just "not single-family"). That broad definition happens to
match how the ACS table (B25024, units 004-009) was summed in the original
comparison, so the 20.72%-vs-23.05% countywide numbers are apples-to-apples
-- but they're answering "how much non-single-family housing is there,"
not "how much missing-middle housing is there" in the term's usual sense.

This script re-scopes both sides to 2-9 unit structures only and reruns
the same tract-level comparison, including whether the structural
candidate detector's flags close the gap under this narrower, more
correct definition.
"""
import pandas as pd
import numpy as np
import json
import os

OUT = os.environ.get("EP_MMH_OUT", "data/output")

RES = ["Single-Family Detached", "Single-Family Detached (Other Residential Subtype)",
       "Mobile/Manufactured Home", "Duplex", "Townhouse", "Condominium", "Apartment/Multifamily",
       "Multi-Unit Residential (Other)", "Apartment (4-6 units)", "Apartment (7-30 units)",
       "Apartment (31-99 units)", "Apartment (100+ units)"]

# 2-9 unit structures only -- excludes Apartment (31-99 units) and (100+ units),
# which the EPCAD-derived is_missing_middle flag otherwise counts as "missing middle."
NARROW_MM_TYPES = ["Duplex", "Townhouse", "Condominium", "Apartment/Multifamily",
                   "Multi-Unit Residential (Other)", "Apartment (4-6 units)", "Apartment (7-30 units)"]

print("Loading parcels and tract join...")
df = pd.read_csv(f"{OUT}/parcels_analysis.csv")
tract_map = pd.read_csv(f"{OUT}/parcel_tract.csv", dtype={"tract_fips": str})
df["PROP_ID_num"] = pd.to_numeric(df["PROP_ID"], errors="coerce")
tract_map["PROP_ID_num"] = pd.to_numeric(tract_map["PROP_ID"], errors="coerce")
m = df.merge(tract_map[["PROP_ID_num", "tract_fips"]], on="PROP_ID_num", how="left")

res = m[m["typology"].isin(RES)].copy()
res["units_filled"] = res["approx_units"].fillna(1)
res["narrow_mm_row"] = np.where(res["typology"].isin(NARROW_MM_TYPES), res["units_filled"], 0)

cand = pd.read_csv(f"{OUT}/missing_middle_candidates.csv")
cand_hi = cand[cand["mm_probability"] >= 0.8].copy()
cand_hi["PROP_ID_num"] = pd.to_numeric(cand_hi["PROP_ID"], errors="coerce")
cand_tract = cand_hi.merge(tract_map[["PROP_ID_num", "tract_fips"]], on="PROP_ID_num", how="left")

g = res.groupby("tract_fips").agg(total_units=("units_filled", "sum"),
                                   narrow_mm_units=("narrow_mm_row", "sum")).reset_index()
g["narrow_mm_share_pct"] = (100 * g["narrow_mm_units"] / g["total_units"]).round(2)

cand_per_tract = cand_tract.groupby("tract_fips").size().rename("n_cand").reset_index()
g = g.merge(cand_per_tract, on="tract_fips", how="left")
g["n_cand"] = g["n_cand"].fillna(0)
g["narrow_mm_plus_model_units"] = g["narrow_mm_units"] + g["n_cand"]
g["total_plus_model"] = g["total_units"] + g["n_cand"]
g["narrow_mm_plus_model_share_pct"] = (100 * g["narrow_mm_plus_model_units"] / g["total_plus_model"]).round(2)

print("Loading ACS narrow (2-9 unit) comparison...")
acs_raw = json.load(open(f"{OUT}/acs_b25024_raw.json"))
rows = []
for geoid, tables in acs_raw["data"].items():
    tract_fips = geoid.split("US")[1]
    est = tables["B25024"]["estimate"]
    total = est.get("B25024001", 0) or 0
    narrow = sum((est.get(f"B25024{i:03d}", 0) or 0) for i in [4, 5, 6])  # 2, 3-4, 5-9 units
    rows.append({"tract_fips": tract_fips, "total_units_acs": total, "narrow_mm_units_acs": narrow})
acs_df = pd.DataFrame(rows)
acs_df["narrow_mm_share_acs_pct"] = (100 * acs_df["narrow_mm_units_acs"] / acs_df["total_units_acs"].replace(0, np.nan)).round(2)

full = g.merge(acs_df, on="tract_fips", how="inner")
full["diff_epcad"] = full["narrow_mm_share_pct"] - full["narrow_mm_share_acs_pct"]
full["diff_epcad_plus_model"] = full["narrow_mm_plus_model_share_pct"] - full["narrow_mm_share_acs_pct"]

print(f"\nN tracts: {len(full)}")
print(f"Mean |EPCAD(narrow) - ACS(narrow)|: {full['diff_epcad'].abs().mean():.2f} pts")
print(f"Mean |EPCAD+model(narrow) - ACS(narrow)|: {full['diff_epcad_plus_model'].abs().mean():.2f} pts")
print(f"Tracts improved by model: {int((full['diff_epcad_plus_model'].abs() < full['diff_epcad'].abs()).sum())} / {len(full)}")

full.to_csv(f"{OUT}/tract_comparison_narrow_mm.csv", index=False)
summary = {
    "definition": "narrow (2-9 unit structures: Duplex, Townhouse, Condo, Apartment/Multifamily, "
                   "Multi-Unit Other, Apartment 4-6, Apartment 7-30; excludes Apartment 31-99 and 100+)",
    "n_tracts_compared": int(len(full)),
    "mean_abs_diff_epcad_pct_points": round(float(full["diff_epcad"].abs().mean()), 2),
    "mean_abs_diff_epcad_plus_model_pct_points": round(float(full["diff_epcad_plus_model"].abs().mean()), 2),
    "tracts_improved_by_model": int((full["diff_epcad_plus_model"].abs() < full["diff_epcad"].abs()).sum()),
    "countywide_epcad_narrow_units": int(full["narrow_mm_units"].sum()),
    "countywide_epcad_narrow_total_units": int(full["total_units"].sum()),
    "countywide_epcad_narrow_pct": round(100 * full["narrow_mm_units"].sum() / full["total_units"].sum(), 2),
    "countywide_epcad_plus_model_narrow_units": int(full["narrow_mm_plus_model_units"].sum()),
    "countywide_epcad_plus_model_narrow_pct": round(100 * full["narrow_mm_plus_model_units"].sum() / full["total_plus_model"].sum(), 2),
    "countywide_acs_narrow_units": int(full["narrow_mm_units_acs"].sum()),
    "countywide_acs_narrow_total_units": int(full["total_units_acs"].sum()),
    "countywide_acs_narrow_pct": round(100 * full["narrow_mm_units_acs"].sum() / full["total_units_acs"].sum(), 2),
}
json.dump(summary, open("data/narrow_mm_comparison_summary.json", "w"), indent=2)
print("\nSaved data/narrow_mm_comparison_summary.json")
