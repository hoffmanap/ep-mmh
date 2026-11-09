"""
Re-runs the EPCAD-vs-ACS tract comparison scoped to missing middle as
the term is actually defined in the literature this project adapts:
Parolek's "roughly duplexes to buildings with up to 19 units," which is
also the exact ACS bucket (2-19 unit structures) Salazar-Miranda & Talen
themselves compared their Chicago measure against.

Why this script exists at all: EPCAD's own is_missing_middle flag
(STATE_CD B1-B6) marks ANY multifamily structure as "missing middle,"
including Apartment (31-99 units) and Apartment (100+ units) complexes --
which no definition of "missing middle" in the planning literature
includes; it's a density/scale category, not just "not single-family."
That broad EPCAD flag happens to roughly match how a NAIVE "any 2+ unit
structure" ACS cut would be scoped (B25024 categories 004-009), so the
original countywide 20.72%-vs-23.05% comparison is apples-to-apples --
but it answers "how much non-single-family housing is there," not "how
much missing middle is there" in the term's actual sense.

This script re-scopes to the literature's actual ceiling (up to 19
units, not 9 -- an earlier pass here used a 2-9 cutoff that was too
conservative and didn't match the project's own cited methodology) and
reruns the same tract-level comparison, including whether the structural
candidate detector's flags close the gap under this corrected scope.

EPCAD's own apartment-size bands top out at QL = "7-30 units," not 19 --
there's no EPCAD band that stops exactly at 19, so "Apartment (7-30
units)" is used as the closest available cut on the EPCAD side (a small
asymmetry worth noting: a handful of EPCAD's 20-30-unit buildings get
counted as "missing middle" that the ACS 2-19 bucket wouldn't). The ACS
side is scoped exactly: B25024 categories 004-007 (2, 3-4, 5-9, 10-19
units).
"""
import pandas as pd
import numpy as np
import json
import os

OUT = os.environ.get("EP_MMH_OUT", "data/output")
DATA = os.environ.get("EP_MMH_DATA", "data")

RES = ["Single-Family Detached", "Single-Family Detached (Other Residential Subtype)",
       "Mobile/Manufactured Home", "Duplex", "Triplex", "Fourplex", "Townhouse", "Condominium", "Apartment/Multifamily",
       "Multi-Unit Residential (Other)", "Apartment (4-6 units)", "Apartment (7-30 units)",
       "Apartment (31-99 units)", "Apartment (100+ units)"]

# EPCAD-side missing-middle cut: everything up through "Apartment (7-30 units)"
# (the closest available EPCAD band to a 19-unit ceiling), excluding the two
# large-complex bands (31-99, 100+).
MM_2_19_TYPES = ["Duplex", "Triplex", "Fourplex", "Townhouse", "Condominium", "Apartment/Multifamily",
                 "Multi-Unit Residential (Other)", "Apartment (4-6 units)", "Apartment (7-30 units)"]

print("Loading parcels and tract join...")
df = pd.read_csv(f"{OUT}/parcels_analysis.csv")
tract_map = pd.read_csv(f"{OUT}/parcel_tract.csv", dtype={"tract_fips": str})
df["PROP_ID_num"] = pd.to_numeric(df["PROP_ID"], errors="coerce")
tract_map["PROP_ID_num"] = pd.to_numeric(tract_map["PROP_ID"], errors="coerce")
m = df.merge(tract_map[["PROP_ID_num", "tract_fips"]], on="PROP_ID_num", how="left")

res = m[m["typology"].isin(RES)].copy()
res["units_filled"] = res["approx_units"].fillna(1)
res["mm_2_19_row"] = np.where(res["typology"].isin(MM_2_19_TYPES), res["units_filled"], 0)

cand = pd.read_csv(f"{OUT}/missing_middle_candidates.csv")
cand_hi = cand[cand["mm_probability"] >= 0.8].copy()
cand_hi["PROP_ID_num"] = pd.to_numeric(cand_hi["PROP_ID"], errors="coerce")
cand_tract = cand_hi.merge(tract_map[["PROP_ID_num", "tract_fips"]], on="PROP_ID_num", how="left")

g = res.groupby("tract_fips").agg(total_units=("units_filled", "sum"),
                                   mm_2_19_units=("mm_2_19_row", "sum")).reset_index()
g["mm_2_19_share_pct"] = (100 * g["mm_2_19_units"] / g["total_units"]).round(2)

cand_per_tract = cand_tract.groupby("tract_fips").size().rename("n_cand").reset_index()
g = g.merge(cand_per_tract, on="tract_fips", how="left")
g["n_cand"] = g["n_cand"].fillna(0)
g["mm_2_19_plus_model_units"] = g["mm_2_19_units"] + g["n_cand"]
g["total_plus_model"] = g["total_units"] + g["n_cand"]
g["mm_2_19_plus_model_share_pct"] = (100 * g["mm_2_19_plus_model_units"] / g["total_plus_model"]).round(2)

print("Loading ACS 2-19 unit comparison...")
acs_raw = json.load(open(f"{OUT}/acs_b25024_raw.json"))
rows = []
for geoid, tables in acs_raw["data"].items():
    tract_fips = geoid.split("US")[1]
    est = tables["B25024"]["estimate"]
    total = est.get("B25024001", 0) or 0
    mm_2_19 = sum((est.get(f"B25024{i:03d}", 0) or 0) for i in [4, 5, 6, 7])  # 2, 3-4, 5-9, 10-19 units
    rows.append({"tract_fips": tract_fips, "total_units_acs": total, "mm_2_19_units_acs": mm_2_19})
acs_df = pd.DataFrame(rows)
acs_df["mm_2_19_share_acs_pct"] = (100 * acs_df["mm_2_19_units_acs"] / acs_df["total_units_acs"].replace(0, np.nan)).round(2)

full = g.merge(acs_df, on="tract_fips", how="inner")
full["diff_epcad"] = full["mm_2_19_share_pct"] - full["mm_2_19_share_acs_pct"]
full["diff_epcad_plus_model"] = full["mm_2_19_plus_model_share_pct"] - full["mm_2_19_share_acs_pct"]

print(f"\nN tracts: {len(full)}")
print(f"Mean |EPCAD(2-19) - ACS(2-19)|: {full['diff_epcad'].abs().mean():.2f} pts")
print(f"Mean |EPCAD+model(2-19) - ACS(2-19)|: {full['diff_epcad_plus_model'].abs().mean():.2f} pts")
print(f"Tracts improved by model: {int((full['diff_epcad_plus_model'].abs() < full['diff_epcad'].abs()).sum())} / {len(full)}")

full.to_csv(f"{OUT}/tract_comparison_mm_2_19.csv", index=False)
summary = {
    "definition": "EPCAD side: Duplex, Townhouse, Condo, Apartment/Multifamily, Multi-Unit Other, "
                   "Apartment 4-6, Apartment 7-30 (EPCAD bands top out at 30, not 19 -- closest available "
                   "cut; excludes Apartment 31-99 and 100+). ACS side: B25024 categories 2, 3-4, 5-9, 10-19 "
                   "units (the exact bucket Parolek / Salazar-Miranda & Talen use for missing middle's "
                   "19-unit ceiling).",
    "n_tracts_compared": int(len(full)),
    "mean_abs_diff_epcad_pct_points": round(float(full["diff_epcad"].abs().mean()), 2),
    "mean_abs_diff_epcad_plus_model_pct_points": round(float(full["diff_epcad_plus_model"].abs().mean()), 2),
    "tracts_improved_by_model": int((full["diff_epcad_plus_model"].abs() < full["diff_epcad"].abs()).sum()),
    "countywide_epcad_mm_2_19_units": int(full["mm_2_19_units"].sum()),
    "countywide_epcad_mm_2_19_total_units": int(full["total_units"].sum()),
    "countywide_epcad_mm_2_19_pct": round(100 * full["mm_2_19_units"].sum() / full["total_units"].sum(), 2),
    "countywide_epcad_plus_model_mm_2_19_units": int(full["mm_2_19_plus_model_units"].sum()),
    "countywide_epcad_plus_model_mm_2_19_pct": round(100 * full["mm_2_19_plus_model_units"].sum() / full["total_plus_model"].sum(), 2),
    "countywide_acs_mm_2_19_units": int(full["mm_2_19_units_acs"].sum()),
    "countywide_acs_mm_2_19_total_units": int(full["total_units_acs"].sum()),
    "countywide_acs_mm_2_19_pct": round(100 * full["mm_2_19_units_acs"].sum() / full["total_units_acs"].sum(), 2),
}
json.dump(summary, open(f"{DATA}/mm_2_19_comparison_summary.json", "w"), indent=2)
print(f"\nSaved {DATA}/mm_2_19_comparison_summary.json")
