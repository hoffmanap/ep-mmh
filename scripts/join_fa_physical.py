"""
Nearest-neighbor join of First American's PHYSICAL-characteristic
fields (bedrooms, bathrooms, living area, effective year built, lot
sqft) onto EPCAD parcels, by situs lat/lon -- same 0.02-mile-tolerance
method scripts/correlations_analysis.py uses descriptively, but saved
here as a reusable file keyed on PROP_ID so these fields can also be
used as model features (scripts/train_mm_detector.py), not just for a
one-off correlation printout.

This is a separate join from scripts/join_first_american.py, which
pulls OWNERSHIP/FINANCING fields (owner-occupied, liens, AVM value,
unit/building counts) into parcels_with_mortgage.csv. That join and
this one use the same source file (Prop48141.txt) and the same
nearest-neighbor method, just different output columns -- kept
separate because they serve different purposes and were added in
different passes of this project.

Coverage caveat: only ~58% of EPCAD parcels match within 0.02 miles
(237,565 of 406,529) -- worse than the ownership/financing join,
because First American's physical-characteristic fields themselves
have more missingness. train_mm_detector.py fills unmatched parcels
with 0 for fa_bathrooms, which is a real limitation (a 0 bathroom
count is indistinguishable from "no data" in that field) rather than
a resolved gap.
"""
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

OUT_DIR = "/sessions/rcw-019pt16gylbtcckzycdh9ppp/mnt/data/el_paso_mm/output"
FA_RAW_DIR = "/sessions/rcw-019pt16gylbtcckzycdh9ppp/mnt/data/first_american"

parcels = pd.read_csv(f"{OUT_DIR}/parcels_analysis.csv")

fa = pd.read_csv(
    f"{FA_RAW_DIR}/Prop48141.txt", sep="|",
    usecols=["PropertyID", "SitusLatitude", "SitusLongitude", "Bedrooms", "BathTotalCalc",
             "BathFull", "SumLivingAreaSqFt", "EffectiveYearBuilt", "LotSizeSqFt"],
    dtype=str, encoding="latin-1", engine="c", quoting=3, low_memory=False,
)
fa["lat"] = pd.to_numeric(fa["SitusLatitude"], errors="coerce")
fa["lon"] = pd.to_numeric(fa["SitusLongitude"], errors="coerce")
fa = fa[(fa["lat"] != 0) & (fa["lon"] != 0) & fa["lat"].notna() & fa["lon"].notna()]

LAT0 = 31.76
MI_LAT, MI_LON = 69.0, 69.0 * np.cos(np.radians(LAT0))
def to_xy(lat, lon):
    return np.column_stack([lon * MI_LON, lat * MI_LAT])

tree = cKDTree(to_xy(fa["lat"].values, fa["lon"].values))
p_valid = parcels.dropna(subset=["lat", "lon"]).copy()
dist, idx = tree.query(to_xy(p_valid["lat"].values, p_valid["lon"].values), k=1, distance_upper_bound=0.02)
matched = dist != np.inf
fa_matched = fa.iloc[idx[matched]].reset_index(drop=True)
p_matched = p_valid.iloc[matched].reset_index(drop=True)

out = pd.DataFrame({
    "PROP_ID": p_matched["PROP_ID"].values,
    "fa_bedrooms": pd.to_numeric(fa_matched["Bedrooms"], errors="coerce").values,
    "fa_bathrooms": pd.to_numeric(fa_matched["BathTotalCalc"], errors="coerce").values,
    "fa_building_sqft": pd.to_numeric(fa_matched["SumLivingAreaSqFt"], errors="coerce").values,
    "fa_effective_year_built": pd.to_numeric(fa_matched["EffectiveYearBuilt"], errors="coerce").values,
    "fa_lot_sqft": pd.to_numeric(fa_matched["LotSizeSqFt"], errors="coerce").values,
    "fa_phys_match_dist_mi": dist[matched],
})
out.to_csv(f"{OUT_DIR}/parcels_with_fa_physical.csv", index=False)
print(f"Matched {len(out):,} of {len(p_valid):,} parcels to FA physical characteristics "
      f"({len(out) / len(p_valid):.1%} coverage)")
print(out.describe())
