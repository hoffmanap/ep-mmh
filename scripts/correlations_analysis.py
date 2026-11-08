"""
Correlations, lot/building-size sweet spots, and zoning concentration
for missing-middle housing in El Paso. Combines the EPCAD parcel
analysis (with homestead/exemption/text-flag columns from
build_dataset.py) with First American's bedroom/bathroom fields.
"""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
import json
import os

OUT = os.environ.get("EP_MMH_OUT", "data/output")
FA_RAW = os.environ.get("EP_MMH_FA_RAW", "data/first_american")

parcels = pd.read_csv(f"{OUT}/parcels_analysis.csv")

RES = ['Single-Family Detached', 'Single-Family Detached (Other Residential Subtype)',
       'Mobile/Manufactured Home', 'Duplex', 'Townhouse', 'Condominium', 'Apartment/Multifamily',
       'Multi-Unit Residential (Other)', 'Apartment (4-6 units)', 'Apartment (7-30 units)',
       'Apartment (31-99 units)', 'Apartment (100+ units)']
res = parcels[parcels["typology"].isin(RES)].copy()

# ---------------------------------------------------------------------
# Pull bedrooms/bathrooms/building sqft from First American (not in EPCAD),
# spatially joined via nearest-neighbor on situs lat/lon (0.02-mile tolerance)
# ---------------------------------------------------------------------
fa_full = pd.read_csv(
    f"{FA_RAW}/Prop48141.txt", sep="|",
    usecols=["PropertyID", "SitusLatitude", "SitusLongitude", "Bedrooms", "BathTotalCalc", "BathFull",
             "SumLivingAreaSqFt", "LotSizeAcres"],
    dtype=str, encoding="latin-1", engine="c", quoting=3, low_memory=False,
)
fa_full["lat"] = pd.to_numeric(fa_full["SitusLatitude"], errors="coerce")
fa_full["lon"] = pd.to_numeric(fa_full["SitusLongitude"], errors="coerce")
fa_full = fa_full[(fa_full["lat"] != 0) & (fa_full["lon"] != 0) & fa_full["lat"].notna()]

LAT0 = 31.76
MI_LAT, MI_LON = 69.0, 69.0 * np.cos(np.radians(LAT0))
def to_xy(lat, lon):
    return np.column_stack([lon * MI_LON, lat * MI_LAT])

tree = cKDTree(to_xy(fa_full["lat"].values, fa_full["lon"].values))
res_valid = res.dropna(subset=["lat", "lon"]).copy()
dist, idx = tree.query(to_xy(res_valid["lat"].values, res_valid["lon"].values), k=1, distance_upper_bound=0.02)
matched = dist != np.inf
fa_matched = fa_full.iloc[idx[matched]].reset_index(drop=True)
res_m = res_valid.iloc[matched].reset_index(drop=True)
res_m["bedrooms"] = pd.to_numeric(fa_matched["Bedrooms"], errors="coerce").values
res_m["bathrooms"] = pd.to_numeric(fa_matched["BathTotalCalc"], errors="coerce").values
res_m["building_sqft"] = pd.to_numeric(fa_matched["SumLivingAreaSqFt"], errors="coerce").values

print(f"Matched {len(res_m):,} of {len(res_valid):,} residential parcels to FA bed/bath data")

# ---------------------------------------------------------------------
# Correlations with is_missing_middle (point-biserial for continuous,
# Pearson for the 0/1 indicator features)
# ---------------------------------------------------------------------
print("\n=== Correlations with is_missing_middle ===")
corr_features = {
    "land_acres": pd.to_numeric(res_m["LAND_ACRES"], errors="coerce"),
    "year_built": res_m["year_built"],
    "dist_to_core_mi": res_m["dist_to_core_mi"],
    "bedrooms": res_m["bedrooms"],
    "bathrooms": res_m["bathrooms"],
    "building_sqft": res_m["building_sqft"],
    "has_homestead": res_m["has_homestead"].astype(int),
    "mailing_mismatch": res_m["mailing_mismatch"].astype(int),
}
corr_results = {}
y = res_m["is_missing_middle"].astype(int)
for name, x in corr_features.items():
    mask = x.notna() & y.notna()
    if mask.sum() > 30:
        if name in ("has_homestead", "mailing_mismatch"):
            r, p = stats.pearsonr(y[mask], x[mask])
        else:
            r, p = stats.pointbiserialr(y[mask], x[mask])
        corr_results[name] = {"r": round(float(r), 4), "p": float(p), "n": int(mask.sum())}
        print(f"  {name:20s} r={r:+.4f}  p={p:.2e}  n={mask.sum():,}")

# ---------------------------------------------------------------------
# Sweet-spot binning: lot acreage and building sqft vs missing-middle share
# ---------------------------------------------------------------------
res_m["acres_num"] = pd.to_numeric(res_m["LAND_ACRES"], errors="coerce")
acre_bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0, 999]
res_m["acre_bin"] = pd.cut(res_m["acres_num"], acre_bins)
lot_sweet_spot = res_m.groupby("acre_bin", observed=True)["is_missing_middle"].agg(["mean", "count"])
lot_sweet_spot["mm_share_pct"] = (lot_sweet_spot["mean"] * 100).round(1)
print("\n=== Missing-middle share by lot size (acres) ===")
print(lot_sweet_spot[["mm_share_pct", "count"]])

sqft_bins = [0, 600, 900, 1200, 1500, 1800, 2200, 2800, 3500, 99999]
res_m["sqft_bin"] = pd.cut(res_m["building_sqft"], sqft_bins)
sqft_sweet_spot = res_m.groupby("sqft_bin", observed=True)["is_missing_middle"].agg(["mean", "count"])
sqft_sweet_spot["mm_share_pct"] = (sqft_sweet_spot["mean"] * 100).round(1)
print("\n=== Missing-middle share by building size (sqft) ===")
print(sqft_sweet_spot[["mm_share_pct", "count"]])

# ---------------------------------------------------------------------
# Zoning concentration: missing-middle share WITHIN each zoning code
# ---------------------------------------------------------------------
zone_g = res.groupby("ZONING")["is_missing_middle"].agg(["mean", "count"])
zone_g["mm_share_pct"] = (zone_g["mean"] * 100).round(1)
zone_g = zone_g[zone_g["count"] >= 30].sort_values("mm_share_pct", ascending=False)
print("\n=== Missing-middle concentration by zoning code (min n=30) ===")
print(zone_g[["mm_share_pct", "count"]].head(20))

out = {
    "correlations": corr_results,
    "lot_size_sweet_spot": [
        {"bin": str(i), "mm_share_pct": float(r["mm_share_pct"]), "n": int(r["count"])}
        for i, r in lot_sweet_spot.iterrows() if r["count"] >= 20
    ],
    "building_size_sweet_spot": [
        {"bin": str(i), "mm_share_pct": float(r["mm_share_pct"]), "n": int(r["count"])}
        for i, r in sqft_sweet_spot.iterrows() if r["count"] >= 20
    ],
    "zoning_concentration": [
        {"zoning": str(i) if str(i) != "nan" else "(unzoned)", "mm_share_pct": float(r["mm_share_pct"]), "n": int(r["count"])}
        for i, r in zone_g.iterrows()
    ],
}
with open(f"{OUT}/correlation_stats.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print("\nSaved correlation_stats.json")
