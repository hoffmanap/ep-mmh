"""
Spatial join: First American property/mortgage records (Prop48141.txt,
397K rows, pipe-delimited) <-> EPCAD parcel-level analysis
(parcels_analysis.csv, from build_dataset.py).

Join strategy: nearest-neighbor match on lat/lon (KDTree), not APN --
First American's APN format ("08MH-000-0000-0120") does not match
EPCAD's GEO_ID scheme ("C34099907500100"), and TaxAccountNumber does
not match EPCAD's PROP_ID either. Both datasets carry a point location
for the same physical parcel, so a tight-tolerance nearest-neighbor
match is the reliable join key.
"""
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

RAW_DIR = "/sessions/rcw-019pt16gylbtcckzycdh9ppp/mnt/data/first_american"
OUT_DIR = "/sessions/rcw-019pt16gylbtcckzycdh9ppp/mnt/data/el_paso_mm/output"

print("Loading EPCAD parcels_analysis.csv...")
epcad = pd.read_csv(f"{OUT_DIR}/parcels_analysis.csv")
print(f"  {len(epcad):,} EPCAD parcels")

print("Loading First American Prop48141.txt (selected columns)...")
FA_COLS = [
    "PropertyID", "APN", "SitusFullStreetAddress", "SitusLatitude", "SitusLongitude",
    "LandUseCode", "StateLandUseCode",
    "AssdTotalValue", "MarketTotalValue", "CurrentAVMValue",
    "YearBuilt", "SumResidentialUnits", "SumBuildingsNbr",
    "OwnerOccupied", "MailingCity", "MailingState", "SitusCity",
    "CurrentSalesPrice", "CurrentSaleRecordingDate",
    "TotalOpenLienNbr", "TotalOpenLienAmt",
    "Mtg1RecordingDate", "Mtg1LoanAmt", "Mtg1Lender", "Mtg1Term",
    "Mtg1LoanType", "Mtg1TypeFinancing", "Mtg1LienPosition",
    "Mtg2LoanAmt", "Mtg2Lender",
    "IsListedFlag", "VacantFlag",
]
fa = pd.read_csv(
    f"{RAW_DIR}/Prop48141.txt", sep="|", usecols=FA_COLS,
    dtype=str, encoding="latin-1", engine="c", quoting=3,
    low_memory=False,
)
print(f"  {len(fa):,} First American records")

fa["lat"] = pd.to_numeric(fa["SitusLatitude"], errors="coerce")
fa["lon"] = pd.to_numeric(fa["SitusLongitude"], errors="coerce")
fa_geo = fa[(fa["lat"].notna()) & (fa["lon"].notna()) & (fa["lat"] != 0) & (fa["lon"] != 0)].copy()
print(f"  {len(fa_geo):,} records with valid lat/lon")

LAT0 = 31.76
MI_PER_DEG_LAT = 69.0
MI_PER_DEG_LON = 69.0 * np.cos(np.radians(LAT0))

def to_xy(lat, lon):
    return np.column_stack([lon * MI_PER_DEG_LON, lat * MI_PER_DEG_LAT])

print("Building KDTree on First American points...")
fa_xy = to_xy(fa_geo["lat"].values, fa_geo["lon"].values)
tree = cKDTree(fa_xy)

epcad_valid = epcad.dropna(subset=["lat", "lon"]).copy()
epcad_xy = to_xy(epcad_valid["lat"].values, epcad_valid["lon"].values)

print("Querying nearest First American record for each EPCAD parcel...")
TOLERANCE_MI = 0.02
dist, idx = tree.query(epcad_xy, k=1, distance_upper_bound=TOLERANCE_MI)

matched = dist != np.inf
print(f"  {matched.sum():,} of {len(epcad_valid):,} EPCAD parcels matched within {TOLERANCE_MI*5280:.0f} ft")

fa_matched_idx = idx[matched]
fa_subset = fa_geo.iloc[fa_matched_idx].reset_index(drop=True)
fa_subset.columns = [f"fa_{c}" for c in fa_subset.columns]

epcad_matched = epcad_valid.iloc[matched].reset_index(drop=True)
joined = pd.concat([epcad_matched, fa_subset], axis=1)
joined["fa_match_dist_ft"] = dist[matched] * 5280

joined.to_csv(f"{OUT_DIR}/parcels_with_mortgage.csv", index=False)
print(f"Saved parcels_with_mortgage.csv ({len(joined):,} rows)")

print("\nSample matches:")
print(joined[["PROP_ID", "GEO_ID", "typology", "fa_APN", "fa_SitusFullStreetAddress", "fa_match_dist_ft"]].head(8))
