"""
Build the El Paso missing-middle housing dataset from real EPCAD data.

Primary classification: GIS parcel layer's STATE_CD (one row per parcel,
already resolved to A1/A2/B1/B2/etc by EPCAD).
Secondary enrichment: Improvements.ClassCD, joined via
  GIS.PROP_ID -> Properties.PropertyId -> Properties.dbId -> Improvements.Property_dbId
used only to sub-split B2 (apartment/multifamily) into size bands via the
QS(4-6)/QL(7-30)/QN(31-99)/QR(100+) codes.
"""
import pyogrio
import pandas as pd
import numpy as np
import json

RAW = "/sessions/rcw-019pt16gylbtcckzycdh9ppp/mnt/data/el_paso_mm/raw"
GIS = "/sessions/rcw-019pt16gylbtcckzycdh9ppp/mnt/data/el_paso_mm/gis/extracted/Parcels2026/New File Geodatabase.gdb"
OUT = "/sessions/rcw-019pt16gylbtcckzycdh9ppp/mnt/data/el_paso_mm/output"
import os
os.makedirs(OUT, exist_ok=True)

DOWNTOWN_LAT, DOWNTOWN_LON = 31.7598, -106.4887

STATE_CD_TYPOLOGY = {
    "A1": ("Single-Family Detached", 1, False),
    # A51/A53/A54/A55/A6/A7 are single-family residential subtypes EPCAD
    # tracks separately from plain A1 (manufactured-on-permanent-
    # foundation, PUD, zero-lot-line, and similar variants) -- not
    # missing-middle, but real urban residential stock that belongs in
    # the RES denominator rather than falling into "Other/Non-
    # Residential" alongside vacant lots and colonias (the "STATE_CD
    # gap" this project hit and fixed earlier: dropping these silently
    # shrank the residential-parcel denominator and distorted every
    # share-of-total stat).
    "A51": ("Single-Family Detached (Other Residential Subtype)", 1, False),
    "A53": ("Single-Family Detached (Other Residential Subtype)", 1, False),
    "A54": ("Single-Family Detached (Other Residential Subtype)", 1, False),
    "A55": ("Single-Family Detached (Other Residential Subtype)", 1, False),
    "A6": ("Single-Family Detached (Other Residential Subtype)", 1, False),
    "A7": ("Single-Family Detached (Other Residential Subtype)", 1, False),
    "A2": ("Mobile/Manufactured Home", 1, False),
    "A3": ("Condominium", 1, True),
    "A4": ("Townhouse", 1, True),
    "B1": ("Duplex", 2, True),
    "B2": ("Apartment/Multifamily", None, True),   # sub-split below via ClassCD
    # B3/B4 split out as Triplex/Fourplex rather than lumped into
    # "Multi-Unit Residential (Other)" -- Texas's statewide SPTB state-
    # code convention (same scheme taxnetusa.com documents for other TX
    # appraisal districts) assigns B3=three-family/triplex and
    # B4=four-or-more-family/fourplex. EPCAD's public site blocks
    # automated lookups (Cloudflare challenge), so this couldn't be
    # confirmed against an EPCAD-published legend directly, but it's
    # corroborated structurally here: B3/B4 parcels average noticeably
    # more "MAIN AREA" improvement records per property (2.06 / 2.57)
    # than B1 duplexes (1.48), consistent with more units per parcel.
    # B5 (n=125) and B6 (n=24) are left lumped as "Multi-Unit Residential
    # (Other)" -- too few parcels and too ambiguous a structural pattern
    # (B5's improvement-count average is actually *lower* than B4's) to
    # confidently assign a specific unit count to them.
    "B3": ("Triplex", 3, True),
    "B4": ("Fourplex", 4, True),
    "B5": ("Multi-Unit Residential (Other)", None, True),
    "B6": ("Multi-Unit Residential (Other)", None, True),
    "C1": ("Vacant Residential Lot", 0, False),
    "C2": ("Colonia Lot/Land Tract", 0, False),
    "C3": ("Vacant Rural/Recreational Lot", 0, False),
    "D1": ("Farm/Ranch Acreage", 0, False),
    "D2": ("Farm/Ranch Acreage", 0, False),
    "D3": ("Farm/Ranch Acreage", 0, False),
    "D4": ("Farm/Ranch Acreage", 0, False),
    "E1": ("Farm/Ranch House", 1, False),
    "E2": ("Farm/Ranch Mobile Home", 1, False),
    "E3": ("Farm/Ranch Other Improvement", 0, False),
    "F1": ("Commercial", None, False),
    "F2": ("Industrial", None, False),
}

APT_SIZE_BAND = {
    "QS": ("Apartment (4-6 units)", 5),
    "QL": ("Apartment (7-30 units)", 18),
    "QN": ("Apartment (31-99 units)", 65),
    "QR": ("Apartment (100+ units)", 150),
}

def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

print("Loading GIS parcel layer...")
gdf = pyogrio.read_dataframe(
    GIS, layer="Parcel_2026",
    columns=["PROP_ID", "GEO_ID", "STATE_CD", "ZONING", "YR_BLT", "SITUS_CITY", "LAND_ACRES",
             "EXEMPTIONS", "ADDR_CITY", "ADDR_STATE", "LEGAL_DESC"],
)
print(f"  {len(gdf):,} parcels loaded, CRS={gdf.crs}")

# Keep the real parcel polygons (not just the centroid) for the building-
# footprint spatial join -- saved separately so this CSV-producing script
# doesn't carry geometry downstream.
print("Saving parcel polygons (for footprint join)...")
gdf[["PROP_ID", "geometry"]].to_parquet(f"{OUT}/parcel_polygons.parquet")

# reproject to WGS84 for lat/lon
gdf = gdf.to_crs(epsg=4326)
gdf["centroid"] = gdf.geometry.centroid
gdf["lon"] = gdf["centroid"].x
gdf["lat"] = gdf["centroid"].y

gdf["STATE_CD_clean"] = gdf["STATE_CD"].astype(str).str.strip()

# ---------------------------------------------------------------------
# Ownership/exemption signals, straight from EPCAD's own 2026 parcel
# layer (current, no First American join needed). Texas homestead
# exemption legally requires owner-occupancy as primary residence, so
# has_homestead is the cleanest owner-occupied signal available.
# ---------------------------------------------------------------------
gdf["has_homestead"] = gdf["EXEMPTIONS"].astype(str).str.contains("HS", na=False)
gdf["has_disabled_vet_exemption"] = gdf["EXEMPTIONS"].astype(str).str.contains("DV", na=False)
gdf["tax_exempt"] = gdf["EXEMPTIONS"].astype(str).str.contains("EX", na=False)
addr_city = gdf["ADDR_CITY"].astype(str).str.strip().str.upper()
situs_city = gdf["SITUS_CITY"].astype(str).str.strip().str.upper()
addr_state = gdf["ADDR_STATE"].astype(str).str.strip().str.upper()
gdf["mailing_mismatch"] = (addr_city != situs_city) | (addr_state != "TX")

# ---------------------------------------------------------------------
# Legal-description text flags: catches cases where EPCAD's free-text
# legal description mentions a typology EPCAD's own STATE_CD doesn't
# reflect (e.g. "CASITA" on a parcel coded A1 single-family) -- a
# direct textual signal of shadow/unlabeled missing-middle housing,
# independent of the structural-feature classifier.
# ---------------------------------------------------------------------
legal_upper = gdf["LEGAL_DESC"].astype(str).str.upper()
gdf["legal_desc_mentions_casita"] = legal_upper.str.contains("CASITA", na=False)
gdf["legal_desc_mentions_duplex"] = legal_upper.str.contains("DUPLEX", na=False)
gdf["legal_desc_mentions_condo"] = legal_upper.str.contains("CONDO", na=False)

def classify(code):
    if code in STATE_CD_TYPOLOGY:
        return STATE_CD_TYPOLOGY[code]
    return ("Other/Non-Residential", None, False)

classified = gdf["STATE_CD_clean"].apply(classify)
gdf["typology"] = classified.apply(lambda t: t[0])
gdf["approx_units"] = classified.apply(lambda t: t[1])
gdf["is_missing_middle"] = classified.apply(lambda t: t[2])

gdf["YR_BLT_num"] = pd.to_numeric(gdf["YR_BLT"], errors="coerce")

# ---------------------------------------------------------------------
# Enrich B2 parcels with apartment-size band from Improvements.ClassCD
# ---------------------------------------------------------------------
print("Loading Properties (for PropertyId -> dbId mapping)...")
props = pd.read_csv(
    f"{RAW}/Properties2027Dump.txt", sep="~", header=None, dtype=str,
    usecols=[0, 1], names=["dbId", "PropertyId"] + [f"c{i}" for i in range(2, 19)],
    encoding="latin-1", engine="c", quoting=3,
).iloc[:, :2]
props.columns = ["dbId", "PropertyId"]
props["PropertyId"] = props["PropertyId"].str.strip()
props["dbId"] = props["dbId"].str.strip()

print("Loading Improvements (ClassCD for B2 parcels)...")
imprv_cols = ["ImprovementId","Type","StateCode","LivingArea","Value","Description",
              "ClassCD","ExteriorWall","YearBuilt","SquareFootage","Year",
              "Property_dbId","BuildingId","TypeCD"]
imprv = pd.read_csv(
    f"{RAW}/Improvements2027Dump.txt", sep="~", header=None, dtype=str,
    names=imprv_cols, encoding="latin-1", engine="c", quoting=3,
)
imprv["ClassCD_clean"] = imprv["ClassCD"].str.strip()
imprv["prefix2"] = imprv["ClassCD_clean"].str[:2]

# only keep rows with an apartment-size-band class code, largest LivingArea per property
apt_rows = imprv[imprv["prefix2"].isin(APT_SIZE_BAND.keys())].copy()
apt_rows["LivingArea_num"] = pd.to_numeric(apt_rows["LivingArea"], errors="coerce")
apt_primary = (
    apt_rows.sort_values("LivingArea_num", ascending=False)
    .drop_duplicates(subset="Property_dbId", keep="first")
)
apt_primary["apt_band"] = apt_primary["prefix2"].map(lambda p: APT_SIZE_BAND[p][0])
apt_primary["apt_units"] = apt_primary["prefix2"].map(lambda p: APT_SIZE_BAND[p][1])

# join: GIS.PROP_ID -> Improvements.Property_dbId (confirmed these are
# the SAME id space, i.e. Property_dbId == Properties.PropertyId directly --
# no need to go through Properties.dbId at all)
gis_prop_id = gdf["PROP_ID"].round().astype("Int64").astype(str)
gis_prop_id = gis_prop_id.replace("<NA>", "")
gdf = gdf.assign(PROP_ID_str=gis_prop_id).merge(
    apt_primary[["Property_dbId", "apt_band", "apt_units"]],
    left_on="PROP_ID_str", right_on="Property_dbId", how="left",
)

# apply sub-split: where typology == Apartment/Multifamily and we have a band, use it
has_band = gdf["apt_band"].notna()
gdf.loc[has_band, "typology"] = gdf.loc[has_band, "apt_band"]
gdf.loc[has_band, "approx_units"] = gdf.loc[has_band, "apt_units"]

print("Computing distance to downtown core...")
gdf["dist_to_core_mi"] = haversine_miles(gdf["lat"], gdf["lon"], DOWNTOWN_LAT, DOWNTOWN_LON)

# ---------------------------------------------------------------------
# Save full analysis table (CSV, no geometry) for scripts/figures
# ---------------------------------------------------------------------
analysis_cols = ["PROP_ID", "GEO_ID", "STATE_CD_clean", "ZONING", "typology",
                  "approx_units", "is_missing_middle", "YR_BLT_num",
                  "dist_to_core_mi", "lat", "lon", "LAND_ACRES", "SITUS_CITY",
                  "has_homestead", "has_disabled_vet_exemption", "tax_exempt",
                  "mailing_mismatch", "legal_desc_mentions_casita",
                  "legal_desc_mentions_duplex", "legal_desc_mentions_condo"]
analysis_df = gdf[analysis_cols].rename(columns={"STATE_CD_clean": "state_cd", "YR_BLT_num": "year_built"})
analysis_df.to_csv(f"{OUT}/parcels_analysis.csv", index=False)
print(f"Saved parcels_analysis.csv ({len(analysis_df):,} rows)")

print(gdf["typology"].value_counts())
