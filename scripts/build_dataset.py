"""
Build the El Paso missing-middle housing dataset from real EPCAD data.

Primary classification: GIS parcel layer's STATE_CD (one row per parcel,
already resolved to A1/A2/B1/B2/etc by EPCAD).
Secondary enrichment: Improvements.ClassCD, joined via
  GIS.PROP_ID -> Properties.PropertyId -> Properties.dbId -> Improvements.Property_dbId
used only to sub-split B2 (apartment/multifamily) into size bands via the
QS(4-6)/QL(7-30)/QN(31-99)/QR(100+) codes.

Inputs expected (see README for download links):
  data/raw/Properties2027Dump.txt
  data/raw/Improvements2027Dump.txt
  data/gis/Parcels2026/New File Geodatabase.gdb

Output:
  data/output/parcels_analysis.csv
"""
import os
import pyogrio
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "data", "raw")
GIS = os.path.join(HERE, "data", "gis", "Parcels2026", "New File Geodatabase.gdb")
OUT = os.path.join(HERE, "data", "output")
os.makedirs(OUT, exist_ok=True)

DOWNTOWN_LAT, DOWNTOWN_LON = 31.7598, -106.4887

# EPCAD state_cd -> (typology label, approx unit count, is_missing_middle)
# Confirmed against EPCAD's "2026 Codes for Export Tables" workbook
# (ptd_state_code table) and validated against real StateCode frequency
# counts in the Improvements export.
STATE_CD_TYPOLOGY = {
    "A1": ("Single-Family Detached", 1, False),
    "A2": ("Mobile/Manufactured Home", 1, False),
    "A3": ("Condominium", 1, True),
    "A4": ("Townhouse", 1, True),
    "B1": ("Duplex", 2, True),
    "B2": ("Apartment/Multifamily", None, True),   # sub-split below via ClassCD
    "B3": ("Multi-Unit Residential (Other)", None, True),
    "B4": ("Multi-Unit Residential (Other)", None, True),
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

# Improvements.ClassCD prefixes used to sub-split generic B2
# (Apartment/Multifamily) parcels into size bands.
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
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def classify(code):
    return STATE_CD_TYPOLOGY.get(code, ("Other/Non-Residential", None, False))


def main():
    print("Loading GIS parcel layer...")
    gdf = pyogrio.read_dataframe(
        GIS, layer="Parcel_2026",
        columns=["PROP_ID", "GEO_ID", "STATE_CD", "ZONING", "YR_BLT", "SITUS_CITY", "LAND_ACRES"],
    )
    print(f"  {len(gdf):,} parcels loaded, CRS={gdf.crs}")

    gdf = gdf.to_crs(epsg=4326)
    gdf["centroid"] = gdf.geometry.centroid
    gdf["lon"] = gdf["centroid"].x
    gdf["lat"] = gdf["centroid"].y

    gdf["STATE_CD_clean"] = gdf["STATE_CD"].astype(str).str.strip()
    classified = gdf["STATE_CD_clean"].apply(classify)
    gdf["typology"] = classified.apply(lambda t: t[0])
    gdf["approx_units"] = classified.apply(lambda t: t[1])
    gdf["is_missing_middle"] = classified.apply(lambda t: t[2])
    gdf["YR_BLT_num"] = pd.to_numeric(gdf["YR_BLT"], errors="coerce")

    # ---- enrich B2 with apartment-size band from Improvements.ClassCD ----
    print("Loading Properties (for PropertyId -> dbId mapping)...")
    props = pd.read_csv(
        f"{RAW}/Properties2027Dump.txt", sep="~", header=None, dtype=str,
        usecols=[0, 1], names=["dbId", "PropertyId"] + [f"c{i}" for i in range(2, 19)],
        encoding="latin-1", on_bad_lines="skip", engine="python",
    ).iloc[:, :2]
    props.columns = ["dbId", "PropertyId"]
    props["PropertyId"] = props["PropertyId"].str.strip()
    props["dbId"] = props["dbId"].str.strip()

    print("Loading Improvements (ClassCD for B2 parcels)...")
    imprv_cols = ["ImprovementId", "Type", "StateCode", "LivingArea", "Value", "Description",
                  "ClassCD", "ExteriorWall", "YearBuilt", "SquareFootage", "Year",
                  "Property_dbId", "BuildingId", "TypeCD"]
    imprv = pd.read_csv(
        f"{RAW}/Improvements2027Dump.txt", sep="~", header=None, dtype=str,
        names=imprv_cols, encoding="latin-1", on_bad_lines="skip", engine="python",
    )
    imprv["ClassCD_clean"] = imprv["ClassCD"].str.strip()
    imprv["prefix2"] = imprv["ClassCD_clean"].str[:2]

    apt_rows = imprv[imprv["prefix2"].isin(APT_SIZE_BAND.keys())].copy()
    apt_rows["LivingArea_num"] = pd.to_numeric(apt_rows["LivingArea"], errors="coerce")
    apt_primary = (
        apt_rows.sort_values("LivingArea_num", ascending=False)
        .drop_duplicates(subset="Property_dbId", keep="first")
    )
    apt_primary["apt_band"] = apt_primary["prefix2"].map(lambda p: APT_SIZE_BAND[p][0])
    apt_primary["apt_units"] = apt_primary["prefix2"].map(lambda p: APT_SIZE_BAND[p][1])

    gis_prop_id = gdf["PROP_ID"].round().astype("Int64").astype(str)
    gis_prop_id = gis_prop_id.replace("<NA>", "")
    gdf = gdf.assign(PROP_ID_str=gis_prop_id).merge(
        props, left_on="PROP_ID_str", right_on="PropertyId", how="left"
    )
    gdf = gdf.merge(
        apt_primary[["Property_dbId", "apt_band", "apt_units"]],
        left_on="dbId", right_on="Property_dbId", how="left",
    )

    has_band = gdf["apt_band"].notna()
    gdf.loc[has_band, "typology"] = gdf.loc[has_band, "apt_band"]
    gdf.loc[has_band, "approx_units"] = gdf.loc[has_band, "apt_units"]

    print("Computing distance to downtown core...")
    gdf["dist_to_core_mi"] = haversine_miles(gdf["lat"], gdf["lon"], DOWNTOWN_LAT, DOWNTOWN_LON)

    analysis_cols = ["PROP_ID", "GEO_ID", "STATE_CD_clean", "ZONING", "typology",
                      "approx_units", "is_missing_middle", "YR_BLT_num",
                      "dist_to_core_mi", "lat", "lon", "LAND_ACRES", "SITUS_CITY"]
    analysis_df = gdf[analysis_cols].rename(columns={"STATE_CD_clean": "state_cd", "YR_BLT_num": "year_built"})
    analysis_df.to_csv(f"{OUT}/parcels_analysis.csv", index=False)
    print(f"Saved parcels_analysis.csv ({len(analysis_df):,} rows)")
    print(gdf["typology"].value_counts())


if __name__ == "__main__":
    main()
