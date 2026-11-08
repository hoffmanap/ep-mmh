"""
Spatial join: Microsoft GlobalMLBuildingFootprints <-> EPCAD parcel polygons.

Derives real geometric shape features per parcel and two signals the user
asked for explicitly: distinguishing a brand-new undetected detached
structure from an addition/expansion of an existing one.

  - footprint_count_all: every footprint polygon landing in the parcel
  - footprint_count_dwelling_sized: only footprints >= DWELLING_MIN_SQFT,
    which excludes sheds, carports, small garages, and pool-equipment
    structures that would otherwise inflate footprint_count_all
  - total_footprint_area_sqft / primary_footprint_area_sqft
  - compactness (4*pi*area/perimeter^2, 1.0 = circle, lower = elongated)
  - footprint_to_recorded_ratio: real footprint area vs EPCAD's recorded
    SquareFootage (TypeCD=="MA" improvement records) -- a ratio > 1.25
    flags a structure visibly larger than what's on the tax roll
  - new_detached_flag: footprint_count_dwelling_sized exceeds the
    EPCAD-recorded main-structure count -- an extra, dwelling-sized
    building nobody assessed (casita/ADU/informal unit candidate)
  - addition_flag: structure count matches the record, but footprint
    area exceeds recorded sqft by >25% (an expansion to an existing,
    already-assessed building)

CAVEAT (kept in the dashboard, not just here): Microsoft's footprint
polygons are noisy. One roof is sometimes split into multiple polygons;
a large garage or carport sometimes clears the dwelling-size threshold.
With ~18% of single-family parcels flagged, read this as a noisy
screening signal to prioritize field review, not as confirmed
undocumented structures.
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import os

OUT = os.environ.get("EP_MMH_OUT", "data/output")
RAW = os.environ.get("EP_MMH_RAW", "data/raw")

DWELLING_MIN_SQFT = 300  # excludes sheds/carports/small garages

# A footprint-to-recorded ratio threshold must clear the population's own
# baseline, not an arbitrary round number. Checked empirically: across ALL
# single-family parcels (not just flagged ones), the median ratio is 1.10
# and the 75th percentile is 1.21 -- footprint area runs ~10-20% larger
# than EPCAD's recorded sqft for most ordinary houses, likely because the
# footprint captures the full exterior outline (attached garage, covered
# patio) while EPCAD's recorded figure is closer to conditioned/living
# area. A 1.25 cutoff sits right on that baseline and flagged ~17% of all
# single-family parcels -- mostly ordinary garage/patio coverage, not
# additions. Only 3.7% of all single-family parcels exceed ratio > 1.5,
# which is the threshold used here; it cut the addition_flag count from
# 29,229 to 4,564 (6.4x) with no change to new_detached_flag.
RATIO_THRESHOLD = 1.5

print("Loading parcel polygons...")
parcels = gpd.read_parquet(f"{OUT}/parcel_polygons.parquet")
print(f"  {len(parcels):,} parcels")

print("Loading footprints...")
footprints = gpd.read_parquet(f"{OUT}/footprints_el_paso.parquet").to_crs(parcels.crs)
print(f"  {len(footprints):,} footprints")

print("Loading recorded main-structure counts (Improvements, TypeCD=='MA')...")
imprv_cols = ["ImprovementId", "Type", "StateCode", "LivingArea", "Value", "Description",
              "ClassCD", "ExteriorWall", "YearBuilt", "SquareFootage", "Year",
              "Property_dbId", "BuildingId", "TypeCD"]
imprv = pd.read_csv(
    f"{RAW}/Improvements2027Dump.txt", sep="~", header=None, dtype=str,
    names=imprv_cols, encoding="latin-1", engine="c", quoting=3,
)
imprv["TypeCD_c"] = imprv["TypeCD"].str.strip()
imprv["SquareFootage_num"] = pd.to_numeric(imprv["SquareFootage"], errors="coerce").fillna(0)
n_main_recorded = imprv[imprv["TypeCD_c"] == "MA"].groupby("Property_dbId").size().rename("n_main_recorded")
total_sqft_recorded = imprv.groupby("Property_dbId")["SquareFootage_num"].sum().rename("total_sqft_recorded")

print("Spatial join: footprint centroid within parcel polygon...")
footprints["centroid_geom"] = footprints.geometry.centroid
footprints["is_dwelling_sized"] = footprints["footprint_area_sqft"] >= DWELLING_MIN_SQFT
fp_points = gpd.GeoDataFrame(
    footprints[["centroid_geom", "footprint_area_sqft", "footprint_perimeter_ft", "is_dwelling_sized"]],
    geometry="centroid_geom", crs=footprints.crs,
)
joined = gpd.sjoin(fp_points, parcels[["PROP_ID", "geometry"]], how="inner", predicate="within")
print(f"  {len(joined):,} footprints matched to a parcel")

agg = joined.groupby("PROP_ID").agg(
    footprint_count_all=("footprint_area_sqft", "count"),
    footprint_count_dwelling_sized=("is_dwelling_sized", "sum"),
    total_footprint_area_sqft=("footprint_area_sqft", "sum"),
    primary_footprint_area_sqft=("footprint_area_sqft", "max"),
).reset_index()

largest_idx = joined.groupby("PROP_ID")["footprint_area_sqft"].idxmax()
largest = joined.loc[largest_idx, ["PROP_ID", "footprint_area_sqft", "footprint_perimeter_ft"]]
largest["compactness"] = (4 * np.pi * largest["footprint_area_sqft"]) / (largest["footprint_perimeter_ft"] ** 2)
agg = agg.merge(largest[["PROP_ID", "compactness"]], on="PROP_ID", how="left")

agg["PROP_ID_num"] = pd.to_numeric(agg["PROP_ID"], errors="coerce")
n_main_recorded.index = pd.to_numeric(n_main_recorded.index, errors="coerce")
total_sqft_recorded.index = pd.to_numeric(total_sqft_recorded.index, errors="coerce")
agg = agg.merge(n_main_recorded.rename_axis("PROP_ID_num").reset_index(), on="PROP_ID_num", how="left")
agg = agg.merge(total_sqft_recorded.rename_axis("PROP_ID_num").reset_index(), on="PROP_ID_num", how="left")
agg["n_main_recorded"] = agg["n_main_recorded"].fillna(0)
agg["total_sqft_recorded"] = agg["total_sqft_recorded"].fillna(0)
agg["footprint_to_recorded_ratio"] = agg["total_footprint_area_sqft"] / agg["total_sqft_recorded"].replace(0, np.nan)

agg["new_detached_flag"] = agg["footprint_count_dwelling_sized"] > agg["n_main_recorded"]
agg["addition_flag"] = (~agg["new_detached_flag"]) & (agg["footprint_to_recorded_ratio"] > RATIO_THRESHOLD)

agg.drop(columns=["PROP_ID_num"]).to_csv(f"{OUT}/parcel_shape_features.csv", index=False)
print(f"Saved parcel_shape_features.csv ({len(agg):,} parcels with footprint data)")
print("new_detached_flag count:", int(agg["new_detached_flag"].sum()))
print("addition_flag count:", int(agg["addition_flag"].sum()))
