"""
Load Microsoft GlobalMLBuildingFootprints tiles covering El Paso County,
build a GeoDataFrame, and save as a working file for the parcel spatial
join (see shape_features.py).

Source: https://github.com/microsoft/GlobalMLBuildingFootprints
Partitioned by Bing Maps quadkey (tile level 9) as gzipped
line-delimited GeoJSON, linked via a dataset-links.csv manifest. El
Paso County is covered by 4 L9 quadkeys (computed from its bounding
box with the standard Bing tile-system lat/lon <-> quadkey math) --
only those 4 tiles (~40MB total) need to be downloaded, not the whole
state.
"""
import gzip
import json
import os
import geopandas as gpd
from shapely.geometry import shape

DIR = os.environ.get("EP_MMH_FOOTPRINT_TILES", "data/gis/footprints")
OUT = os.environ.get("EP_MMH_OUT", "data/output")

# the 4 Bing Maps L9 quadkeys covering El Paso County's bounding box
TILES = ["023103222", "023103223", "023121000", "023121001"]

records = []
for t in TILES:
    with gzip.open(f"{DIR}/{t}.csv.gz", "rt") as f:
        for line in f:
            rec = json.loads(line)
            records.append({
                "height": rec["properties"].get("height"),
                "confidence": rec["properties"].get("confidence"),
                "geometry": shape(rec["geometry"]),
            })
    print(f"  loaded tile {t}: running total {len(records):,}")

gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
print(f"Total footprints loaded: {len(gdf):,}")

# shape metrics in a projected CRS (feet) to match EPCAD's parcel layer
gdf_proj = gdf.to_crs(epsg=2277)  # Texas State Plane Central, feet
gdf["footprint_area_sqft"] = gdf_proj.geometry.area
gdf["footprint_perimeter_ft"] = gdf_proj.geometry.length
gdf["centroid_lon"] = gdf.geometry.centroid.x
gdf["centroid_lat"] = gdf.geometry.centroid.y

# drop tiny slivers/artifacts
gdf = gdf[gdf["footprint_area_sqft"] > 30].reset_index(drop=True)
print(f"After filtering slivers: {len(gdf):,}")

gdf.to_parquet(f"{OUT}/footprints_el_paso.parquet")
print("saved footprints_el_paso.parquet")
