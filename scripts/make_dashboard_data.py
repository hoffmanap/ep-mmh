"""
Aggregate scripts/build_dataset.py's output (data/output/parcels_analysis.csv)
into the JSON + GeoJSON files the dashboard (index.html) consumes:
  data/dashboard_stats.json
  data/parcels_missing_middle.geojson
"""
import os
import pandas as pd
import json

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "data", "output")
DATA = os.path.join(HERE, "data")

RESIDENTIAL_TYPES = [
    "Single-Family Detached", "Mobile/Manufactured Home", "Duplex", "Townhouse",
    "Condominium", "Apartment/Multifamily", "Multi-Unit Residential (Other)",
    "Apartment (4-6 units)", "Apartment (7-30 units)", "Apartment (31-99 units)", "Apartment (100+ units)",
    "Farm/Ranch House", "Farm/Ranch Mobile Home",
]


def main():
    df = pd.read_csv(f"{OUT}/parcels_analysis.csv")
    res = df[df["typology"].isin(RESIDENTIAL_TYPES)].copy()

    total = len(res)
    mm = int(res["is_missing_middle"].sum())
    stats = {
        "total_residential_parcels": int(total),
        "missing_middle_parcels": mm,
        "missing_middle_share_pct": round(mm / total * 100, 2),
        "typology_counts": res["typology"].value_counts().to_dict(),
    }

    res_d = res.dropna(subset=["dist_to_core_mi"]).copy()
    bins = list(range(0, 16))
    res_d["dist_bin"] = pd.cut(res_d["dist_to_core_mi"], bins=bins, right=False)
    grouped = res_d.groupby("dist_bin", observed=True)["is_missing_middle"].agg(["sum", "count"])
    grouped["share_pct"] = (grouped["sum"] / grouped["count"] * 100).round(2)
    distance_gradient = [
        {"bin_start": interval.left, "bin_end": interval.right,
         "mm_share_pct": float(row["share_pct"]) if pd.notna(row["share_pct"]) else None,
         "n": int(row["count"])}
        for interval, row in grouped.iterrows()
    ]

    res_y = res.dropna(subset=["year_built"]).copy()
    res_y = res_y[(res_y["year_built"] >= 1900) & (res_y["year_built"] <= 2026)]
    res_y["decade"] = (res_y["year_built"] // 10 * 10).astype(int)
    decade_typology = res_y.groupby(["decade", "typology"]).size().unstack(fill_value=0)
    decade_data = {
        "decades": decade_typology.index.tolist(),
        "typologies": decade_typology.columns.tolist(),
        "matrix": decade_typology.values.tolist(),
    }

    res_z = res.dropna(subset=["ZONING"]).copy()
    res_z["ZONING"] = res_z["ZONING"].astype(str).str.strip()
    res_z = res_z[res_z["ZONING"] != "NULL"]
    top_zones = res_z["ZONING"].value_counts().head(12).index.tolist()
    res_z = res_z[res_z["ZONING"].isin(top_zones)]
    zoning_typology = res_z.groupby(["ZONING", "typology"]).size().unstack(fill_value=0)
    zoning_data = {
        "zones": zoning_typology.index.tolist(),
        "typologies": zoning_typology.columns.tolist(),
        "matrix": zoning_typology.values.tolist(),
    }

    with open(f"{DATA}/dashboard_stats.json", "w") as f:
        json.dump({
            "stats": stats,
            "distance_gradient": distance_gradient,
            "decade_typology": decade_data,
            "zoning_typology": zoning_data,
        }, f, indent=2, default=str)

    mm_df = res[res["is_missing_middle"] & res["lat"].notna() & res["lon"].notna()]
    features = []
    for _, r in mm_df.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(r["lon"], 6), round(r["lat"], 6)]},
            "properties": {
                "typology": r["typology"],
                "year_built": int(r["year_built"]) if pd.notna(r["year_built"]) else None,
                "dist_to_core_mi": round(float(r["dist_to_core_mi"]), 2) if pd.notna(r["dist_to_core_mi"]) else None,
                "zoning": r["ZONING"] if pd.notna(r["ZONING"]) else None,
            },
        })
    with open(f"{DATA}/parcels_missing_middle.geojson", "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)

    print(f"Residential parcels: {total:,}")
    print(f"Missing-middle parcels: {mm:,} ({stats['missing_middle_share_pct']}%)")
    print(f"GeoJSON features (missing-middle only): {len(features):,}")
    print("Saved data/dashboard_stats.json and data/parcels_missing_middle.geojson")


if __name__ == "__main__":
    main()
