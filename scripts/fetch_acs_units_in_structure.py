"""
Pull ACS 5-year table B25024 (Units in Structure) for every census tract
in El Paso County, TX via Census Reporter's open API (no key required,
unlike the Census Bureau's own API which now gates all requests behind
a key). This is the external validation benchmark: an independent survey
estimate of how many housing units sit in 2+-unit structures, to compare
against EPCAD's own parcel-level classification and the structural
candidate detector's flagged counts.

B25024 categories: 001=total, 002=1-unit detached, 003=1-unit attached,
004=2 units, 005=3-4, 006=5-9, 007=10-19, 008=20-49, 009=50+,
010=mobile home, 011=boat/RV/van.
"""
import json
import urllib.request
import os

OUT = os.environ.get("EP_MMH_OUT", "data")

GEO_ID = "05000US48141"  # El Paso County, TX FIPS
url = f"https://api.censusreporter.org/1.0/data/show/latest?table_ids=B25024&geo_ids=140|{GEO_ID}"

req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research script)"})
with urllib.request.urlopen(req, timeout=30) as resp:
    raw = json.loads(resp.read())

print(f"Release: {raw['release']['name']}, {len(raw['data'])} tracts")

rows = []
total_units = total_single = total_mm = total_mobile = 0
for geoid, tables in raw["data"].items():
    tract_fips = geoid.split("US")[1]
    est = tables["B25024"]["estimate"]
    total = est.get("B25024001", 0) or 0
    single = (est.get("B25024002", 0) or 0) + (est.get("B25024003", 0) or 0)
    mm_2plus = sum((est.get(f"B25024{i:03d}", 0) or 0) for i in [4, 5, 6, 7, 8, 9])
    mobile = est.get("B25024010", 0) or 0
    total_units += total
    total_single += single
    total_mm += mm_2plus
    total_mobile += mobile
    rows.append({
        "tract_fips": tract_fips,
        "total_units": total,
        "single_family_units": single,
        "missing_middle_units_2plus": mm_2plus,
        "mobile_home_units": mobile,
    })

summary = {
    "source": "Census Reporter API, ACS 2020-2024 5-year, table B25024",
    "geography": "El Paso County, TX (FIPS 48141)",
    "n_tracts": len(rows),
    "total_housing_units": total_units,
    "total_single_family_units": total_single,
    "total_missing_middle_units_2plus": total_mm,
    "total_mobile_home_units": total_mobile,
    "missing_middle_share_pct_of_units": round(100 * total_mm / total_units, 2),
    "tracts": rows,
}
with open(f"{OUT}/acs_units_in_structure.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"Total housing units: {total_units:,}")
print(f"Missing-middle (2+ units in structure): {total_mm:,} ({summary['missing_middle_share_pct_of_units']}%)")
print("Saved acs_units_in_structure.json")
