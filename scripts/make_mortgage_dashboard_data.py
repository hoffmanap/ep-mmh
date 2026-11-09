"""
Aggregate the First American join (parcels_with_mortgage.csv) into
financing-related dashboard stats: investor/absentee ownership rate,
active-lien prevalence, and loan-type mix, each broken down by
missing-middle typology vs. single-family, and by distance from core
for missing-middle parcels specifically.
"""
import pandas as pd
import numpy as np
import json

OUT = "/sessions/rcw-019pt16gylbtcckzycdh9ppp/mnt/data/el_paso_mm/output"

j = pd.read_csv(f"{OUT}/parcels_with_mortgage.csv")

RESIDENTIAL_TYPES = [
    "Single-Family Detached", "Mobile/Manufactured Home", "Duplex", "Triplex", "Fourplex", "Townhouse",
    "Condominium", "Apartment/Multifamily", "Multi-Unit Residential (Other)",
    "Apartment (4-6 units)", "Apartment (7-30 units)", "Apartment (31-99 units)",
    "Apartment (100+ units)",
]
res = j[j["typology"].isin(RESIDENTIAL_TYPES)].copy()

res["investor_owned"] = res["fa_OwnerOccupied"] != "Y"
res["has_open_lien"] = pd.to_numeric(res["fa_TotalOpenLienNbr"], errors="coerce").fillna(0) > 0
res["mtg1_amt"] = pd.to_numeric(res["fa_Mtg1LoanAmt"], errors="coerce")
res["avm"] = pd.to_numeric(res["fa_CurrentAVMValue"], errors="coerce")
res["assd"] = pd.to_numeric(res["AssdTotalValue"] if "AssdTotalValue" in res.columns else np.nan, errors="coerce")

# ---------------------------------------------------------------------
# Investor ownership + active-lien rate by typology (missing-middle
# typologies vs. single-family)
# ---------------------------------------------------------------------
by_typology = (
    res.groupby("typology")
    .agg(
        n=("PROP_ID", "count"),
        investor_pct=("investor_owned", lambda s: round(s.mean() * 100, 1)),
        open_lien_pct=("has_open_lien", lambda s: round(s.mean() * 100, 1)),
        median_avm=("avm", lambda s: round(s.median(), 0) if s.notna().any() else None),
    )
    .reset_index()
    .to_dict("records")
)

# ---------------------------------------------------------------------
# Investor ownership rate for missing-middle parcels, by distance band
# (same 1-mile bins as the main distance-gradient chart)
# ---------------------------------------------------------------------
mm = res[res["is_missing_middle"] == True].dropna(subset=["dist_to_core_mi"]).copy()
bins = list(range(0, 16))
mm["dist_bin"] = pd.cut(mm["dist_to_core_mi"], bins=bins, right=False)
grouped = mm.groupby("dist_bin", observed=True).agg(
    n=("PROP_ID", "count"),
    investor_pct=("investor_owned", lambda s: round(s.mean() * 100, 1) if len(s) else None),
)
investor_by_distance = [
    {
        "bin_start": str(int(interval.left)),
        "bin_end": str(int(interval.right)),
        "investor_pct": row["investor_pct"],
        "n": int(row["n"]),
    }
    for interval, row in grouped.iterrows()
    if row["n"] >= 10
]

# ---------------------------------------------------------------------
# Financing type mix (Fixed / Adjustable / Variable) -- small sample
# (First American's TypeFinancing field is only populated where a
# concurrent mortgage was recorded with FA), but genuinely informative:
# missing-middle vs. single-family
# ---------------------------------------------------------------------
fin = res.dropna(subset=["fa_Mtg1TypeFinancing"]).copy()
fin["mm_or_sf"] = np.where(fin["is_missing_middle"] == True, "Missing-middle", "Single-family")
financing_mix = (
    fin.groupby(["mm_or_sf", "fa_Mtg1TypeFinancing"]).size().reset_index(name="n")
    .pivot(index="mm_or_sf", columns="fa_Mtg1TypeFinancing", values="n")
    .fillna(0).astype(int).to_dict("index")
)

stats = {
    "match_coverage": {
        "residential_parcels": int(len(res)),
        "first_american_dataset": "Prop48141.txt (El Paso County, FIPS 48141, First American, snapshot 2020-04-09)",
    },
    "by_typology": by_typology,
    "investor_ownership_by_distance": investor_by_distance,
    "financing_mix": financing_mix,
}

with open(f"{OUT}/mortgage_stats.json", "w") as f:
    json.dump(stats, f, indent=2, default=str)

print(json.dumps(stats, indent=2, default=str))
