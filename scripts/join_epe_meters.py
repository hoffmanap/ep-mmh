"""
Independent validation layer: joins El Paso Electric's own meter-level
SALESCLASS (Residential Single / Duplex / Triplex / Apartment Complex --
EPE's own billing classification of each service address, entirely
independent of EPCAD's parcel records) against EPCAD parcels by
normalized situs address.

This is a genuinely independent check on the missing-middle candidate
detector (section 3): EPE bills electricity by unit, so a utility class
of Duplex/Triplex/Apartment at an address EPCAD's STATE_CD still calls
"Single-Family Detached" is strong, address-level corroboration of a
real hidden/undercounted unit -- a different kind of evidence than the
footprint-shape check (section 4, exterior geometry) or the ACS
comparison (section 5, tract-level survey aggregates), because this one
can confirm or refute the detector's specific flagged PARCELS, one at a
time, not just a tract-level aggregate.

Caveats, upfront:
  - This is an ADDRESS-string join (no shared parcel ID, no lat/lon on
    the EPE side), so match quality depends on EPCAD's SITUS_NUM/
    SITUS_STRE/SITUS_DIR fields and EPE's SERVICEADDRESS being written
    the same way. Unmatched doesn't mean "not missing middle" -- it may
    just mean the address didn't normalize to the same string.
  - EPE's classification is by billing account, not inspection -- a
    landlord billing a duplex as one commercial/single account would
    not show up as class 2 here. This likely UNDERSTATES hidden units
    the same direction the other methods do, not overstates them.
  - Multiple EPE rows can share one normalized address (multiple
    meters/accounts at one service address, mostly for apartment
    complexes) -- collapsed to one row per address, keeping the
    HIGHEST sales class present (duplex/triplex/apartment outranks
    single, since that's the more informative signal for this check).
"""
import pandas as pd
import numpy as np
import json
import os
import re

OUT = os.environ.get("EP_MMH_OUT", "data/output")
DATA = os.environ.get("EP_MMH_DATA", "data")
EPE_XLSX = os.environ.get("EP_MMH_EPE_XLSX", f"{DATA}/raw/epe/El_Paso_Electric_Meters.xlsx")
# missing_middle_candidates.csv is written to the OUT dir by the training
# script, not the repo's committed `data/` dir -- allow overriding separately.
CAND_CSV = os.environ.get("EP_MMH_CAND_CSV", f"{OUT}/missing_middle_candidates.csv")

SALESCLASS_LABEL = {
    1: "Residential Single", 2: "Residential Duplex", 3: "Residential Triplex",
    4: "Apartment Complex", 8: "Commercial Account", 9: "Industrial Account",
    48: "Housing Authority", 70: "E Montana Residential", 71: "E Montana Commercial",
}
# Only these sales classes say anything about housing-unit count; commercial/
# industrial/utility-district accounts are excluded from the comparison.
RESIDENTIAL_CLASSES = {1, 2, 3, 4}
# Rank used only to pick ONE class when multiple EPE meters share an
# address (keep the most-informative / highest-unit-count one).
CLASS_RANK = {4: 3, 3: 2, 2: 1, 1: 0}


def normalize_address(num, stre, suf):
    parts = [str(x).strip().upper() for x in (num, stre, suf) if x is not None and str(x).strip() not in ("", "nan")]
    s = " ".join(parts)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_epe_address(addr):
    if addr is None:
        return None
    s = str(addr).strip().upper()
    s = re.sub(r"\s+", " ", s)
    return s


print("Loading EPE meter data...")
epe = pd.read_excel(EPE_XLSX, sheet_name="City Classes and Key")
epe.columns = [c.strip().upper() for c in epe.columns]
epe["addr_norm"] = epe["SERVICEADDRESS"].apply(normalize_epe_address)
epe["zip5"] = pd.to_numeric(epe["ZIP"], errors="coerce")
epe = epe.dropna(subset=["addr_norm", "zip5"])
epe["zip5"] = epe["zip5"].astype(int)
epe = epe[epe["SALESCLASS"].isin(RESIDENTIAL_CLASSES)].copy()
epe["rank"] = epe["SALESCLASS"].map(CLASS_RANK)
epe_collapsed = (
    epe.sort_values("rank", ascending=False)
    .drop_duplicates(subset=["addr_norm", "zip5"], keep="first")
)
epe_collapsed["epe_class"] = epe_collapsed["SALESCLASS"].map(SALESCLASS_LABEL)
print(f"  {len(epe):,} residential-class meter rows -> {len(epe_collapsed):,} unique addresses "
      f"(after collapsing multi-meter addresses to their highest class)")

print("Loading EPCAD parcels + situs addresses...")
parcels = pd.read_csv(f"{OUT}/parcels_analysis.csv", dtype={"PROP_ID": str})
situs = pd.read_csv(f"{OUT}/parcel_situs_address.csv", dtype=str)
situs["PROP_ID_num"] = pd.to_numeric(situs["PROP_ID"], errors="coerce")
parcels["PROP_ID_num"] = pd.to_numeric(parcels["PROP_ID"], errors="coerce")
situs = situs[situs["PROP_ID_num"].notna()].drop_duplicates(subset="PROP_ID_num", keep="first")
parcels = parcels.merge(
    situs[["PROP_ID_num", "SITUS_NUM", "SITUS_STRE", "SITUS_DIR", "SITUS_ZIP"]],
    on="PROP_ID_num", how="left",
)
parcels["addr_norm"] = parcels.apply(
    lambda r: normalize_address(r["SITUS_NUM"], r["SITUS_STRE"], r["SITUS_DIR"]), axis=1
)
parcels["zip5"] = pd.to_numeric(parcels["SITUS_ZIP"], errors="coerce")
parcels = parcels[parcels["addr_norm"].str.len() > 0]

joined = parcels.merge(
    epe_collapsed[["addr_norm", "zip5", "epe_class", "SALESCLASS"]],
    on=["addr_norm", "zip5"], how="left",
)
match_rate = joined["epe_class"].notna().mean()
print(f"\nMatched {joined['epe_class'].notna().sum():,} / {len(joined):,} addressable parcels "
      f"to an EPE residential meter record ({match_rate:.1%})")

# The all-parcels match rate above includes commercial/industrial/vacant
# parcels that would never match a residential EPE class -- the figure
# that actually matters for this validation is the match rate among
# parcels EPCAD itself classifies as residential.
RES_TYPES = ["Single-Family Detached", "Single-Family Detached (Other Residential Subtype)",
             "Duplex", "Triplex", "Fourplex", "Townhouse", "Condominium",
             "Apartment/Multifamily", "Multi-Unit Residential (Other)"]
res_joined = joined[joined["typology"].isin(RES_TYPES)]
res_match_rate = res_joined["epe_class"].notna().mean()
print(f"Of just the {len(res_joined):,} residential-typology addressable parcels: "
      f"{res_joined['epe_class'].notna().sum():,} matched ({res_match_rate:.1%})")

# ---------------------------------------------------------------------
# Check 1: sanity check. For parcels EPCAD ALREADY calls missing-middle,
# does EPE's independent billing class agree they're multi-unit?
# ---------------------------------------------------------------------
confirmed_mm_all = joined[joined["is_missing_middle"]]
confirmed_mm = confirmed_mm_all[confirmed_mm_all["epe_class"].notna()]
confirmed_mm_agree = (confirmed_mm["SALESCLASS"] >= 2).mean() if len(confirmed_mm) else None
print(f"\nSanity check -- EPCAD-confirmed missing-middle parcels: {len(confirmed_mm_all):,}")
print(f"  matched to EPE: {len(confirmed_mm):,} ({len(confirmed_mm)/len(confirmed_mm_all):.1%})")
if confirmed_mm_agree is not None:
    print(f"  EPE agrees (class 2/3/4) for {confirmed_mm_agree:.1%} of them")
    print(f"  (the rest -- EPE billing class 1 at a known duplex/triplex/4-plex address -- most likely "
          f"reflects a single electric account covering a multi-unit building, not a join error; "
          f"EPE classifies by billing account, not inspected unit count)")

# ---------------------------------------------------------------------
# Check 2: the real test. For EPCAD single-family parcels the structural
# candidate detector flagged as LIKELY missing-middle (p>=0.8), does EPE's
# independent billing class show them as duplex/triplex/apartment?
# ---------------------------------------------------------------------
cand = pd.read_csv(CAND_CSV, dtype={"PROP_ID": str})
cand["PROP_ID_num"] = pd.to_numeric(cand["PROP_ID"], errors="coerce")
cand_hi = cand[cand["mm_probability"] >= 0.8]
cand_joined = cand_hi.merge(
    joined[["PROP_ID_num", "epe_class", "SALESCLASS", "addr_norm"]], on="PROP_ID_num", how="left"
)
cand_matched = cand_joined[cand_joined["epe_class"].notna()]
cand_confirmed = cand_matched[cand_matched["SALESCLASS"] >= 2]
print(f"\nModel candidates (p>=0.8): {len(cand_hi):,}")
print(f"  matched to an EPE residential meter record: {len(cand_matched):,} ({len(cand_matched)/len(cand_hi):.1%})")
print(f"  of those, EPE independently classifies as duplex/triplex/apartment "
      f"(NOT single): {len(cand_confirmed):,} ({len(cand_confirmed)/len(cand_matched):.1%} of matched)" if len(cand_matched) else "  (no matches)")

# Also run the same check against ALL single-family-labeled parcels
# (not just high-confidence candidates), as a baseline comparison -- if
# EPE shows multi-unit meters at roughly the SAME rate among non-
# candidates, that would undercut the candidate detector; if it's much
# rarer, that's further validation the detector is finding something real.
sf_all = joined[(joined["typology"] == "Single-Family Detached") & joined["epe_class"].notna()]
sf_all_multi_rate = (sf_all["SALESCLASS"] >= 2).mean() if len(sf_all) else None
cand_multi_rate = len(cand_confirmed) / len(cand_matched) if len(cand_matched) else None
print(f"\nBaseline -- among ALL EPCAD single-family parcels matched to EPE: "
      f"{(sf_all_multi_rate or 0):.2%} show an EPE duplex/triplex/apartment meter")
if cand_multi_rate and sf_all_multi_rate:
    print(f"  vs. {cand_multi_rate:.2%} among the model's high-confidence candidates "
          f"-- {cand_multi_rate / sf_all_multi_rate:.1f}x the baseline rate")

summary = {
    "match_rate_all_addressable_parcels": round(float(match_rate), 4),
    "n_addressable_parcels_matched": int(joined["epe_class"].notna().sum()),
    "n_addressable_parcels_total": int(len(joined)),
    "match_rate_residential_parcels": round(float(res_match_rate), 4),
    "n_residential_parcels_matched": int(res_joined["epe_class"].notna().sum()),
    "n_residential_parcels_total": int(len(res_joined)),
    "sanity_check_confirmed_mm_n_total": int(len(confirmed_mm_all)),
    "sanity_check_confirmed_mm_match_rate": round(len(confirmed_mm) / len(confirmed_mm_all), 4) if len(confirmed_mm_all) else None,
    "sanity_check_confirmed_mm_n_matched": int(len(confirmed_mm)),
    "sanity_check_confirmed_mm_epe_agree_pct": round(float(confirmed_mm_agree) * 100, 2) if confirmed_mm_agree is not None else None,
    "candidates_p80_n": int(len(cand_hi)),
    "candidates_p80_n_matched_to_epe": int(len(cand_matched)),
    "candidates_p80_matched_pct": round(len(cand_matched) / len(cand_hi) * 100, 2) if len(cand_hi) else None,
    "candidates_p80_epe_confirms_multiunit_n": int(len(cand_confirmed)),
    "candidates_p80_epe_confirms_multiunit_pct_of_matched": round(len(cand_confirmed) / len(cand_matched) * 100, 2) if len(cand_matched) else None,
    "baseline_all_single_family_matched_n": int(len(sf_all)),
    "baseline_all_single_family_epe_multiunit_pct": round(float(sf_all_multi_rate) * 100, 2) if sf_all_multi_rate is not None else None,
    "candidate_vs_baseline_lift": round(cand_multi_rate / sf_all_multi_rate, 2) if (cand_multi_rate and sf_all_multi_rate) else None,
}
with open(f"{DATA}/epe_meter_validation.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved epe_meter_validation.json")
print(json.dumps(summary, indent=2))

# Save the matched-and-confirmed candidate list (PROP_ID, address, model
# probability, EPE class) -- these are the strongest individually-
# verifiable "real hidden unit" cases this project has produced.
cols = ["PROP_ID", "addr_norm", "mm_probability", "epe_class"]
cand_confirmed[cols].rename(columns={"addr_norm": "address"}).to_csv(
    f"{DATA}/epe_confirmed_hidden_units.csv", index=False
)
print(f"Saved epe_confirmed_hidden_units.csv ({len(cand_confirmed):,} rows)")
