"""
Train a classifier to predict missing-middle housing from STRUCTURAL
features alone (no STATE_CD, no ClassCD) -- then score every parcel
EPCAD labeled single-family (A1) to find candidates the appraisal
district's own classification may have missed: informal conversions,
multi-structure lots, undercounted duplexes.

Ground truth: STATE_CD-derived missing-middle label (B1/B2/B3-B6/A3/A4)
vs. A1 (single-family). The model never sees STATE_CD/ClassCD as an
input feature -- only things you could infer independently: how many
separate dwelling structures sit on the parcel, how much living area
relative to lot size, improvement value density, year-built pattern,
zoning, and distance to downtown.
"""
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve
from sklearn.inspection import permutation_importance

RAW = "/sessions/rcw-019pt16gylbtcckzycdh9ppp/mnt/data/el_paso_mm/raw"
OUT = "/sessions/rcw-019pt16gylbtcckzycdh9ppp/mnt/data/el_paso_mm/output"

print("Loading parcels_analysis.csv (from build_dataset.py)...")
parcels = pd.read_csv(f"{OUT}/parcels_analysis.csv")

print("Loading Improvements for structural features...")
imprv_cols = ["ImprovementId","Type","StateCode","LivingArea","Value","Description",
              "ClassCD","ExteriorWall","YearBuilt","SquareFootage","Year",
              "Property_dbId","BuildingId","TypeCD"]
imprv = pd.read_csv(
    f"{RAW}/Improvements2027Dump.txt", sep="~", header=None, dtype=str,
    names=imprv_cols, encoding="latin-1", engine="c", quoting=3,
)
imprv["LivingArea_num"] = pd.to_numeric(imprv["LivingArea"], errors="coerce").fillna(0)
imprv["SquareFootage_num"] = pd.to_numeric(imprv["SquareFootage"], errors="coerce").fillna(0)
imprv["YearBuilt_num"] = pd.to_numeric(imprv["YearBuilt"], errors="coerce")
imprv["Value_num"] = pd.to_numeric(imprv["Value"], errors="coerce").fillna(0)
imprv["TypeCD_clean"] = imprv["TypeCD"].str.strip()
imprv["Description_clean"] = imprv["Description"].str.strip()

# "Main" dwelling structures: TypeCD MA (main area), as opposed to
# garages, porches, sheds, amenities etc. Multiple MA records on one
# property is the strongest independent signal of multiple dwelling
# units -- a true duplex/accessory-unit conversion shows up here even
# when STATE_CD still says A1.
main_structures = imprv[imprv["TypeCD_clean"] == "MA"].copy()

agg = imprv.groupby("Property_dbId").agg(
    n_improvement_records=("ImprovementId", "count"),
    total_living_area=("LivingArea_num", "sum"),
    total_sqft=("SquareFootage_num", "sum"),
    total_improvement_value=("Value_num", "sum"),
    earliest_year=("YearBuilt_num", "min"),
    latest_year=("YearBuilt_num", "max"),
).reset_index()

main_agg = main_structures.groupby("Property_dbId").agg(
    n_main_structures=("ImprovementId", "count"),
    main_living_area_total=("LivingArea_num", "sum"),
    main_living_area_max=("LivingArea_num", "max"),
    main_living_area_std=("LivingArea_num", "std"),
).reset_index()

features = agg.merge(main_agg, on="Property_dbId", how="left")
features["n_main_structures"] = features["n_main_structures"].fillna(0)
features["main_living_area_std"] = features["main_living_area_std"].fillna(0)
features["year_span"] = features["latest_year"] - features["earliest_year"]
features["year_span"] = features["year_span"].fillna(0).clip(lower=0)

# ---------------------------------------------------------------------
# Join to parcel-level data. Improvements.Property_dbId IS
# Properties.PropertyId directly (confirmed: 99.95% overlap) -- no
# detour through Properties.dbId needed.
# ---------------------------------------------------------------------
features["Property_dbId_num"] = pd.to_numeric(features["Property_dbId"], errors="coerce")
parcels["PROP_ID_num"] = pd.to_numeric(parcels["PROP_ID"], errors="coerce")
df = parcels.merge(features, left_on="PROP_ID_num", right_on="Property_dbId_num", how="left")

# ---------------------------------------------------------------------
# Feature set (deliberately excludes state_cd / ClassCD -- the model
# must learn to recognize missing-middle from structure, not from the
# label we're trying to predict)
# ---------------------------------------------------------------------
df["land_acres_num"] = pd.to_numeric(df["LAND_ACRES"], errors="coerce")
df["living_area_per_acre"] = df["total_living_area"] / df["land_acres_num"].replace(0, np.nan)
df["value_per_sqft_land"] = df["total_improvement_value"] / (df["land_acres_num"] * 43560).replace(0, np.nan)
df["avg_main_structure_size"] = df["main_living_area_total"] / df["n_main_structures"].replace(0, np.nan)

FEATURE_COLS = [
    "n_improvement_records", "n_main_structures",
    "total_living_area", "total_sqft", "total_improvement_value",
    "main_living_area_total", "main_living_area_max", "main_living_area_std",
    "avg_main_structure_size",
    "land_acres_num", "living_area_per_acre", "value_per_sqft_land",
    "year_span", "year_built", "dist_to_core_mi",
]

# train only on confidently-labeled parcels: A1 (negative) vs B1/B2/B3-6/A3/A4 (positive)
TRAIN_POSITIVE = ["Duplex","Townhouse","Condominium","Apartment/Multifamily",
                   "Multi-Unit Residential (Other)","Apartment (4-6 units)",
                   "Apartment (7-30 units)","Apartment (31-99 units)","Apartment (100+ units)"]
train_df = df[df["typology"].isin(TRAIN_POSITIVE + ["Single-Family Detached"])].copy()
train_df["label"] = train_df["typology"].isin(TRAIN_POSITIVE).astype(int)
train_df = train_df.dropna(subset=FEATURE_COLS, how="all")

X = train_df[FEATURE_COLS].fillna(0)
y = train_df["label"]

print(f"Training set: {len(X):,} parcels ({y.sum():,} positive / {(y==0).sum():,} negative)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

model = HistGradientBoostingClassifier(
    max_iter=200, max_depth=6, learning_rate=0.08,
    class_weight="balanced", random_state=42,
)
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nTest AUC: {auc:.4f}")
print(classification_report(y_test, model.predict(X_test), target_names=["single-family", "missing-middle"]))

# ---- feature importance via permutation (model-agnostic, works for HGB) ----
print("Computing permutation feature importance...")
perm = permutation_importance(model, X_test, y_test, n_repeats=8, random_state=42, n_jobs=-1)
importance = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False)
print("\nFeature importance (permutation, AUC drop):")
print(importance.to_string(index=False))

importance.to_csv(f"{OUT}/feature_importance.csv", index=False)
with open(f"{OUT}/model_metrics.json", "w") as f:
    json.dump({
        "test_auc": round(float(auc), 4),
        "n_train": len(X_train), "n_test": len(X_test),
        "n_positive_train": int(y_train.sum()), "n_positive_test": int(y_test.sum()),
    }, f, indent=2)

# ---------------------------------------------------------------------
# Score ALL parcels currently labeled Single-Family Detached (A1) --
# find ones that look structurally like missing middle despite the label
# ---------------------------------------------------------------------
print("\nScoring all Single-Family Detached (A1) parcels for missed missing-middle candidates...")
a1_df = df[df["typology"] == "Single-Family Detached"].copy()
a1_X = a1_df[FEATURE_COLS].fillna(0)
a1_df["mm_probability"] = model.predict_proba(a1_X)[:, 1]

candidates = a1_df[a1_df["mm_probability"] >= 0.5].sort_values("mm_probability", ascending=False)
print(f"Flagged {len(candidates):,} A1-labeled parcels as likely missing-middle (p >= 0.5)")
print(f"  of which {(candidates['mm_probability'] >= 0.8).sum():,} have p >= 0.8 (high confidence)")

out_cols = ["PROP_ID", "GEO_ID", "lat", "lon", "ZONING", "year_built",
            "dist_to_core_mi", "n_main_structures", "total_living_area",
            "land_acres_num", "mm_probability"]
candidates[out_cols].to_csv(f"{OUT}/missing_middle_candidates.csv", index=False)
print("Saved feature_importance.csv, model_metrics.json, missing_middle_candidates.csv")
