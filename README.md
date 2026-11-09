# El Paso Missing Middle Housing Analysis

Adapted from the methodology of Salazar-Miranda & Talen,
*"Mapping the Missing Middle: AI-Based Detection of Mid-Density Housing"*
(replication package: `github.com/ariannasm/missing-middle-housing`).

Live dashboard: `index.html` (open directly, or via GitHub Pages) at: 

## https://hoffmanap.github.io/ep-mmh/

## 1. The problem

Missing-middle housing — duplexes, triplexes, courtyard apartments, small
multi-unit buildings mixed into single-family blocks — almost certainly
exists in greater quantity in El Paso than official county records show,
because in most of the city it is **not legal to build**: single-family
zoning (`R3A`, `R3`, `R2`, etc.) covers the overwhelming majority of
residential land, so any informal second unit, converted garage, or
backyard casita has every incentive to stay off the books. A property
owner who adds a unit doesn't file a permit for something zoning
forbids. That means the central appraisal district's own classification
— built from permitted, assessed improvements — is a **floor**, not a
ceiling, on how much missing-middle housing actually exists. This
project tries to size that gap from several independent angles rather
than taking any single source's count at face value.

> **Bottom line, up front:** EPCAD's own records say 21.56% of El
> Paso's housing units are missing-middle (duplex through small
> apartment building). But most of that is actually large apartment
> complexes — scoped to missing middle's real definition (2-19 units,
> per Parolek and the paper this project adapts), EPCAD's figure drops
> to **10.11%**, while the Census's own household survey (ACS) implies
> **16.96%** exists. EPCAD's classification — a legal record of what's
> been permitted and assessed — is capturing only about **60%** of the
> missing-middle housing that independently gathered survey data says
> is actually there. See section 5 for the full comparison, and
> section 6 for what ties all four pieces of evidence together.

## 2. Baseline data — how EPCAD classifies housing, and where it can't see

El Paso Central Appraisal District (EPCAD) already publishes an
authoritative, deterministic classification per parcel — no model
needed for the baseline:

- **Primary classification** uses the GIS parcel layer's `STATE_CD`
  field (one row per parcel): `A1`=single-family, `A2`=mobile home,
  `A3`=condominium, `A4`=townhouse, `B1`=duplex, `B2`=apartment/
  multifamily, `B3`=triplex, `B4`=fourplex, `B5`/`B6`=other multi-unit
  residential subtypes (too few parcels — 125 and 24 — and too
  ambiguous a structural signal to confidently assign them a specific
  unit count; see "Known data bugs" below).
- **Secondary enrichment** sub-splits generic `B2` parcels into
  apartment-size bands (4-6 / 7-30 / 31-99 / 100+ units) using the
  `Improvements.ClassCD` field's `QS`/`QL`/`QN`/`QR` prefixes, joined
  via `Improvements.Property_dbId == Properties.PropertyId` directly
  (see "Known data bugs" below — the documented join key was wrong).
- **Zoning** comes from the GIS layer's own `ZONING` field.

Classified this way: **252,626** residential parcels (includes the
STATE_CD-gap fix below), of which **8,883** parcels are labeled
missing-middle by EPCAD's own code — **3.52%** of parcels. But that's a
parcel count, and parcels aren't units — one apartment building is a
single parcel holding 50+ units. Converted to a unit-weighted basis
(each parcel's typology's recorded/typical unit count, summed): EPCAD's
own records imply **70,504 of 327,081 residential units (21.56%)** are
missing-middle — a very different number from the 3.52% parcel-count
headline, and the first sign that parcel-level and unit-level views of
"how much missing middle exists" tell very different stories. See
section 5 for how this compares to the Census's independent count.

### Where missing middle exists outside EPCAD's own classification

Three signals live inside EPCAD's own data but outside its `STATE_CD`
label — direct evidence of missing-middle housing EPCAD's primary
classification doesn't catch:

- **Legal description (`LEGAL_DESC`) free-text scan**: 148 parcels
  classified `A1` (single-family) mention "CASITA" and 362 mention
  "CONDO" in their legal description — housing EPCAD's own structural
  code doesn't reflect, found independent of any model.
- **Homestead exemption (`EXEMPTIONS` contains `HS`)** — Texas's legal
  requirement that the exemption-holder occupy the property as their
  primary residence, a current, authoritative owner-occupancy signal.
  **75.9%** of single-family parcels claim it, down to **2.4-3.4%**
  for duplexes/multi-unit residential, and **0%** for apartments —
  exactly the pattern expected if non-owner-occupied rental housing
  (often informal, often missing-middle) clusters outside the
  single-family label.
- **Mailing-address mismatch** (owner's mailing city/state differs
  from the parcel's situs city, or isn't in Texas) — a weaker,
  secondary absentee-ownership signal (r = +0.028 with missing-middle
  status).

## 3. The model — learning what missing middle looks like structurally

A `HistGradientBoostingClassifier` was trained **only** on EPCAD's
confirmed missing-middle parcels and confirmed single-family parcels,
using structural features alone, with `STATE_CD`/`ClassCD` (the fields
that *define* the label) explicitly excluded from the model's inputs.
The point is to learn what missing-middle housing looks like
structurally, then use that signature to flag single-family-labeled
parcels EPCAD's own classification might have missed — informal,
undocumented missing middle, which is exactly the population the
problem statement above says should exist.

**Four feature groups feed the model, 37 features total** (an earlier
pass used only the first group):

1. **Structural (15 features, original):** parcel size, number of
   distinct main dwelling structures, total square footage, year
   built, distance to downtown, value density.
2. **Building shape/footprint (8 features):** footprint count and
   size, compactness, footprint-to-recorded-area ratio, new-detached
   and addition flags (`scripts/shape_features.py` — see section 4).
   These had previously been used only as an *independent* validation
   check, never fed to the classifier itself.
3. **Ownership/financing (13 features):** EPCAD's own homestead,
   disabled-veteran, tax-exempt, mailing-mismatch and legal-description
   flags, plus First American fields — `fa_SumResidentialUnits` and
   `fa_SumBuildingsNbr` (an independent third-party unit/building
   count), owner-occupied and vacancy flags, and a value-gap feature
   between First American's AVM estimate and EPCAD's assessed value.
   **First American data had previously been used only descriptively**
   (the ownership/financing appendix below), never as a predictive
   feature — this was a real gap, not an oversight that turned out to
   be nothing.
4. **Physical characteristic (1 feature):** `fa_bathrooms` — First
   American's total bathroom count, joined separately
   (`scripts/join_fa_physical.py`) because it comes from a different
   part of the First American record than the ownership/financing
   fields above. Four other physical fields (bedrooms, living-area
   sqft, effective year built, lot sqft) were tested and dropped —
   they didn't earn their keep against this join's lower coverage
   (58% of parcels match, vs. near-complete for the other feature
   groups). Bathrooms did: the rationale is that converting a
   single-family home into multiple units requires enough existing
   plumbing to physically support the split, so bathroom count is an
   independent capacity signal for conversion, distinct from the
   square-footage and structure-count signals already in the model.

**Result: the richer feature set is a measurably better classifier,
but a worse candidate-finder** — those are different claims, and the
project needs to be honest that they diverged:

| | v1 (15 structural features) | v2 (36 features) | current (37, + `fa_bathrooms`) |
|---|---|---|---|
| Test AUC | 0.986 | 0.996 | **0.997** |
| Precision on missing-middle class (p≥0.5) | lower | 0.66 | **0.67** |
| Recall on missing-middle class (p≥0.5) | — | 0.95 | 0.96 |
| Candidates flagged, p≥0.5 | 7,909 | 4,439 | **4,072** |
| Candidates at ≥80% confidence | 2,622 | 1,569 | **1,471** |

The strongest tells in the current model, in order: **number of main
structures on the parcel**, **distance to downtown core**, **year
built**, **parcel size (acres)**, and **`fa_bathrooms`** — bathroom
count ranks 5th, ahead of every footprint-shape and ownership feature
except `primary_footprint_area_sqft` (6th). Full ranking:
`data/output/feature_importance.csv`.

**What "test AUC" actually measures, and whether 0.997 is good.**
AUC (area under the ROC curve) answers one specific question: pick one
random confirmed missing-middle parcel and one random confirmed
single-family parcel from the held-out test set (59,831 parcels the
model never trained on) — how often does the model rank the
missing-middle one as more likely to be missing-middle? 0.5 is a coin
flip, 1.0 is perfect separation. At **0.997**, it gets that ordering
right 99.7% of the time, which is a strong score — well above what
most real-world classification tasks achieve — and means the
structural/shape/ownership/physical-capacity signature of missing-middle
housing really is distinct from single-family housing in this data, not
something the model is guessing at.

AUC alone doesn't say how many candidates to flag, though — that's
**precision** (of parcels flagged missing-middle, how many actually
are) and **recall** (of parcels that actually are missing-middle, how
many got flagged). At the 0.5 threshold used to generate candidates,
the model scores **0.66 precision / 0.95 recall** on the missing-
middle class: it catches 95% of true missing-middle parcels in the
test set, at the cost of roughly one false positive for every two true
ones. That's a reasonable trade-off for a *screening* tool meant to
flag parcels worth a closer look, not a tool meant to make a final
determination on its own.

**Crucially, none of this — AUC, precision, recall — answers "did the
model find real undercounted missing-middle housing."** All three are
computed against EPCAD's own labels, i.e. parcels the county has
already classified one way or the other. The candidates below are
single-family-*labeled* parcels the model thinks share the missing-
middle signature; EPCAD has made no determination on them, so there's
no ground truth to score the candidates against directly — only the
downstream ACS tract comparison in section 5, which is a much more
modest, much more honest answer than the AUC alone would suggest.

**Why better accuracy means fewer, not more, candidates.** The new
features let the model separate confirmed missing-middle from
confirmed single-family more cleanly, so its probability estimates
tighten — fewer borderline single-family parcels cross the 0.5 or 0.8
threshold by accident. That is the correct behavior for the task the
model was *trained and tested* on. But the candidate-detector's real
job is a different task: finding undercounted missing-middle among
parcels EPCAD never labeled as such in the first place, where there's
no ground truth to score against directly — only the downstream
ACS tract comparison in section 5. And on that downstream test, the
smaller v2 candidate set performs *slightly worse*: it improves the
EPCAD-vs-ACS match in 88 of 182 tracts (49%) versus v1's 96 of 182
(53%), and adds less to the countywide share (9.15% → 9.59% vs.
9.15% → 9.88% — see section 5 for the full table). A model that is
better at the labeled task isn't automatically better at the
unlabeled one; they are genuinely different questions, and this
project reports both honestly rather than picking whichever looks
better.

**Adding `fa_bathrooms` (current model) continues that same pattern,
mildly.** AUC ticked up again (0.996 → 0.997) and bathrooms earned a
real top-5 importance ranking, but the downstream ACS test barely
moved — 88 of 182 tracts improved either way, countywide share 9.56%
vs. v2's 9.59%, functionally a wash. It's kept anyway: the rationale
for bathroom count (existing plumbing capacity constrains whether a
conversion is physically feasible) is independently sound regardless
of this one downstream test, and the metric that test measures —
closing today's EPCAD/ACS gap — isn't the only thing this feature is
useful for; it's also a plausible forward-looking signal for which
single-family homes *could* convert, not just which already-converted
ones EPCAD missed. See `scripts/train_mm_detector.py`'s docstring for
the full before/after on all the physical-characteristic fields
tested, including the four that were tried and dropped.
- These appear on the dashboard map as hollow rust-colored markers,
  distinct from confirmed missing-middle parcels

**The descriptive statistics behind why this works** — the
correlations and non-linear patterns the model is implicitly learning:

Point-biserial/Pearson correlations between structural features and
`is_missing_middle`, computed on 218,009 residential parcels
(bedrooms/bathrooms/building sqft pulled from the First American join,
0.02-mile nearest-neighbor match):

| feature | r | direction |
|---|---|---|
| has_homestead | -0.274 | owner-occupied strongly predicts *not* missing-middle |
| land_acres | +0.197 | larger lots lean missing-middle (see U-shape below) |
| dist_to_core_mi | -0.151 | closer to downtown → more missing-middle |
| bedrooms | -0.073 | fewer bedrooms → more missing-middle |
| building_sqft | +0.068 | weak linear read — see U-shape below |
| year_built | -0.069 | older stock → more missing-middle |
| bathrooms | -0.039 | weak |
| mailing_mismatch | +0.028 | weak |

**The linear correlations understate the real pattern.** Binning lot
size and building size reveals a **U-shape**, not a straight line —
both very small *and* very large lots/buildings over-index on missing
middle, while "typical single-family" mid-range values under-index:

- Lot size: **47.8%** missing-middle share on lots under 1,500 sqft,
  falling to around **1.6%** at 4,500-8,700 sqft (the single-family
  sweet spot), then climbing back to **41.2%** at 87,100+ sqft / 2+
  acres (large multi-family/apartment sites).
- Building size: **10.1%** share at 600-900 sqft (small/casita-sized),
  dropping to **2.3%** at 1500-1800 sqft, then rising to **16.7%** at
  3500+ sqft (large apartment buildings).

**Zoning concentration** (missing-middle share *within* each zoning
code, not raw counts, min n=100), using the **official City of El Paso
zoning layer** (`gis.elpasotexas.gov` Planning/Zoning FeatureServer,
spatially joined to parcel centroids — 214,558/400,274 parcels matched;
unmatched parcels are outside city limits/ETJ with no city zoning
record): `C-1` (50.1%), `C-3` (50.0%), `A-2` (33.5%), `S-D` (33.5%),
`A-O` (30.3%) lead. This replaces an earlier pass using EPCAD's raw
GIS `ZONING` field, which had inconsistent duplicate spellings for the
same district (`R-3`, `R3A`, `R 3A` all meaning similar things) —
the official layer's `ZONELABEL` field is clean. Full numbers:
`data/shape_and_signal_stats.json`. Reproduce with
`scripts/correlations_analysis.py` and
`scripts/join_tracts_and_compare.py` (requires the First American
extract — see Financing section below).

## 4. Validation, layer 1 — building footprints (external, geometric)

The original Salazar-Miranda & Talen paper detects housing typology
from a street-view CV model's read of building *shape*. El Paso has no
imagery pipeline, but real building footprint polygons (Microsoft's
[GlobalMLBuildingFootprints](https://github.com/microsoft/GlobalMLBuildingFootprints),
~384K buildings across El Paso County) let us compute actual shape
features and cross-check EPCAD's own structure counts against physical
reality — an independent, geometric check on whether more housing
exists on a parcel than EPCAD has on record.

`scripts/shape_features.py` spatially joins footprint centroids into
parcels and derives two signals distinguishing **a brand-new undetected
detached structure** from **an addition to an existing one**:

- **`new_detached_flag`**: the parcel has more dwelling-sized
  (≥300 sqft, to exclude sheds/carports) footprints than EPCAD has
  recorded main structures — a candidate undocumented ADU/casita.
- **`addition_flag`**: structure count matches the record, but total
  footprint area exceeds recorded square footage by more than a
  **1.5x** ratio — a candidate undocumented expansion. (This threshold
  was calibrated empirically against the population baseline — see
  commit history; an earlier 1.25x cutoff mostly caught ordinary
  garage/patio coverage, not real additions.)

Of 202,333 single-family (`A1`) parcels with footprint coverage,
**12,955** trip `new_detached_flag` and **4,564** trip `addition_flag`
(**17,519** total, ~8.7% of single-family stock) — a geometric
corroboration, independent of the structural classifier, that
undocumented housing on "single-family" parcels is a real and sizeable
phenomenon, though **not** a confirmed count on its own (footprint
polygons are noisy — see "Known caveats").

**Does the shape signal hold up across typology, not just
single-family?** Breaking the same footprint features out by typology
(`data/shape_by_typology.json`) — using medians, not means, since a
few parcels with implausibly large ratios skew the mean badly for the
smaller attached-housing categories:

| typology | n with footprint data | median footprint-to-recorded ratio | addition flag rate | new-detached flag rate |
|---|---|---|---|---|
| Single-Family Detached | 202,332 | 1.10 | 2.3% | 6.4% |
| Duplex | 3,271 | 1.18 | **25.0%** | 4.0% |
| Mobile/Manufactured Home | 2,472 | 1.01 | 5.9% | 20.5% |
| Multi-Unit Residential (Other) | 1,079 | 1.08 | 13.3% | 1.9% |
| Townhouse | 471 | 1.47 | 44.2% | 6.8% |
| Condominium | 49 | 3.26 | 49.0% | 16.3% |
| Apartment (7-30 units) | 735 | 0.47 | 1.4% | 6.3% |
| Apartment (31-99 units) | 215 | 0.34 | 0.5% | 21.9% |
| Apartment (100+ units) | 217 | 0.32 | 0.0% | **41.9%** |

Two real findings and one data-quality catch:

- **Duplexes show a genuinely elevated addition-flag rate (25% vs.
  2.3% for single-family) without the extreme outliers that distort
  the attached-housing rows below** — the 90th-percentile ratio for
  duplexes is a modest 2.36, not the 8-25x seen for condos/townhouses.
  That's a real structural signature: duplex parcels routinely carry
  more footprint area than what's recorded, independent of the
  structural classifier or the EPCAD typology code itself.
- **Large apartment complexes (31-99 and 100+ units) show high
  new-detached-flag rates (22%/42%)** — expected and not itself a
  "shadow housing" finding: big complexes legitimately have many
  separate buildings on one parcel, and EPCAD's "main structure" count
  per parcel doesn't always keep pace with how many buildings a large
  complex actually has.
- **The footprint-ratio signal breaks down for townhouses and condos**
  (median ratio 1.47/3.26, but 90th-percentile 4.4x/25.6x) — not
  because these are riddled with real additions, but because the
  footprint-centroid spatial join sometimes assigns a *shared*
  building's full footprint to an individual unit's small parcel
  (several attached units, one footprint polygon, each unit's own
  recorded square footage is a fraction of the whole building). The
  1.5x `addition_flag` threshold was calibrated against the
  single-family population baseline (see above) and was never meant
  to apply to attached/shared-structure typologies — their
  addition-flag rates (44%/49%) should be read as spatial-join noise,
  not confirmed undocumented expansions.

**Bottom line:** the footprint-shape validation layer is reliable for
single-family and mobile-home parcels (where it was calibrated) and
surfaces one genuine cross-typology signal — duplexes running hot on
footprint-vs-recorded ratio — but it is not a reliable check for
attached/shared-structure typologies (townhouse, condo) without
unit-splitting the shared footprints first, which this project doesn't
currently do.

## 5. Validation, layer 2 — ACS units-in-structure (external, survey-based)

The Census Bureau's American Community Survey asks households directly
how many units are in their building (table B25024), independent of any
local permitting or assessment process — a third benchmark alongside
EPCAD's parcel records and the footprint cross-check. This answers the
question directly: **whose count is closer to reality, ours (EPCAD-
based) or the Census's household-reported figure?**

Pulled via Census Reporter's open API (2020-2024 ACS 5-year, El Paso
County, all 188 tracts — `scripts/fetch_acs_units_in_structure.py`),
each parcel matched to its tract via a spatial join against TIGERweb
tract boundaries (`scripts/join_tracts_and_compare.py`, 400,183/400,274
parcels matched, 182/188 tracts had both EPCAD and ACS data to compare):

**Countywide:**

| source | missing-middle units | total units | share |
|---|---|---|---|
| EPCAD, parcel count | 8,883 parcels | 252,626 parcels | 3.52% (of *parcels*) |
| EPCAD, unit-weighted (exact) | 70,504 | 327,081 | **21.56%** |
| ACS B25024 (2+ units in structure), 182 tracts matched to EPCAD | 73,463 | 318,700 | **23.05%** |

Once EPCAD's own records are converted to the same unit basis ACS
uses, the headline 3.52%-vs-23% gap mostly disappears — it was largely
a unit-of-analysis artifact (parcels vs. units), not undercounting.
What's left is a real but modest **2.3-point gap**, with ACS running
*higher* than EPCAD — consistent with some degree of actual
undercounting (informal/undocumented units a self-reported survey
would catch that a parcel classification wouldn't), just far smaller
than the raw parcel-count headline implied.

**Tract-level tells a sharper story than the countywide average**:
mean absolute difference between EPCAD and ACS is **7.16 percentage
points per tract** — real local disagreement hiding behind a close
countywide match, because some tracts run EPCAD-high and others run
EPCAD-low, and they roughly cancel out countywide. The biggest gaps:
tract 48141001700 (EPCAD 53.0% vs. ACS 94.3%, a 41.3-point EPCAD
undercount), tract 48141003402 (EPCAD 17.8% vs. ACS 55.6%, 37.8
points), tract 48141002900 (EPCAD 32.6% vs. ACS 64.4%, 31.8 points).

**Adding the structural model's high-confidence candidates does not
close the gap** — a genuinely useful negative result. Treating each
≥80%-confidence candidate as one extra undocumented unit moves the
mean absolute difference from 7.16 to **7.28** (slightly worse), and
only improves the match in 57 of 182 tracts (worse than a coin flip).
**So whose count is more accurate?** For the countywide total, EPCAD's
unit-weighted figure (21.56%) and ACS (23.05%) are close enough that
neither should be treated as definitively right — they're measuring
related but not identical things (EPCAD: what's been assessed and
classified; ACS: what households self-report, survey-sampled, with its
own margins of error at the tract level). What the comparison does show
clearly is that the discrepancy is **not primarily "undocumented units
the structural model can find"** — if it were, model-adjusted EPCAD
numbers would track ACS more closely than raw EPCAD numbers, and they
don't. The gap is more likely driven by differences in what counts as
a "unit" per typology, EPCAD assessment lag, or ACS sampling noise at
the tract level (several of the biggest-gap tracts have small total
unit counts, e.g. 419-621 units, where ACS's margin of error is
largest).

Full tract-level numbers: `data/tract_comparison_epcad_acs.json`.

**A scope problem undermines that close countywide match, though.**
EPCAD's `is_missing_middle` flag — inherited from the STATE_CD
classification (B1–B6) — marks *any* non-single-family residential
structure as "missing middle," including large apartment complexes. Of
the 67,050 unit-weighted "missing middle" units above, 39,695 (59%) are
from Apartment (31-99 units) and Apartment (100+ units) buildings —
not missing middle by any definition in the planning literature, just
large multifamily that happens to share a STATE_CD prefix. The ACS
side of the original comparison was scoped the same broad way (B25024
categories 004-009, any 2+-unit structure), which is why the two
numbers lined up — they're both answering "how much non-single-family
housing is there," not "how much missing middle, specifically, is
there."

**The actual definition — and the one the paper this project adapts
uses for its own Chicago/ACS comparison — caps at 19 units, not any
size.** Parolek (2020), the term's originator, puts missing middle at
"duplexes to buildings with up to roughly 19 units." Salazar-Miranda &
Talen compare their Chicago measure to "ACS estimates of housing units
in 2- to 19-unit structures (44%)" directly — the same ACS bucket used
here. Re-running the comparison with that correct ceiling
(`scripts/mm_2_19_comparison.py`; EPCAD-side types: Duplex, Townhouse,
Condo, Multi-Unit Other, Apartment 4-6, Apartment 7-30 — EPCAD's own
bands top out at 30, not 19, so Apartment 7-30 is the closest available
cut; ACS side: B25024 categories 004-007, units 2 through 19 exactly)
changes the finding substantially:

| source | missing-middle units (2-19) | total units | share |
|---|---|---|---|
| EPCAD, unit-weighted | 33,055 | 327,029 | **10.11%** |
| EPCAD + model candidates (current, 37 features) | 31,069 | 324,998 | 9.56% |
| EPCAD + model candidates (v2, 36 features) | 31,171 | 325,145 | 9.59% |
| EPCAD + model candidates (v1, 15 features) | 32,222 | 326,196 | 9.88% |
| ACS B25024 (2-19 units in structure) | 54,042 | 318,700 | **16.96%** |

**This is the real headline number of the ACS validation: EPCAD
captures only about 60% of the missing-middle housing ACS implies
actually exists in El Paso County** — a **6.85-point countywide gap**,
nearly a factor-of-two undercount, not the reassuring 2.3-point gap the
broad (any-size) comparison suggested. The tract-level mean absolute
difference is **10.22 points**. This is also the most direct possible
confirmation of the problem statement in section 1: EPCAD's own
classification, built from permitted and assessed improvements, really
is a *floor*, and a materially low one, on how much missing-middle
housing exists.

Here, unlike at the broader or narrower cuts, the structural model's
high-confidence candidates help slightly more often than not — with
the **prior (v1), structural-only model**: adding its 2,622
high-confidence candidates as extra units moved the mean absolute
difference from 10.36 (pre-B3/B4-split baseline) to **10.31** and improved the match in **96 of
182 tracts (53%)**, the first time across any definition tested that
the model's candidates net positive rather than negative.

The **v2 model** — retrained with footprint/shape and
ownership/financing features added (section 3) — is a better
classifier (AUC 0.996 vs. 0.986) but produces a *smaller* high-
confidence candidate set (1,569 vs. 2,622), and that smaller set helps
slightly less on this specific test: mean absolute difference moves
from 10.36 (pre-B3/B4-split baseline) to **10.30** (still an improvement, just a smaller one) and
the match improves in **88 of 182 tracts (49%)** — just under half,
versus v1's just-over-half. The **current model** (v2 + `fa_bathrooms`,
section 3) barely moves this further: 1,471 high-confidence candidates,
mean absolute difference now **10.19** (10.22 EPCAD-alone baseline), 84 of 182 tracts,
countywide share 9.56% instead of 9.59%. All three post-v1 results read
as weak, close-to-a-coin-flip evidence either way; none of the models'
candidates clear the bar of "found a lot of real undocumented units."
This is a genuine, somewhat counterintuitive finding worth sitting
with: a classifier can get measurably *better* at the labeled task
(telling confirmed missing-middle from confirmed single-family apart)
while getting marginally *worse* at the unlabeled one (guessing at
undercounted missing-middle EPCAD never flagged at all) — tighter
probability estimates mean fewer parcels clear the confidence
threshold, and some of what the noisier v1 model was catching at
p≥0.8 was apparently true positives this stricter model no longer
reaches that threshold on.

Either way, even the larger (v1) candidate count is far too few to
meaningfully close a gap this size (ACS implies roughly 21,000 more
missing-middle units exist than EPCAD has on record) — closing it
would take a different kind of evidence than a structural anomaly
detector of this scale can provide, regardless of which version of
the model is used.

Full tract-level numbers: `data/mm_2_19_comparison_summary.json`
(checked in); per-tract detail CSV regenerates via
`scripts/mm_2_19_comparison.py`.

## 6. What this means, put together

Four independent pieces of evidence, each with its own blind spots, point
the same direction:

1. **The legal mechanism (section 1).** Single-family zoning makes most
   missing-middle housing illegal to build in most of the city, so
   there's a structural reason for it to go undocumented — property
   owners don't permit what zoning forbids.
2. **EPCAD's own data (section 2-3)** already shows missing-middle
   housing concentrated exactly where that mechanism predicts: near
   downtown, on oddly small or oddly large lots, in older housing stock,
   strongly anti-correlated with owner-occupancy. The structural model
   learns this pattern well enough to separate missing-middle from
   single-family parcels with 0.997 AUC — using structure, shape,
   ownership, and physical-capacity (bathroom count) signal, never the
   EPCAD code itself.
3. **The footprint check (section 4)**, an entirely separate data
   source (satellite-derived building outlines, not EPCAD's own
   records), independently confirms that ~8.7% of single-family-labeled
   parcels carry more or larger structures than EPCAD has on record —
   and duplexes specifically show an elevated version of the same
   signal, a structural fingerprint that doesn't depend on EPCAD's
   classification being right in the first place.
4. **The ACS check (section 5)**, a third, completely independent data
   source (self-reported household survey, not administrative records
   at all), puts a number on the size of the gap: at missing middle's
   real definition, EPCAD accounts for only about **half** of what ACS
   implies exists.

**The key finding, reasoned through.** Put these four pieces together
and a specific explanation for *where* the undercount lives falls out
— not a new measurement, but the most parsimonious reading of what the
measurements above do and don't find:

- A model trained purely on structure, footprint shape, and ownership
  signal — never on EPCAD's own classification — can tell confirmed
  missing-middle parcels from confirmed single-family parcels apart
  with 0.997 AUC (section 3). That's strong evidence that **converted
  missing-middle housing leaves a detectable structural fingerprint
  even with no building permit on file** — the fingerprint is real and
  learnable, independent of whether the county ever classified or
  permitted the conversion.
- But that same model, extended with footprint-shape features
  specifically built to catch *exterior* evidence of conversion
  (extra structures, oversized footprints relative to recorded square
  footage), still only turns up 1,471-2,622 high-confidence candidates
  against an ACS-implied gap of roughly 21,000 missing-middle units
  (section 5) — and the independent footprint check alone finds
  exterior anomalies on only ~8.7% of single-family-labeled parcels
  (section 4). Two separate exterior-evidence methods, same result:
  neither comes close to the size of the gap.
- The straightforward reading of "the exterior-evidence methods
  consistently undershoot, by an order of magnitude, while a model
  that *can* tell converted from unconverted housing apart doesn't
  find enough convertible-looking candidates to explain the gap" is
  that **most of the undercounted missing-middle housing was converted
  without any exterior change** — internal subdivisions, a kitchen and
  a second entrance added to an existing floor plan, a garage or
  basement converted to a separate unit without changing the building's
  footprint or roofline. That's a conversion no satellite footprint
  comparison and no permit record would ever catch, and it's
  consistent with every piece of evidence above without requiring any
  of it to be wrong. It is an inference from the pattern of what the
  detection methods did and didn't find, not a direct measurement —
  this project has no interior inspection or permit-matching data that
  could confirm it directly — but it's the explanation that requires
  the fewest assumptions.

**That has three specific implications, each one corroborated by
something already in this analysis rather than asserted on its own:**

1. **Single-family homes can reasonably be converted to missing-middle
   housing, but for zoning and building codes.** Section 1 establishes
   the legal mechanism directly: the housing types this project finds
   evidence of already exist in El Paso's housing stock (section 2-3)
   and are structurally learnable from existing homes (section 3) —
   the conversions are evidently physically and economically feasible,
   since the pattern above implies they are already happening. What
   stands between "feasible and happening informally" and "legal and
   on the record" is the zoning and permitting framework from section
   1, not a technical or market barrier.
2. **These renovations don't require changes that would alter a
   neighborhood's exterior character.** This follows directly from
   the gap between the two exterior-evidence methods (sections 3-4:
   8.7% footprint-anomaly rate, 1,471-2,622 model candidates) and the
   size of the ACS-implied undercount (section 5: ~21,000 units) — if
   most of the missing units required visible exterior changes, the
   footprint check and the shape-aware model would have found far more
   of them than they did. A policy concerned with preserving
   streetscape character is, on this evidence, largely already
   getting that outcome from this kind of conversion, whether or not
   it's legal.
3. **This data can help target where zoning and building-code reform
   would matter most, because the evidence suggests informal
   conversion is already happening there.** Section 2-3's concentration
   pattern (near downtown, on older housing stock, strongly
   anti-correlated with owner-occupancy) identifies specific,
   mappable tracts where this is occurring organically, ahead of and
   outside of any policy change — which is exactly the population a
   legalization, amnesty, or targeted code-reform effort would want to
   prioritize, both because the demand is evidently already there and
   because formalizing existing conditions is a smaller lift than
   inducing new ones.

**What this project does *not* show:** that the structural candidate
detector has found the missing units, or that internal conversion is
the *only* driver of the gap. Across every definition tested — broad,
2-9-unit, and the correct 2-19-unit cut — the model's high-confidence
candidates close little to none of the EPCAD/ACS gap (worse at the
broad and 2-9 cuts, marginally better than a coin flip at the correct
2-19 cut). Some of the remaining gap is almost certainly EPCAD
assessment lag or genuine ACS sampling noise at the tract level
(several of the biggest-gap tracts have small total unit counts,
where ACS's margin of error is largest) rather than undocumented
housing of any kind — this project can't cleanly separate how much of
the ~21,000-unit gap is "real but invisible conversions" versus
"measurement noise between two different data sources." Confirming the
internal-conversion explanation specifically would take either field
verification in the highest-gap tracts (listed in section 5) or a
different kind of evidence entirely — permit-vs-utility-hookup
cross-referencing, interior inspection, or resident surveys — that
this project's geometry- and classification-based methods can't
provide on their own.

**For policy purposes**, the practical takeaway is the one in the
bottom line: El Paso almost certainly has substantially more
missing-middle housing than its own records show, concentrated in
specific, identifiable places (inner-ring, pre-war, non-owner-occupied
tracts), and the best-supported explanation for why it's invisible to
both administrative records and satellite-based detection is that it's
happening inside existing buildings, not through additions or new
construction that would already have been visible. That combination —
already happening, already low-impact on neighborhood character,
already concentrated in identifiable places — is exactly the case for
treating this as a legalization and code-reform question rather than
a detection problem to keep refining.

## Ownership and financing

El Paso County's First American property/ownership/lien extract
(`Prop48141.txt`, FIPS 48141, 397,150 records, snapshot 2020-04-09)
was spatially joined to the EPCAD parcel set via nearest-neighbor
matching on situs lat/lon. 86.8% of residential parcels matched within
100 feet.

- **Investor ownership is sharply higher for missing-middle types.**
  Duplexes are **73.9%** investor-owned (non-owner-occupant) vs.
  **16.4%** for single-family detached homes; 4-30-unit apartments
  run 92-98% investor-owned.
- Active-lien prevalence and median AVM (automated valuation) by
  typology are also broken out, along with a small (~11,700-record)
  fixed/adjustable/variable financing-type split.
- Caveat: this is a single 2020 snapshot, not current.
- Beyond this descriptive use, several First American fields
  (`fa_SumResidentialUnits`, `fa_SumBuildingsNbr` — an independent
  third-party unit/building count — owner-occupied flag, vacancy
  flag, and an AVM-vs-assessed value-gap feature) are now also fed
  into the candidate-detector classifier in section 3, alongside
  building-footprint shape features. Previously this dataset was used
  only for the descriptive stats above, never as a predictive input.

## Known data bugs found and fixed

Two bugs silently corrupted the first version of this analysis:

1. **Silent CSV-parsing failure.** The EPCAD flat files contain
   embedded NULL bytes and stray quote characters. Reading them with
   `engine="python", on_bad_lines="skip"` caused pandas to silently
   drop *every row* with no visible error. Fixed by switching to
   `engine="c", quoting=3` (`csv.QUOTE_NONE`).
2. **Wrong join key.** The documented schema implies
   `Improvements.Property_dbId == Properties.dbId`, requiring a
   two-hop join. Real data shows `Improvements.Property_dbId` equals
   `Properties.PropertyId` **directly** (99.95% overlap) — fixing this
   resolved the apartment-size sub-split almost completely and raised
   the candidate-detector's test AUC from ~0.93 to 0.986.
3. **Incomplete `STATE_CD` typology map.** Several codes with real
   volume (`A6`, `A51`, `A53`-`A55`, `A7`, `C10`, `D5`, `D6`) were
   missing from the typology dictionary, silently dropping ~10,640
   parcels from the residential denominator. Checked against EPCAD's
   official `2026_Codes for Export Tables_All.xlsx` data dictionary —
   none of these local extensions are defined there either, so their
   typology assignment remains inference from acreage/legal-
   description patterns, flagged as such in `build_dataset.py`.
4. **`B3`/`B4` lumped into a catch-all, and under-counted on top of
   it.** EPCAD does track triplexes (`B3`, 368 parcels) and fourplexes
   (`B4`, 906 parcels) as distinct `STATE_CD` values, but an earlier
   version of this pipeline lumped `B3`, `B4`, `B5`, and `B6` together
   into one "Multi-Unit Residential (Other)" typology — and every
   parcel in that bucket was then counted as exactly **one unit** in
   every unit-weighted statistic on this page (`approx_units` was
   `None` for all four codes, which `.fillna(1)` silently turned into
   1). A real triplex was being counted the same as a studio
   apartment. Fixed by splitting `B3`→Triplex (3 units) and
   `B4`→Fourplex (4 units) in `STATE_CD_TYPOLOGY`, based on the Texas
   statewide SPTB state-code convention other Texas appraisal
   districts document for these codes (EPCAD's own site blocks
   automated lookups via a Cloudflare challenge, so this couldn't be
   confirmed against an EPCAD-published legend directly), corroborated
   here by B3/B4 parcels averaging more "MAIN AREA" improvement
   records per parcel (2.06 / 2.57) than B1 duplexes (1.48). `B5`
   (125 parcels) and `B6` (24) are left lumped as "Multi-Unit
   Residential (Other)" — too few parcels and too ambiguous a
   structural pattern (B5's average is actually *lower* than B4's) to
   confidently assign them a specific unit count. This wasn't
   cosmetic: it moved this page's headline numbers (see sections 1
   and 5) — countywide unit-weighted missing-middle share from 20.72%
   to 21.56%, and the section-5 scoped-definition EPCAD share from
   9.15% to 10.11% of what ACS implies exists.

## Setup

1. Download EPCAD's flat-file exports from `epcad.org/OpenGovernment`
   → "Property Search Dataset Downloads":
   `Properties2027Dump.zip`, `Improvements2027Dump.zip` → unzip into
   `data/raw/`
2. Download the EPCAD parcel GIS layer (same page, "GIS Data" section,
   `parcels2026.zip`) → unzip into `data/gis/` (it's a File
   Geodatabase — `Parcels2026/New File Geodatabase.gdb`)
3. Install dependencies:
   ```
   pip install pandas pyogrio numpy geopandas scipy scikit-learn
   ```

## Known gotchas (cross-project, not just data bugs)

- **CartoDB basemap tiles require an API key now — do not use them.**
  `*.basemaps.cartocdn.com` (Positron/`light_all`, Dark Matter/
  `dark_all`, etc.) used to be usable anonymously and was this
  project's basemap at one point. As of late 2026 it requires a CARTO
  API key for unauthenticated requests: without one, it serves a
  visible **"API KEY REQUIRED" watermark tile instead of the map**,
  with no error thrown — the HTTP request succeeds, so this fails
  silently and looks like a CSS, zoom-level, or caching problem rather
  than a dead tile source, which wastes real debugging time. This
  project now uses Esri's `World_Light_Gray_Base`
  (`server.arcgisonline.com/ArcGIS/rest/services/Canvas/
  World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}`, max zoom 16, no
  key required, `index.html`'s `renderMap()`), which has the same
  quiet/low-contrast intent as Positron. If a future basemap swap is
  needed, check whether the provider requires a key *before* wiring it
  in, not after the map silently breaks.

## Running the pipeline

```
python scripts/build_dataset.py              # classify parcels -> data/output/parcels_analysis.csv
python scripts/make_dashboard_data.py        # aggregate -> data/dashboard_stats.json, data/parcels_missing_middle.geojson
python scripts/train_mm_detector.py          # the model (section 3)
python scripts/correlations_analysis.py      # correlations/sweet-spots/zoning
python scripts/load_footprints.py            # building footprints (section 4)
python scripts/shape_features.py
python scripts/fetch_acs_units_in_structure.py  # ACS comparison (section 5)
python scripts/join_tracts_and_compare.py       # official zoning + tract-level EPCAD/model/ACS comparison (broad definition)
python scripts/mm_2_19_comparison.py            # same comparison re-scoped to the literature's actual 2-19 unit definition (section 5)
```

Then open `index.html` — it reads the generated JSON/GeoJSON files
directly.

## Structure

```
ep-mmh/
├── config.py                       # downtown anchor point, typology color tokens
├── index.html                      # Field Ledger-styled Leaflet + Chart.js dashboard
├── scripts/
│   ├── build_dataset.py             # load GIS + EPCAD flat files, classify typology, compute distance
│   ├── make_dashboard_data.py       # aggregate into dashboard JSON/GeoJSON
│   ├── train_mm_detector.py         # classifier (structural + footprint-shape + ownership/financing features) + candidate scoring (section 3)
│   ├── correlations_analysis.py     # correlations, size sweet-spots, zoning concentration
│   ├── load_footprints.py           # load Microsoft building-footprint tiles (section 4)
│   ├── shape_features.py            # spatial join footprints <-> parcels; shadow-structure signals
│   ├── fetch_acs_units_in_structure.py  # ACS B25024 comparison (section 5)
│   ├── join_tracts_and_compare.py   # official zoning join + tract-level EPCAD/model/ACS comparison
│   ├── join_first_american.py       # spatial join: First American ownership/financing data <-> EPCAD parcels
│   ├── join_fa_physical.py          # separate spatial join: First American physical characteristics (bed/bath/sqft) <-> EPCAD parcels (section 3)
│   └── make_mortgage_dashboard_data.py  # aggregate financing stats for the dashboard
├── data/
│   ├── raw/                         # EPCAD flat-file exports (not checked in — see Setup)
│   ├── gis/                         # EPCAD parcel geodatabase + footprint tiles + zoning/tract boundaries (not checked in — see Setup)
│   ├── output/                      # generated intermediate CSVs (not checked in, large)
│   ├── dashboard_stats.json         # generated: aggregated stats for the dashboard (checked in)
│   ├── shape_and_signal_stats.json  # generated: correlations/sweet-spots/zoning/shape-feature stats (checked in)
│   ├── lot_size_typology_breakdown.json  # generated: typology counts per lot-size bin, for the drill-down chart (checked in)
│   ├── shape_by_typology.json       # generated: footprint-shape features (ratio, flags) broken out by typology (checked in)
│   ├── acs_units_in_structure.json  # generated: ACS B25024 comparison (checked in)
│   ├── tract_comparison_epcad_acs.json  # generated: tract-level EPCAD vs. model vs. ACS comparison, broad (any-size) definition (checked in)
│   ├── mm_2_19_comparison_summary.json  # generated: same comparison re-scoped to the literature's 2-19 unit definition (checked in)
│   ├── zoning_simplified.geojson    # generated: official City zoning polygons, dissolved/simplified for the map overlay (checked in)
│   ├── legalize_priority_tracts.geojson  # generated: census tracts + EPCAD/ACS gap per tract, for the "where to legalize" map choropleth (checked in)
│   ├── parcels_missing_middle.geojson   # generated: confirmed missing-middle parcel points (checked in)
│   ├── candidates_high_confidence.geojson  # generated: model-flagged candidate parcels (checked in)
│   ├── missing_middle_candidates.csv    # generated: full candidate list w/ probabilities (checked in)
│   ├── feature_importance.csv       # generated: classifier permutation importances (checked in)
│   ├── model_metrics.json           # generated: classifier test AUC etc. (checked in)
│   └── mortgage_stats.json          # generated: First American financing aggregates (checked in)
```

## Known caveats

- `B3`-`B6` and the inferred `STATE_CD` extensions above are EPCAD
  local codes with no documented meaning anywhere, including EPCAD's
  own official data dictionary. Their typology assignment is inference
  from acreage/legal-description patterns, flagged as such in code.
- The unit-weighted EPCAD share (21.56%) fills two small typologies
  with no documented typical-unit-count (`Multi-Unit Residential
  (Other)`, generic `Apartment/Multifamily` — 1,437 parcels combined)
  at a conservative 1 unit each, so the true figure is likely a touch
  higher than 21.56%, not lower.
- ACS figures are 5-year survey estimates with sampling error, largest
  in tracts with small total unit counts — several of the biggest
  EPCAD-vs-ACS tract gaps are in tracts under 1,000 units.
- The apartment-size sub-split only covers `B2` parcels whose
  `Improvements.ClassCD` happens to carry a `QS`/`QL`/`QN`/`QR` code.
- Microsoft's building-footprint polygons (section 4) are noisy — one
  roof is sometimes split into multiple polygons by the detection
  model. Read footprint-based flags as a screening signal, not a
  confirmed count.
- The footprint-to-recorded ratio (`addition_flag`) is only reliable
  for single-family and mobile-home parcels. For townhouses and
  condos, a shared building's footprint sometimes gets assigned to one
  attached unit's small individually-recorded parcel, producing
  implausible ratios (seen up to 25.6x at the 90th percentile) — their
  44-49% addition-flag rates in the typology breakdown (section 4) are
  mostly this join artifact, not real undocumented expansions.
- Distance is measured from a single downtown point (San Jacinto
  Plaza), straight-line, not drive-time.
- The headline "missing middle" figures in sections 2-3, and the
  broad (any-size) cut in section 5, count *any* non-single-family
  residential structure, including large (31-99 and 100+ unit)
  apartment buildings — not "missing middle" in the 2-19-unit sense
  Parolek and Salazar-Miranda & Talen actually use. Those large-
  building units are 59% of the 67,050 unit-weighted total. Section
  5's corrected 2-19-unit recheck is the one to use if "missing
  middle" specifically (not "non-single-family generally") is what
  matters for your question — and it's the project's central finding:
  EPCAD captures only about 60% of what ACS implies exists at that
  definition (10.11% vs. 16.96%), an undercount the structural model's
  candidates only marginally help close.
- First American ownership/financing data (Ownership and financing
  section) is a single 2020 snapshot, not current.
- The candidate-detector model (section 3) was retrained with
  footprint-shape and ownership/financing features added, which
  improved its AUC (0.986 → 0.996) but *shrank* the candidate set
  (7,909 → 4,439 at p≥0.5; 2,622 → 1,471 at p≥0.8) and slightly
  reduced how much those candidates help close the section-5 gap
  (96/182 tracts improved → 88/182). Better discrimination on the
  labeled training task and better performance on the unlabeled
  real-world task are not the same thing — see section 3 for the
  full explanation.
