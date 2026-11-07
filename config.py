"""
Shared constants for the El Paso Missing Middle Housing analysis.

The actual classification logic (StateCode -> typology, ClassCD-based
apartment-size sub-split) lives in scripts/build_dataset.py, alongside
the data loading it depends on. This file just holds values shared
across scripts: the downtown anchor point for the distance-gradient
analysis, and typology color tokens for any plotting/dashboard code.
"""

# El Paso downtown core (San Jacinto Plaza) -- the center point for the
# distance-gradient analysis, mirroring the original paper's use of the
# Chicago Loop as its CBD reference point.
DOWNTOWN_CORE_LAT = 31.7598
DOWNTOWN_CORE_LON = -106.4887

TYPOLOGY_COLORS = {
    "Single-Family Detached": "#9e9b8f",
    "Mobile/Manufactured Home": "#c2bfae",
    "Duplex": "#3D5A6C",
    "Townhouse": "#B06A6A",
    "Condominium": "#9C7A96",
    "Apartment/Multifamily": "#A6531C",
    "Multi-Unit Residential (Other)": "#6B4A7A",
    "Apartment (4-6 units)": "#4F8C85",
    "Apartment (7-30 units)": "#8f8a7a",
    "Apartment (31-99 units)": "#5C7A4A",
    "Apartment (100+ units)": "#8C3B2E",
    "Farm/Ranch House": "#c9bfa4",
}
