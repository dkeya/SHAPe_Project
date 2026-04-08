# core/transforms.py
from __future__ import annotations

import io
import re
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import pandas as pd

from core.schema import CANONICAL_FIELDS, REQUIRED_FOR_STABILITY, RECOMMENDED_FOR_EXEC


def _norm_col(s: str) -> str:
    """Strong normalization for matching columns across raw exports / processed workbooks."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = s.strip().lower()
    s = re.sub(r"[\t\r\n]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _slug(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Trim colnames, fix odd apostrophes; ensure uniqueness."""
    if df is None or df.empty:
        return pd.DataFrame()

    cols = []
    seen = {}
    for c in df.columns:
        c2 = str(c).strip().replace("’", "'")
        key = c2.lower()
        if key in seen:
            seen[key] += 1
            c2 = f"{c2}__dup{seen[key]}"
        else:
            seen[key] = 0
        cols.append(c2)

    out = df.copy()
    out.columns = cols
    return out


def _read_excel_bytes(xlsx_bytes: bytes) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(io.BytesIO(xlsx_bytes))
    sheets = {}
    for name in xls.sheet_names:
        sheets[name] = pd.read_excel(xls, sheet_name=name)
    return sheets


def detect_source_type(sheets: Dict[str, pd.DataFrame]) -> str:
    if not sheets:
        return "unknown"
    sheet_names = {str(s).strip().lower() for s in sheets.keys()}
    if "baseline" in sheet_names:
        return "processed"
    return "raw"


def _first_sheet(sheets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not sheets:
        return pd.DataFrame()
    first = list(sheets.keys())[0]
    return sheets[first].copy()


def _find_matching_col(df_cols: List[str], synonyms: List[str]) -> Optional[str]:
    """Find first match by normalized equality."""
    if not df_cols or not synonyms:
        return None
    df_map = {_norm_col(c): c for c in df_cols}
    for syn in synonyms:
        syn_n = _norm_col(syn)
        if syn_n in df_map:
            return df_map[syn_n]
    return None


def _to_bool_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="bool")
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.60:
        return (num.fillna(0) > 0).astype(bool)
    st = s.astype("string").fillna("").str.strip().str.lower()
    return st.isin(["yes", "y", "true", "1", "approved", "compliant", "available"])


def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _to_str_series(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.strip()

def _to_datetime_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(s, errors="coerce")

# ==========================================================
# AEZ Mapping Function
# ==========================================================
def map_gps_to_aez(lat: float, lon: float) -> str:
    """
    Map GPS coordinates to Kenya's Agro-Ecological Zones (AEZ).
    This is a simplified version - in production, you would use a shapefile
    or a more comprehensive mapping database.
    
    Kenya's AEZ classification (simplified):
    - Zone I: Alpine (High altitude, >3000m) - Very cold
    - Zone II: High Potential (2000-3000m) - Tea, coffee zones
    - Zone III: Medium Potential (1500-2000m) - Mixed farming
    - Zone IV: Semi-arid (1000-1500m) - Livestock, drought crops
    - Zone V: Arid (<1000m) - Pastoralism
    
    Returns AEZ classification string.
    """
    if pd.isna(lat) or pd.isna(lon):
        return "Unknown"
    
    # Simplified altitude-based AEZ mapping
    # In production, you would use actual AEZ shapefile data
    altitude = None
    
    # Placeholder - actual implementation would use:
    # 1. Digital Elevation Model (DEM) data
    # 2. AEZ shapefile from Kenya Agricultural Research Institute
    # 3. Or pre-computed AEZ lookup table
    
    # For now, return based on approximate altitude ranges
    # This should be replaced with actual altitude data from GPS
    
    return "Unknown"  # Will be populated from GPS altitude in real implementation


def assign_aez_from_altitude(df: pd.DataFrame) -> pd.Series:
    """
    Assign AEZ based on altitude data from GPS coordinates.
    
    Kenya AEZ by altitude (simplified):
    - Alpine (>3000m): Zone I
    - Highland (2000-3000m): Zone II
    - Medium (1500-2000m): Zone III
    - Lowland (1000-1500m): Zone IV
    - Arid (<1000m): Zone V
    """
    if "altitude" not in df.columns:
        return pd.Series(["Unknown"] * len(df), index=df.index, dtype="string")
    
    def _assign(alt):
        if pd.isna(alt):
            return "Unknown"
        if alt > 3000:
            return "Zone I (Alpine)"
        elif alt > 2000:
            return "Zone II (High Potential)"
        elif alt > 1500:
            return "Zone III (Medium Potential)"
        elif alt > 1000:
            return "Zone IV (Semi-arid)"
        else:
            return "Zone V (Arid)"
    
    return df["altitude"].apply(_assign).astype("string")


# ----------------------------------------------------------
# Extra fields (robust fallback, even if schema not updated)
# ----------------------------------------------------------
EXTRA_FIELDS = [
    # Production
    {
        "name": "harvest_kg",
        "dtype": "float",
        "synonyms": [
            "4.2 Total Harvest Last Season (kg)",
            "Total Harvest Last Season (kg)",
            "Harvest Last Season (kg)",
            "Total Harvest (kg)",
            "Harvest (kg)",
            "harvest_kg",
        ],
    },
    # Farmer Profile
    {
        "name": "gender",
        "dtype": "string",
        "synonyms": [
            "1.11 Gender",
            "Gender",
            "gender",
        ],
    },
    {
        "name": "age",
        "dtype": "float",
        "synonyms": [
            "1.12 Age",
            "Age",
            "age",
        ],
    },
    {
        "name": "education",
        "dtype": "string",
        "synonyms": [
            "1.13 Formal education level",
            "Formal education level",
            "Education Level",
            "education",
            "Highest education",
            "Educational attainment",
        ],
    },
    {
        "name": "experience",
        "dtype": "float",
        "synonyms": [
            "1.14 Experience in Avocado farming in years",
            "Experience in Avocado farming in years",
            "Farming Experience",
            "Avocado farming experience",
            "Experience years",
            "experience",
        ],
    },
    # Legal Registration
    {
        "name": "orchard_group",
        "dtype": "bool",
        "synonyms": [
            "1.24 Is this orchard registered as part of a Group",
            "Orchard Group Registration",
        ],
    },
    # Production Practices
    {
        "name": "fertilizer_organic",
        "dtype": "bool",
        "synonyms": [
            "3.3 Type of Fertilizer Used/Organic",
            "Organic Fertilizer",
        ],
    },
    {
        "name": "fertilizer_inorganic",
        "dtype": "bool",
        "synonyms": [
            "3.3 Type of Fertilizer Used/Inorganic",
            "Inorganic Fertilizer",
        ],
    },
    {
        "name": "fertilizer_quantity_liquid_organic",
        "dtype": "float",
        "synonyms": [
            "3.311 Provide the annual quantity of Liquid Organic Fertilizer used per tree (in liters)",
            "Liquid Organic Fertilizer quantity",
        ],
    },
    {
        "name": "fertilizer_quantity_solid_organic",
        "dtype": "float",
        "synonyms": [
            "3.312 Provide the annual quantity of Solid Organic Fertilizer used per tree (in kgs)",
            "Solid Organic Fertilizer quantity",
        ],
    },
    {
        "name": "fertilizer_quantity_liquid_inorganic",
        "dtype": "float",
        "synonyms": [
            "3.313 Provide the annual quantity of Liquid Inorganic Fertilizer used per tree (in liters)",
            "Liquid Inorganic Fertilizer quantity",
        ],
    },
    {
        "name": "fertilizer_quantity_solid_inorganic",
        "dtype": "float",
        "synonyms": [
            "3.314 Provide the annual quantity of Solid Inorganic Fertilizer used per tree (in kgs)",
            "Solid Inorganic Fertilizer quantity",
        ],
    },
    {
        "name": "soil_conservation",
        "dtype": "string",
        "synonyms": [
            "3.4 Soil Conservation Measures Applied",
            "Soil Conservation",
        ],
    },
    {
        "name": "irrigation_drip",
        "dtype": "bool",
        "synonyms": [
            "3.5 Irrigation Practices/Drip",
            "Drip Irrigation",
        ],
    },
    {
        "name": "irrigation_sprinkler",
        "dtype": "bool",
        "synonyms": [
            "3.5 Irrigation Practices/Sprinkler",
            "Sprinkler Irrigation",
        ],
    },
    {
        "name": "irrigation_rainfed",
        "dtype": "bool",
        "synonyms": [
            "3.5 Irrigation Practices/Rainfed",
            "Rainfed",
        ],
    },
    {
        "name": "sanitation_record",
        "dtype": "bool",
        "synonyms": [
            "3.6 Is there a record of sanitation conditions?",
            "Sanitation record",
        ],
    },

    # ==========================================================
    # NEW: Enhanced Production Practices (from updated document)
    # ==========================================================
    {
        "name": "dropped_fruits_elimination",
        "dtype": "bool",
        "synonyms": [
            "3.7 Is there prompt elimination of dropped fruits?",
            "Prompt elimination of dropped fruits",
            "Dropped fruits elimination",
            "Elimination of dropped fruits",
            "Dropped fruits removal",
        ],
    },
    {
        "name": "weed_management",
        "dtype": "bool",
        "synonyms": [
            "Weed management",
            "Weed control",
            "Weeding",
            "Weed Management",
            "Weed management practices",
        ],
    },
    {
        "name": "pruning_practices",
        "dtype": "bool",
        "synonyms": [
            "Pruning",
            "Tree Pruning",
            "Pruning of trees",
            "Pruning practices",
            "6.3 Record-Keeping Practices/Pruning",
        ],
    },

    # ==========================================================
    # NEW: Detailed IPM Measures (from updated document)
    # ==========================================================
    {
        "name": "ipm_traps",
        "dtype": "bool",
        "synonyms": [
            "3.91 Fly Traps",
            "Fly Traps",
            "Use of traps",
            "Traps",
            "Fly traps",
            "3.84 Fly Traps",
        ],
    },
    {
        "name": "ipm_chemical",
        "dtype": "bool",
        "synonyms": [
            "3.92 Chemical Control",
            "Chemical control",
            "IPM Chemical",
            "Chemical Control",
            "3.82 Chemical Control",
        ],
    },
    {
        "name": "ipm_biological",
        "dtype": "bool",
        "synonyms": [
            "3.93 Biological Control",
            "Biological control",
            "IPM Biological",
            "Biological Control",
            "3.83 Biological Control",
        ],
    },
    {
        "name": "ipm_mating_disruption",
        "dtype": "bool",
        "synonyms": [
            "3.94 Mating Disruption",
            "Mating disruption",
            "IPM Mating disruption",
            "Mating Disruption",
        ],
    },
    {
        "name": "ipm_pest_monitoring",
        "dtype": "bool",
        "synonyms": [
            "3.81 Pest Monitoring",
            "Pest monitoring",
            "Is pest monitoring conducted",
        ],
    },

    # ==========================================================
    # NEW: Certification Compliance Training (from updated document)
    # ==========================================================
    {
        "name": "cert_compliance_training",
        "dtype": "bool",
        "synonyms": [
            "8.6 What are your most pressing training/extension needs/Certification Compliance",
            "Certification Compliance Training",
            "Training on certification compliance",
            "Certification training",
            "8.2 Training Received in the Last 6-months/Certification Compliance",
        ],
    },

    # ==========================================================
    # NEW: Grade 2 Share (from updated document)
    # ==========================================================
    {
        "name": "grade2_share_last",
        "dtype": "float",
        "synonyms": [
            "5.5 What proportion of your harvest did you retain for own use/gifting last season (%)?",
            "Grade 2 Share Last Season",
            "Own use share",
            "Proportion retained for own use",
        ],
    },
    {
        "name": "grade2_share_current",
        "dtype": "float",
        "synonyms": [
            "5.9 What proportion of your harvest did you retain for own use/gifting this season (%)?",
            "Grade 2 Share Current Season",
        ],
    },

    # ==========================================================
    # NEW: Fruits per tree metrics (for income per tree calculations)
    # ==========================================================
    {
        "name": "fruits_per_tree_0_3",
        "dtype": "float",
        "synonyms": [
            "4.8 Average No. of Fruits per avocado tree aged 0-3 years",
            "Fruits per tree 0-3 years",
        ],
    },
    {
        "name": "fruits_per_tree_4_7",
        "dtype": "float",
        "synonyms": [
            "4.81 Average No. of Fruits per avocado tree aged 4-7 years",
            "Fruits per tree 4-7 years",
        ],
    },
    {
        "name": "fruits_per_tree_8_plus",
        "dtype": "float",
        "synonyms": [
            "4.82 Average No. of Fruits per avocado tree aged 8+ years",
            "Fruits per tree 8+ years",
        ],
    },
    
    # ==========================================================
    # NEW: Yield per tree by age
    # ==========================================================
    {
        "name": "yield_per_tree_0_3",
        "dtype": "float",
        "synonyms": [
            "4.1 Average Yield per avocado Tree aged 0-3 years last season (kg)",
            "Yield per tree 0-3 years",
        ],
    },
    {
        "name": "yield_per_tree_4_7",
        "dtype": "float",
        "synonyms": [
            "4.11 Average Yield per avocado Tree aged 4-7 years last season (kg)",
            "Yield per tree 4-7 years",
        ],
    },
    {
        "name": "yield_per_tree_8_plus",
        "dtype": "float",
        "synonyms": [
            "4.12 Average Yield per avocado Tree aged 8+years last season (kg)",
            "Yield per tree 8+ years",
        ],
    },
    
    # Compliance
    {
        "name": "sanitation_tools",
        "dtype": "bool",
        "synonyms": [
            "6.4 Use of Clean Harvesting Tools",
            "Clean Harvesting Tools",
        ],
    },
    {
        "name": "ipm_use",
        "dtype": "bool",
        "synonyms": [
            "3.8 Is an Integrated Pest Management (IPM) program implemented?",
            "IPM Implemented",
        ],
    },
    {
        "name": "biological_control",
        "dtype": "bool",
        "synonyms": [
            "3.83 Biological Control",
            "Biological Control",
        ],
    },
    {
        "name": "pesticide_use",
        "dtype": "bool",
        "synonyms": [
            "6.7 Use of Approved Pesticides Only",
            "Approved pesticides",
        ],
    },
    {
        "name": "pesticide_compliance",
        "dtype": "bool",
        "synonyms": [
            "6.6 Compliance with Pesticide Withdrawal Period (REI/ PHI)",
            "Pesticide withdrawal compliance",
        ],
    },
    # Market
    {
        "name": "hass_price_current_ksh_per_kg",
        "dtype": "float",
        "synonyms": [
            "5.6 Average Selling Price of (Hass variety) per kg this Season (KSH)",
            "Current Hass price per kg",
        ],
    },
    {
        "name": "grade1_share_last",
        "dtype": "float",
        "synonyms": [
            "5.4 What proportion of your harvest did you sell as Grade 1 last season (%)?",
            "Grade 1 Share Last Season",
        ],
    },
    {
        "name": "grade1_share_current",
        "dtype": "float",
        "synonyms": [
            "5.8 What proportion of your harvest did you sell as Grade 1 this season (%)?",
            "Grade 1 Share Current Season",
        ],
    },
    {
        "name": "market_constraints",
        "dtype": "string",
        "synonyms": [
            "5.10 Challenges in Market Access",
            "Market Access Challenges",
        ],
    },
    # Training
    {
        "name": "sps_training",
        "dtype": "bool",
        "synonyms": [
            "6.1 Training on SPS & Compliance Received",
            "SPS Training Received",
        ],
    },
    {
        "name": "training_last_year",
        "dtype": "string",
        "synonyms": [
            "8.1Training Received in the Last Year",
            "Training last year",
        ],
    },
    {
        "name": "training_last_6months",
        "dtype": "string",
        "synonyms": [
            "8.2 Training Received in the Last 6-months",
            "Training last 6 months",
        ],
    },
    {
        "name": "training_provider",
        "dtype": "string",
        "synonyms": [
            "8.3 Who provided the training",
            "Training Provider",
        ],
    },
    {
        "name": "record_keeping",
        "dtype": "bool",
        "synonyms": [
            "6.3 Record-Keeping Practices",
            "Record Keeping",
        ],
    },
    {
        "name": "extension_access",
        "dtype": "string",
        "synonyms": [
            "8.4 Extension Services Accessed",
            "Extension Services",
        ],
    },
    {
        "name": "training_needs_text",
        "dtype": "string",
        "synonyms": [
            "8.6 What are your most pressing training/extension needs",
            "Training needs",
        ],
    },
    # Sustainability
    {
        "name": "water_source",
        "dtype": "string",
        "synonyms": [
            "7.1 Water Source for Irrigation",
            "Water Source",
        ],
    },
    {
        "name": "waste_management",
        "dtype": "string",
        "synonyms": [
            "7.2 Waste Management Practices",
            "Waste Management",
        ],
    },
    {
        "name": "biodiversity",
        "dtype": "string",
        "synonyms": [
            "7.3 Biodiversity Conservation Practices",
            "Biodiversity Practices",
        ],
    },
    {
        "name": "additional_trees",
        "dtype": "string",
        "synonyms": [
            "7.4 What other trees are planted around/within the orchard",
            "Other Trees Planted",
        ],
    },
    {
        "name": "other_value_chains",
        "dtype": "string",
        "synonyms": [
            "7.5 What other value chains is the farmer involved in",
            "Other Value Chains",
        ],
    },
    # Losses
    {
        "name": "harvest_losses_kg",
        "dtype": "float",
        "synonyms": [
            "4.3 Avocado Losses last season (kg)",
            "Avocado Losses (kg)",
            "Losses (kg)",
        ],
    },
    {
        "name": "loss_causes",
        "dtype": "string",
        "synonyms": [
            "4.31 Primary Cause of Loss last season",
            "Primary Cause of Loss",
        ],
    },
    {
        "name": "loss_causes_other",
        "dtype": "string",
        "synonyms": [
            "4.32 Other Causes of Loss last season",
            "Other Loss Causes",
        ],
    },
]


def _add_extra_field(
    raw_df: pd.DataFrame,
    out: pd.DataFrame,
    used: Dict[str, str],
    name: str,
    dtype: str,
    synonyms: List[str],
) -> None:
    """Enhanced version with better error handling and logging."""
    # Only skip if we have already successfully mapped this field
    if name in used:
        return

    src = _find_matching_col(list(raw_df.columns), synonyms or [])
    if src is None:
        if name not in out.columns:
            out[name] = np.nan
        return

    used[name] = src
    s = raw_df[src]
    
    try:
        if dtype == "float":
            out[name] = _to_float_series(s)
        elif dtype == "int":
            out[name] = _to_float_series(s).round(0)
        elif dtype == "bool":
            out[name] = _to_bool_series(s)
        elif dtype == "datetime":
            out[name] = _to_datetime_series(s)
        else:
            out[name] = _to_str_series(s)
    except Exception as e:
        # Fallback to string if conversion fails
        print(f"Warning: Failed to convert field {name} to {dtype}, using string. Error: {e}")
        out[name] = _to_str_series(s)


# ----------------------------------------------------------
# Derive sub_county from raw one-hot columns (1.4* SubCounty ...)
# ----------------------------------------------------------
def _derive_sub_county_from_onehots(raw_df: pd.DataFrame) -> pd.Series:
    """
    Raw exports encode sub-county as many one-hot columns, e.g.:
      '1.4P SubCounty Baringo/Baringo Central'
      '1.4A Subcounty Meru'
      '1.4V SubCounty  Vihiga/Hamisi'
    We return a per-row string (single best label; if multiple selected, join with '; ').
    """
    if raw_df is None or raw_df.empty:
        return pd.Series("", index=pd.Index([]), dtype="string")

    cols = list(raw_df.columns)

    # Match any column that contains "subcounty" (case-insensitive),
    # and starts with the questionnaire prefix "1.4" (tolerant to spacing/letter codes).
    sub_cols = []
    for c in cols:
        c_str = str(c)
        c_norm = c_str.lower()
        if "subcounty" in c_norm and c_norm.strip().startswith("1.4"):
            sub_cols.append(c_str)

    if not sub_cols:
        return pd.Series("", index=raw_df.index, dtype="string")

    # Convert to 0/1 selections
    sub_df = raw_df[sub_cols].copy()
    for c in sub_cols:
        sub_df[c] = (pd.to_numeric(sub_df[c], errors="coerce").fillna(0) > 0).astype(int)

    def _label_from_col(colname: str) -> str:
        # Remove the leading "1.4X SubCounty" part
        s = str(colname)

        # strip duplicate marker if present
        s = re.sub(r"__dup\d+$", "", s).strip()

        # Remove the leading "1.4<code> SubCounty" tokens
        s2 = re.sub(r"^1\.4[a-z0-9]*\s*subcounty\s*", "", s, flags=re.IGNORECASE).strip()

        # If label still has a county prefix like "Baringo/Baringo Central", keep the most specific segment
        if "/" in s2:
            s2 = s2.split("/")[-1].strip()

        # Final cleanup spacing
        s2 = re.sub(r"\s+", " ", s2).strip()
        return s2

    labels = {c: _label_from_col(c) for c in sub_cols}

    def pick_row(row: pd.Series) -> str:
        picked = [labels[c] for c in sub_cols if int(row.get(c, 0)) == 1 and labels.get(c)]
        if not picked:
            return ""
        seen = set()
        out = []
        for x in picked:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return "; ".join(out)

    return sub_df.apply(pick_row, axis=1).astype("string")


def map_to_canonical(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    raw_df = clean_column_names(raw_df)
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(), {}

    used: Dict[str, str] = {}
    out = pd.DataFrame(index=raw_df.index)

    # 1) Schema-driven mapping
    for field in CANONICAL_FIELDS:
        src = _find_matching_col(list(raw_df.columns), (field.synonyms or []))
        if src is None:
            out[field.name] = np.nan
            continue

        used[field.name] = src
        s = raw_df[src]

        if field.dtype == "float":
            out[field.name] = _to_float_series(s)
        elif field.dtype == "int":
            out[field.name] = _to_float_series(s).round(0)
        elif field.dtype == "bool":
            out[field.name] = _to_bool_series(s)
        elif field.dtype == "datetime":
            out[field.name] = _to_datetime_series(s)
        else:
            out[field.name] = _to_str_series(s)

    # 2) Add extra fields (works even before schema changes)
    for f in EXTRA_FIELDS:
        _add_extra_field(raw_df, out, used, f["name"], f["dtype"], f.get("synonyms", []))

    # 3) sub_county derived from raw one-hots if sub_county is missing/empty
    if ("sub_county" not in out.columns) or (out["sub_county"].astype("string").str.strip().replace("", np.nan).isna().all()):
        derived_sub = _derive_sub_county_from_onehots(raw_df)
        if isinstance(derived_sub, pd.Series) and len(derived_sub) == len(out):
            out["sub_county"] = derived_sub
            if "sub_county" not in used:
                used["sub_county"] = "derived:1.4*_SubCounty_onehots"

    # 4) AEZ assignment based on altitude
    out["aez_zone"] = assign_aez_from_altitude(out)

    # 5) Derived stability fields
    out["trees_per_acre"] = np.nan
    if "trees_total" in out.columns and "area_acres" in out.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["trees_per_acre"] = out["trees_total"] / out["area_acres"]

    out["income_per_acre"] = np.nan
    if "income_ksh_last_season" in out.columns and "area_acres" in out.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["income_per_acre"] = out["income_ksh_last_season"] / out["area_acres"]

    out["yield_per_acre"] = np.nan
    if "harvest_kg" in out.columns and "area_acres" in out.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["yield_per_acre"] = out["harvest_kg"] / out["area_acres"]

    # ==========================================================
    # NEW: Additional derived metrics (from updated document)
    # ==========================================================

    # Income per Tree
    out["income_per_tree"] = np.nan
    if "income_ksh_last_season" in out.columns and "trees_total" in out.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["income_per_tree"] = out["income_ksh_last_season"] / out["trees_total"]

    # Income per Tree by Age Group using yield data if available
    out["income_per_tree_0_3"] = np.nan
    out["income_per_tree_4_7"] = np.nan
    out["income_per_tree_8_plus"] = np.nan
    
    # Try to use actual yield data first
    if "hass_price_ksh_per_kg" in out.columns:
        price = out["hass_price_ksh_per_kg"].fillna(0)
        
        if "yield_per_tree_0_3" in out.columns:
            out["income_per_tree_0_3"] = out["yield_per_tree_0_3"] * price
        elif "fruits_per_tree_0_3" in out.columns:
            # Estimate kg per fruit (0.2 kg typical for Hass)
            kg_per_fruit = 0.2
            out["income_per_tree_0_3"] = out["fruits_per_tree_0_3"] * kg_per_fruit * price
        
        if "yield_per_tree_4_7" in out.columns:
            out["income_per_tree_4_7"] = out["yield_per_tree_4_7"] * price
        elif "fruits_per_tree_4_7" in out.columns:
            kg_per_fruit = 0.2
            out["income_per_tree_4_7"] = out["fruits_per_tree_4_7"] * kg_per_fruit * price
        
        if "yield_per_tree_8_plus" in out.columns:
            out["income_per_tree_8_plus"] = out["yield_per_tree_8_plus"] * price
        elif "fruits_per_tree_8_plus" in out.columns:
            kg_per_fruit = 0.2
            out["income_per_tree_8_plus"] = out["fruits_per_tree_8_plus"] * kg_per_fruit * price

    # Total Grade 1 + Grade 2 should equal 100% (for validation)
    out["grade_total_completeness"] = np.nan
    if "grade1_share_last" in out.columns and "grade2_share_last" in out.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["grade_total_completeness"] = out["grade1_share_last"] + out["grade2_share_last"]

    # Organic users vs Organic certified comparison flag
    out["organic_alignment"] = np.nan
    if "fertilizer_organic" in out.columns and "cert_organic" in out.columns:
        # Flag farms that use organic fertilizer but are NOT organically certified
        out["organic_alignment"] = (out["fertilizer_organic"] == True) & (out["cert_organic"] == False)

    def _dominant_age(row) -> str:
        a = row.get("trees_0_3", np.nan)
        b = row.get("trees_4_7", np.nan)
        c = row.get("trees_8_plus", np.nan)
        vals = {"0-3 years": a, "4-7 years": b, "8+ years": c}
        if all(pd.isna(v) for v in vals.values()):
            return ""
        vals2 = {k: (0.0 if pd.isna(v) else float(v)) for k, v in vals.items()}
        return max(vals2, key=vals2.get)

    out["dominant_age_group"] = out.apply(_dominant_age, axis=1)
    
    # Farm size classification
    out["farm_size_classification"] = "Unknown"
    if "area_acres" in out.columns:
        def classify_farm_size(acres):
            if pd.isna(acres):
                return "Unknown"
            if acres < 2:
                return "Micro (0-2 acres)"
            elif acres < 10:
                return "Small (2-10 acres)"
            elif acres < 50:
                return "Medium (10-50 acres)"
            else:
                return "Large (50+ acres)"
        out["farm_size_classification"] = out["area_acres"].apply(classify_farm_size)
    
    # IPM adoption completeness score (0-100%)
    ipm_components = ["ipm_traps", "ipm_chemical", "ipm_biological", "ipm_mating_disruption", "ipm_pest_monitoring"]
    ipm_available = [c for c in ipm_components if c in out.columns]
    if ipm_available:
        out["ipm_adoption_score"] = out[ipm_available].fillna(False).astype(bool).mean(axis=1) * 100
    
    # GACC Progress Tracking
    out["is_gacc_approved"] = out.get("gacc_status", pd.Series([False] * len(out))).fillna(False).astype(bool)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out, used


def extract_training_onehots(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(index=pd.Index([])), []

    raw_df = clean_column_names(raw_df)

    prefix = "8.6 What are your most pressing training/extension needs/"
    cols = [c for c in raw_df.columns if str(c).startswith(prefix)]
    if not cols:
        return pd.DataFrame(index=raw_df.index), []

    out = pd.DataFrame(index=raw_df.index)
    created = []
    for c in cols:
        option = str(c).replace(prefix, "").strip()
        cname = f"training_need__{_slug(option)}"
        out[cname] = (pd.to_numeric(raw_df[c], errors="coerce").fillna(0) > 0).astype(int)
        created.append(cname)
    return out, created


def build_derived_sheets(canonical_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    d = canonical_df.copy()
    if d is None or d.empty:
        return {
            "Baseline": pd.DataFrame(),
            "Metrics": pd.DataFrame(),
            "Certifications": pd.DataFrame(),
            "Training Needs": pd.DataFrame(),
            "Market Summary": pd.DataFrame(),
            "GACC Tracking": pd.DataFrame(),
            "AEZ Distribution": pd.DataFrame(),
        }

    agg_spec = dict(
        Total_Farmers=("exporter", "size"),
        Total_Acres=("area_acres", "sum"),
        Total_Trees=("trees_total", "sum"),
        China_Approved_Farms=("gacc_status", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
    )

    if "harvest_kg" in d.columns:
        agg_spec["Total_Harvest_Kg"] = ("harvest_kg", "sum")

    if "yield_per_acre" in d.columns:
        agg_spec["Median_Yield_Kg_per_Acre"] = ("yield_per_acre", "median")

    if "income_per_acre" in d.columns:
        agg_spec["Median_Income_KSh_per_Acre"] = ("income_per_acre", "median")

    if "income_per_tree" in d.columns:
        agg_spec["Median_Income_KSh_per_Tree"] = ("income_per_tree", "median")

    metrics = (
        d.groupby("exporter", dropna=False)
        .agg(**agg_spec)
        .reset_index()
        .rename(columns={"exporter": "Exporter"})
    )

    cert_cols = [
        "exporter",
        "farmer_name",
        "orchard_name",
        "gacc_status",
        "kephis_status",
        "sanitation_record",
        "pest_monitoring",
        "approved_pesticides",
        "cert_globalgap",
        "cert_organic",
        "cert_fairtrade",
        # NEW: Additional compliance fields
        "dropped_fruits_elimination",
        "weed_management",
        "pruning_practices",
        "ipm_traps",
        "ipm_chemical",
        "ipm_biological",
        "ipm_mating_disruption",
        "ipm_pest_monitoring",
        "ipm_adoption_score",
    ]
    cert = d[[c for c in cert_cols if c in d.columns]].copy()

    onehot_cols = [c for c in d.columns if str(c).startswith("training_need__")]
    if onehot_cols and "exporter" in d.columns:
        sums = d.groupby("exporter")[onehot_cols].sum(numeric_only=True)

        def pick_top(row):
            if row.max() <= 0:
                return "None"
            top = row.idxmax()
            return str(top).replace("training_need__", "").replace("_", " ").title()

        top = sums.apply(pick_top, axis=1).reset_index()
        top.columns = ["Exporter", "Top_Training_Need"]
        training = top
    else:
        training = pd.DataFrame(columns=["Exporter", "Top_Training_Need"])

    market_rows = []
    if "exporter" in d.columns:
        for exp, g in d.groupby("exporter"):
            outlet = g.get("market_outlet", pd.Series(dtype="string"))
            outlet = outlet.astype("string").str.strip()
            outlet = outlet[outlet != ""]
            top_outlet = outlet.value_counts().index[0] if len(outlet) else ""

            prices = pd.to_numeric(g.get("hass_price_ksh_per_kg", pd.Series(dtype="float")), errors="coerce").dropna()
            current_prices = pd.to_numeric(g.get("hass_price_current_ksh_per_kg", pd.Series(dtype="float")), errors="coerce").dropna()
            incomes = pd.to_numeric(g.get("income_ksh_last_season", pd.Series(dtype="float")), errors="coerce").dropna()

            market_rows.append(
                {
                    "Exporter": exp,
                    "Top_Outlet": top_outlet,
                    "Median_Hass_Price_KSh_per_Kg": float(prices.median()) if len(prices) else np.nan,
                    "Median_Hass_Price_Current_KSh_per_Kg": float(current_prices.median()) if len(current_prices) else np.nan,
                    "Median_Income_KSh_Last_Season": float(incomes.median()) if len(incomes) else np.nan,
                }
            )
    market = pd.DataFrame(market_rows)
    
    # ==========================================================
    # NEW: GACC Tracking Sheet
    # ==========================================================
    if "is_gacc_approved" in d.columns:
        gacc_tracking = pd.DataFrame({
            "Metric": [
                "SHAPe GACC Approved Farms (Current)",
                "Total KEPHIS-China Coded Farms in Kenya (External Reference)",
                "Newly Inspected Farms (External Reference)",
                "SHAPe % of National Total",
            ],
            "Value": [
                int(d["is_gacc_approved"].sum()),
                270,  # External reference - total KEPHIS-China coded farms in Kenya
                41,   # External reference - newly inspected farms
                f"{(int(d['is_gacc_approved'].sum()) / 270 * 100):.1f}%" if int(d['is_gacc_approved'].sum()) > 0 else "0%",
            ]
        })
    else:
        gacc_tracking = pd.DataFrame(columns=["Metric", "Value"])
    
    # ==========================================================
    # NEW: AEZ Distribution Sheet
    # ==========================================================
    if "aez_zone" in d.columns:
        aez_dist = d["aez_zone"].value_counts().reset_index()
        aez_dist.columns = ["AEZ Zone", "Number of Farms"]
        aez_dist["Percentage"] = (aez_dist["Number of Farms"] / len(d) * 100).round(1)
        
        # Add altitude distribution by AEZ
        if "altitude" in d.columns:
            aez_altitude = d.groupby("aez_zone")["altitude"].agg(["count", "mean", "min", "max"]).reset_index()
            aez_altitude.columns = ["AEZ Zone", "Farm Count", "Mean Altitude (m)", "Min Altitude (m)", "Max Altitude (m)"]
        else:
            aez_altitude = pd.DataFrame()
    else:
        aez_dist = pd.DataFrame()
        aez_altitude = pd.DataFrame()

    return {
        "Baseline": d,
        "Metrics": metrics,
        "Certifications": cert,
        "Training Needs": training,
        "Market Summary": market,
        "GACC Tracking": gacc_tracking,
        "AEZ Distribution": aez_dist,
        "AEZ Altitude": aez_altitude,
    }


def coverage_diagnostics(canonical_df: pd.DataFrame, used_mapping: Dict[str, str]) -> Dict[str, Any]:
    if canonical_df is None or canonical_df.empty:
        return {
            "rows": 0,
            "missing_required": REQUIRED_FOR_STABILITY.copy(),
            "coverage": [],
            "mapping_used": used_mapping,
            "flags": ["No data rows loaded."],
        }

    d = canonical_df
    rows = int(len(d))
    flags = []

    missing_required = []
    for f in REQUIRED_FOR_STABILITY:
        if f not in d.columns or d[f].isna().all():
            missing_required.append(f)

    recommended_fields = list(dict.fromkeys(list(RECOMMENDED_FOR_EXEC) + ["harvest_kg", "yield_per_acre", "income_per_acre", "income_per_tree", "education", "experience"]))
    cov = []
    for f in recommended_fields:
        if f not in d.columns:
            cov.append({"field": f, "present_pct": 0.0, "non_null_count": 0, "used_from_raw": ""})
            continue
        non_null = int(d[f].notna().sum())
        pct = float(non_null / max(rows, 1)) * 100.0
        cov.append(
            {
                "field": f,
                "present_pct": pct,
                "non_null_count": non_null,
                "used_from_raw": used_mapping.get(f, ""),
            }
        )

    if "lat" in d.columns and "lon" in d.columns:
        if int(d[["lat", "lon"]].dropna().shape[0]) == 0:
            flags.append("No valid GPS coordinates detected (lat/lon missing).")

    if "market_outlet" in d.columns and d["market_outlet"].astype("string").str.strip().replace("", np.nan).isna().mean() > 0.75:
        flags.append("Market outlet is mostly missing (outlet coverage is low).")

    if "submit_date" in d.columns and d["submit_date"].notna().sum() == 0:
        flags.append("No submission/interview dates detected (trend analysis will be limited).")

    if "harvest_kg" not in d.columns or d["harvest_kg"].notna().sum() == 0:
        flags.append("Harvest (kg) is missing/empty (yield analytics will be limited).")
    
    if "education" not in d.columns or d["education"].notna().sum() == 0:
        flags.append("Education data not available - column mapping may need adjustment.")
    
    if "experience" not in d.columns or d["experience"].notna().sum() == 0:
        flags.append("Experience data not available - column mapping may need adjustment.")

    return {
        "rows": rows,
        "missing_required": missing_required,
        "coverage": cov,
        "mapping_used": used_mapping,
        "flags": flags,
    }


def build_export_workbook_bytes(sheets: Dict[str, pd.DataFrame], report: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name in ["Baseline", "Metrics", "Certifications", "Training Needs", "Market Summary", "GACC Tracking", "AEZ Distribution", "AEZ Altitude"]:
            df = sheets.get(name, pd.DataFrame())
            if not df.empty:
                df.to_excel(writer, sheet_name=name[:31], index=False)

        cov = pd.DataFrame(report.get("coverage", []))
        cov.to_excel(writer, sheet_name="Data_Quality_Report", index=False)

        mapping_used = report.get("mapping_used", {}) or {}
        mp = pd.DataFrame([{"canonical_field": k, "raw_column_used": v} for k, v in mapping_used.items()])
        mp.to_excel(writer, sheet_name="Mapping_Used", index=False)

        flags = report.get("flags", []) or []
        fl = pd.DataFrame([{"flag": x} for x in flags])
        fl.to_excel(writer, sheet_name="Flags", index=False)

    return buf.getvalue()


def load_and_prepare_from_excel_bytes(xlsx_bytes: bytes) -> Dict[str, Any]:
    sheets = _read_excel_bytes(xlsx_bytes)
    source_type = detect_source_type(sheets)

    if source_type == "processed":
        base_src = None
        for k, v in sheets.items():
            if str(k).strip().lower() == "baseline":
                base_src = v
                break
        if base_src is None:
            base_src = _first_sheet(sheets)

        base_src = clean_column_names(base_src)
        canonical_df, used = map_to_canonical(base_src)

        derived = build_derived_sheets(canonical_df)
        report = coverage_diagnostics(canonical_df, used)
        wb_bytes = build_export_workbook_bytes(derived, report)

        return {
            "source_type": "processed",
            "baseline_df": canonical_df,
            "derived_sheets": derived,
            "report": report,
            "export_workbook_bytes": wb_bytes,
        }

    raw = _first_sheet(sheets)
    raw = clean_column_names(raw)

    canonical_df, used = map_to_canonical(raw)

    train_oh, _ = extract_training_onehots(raw)
    if not train_oh.empty:
        canonical_df = canonical_df.join(train_oh)

    derived = build_derived_sheets(canonical_df)
    report = coverage_diagnostics(canonical_df, used)
    wb_bytes = build_export_workbook_bytes(derived, report)

    return {
        "source_type": "raw",
        "baseline_df": canonical_df,
        "derived_sheets": derived,
        "report": report,
        "export_workbook_bytes": wb_bytes,
    }