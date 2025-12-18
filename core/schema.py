# core/schema.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _norm(s: str) -> str:
    """Normalize a label for matching (lower, strip, collapse spaces)."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = s.strip().lower()
    return " ".join(s.split())


@dataclass(frozen=True)
class SchemaField:
    name: str
    dtype: str  # "string" | "float" | "int" | "bool" | "datetime"
    required: bool = False
    synonyms: List[str] = None  # raw/legacy names

    def norm_synonyms(self) -> List[str]:
        syns = self.synonyms or []
        return [_norm(x) for x in syns if x is not None]


# -------------------------------------------------------------------
# Canonical Monitoring schema
# -------------------------------------------------------------------
# Notes:
# - We keep canonical names short and stable.
# - synonyms include your known raw question labels + common variants.
# - Dashboard should read canonical columns only.
CANONICAL_FIELDS: List[SchemaField] = [
    # Identity / scope
    SchemaField(
        name="exporter",
        dtype="string",
        required=True,
        synonyms=[
            "1.1 Company Name",
            "Company",
            "Exporter",
            "exporter",
        ],
    ),
    SchemaField(
        name="farmer_name",
        dtype="string",
        required=False,
        synonyms=[
            "1.10 Farmer's Name (Three Names)",
            "1.10 Farmer’s Name (Three Names)",
            "Farmer",
            "Farmer Name",
        ],
    ),
    SchemaField(
        name="orchard_name",
        dtype="string",
        required=False,
        synonyms=[
            "1.22 Orchard Name/ Name of farm",
            "Orchard",
            "Farm Name",
        ],
    ),

    # Geography
    SchemaField(
        name="county",
        dtype="string",
        required=False,
        synonyms=[
            "1.2 County",
            "County",
        ],
    ),
    SchemaField(
        name="sub_county",
        dtype="string",
        required=False,
        synonyms=[
            "1.3 Sub County",
            "Sub County",
            "Sub_County",
        ],
    ),
    SchemaField(
        name="ward",
        dtype="string",
        required=False,
        synonyms=[
            "1.4 Ward",
            "Ward",
        ],
    ),
    SchemaField(
        name="lat",
        dtype="float",
        required=False,
        synonyms=[
            "_1.21 GPS Coordinates of Orchard_latitude",
            "1.21 GPS Coordinates of Orchard_latitude",
            "GPS_Latitude",
            "Latitude",
            "lat",
        ],
    ),
    SchemaField(
        name="lon",
        dtype="float",
        required=False,
        synonyms=[
            "_1.21 GPS Coordinates of Orchard_longitude",
            "1.21 GPS Coordinates of Orchard_longitude",
            "GPS_Longitude",
            "Longitude",
            "lon",
        ],
    ),

    # Scale
    SchemaField(
        name="area_acres",
        dtype="float",
        required=False,
        synonyms=[
            "2.2 Total Area under Avocado Cultivation (Acres)",
            "Total Area under Avocado Cultivation (Acres)",
            "Avocado Area (Acres)",
        ],
    ),
    SchemaField(
        name="trees_total",
        dtype="float",
        required=False,
        synonyms=[
            "2.3 Number of Avocado Trees Planted",
            "Number of Avocado Trees Planted",
            "Total Trees",
        ],
    ),

    # ✅ NEW: Production (for yield_per_acre and analytics models)
    SchemaField(
        name="harvest_kg",
        dtype="float",
        required=False,
        synonyms=[
            "4.2 Total Harvest Last Season (kg)",
            "Total Harvest Last Season (kg)",
            "Harvest (kg)",
            "harvest_kg",
        ],
    ),

    # Age portfolio (tree counts)
    SchemaField(
        name="trees_0_3",
        dtype="float",
        required=False,
        synonyms=[
            "2.41 Number of trees for Age class 0-3 years",
            "Trees 0-3",
        ],
    ),
    SchemaField(
        name="trees_4_7",
        dtype="float",
        required=False,
        synonyms=[
            "2.42 Number of trees for Age class 4-7 years",
            "Trees 4-7",
        ],
    ),
    SchemaField(
        name="trees_8_plus",
        dtype="float",
        required=False,
        synonyms=[
            "2.43 Number of trees for Age class 8+ years",
            "Trees 8+",
        ],
    ),

    # Fruits per tree (yield proxy)
    SchemaField(
        name="fruits_per_tree_0_3",
        dtype="float",
        required=False,
        synonyms=[
            "4.8 Average No. of Fruits per avocado tree aged 0-3 years",
            "Fruits per tree 0-3 years",
        ],
    ),
    SchemaField(
        name="fruits_per_tree_4_7",
        dtype="float",
        required=False,
        synonyms=[
            "4.81 Average No. of Fruits per avocado tree aged 4-7 years",
            "Fruits per tree 4-7 years",
        ],
    ),
    SchemaField(
        name="fruits_per_tree_8_plus",
        dtype="float",
        required=False,
        synonyms=[
            "4.82 Average No. of Fruits per avocado tree aged 8+ years",
            "Fruits per tree 8+ years",
        ],
    ),

    # Certifications / compliance
    SchemaField(
        name="gacc_status",
        dtype="bool",
        required=False,
        synonyms=[
            "1.26 General Administration of Customs of the Peoples Republic of China (GACC ) Approval Status",
            "1.26 General Administration of Customs of the Peoples Republic of China (GACC ) Approval Status ",
            "GACC approval",
        ],
    ),
    SchemaField(
        name="kephis_status",
        dtype="bool",
        required=False,
        synonyms=[
            "1.25 KEPHIS Registration Status",
            "KEPHIS registration",
        ],
    ),
    SchemaField(
        name="sanitation_record",
        dtype="bool",
        required=False,
        synonyms=[
            "3.6 Is there a record of sanitation conditions?",
            "Sanitation records",
        ],
    ),
    SchemaField(
        name="pest_monitoring",
        dtype="bool",
        required=False,
        synonyms=[
            "3.81 Pest Monitoring",
            "Pest Monitoring",
        ],
    ),
    SchemaField(
        name="approved_pesticides",
        dtype="bool",
        required=False,
        synonyms=[
            "6.7 Use of Approved Pesticides Only",
            "Use of Approved Pesticides Only",
        ],
    ),

    # Certification types (checkbox style)
    SchemaField(
        name="cert_globalgap",
        dtype="bool",
        required=False,
        synonyms=[
            "6.2 Which Certifications is the Orchard compliant for this season?/Global GAP",
            "Global GAP",
            "GlobalGAP",
        ],
    ),
    SchemaField(
        name="cert_organic",
        dtype="bool",
        required=False,
        synonyms=[
            "6.2 Which Certifications is the Orchard compliant for this season?/Organic",
            "Organic",
        ],
    ),
    SchemaField(
        name="cert_fairtrade",
        dtype="bool",
        required=False,
        synonyms=[
            "6.2 Which Certifications is the Orchard compliant for this season?/FairTrade",
            "FairTrade",
        ],
    ),

    # Market & income
    SchemaField(
        name="market_outlet",
        dtype="string",
        required=False,
        synonyms=[
            "5.1 Main Market Outlet",
            "Main Market Outlet",
        ],
    ),
    SchemaField(
        name="hass_price_ksh_per_kg",
        dtype="float",
        required=False,
        synonyms=[
            "5.2 Average Selling Price of (Hass variety) per kg last Season (KSH) ",
            "Hass price per kg",
        ],
    ),
    SchemaField(
        name="income_ksh_last_season",
        dtype="float",
        required=False,
        synonyms=[
            "5.3 Total Income from Avocado Sales (KSH last season)",
            "Income from avocado sales (last season)",
        ],
    ),

    # Dates
    SchemaField(
        name="submit_date",
        dtype="datetime",
        required=False,
        synonyms=[
            "submitdate",
            "submissiondate",
            "Submission Date",
            "today",
            "Date of interview:",
            "start",
            "end",
        ],
    ),
]


REQUIRED_FOR_STABILITY = [
    "exporter",
]

RECOMMENDED_FOR_EXEC = [
    "county",
    "sub_county",
    "ward",
    "lat",
    "lon",
    "area_acres",
    "trees_total",
    "harvest_kg",  # ✅ NEW: recommended for yield analytics
    "trees_0_3",
    "trees_4_7",
    "trees_8_plus",
    "market_outlet",
    "hass_price_ksh_per_kg",
    "income_ksh_last_season",
    "submit_date",
]


def canonical_field_map() -> Dict[str, SchemaField]:
    return {f.name: f for f in CANONICAL_FIELDS}


def all_canonical_names() -> List[str]:
    return [f.name for f in CANONICAL_FIELDS]