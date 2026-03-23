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
CANONICAL_FIELDS: List[SchemaField] = [
    # ==========================================================
    # Identity / scope
    # ==========================================================
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

    # ==========================================================
    # Farmer Profile
    # ==========================================================
    SchemaField(
        name="gender",
        dtype="string",
        required=False,
        synonyms=[
            "1.11 Gender",
            "Gender",
            "gender",
        ],
    ),
    SchemaField(
        name="age",
        dtype="float",
        required=False,
        synonyms=[
            "1.12 Age",
            "Age",
            "age",
        ],
    ),
    SchemaField(
        name="education",
        dtype="string",
        required=False,
        synonyms=[
            "1.13 Formal education level",
            "Formal education level",
            "Education Level",
            "education",
        ],
    ),
    SchemaField(
        name="experience",
        dtype="float",
        required=False,
        synonyms=[
            "1.14 Experience in Avocado farming in years",
            "Experience in Avocado farming in years",
            "Farming Experience",
            "experience",
        ],
    ),

    # ==========================================================
    # Geography
    # ==========================================================
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

    # ==========================================================
    # Legal Registration
    # ==========================================================
    SchemaField(
        name="orchard_group",
        dtype="bool",
        required=False,
        synonyms=[
            "1.24 Is this orchard registered as part of a Group",
            "Orchard Group Registration",
            "Group Registration",
        ],
    ),

    # ==========================================================
    # Scale
    # ==========================================================
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

    # ==========================================================
    # Age portfolio (tree counts)
    # ==========================================================
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

    # ==========================================================
    # Fruits per tree (yield proxy)
    # ==========================================================
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

    # ==========================================================
    # Production (harvest & losses)
    # ==========================================================
    SchemaField(
        name="harvest_kg",
        dtype="float",
        required=False,
        synonyms=[
            "4.2 Total Harvest Last Season (kg)",
            "Total Harvest Last Season (kg)",
            "Harvest Last Season (kg)",
            "Total Harvest (kg)",
            "Harvest (kg)",
            "harvest_kg",
        ],
    ),
    SchemaField(
        name="harvest_losses_kg",
        dtype="float",
        required=False,
        synonyms=[
            "4.3 Avocado Losses last season (kg)",
            "Avocado Losses (kg)",
            "Losses (kg)",
        ],
    ),
    SchemaField(
        name="loss_causes",
        dtype="string",
        required=False,
        synonyms=[
            "4.31 Primary Cause of Loss last season",
            "Primary Cause of Loss",
        ],
    ),

    # ==========================================================
    # Production Practices
    # ==========================================================
    SchemaField(
        name="fertilizer_organic",
        dtype="bool",
        required=False,
        synonyms=[
            "3.3 Type of Fertilizer Used/Organic",
            "Organic Fertilizer",
        ],
    ),
    SchemaField(
        name="fertilizer_inorganic",
        dtype="bool",
        required=False,
        synonyms=[
            "3.3 Type of Fertilizer Used/Inorganic",
            "Inorganic Fertilizer",
        ],
    ),
    SchemaField(
        name="fertilizer_quantity",
        dtype="float",
        required=False,
        synonyms=[
            "3.312 Provide the annual quantity of Solid Organic Fertilizer used per tree (in kgs)",
            "3.314 Provide the annual quantity of Solid Inorganic Fertilizer used per tree (in kgs)",
            "Fertilizer Quantity (kg/tree)",
        ],
    ),
    SchemaField(
        name="soil_conservation",
        dtype="string",
        required=False,
        synonyms=[
            "3.4 Soil Conservation Measures Applied",
            "Soil Conservation",
        ],
    ),
    SchemaField(
        name="irrigation_drip",
        dtype="bool",
        required=False,
        synonyms=[
            "3.5 Irrigation Practices/Drip",
            "Drip Irrigation",
        ],
    ),
    SchemaField(
        name="irrigation_sprinkler",
        dtype="bool",
        required=False,
        synonyms=[
            "3.5 Irrigation Practices/Sprinkler",
            "Sprinkler Irrigation",
        ],
    ),
    SchemaField(
        name="irrigation_rainfed",
        dtype="bool",
        required=False,
        synonyms=[
            "3.5 Irrigation Practices/Rainfed",
            "Rainfed",
        ],
    ),

    # ==========================================================
    # Compliance & Certifications
    # ==========================================================
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
        name="sanitation_tools",
        dtype="bool",
        required=False,
        synonyms=[
            "6.4 Use of Clean Harvesting Tools",
            "Clean Harvesting Tools",
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
    SchemaField(
        name="ipm_use",
        dtype="bool",
        required=False,
        synonyms=[
            "3.8 Is an Integrated Pest Management (IPM) program implemented?",
            "IPM Implemented",
        ],
    ),
    SchemaField(
        name="biological_control",
        dtype="bool",
        required=False,
        synonyms=[
            "3.83 Biological Control",
            "Biological Control",
        ],
    ),
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

    # ==========================================================
    # Market & Income
    # ==========================================================
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
        name="hass_price_current_ksh_per_kg",
        dtype="float",
        required=False,
        synonyms=[
            "5.6 Average Selling Price of (Hass variety) per kg this Season (KSH)",
            "Current Hass price per kg",
        ],
    ),
    SchemaField(
        name="grade1_share_last",
        dtype="float",
        required=False,
        synonyms=[
            "5.4 What proportion of your harvest did you sell as Grade 1 last season (%)?",
            "Grade 1 Share Last Season",
        ],
    ),
    SchemaField(
        name="grade1_share_current",
        dtype="float",
        required=False,
        synonyms=[
            "5.8 What proportion of your harvest did you sell as Grade 1 this season (%)?",
            "Grade 1 Share Current Season",
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
    SchemaField(
        name="market_constraints",
        dtype="string",
        required=False,
        synonyms=[
            "5.10 Challenges in Market Access",
            "Market Access Challenges",
        ],
    ),

    # ==========================================================
    # Training & Extension
    # ==========================================================
    SchemaField(
        name="sps_training",
        dtype="bool",
        required=False,
        synonyms=[
            "6.1 Training on SPS & Compliance Received",
            "SPS Training Received",
        ],
    ),
    SchemaField(
        name="training_provider",
        dtype="string",
        required=False,
        synonyms=[
            "8.3 Who provided the training",
            "Training Provider",
        ],
    ),
    SchemaField(
        name="record_keeping",
        dtype="bool",
        required=False,
        synonyms=[
            "6.3 Record-Keeping Practices",
            "Record Keeping",
        ],
    ),
    SchemaField(
        name="extension_access",
        dtype="string",
        required=False,
        synonyms=[
            "8.4 Extension Services Accessed",
            "Extension Services",
        ],
    ),
    SchemaField(
        name="training_needs_text",
        dtype="string",
        required=False,
        synonyms=[
            "8.6 What are your most pressing training/extension needs",
            "Training Needs",
            "Suggestions for the Shape Program Improvement",
        ],
    ),

    # ==========================================================
    # Sustainability
    # ==========================================================
    SchemaField(
        name="water_source",
        dtype="string",
        required=False,
        synonyms=[
            "7.1 Water Source for Irrigation",
            "Water Source",
        ],
    ),
    SchemaField(
        name="waste_management",
        dtype="string",
        required=False,
        synonyms=[
            "7.2 Waste Management Practices",
            "Waste Management",
        ],
    ),
    SchemaField(
        name="biodiversity",
        dtype="string",
        required=False,
        synonyms=[
            "7.3 Biodiversity Conservation Practices",
            "Biodiversity Practices",
        ],
    ),
    SchemaField(
        name="additional_trees",
        dtype="string",
        required=False,
        synonyms=[
            "7.4 What other trees are planted around/within the orchard",
            "Other Trees Planted",
        ],
    ),
    SchemaField(
        name="other_value_chains",
        dtype="string",
        required=False,
        synonyms=[
            "7.5 What other value chains is the farmer involved in",
            "Other Value Chains",
        ],
    ),

    # ==========================================================
    # Dates
    # ==========================================================
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
    # Identity
    "exporter",
    "farmer_name",
    "orchard_name",
    # Farmer Profile
    "gender",
    "age",
    "education",
    "experience",
    # Geography
    "county",
    "sub_county",
    "ward",
    "lat",
    "lon",
    # Scale
    "area_acres",
    "trees_total",
    # Age portfolio
    "trees_0_3",
    "trees_4_7",
    "trees_8_plus",
    # Fruits per tree
    "fruits_per_tree_0_3",
    "fruits_per_tree_4_7",
    "fruits_per_tree_8_plus",
    # Production
    "harvest_kg",
    "harvest_losses_kg",
    "loss_causes",
    # Production Practices
    "fertilizer_organic",
    "fertilizer_inorganic",
    "soil_conservation",
    "irrigation_drip",
    "irrigation_sprinkler",
    "irrigation_rainfed",
    # Compliance
    "gacc_status",
    "kephis_status",
    "sanitation_record",
    "sanitation_tools",
    "pest_monitoring",
    "approved_pesticides",
    "ipm_use",
    "biological_control",
    # Certifications
    "cert_globalgap",
    "cert_organic",
    "cert_fairtrade",
    # Market
    "market_outlet",
    "hass_price_ksh_per_kg",
    "hass_price_current_ksh_per_kg",
    "grade1_share_last",
    "grade1_share_current",
    "income_ksh_last_season",
    "market_constraints",
    # Training
    "sps_training",
    "training_provider",
    "record_keeping",
    "extension_access",
    "training_needs_text",
    # Sustainability
    "water_source",
    "waste_management",
    "biodiversity",
    "additional_trees",
    "other_value_chains",
    # Date
    "submit_date",
]


def canonical_field_map() -> Dict[str, SchemaField]:
    return {f.name: f for f in CANONICAL_FIELDS}


def all_canonical_names() -> List[str]:
    return [f.name for f in CANONICAL_FIELDS]