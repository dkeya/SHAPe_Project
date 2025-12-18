# core/data.py
from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from core.transforms import load_and_prepare_from_excel_bytes

# Yield reference used across dashboard (kg/tree is used in forecast)
YIELD_REFERENCE = {
    "0-3": {"fruits": 275, "kg": 45.8},
    "4-7": {"fruits": 350, "kg": 58.3},
    "8+": {"fruits": 900, "kg": 150.0},
}


@dataclass
class DataPackage:
    source_type: str
    baseline_df: pd.DataFrame
    sheets: Dict[str, pd.DataFrame]
    report: Dict[str, Any]
    workbook_bytes: bytes


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------
def _bytes_hash(b: bytes) -> str:
    return hashlib.sha256(b or b"").hexdigest()


@st.cache_data(show_spinner=False)
def _prepare_cached(xlsx_bytes: bytes, _hash: str) -> Dict[str, Any]:
    # _hash forces cache invalidation when bytes change
    return load_and_prepare_from_excel_bytes(xlsx_bytes)


def get_uploaded_workbook_bytes_from_session() -> Optional[bytes]:
    b = st.session_state.get("shape_workbook_bytes", None)
    if isinstance(b, (bytes, bytearray)) and len(b) > 0:
        return bytes(b)
    return None


def set_uploaded_workbook_bytes_in_session(xlsx_bytes: bytes):
    if isinstance(xlsx_bytes, (bytes, bytearray)) and len(xlsx_bytes) > 0:
        st.session_state["shape_workbook_bytes"] = bytes(xlsx_bytes)


def _ensure_df(x) -> pd.DataFrame:
    return x if isinstance(x, pd.DataFrame) else pd.DataFrame()


# -------------------------------------------------------------------
# Enterprise-grade loader (single source of truth)
# -------------------------------------------------------------------
def load_and_prepare_data(uploaded_file=None) -> DataPackage:
    """
    Enterprise-grade loader:
    - If user uploaded a workbook: use it as the baseline source
    - Else if a previous upload exists in session_state: reuse it (multipage friendly)
    - Else fall back to local shape_data.xlsx ONLY if it exists (demo convenience),
      but the primary workflow is always upload -> transform -> export.
    """
    xlsx_bytes: Optional[bytes] = None

    if uploaded_file is not None:
        # Streamlit UploadedFile
        xlsx_bytes = uploaded_file.getvalue()
        set_uploaded_workbook_bytes_in_session(xlsx_bytes)
    else:
        xlsx_bytes = get_uploaded_workbook_bytes_from_session()

    if xlsx_bytes is None:
        # Optional fallback for demo / local dev
        if os.path.exists("shape_data.xlsx"):
            with open("shape_data.xlsx", "rb") as f:
                xlsx_bytes = f.read()
        else:
            return DataPackage(
                source_type="none",
                baseline_df=pd.DataFrame(),
                sheets={},
                report={
                    "rows": 0,
                    "missing_required": ["exporter"],
                    "coverage": [],
                    "mapping_used": {},
                    "flags": ["No file uploaded."],
                },
                workbook_bytes=b"",
            )

    h = _bytes_hash(xlsx_bytes)
    payload = _prepare_cached(xlsx_bytes, h)

    baseline_df = _ensure_df(payload.get("baseline_df"))
    derived = payload.get("derived_sheets")
    derived_sheets = derived if isinstance(derived, dict) else {}

    report = payload.get("report")
    report = report if isinstance(report, dict) else {}

    workbook_bytes = payload.get("export_workbook_bytes")
    workbook_bytes = workbook_bytes if isinstance(workbook_bytes, (bytes, bytearray)) else b""

    return DataPackage(
        source_type=str(payload.get("source_type", "unknown")),
        baseline_df=baseline_df,
        sheets=derived_sheets,
        report=report,
        workbook_bytes=bytes(workbook_bytes),
    )


# -------------------------------------------------------------------
# Backwards-compatible helpers (keep existing imports working)
# -------------------------------------------------------------------
def load_farmer_data(uploaded_file=None) -> pd.DataFrame:
    pkg = load_and_prepare_data(uploaded_file=uploaded_file)
    return pkg.baseline_df.copy()


def load_exporter_metrics(uploaded_file=None) -> pd.DataFrame:
    pkg = load_and_prepare_data(uploaded_file=uploaded_file)
    df = pkg.sheets.get("Metrics", None)
    return _ensure_df(df).copy()


def load_certifications(uploaded_file=None) -> pd.DataFrame:
    pkg = load_and_prepare_data(uploaded_file=uploaded_file)
    df = pkg.sheets.get("Certifications", None)
    return _ensure_df(df).copy()


def load_training_needs(uploaded_file=None) -> pd.DataFrame:
    pkg = load_and_prepare_data(uploaded_file=uploaded_file)
    df = pkg.sheets.get("Training Needs", None)
    return _ensure_df(df).copy()


# -------------------------------------------------------------------
# Admin helper: key columns summary (expected by pages/99_Admin.py)
# -------------------------------------------------------------------
def key_columns_summary(df: Optional[pd.DataFrame] = None, uploaded_file=None) -> pd.DataFrame:
    """
    Returns a compact completeness / detection summary of key fields.

    This is intentionally tolerant:
    - Works on canonical baseline (preferred)
    - Works on raw baseline that still contains survey-style column names
    - Never crashes if columns are missing
    """
    if df is None:
        df = load_farmer_data(uploaded_file=uploaded_file)

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(
            {
                "Field": ["(no data)"],
                "Detected Column": [""],
                "Completeness (%)": [0.0],
                "Missing (%)": [100.0],
                "Notes": ["Upload a workbook to view column coverage."],
            }
        )

    # Logical field -> candidate column names (canonical first, then legacy raw)
    KEY_COLS: Dict[str, List[str]] = {
        "Exporter": ["exporter", "1.1 Company Name", "Exporter", "Company"],
        "Farmer name": ["farmer_name", "1.10 Farmer's Name (Three Names)", "Farmer", "Farmer_Name"],
        "Farm/Orchard name": ["farm_name", "orchard_name", "1.22 Orchard Name/ Name of farm", "Farm_Name"],
        "County": ["county", "1.2 County", "County"],
        "Sub-county": ["sub_county", "1.3 Sub County", "Sub_County", "Sub County"],
        "Ward": ["ward", "1.4 Ward", "Ward"],
        "Latitude": ["lat", "_1.21 GPS Coordinates of Orchard_latitude", "GPS_Latitude", "latitude"],
        "Longitude": ["lon", "_1.21 GPS Coordinates of Orchard_longitude", "GPS_Longitude", "longitude"],
        "Avocado area (acres)": ["avocado_area_acres", "2.2 Total Area under Avocado Cultivation (Acres)", "Total Area under Avocado Cultivation (Acres)"],
        "Total trees": ["trees_planted", "2.3 Number of Avocado Trees Planted", "Number of Avocado Trees Planted"],
        "Trees age 0–3": ["trees_0_3", "2.41 Number of trees for Age class 0-3 years"],
        "Trees age 4–7": ["trees_4_7", "2.42 Number of trees for Age class 4-7 years"],
        "Trees age 8+": ["trees_8_plus", "trees_8p", "2.43 Number of trees for Age class 8+ years"],
        "GACC approval": ["gacc_approval", "1.26 General Administration of Customs of the Peoples Republic of China (GACC ) Approval Status"],
        "Main market outlet": ["main_market_outlet", "5.1 Main Market Outlet"],
        "Hass price (KSh/kg)": ["hass_price_ksh_per_kg", "5.2 Average Selling Price of (Hass variety) per kg last Season (KSH)", "5.2 Average Selling Price of (Hass variety) per kg last Season (KSH) "],
        "Income (KSh, last season)": ["income_ksh_last_season", "5.3 Total Income from Avocado Sales (KSH last season)"],
        "Submission/interview date": ["submitdate", "submit_date", "data_time", "SubmissionDate", "Date of interview:"],
    }

    cols_lower = {c.lower().strip(): c for c in df.columns}

    def pick_col(candidates: List[str]) -> Optional[str]:
        for cand in candidates:
            k = cand.lower().strip()
            if k in cols_lower:
                return cols_lower[k]
        # exact fallback
        for cand in candidates:
            if cand in df.columns:
                return cand
        return None

    rows: List[Dict[str, Any]] = []
    n = len(df)

    for logical, candidates in KEY_COLS.items():
        col = pick_col(candidates)
        if not col:
            rows.append(
                {
                    "Field": logical,
                    "Detected Column": "",
                    "Completeness (%)": 0.0,
                    "Missing (%)": 100.0,
                    "Notes": "Not found",
                }
            )
            continue

        s = df[col]
        # treat empty strings as missing too
        missing = s.isna() | (s.astype("string").str.strip() == "")
        miss_rate = float(missing.mean() * 100.0) if n else 100.0
        comp_rate = 100.0 - miss_rate

        note = ""
        if logical in ("Latitude", "Longitude"):
            # geo sanity: how many plausible numeric points?
            sn = pd.to_numeric(s, errors="coerce")
            ok = sn.notna().mean() * 100.0
            note = f"Numeric coverage: {ok:.0f}%"
        elif "price" in logical.lower():
            sn = pd.to_numeric(s, errors="coerce")
            note = f"Numeric coverage: {(sn.notna().mean()*100.0):.0f}%"
        elif "area" in logical.lower() or "trees" in logical.lower() or "income" in logical.lower():
            sn = pd.to_numeric(s, errors="coerce")
            note = f"Numeric coverage: {(sn.notna().mean()*100.0):.0f}%"

        rows.append(
            {
                "Field": logical,
                "Detected Column": col,
                "Completeness (%)": round(comp_rate, 1),
                "Missing (%)": round(miss_rate, 1),
                "Notes": note,
            }
        )

    out = pd.DataFrame(rows)
    # Put most problematic on top
    out = out.sort_values(["Missing (%)", "Field"], ascending=[False, True]).reset_index(drop=True)
    return out
