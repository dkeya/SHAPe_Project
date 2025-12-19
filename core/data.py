# core/data.py
from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

import pandas as pd
import streamlit as st
import urllib.request

from core.transforms import load_and_prepare_from_excel_bytes

# Yield reference used across dashboard (kg/tree is used in forecast)
YIELD_REFERENCE = {
    "0-3": {"fruits": 275, "kg": 45.8},
    "4-7": {"fruits": 350, "kg": 58.3},
    "8+": {"fruits": 900, "kg": 150.0},
}

# -------------------------------------------------------------------
# Default dataset locations
# -------------------------------------------------------------------
# Local dev (your machine only)
DEFAULT_LOCAL_WORKBOOK_PATH = r"C:\Users\dkeya\Documents\SBS\2025\shape_project\shape_raw_data.xlsx"

# Repo-bundled demo defaults (works in deployed app if committed)
DEFAULT_REPO_WORKBOOK_PATHS = [
    os.path.join("data", "shape_raw_data.xlsx"),
    "shape_raw_data.xlsx",
    "shape_data.xlsx",
]


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


def clear_uploaded_workbook_bytes_from_session():
    if "shape_workbook_bytes" in st.session_state:
        del st.session_state["shape_workbook_bytes"]


def _ensure_df(x) -> pd.DataFrame:
    return x if isinstance(x, pd.DataFrame) else pd.DataFrame()


def _safe_read_file_bytes(path: str) -> Optional[bytes]:
    try:
        if path and os.path.exists(path) and os.path.isfile(path):
            with open(path, "rb") as f:
                return f.read()
    except Exception:
        return None
    return None


def _download_bytes_from_url(url: str, timeout: int = 25) -> Optional[bytes]:
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        return None


def _get_data_mode() -> str:
    """
    Order:
      1) Session override (admin switch)
      2) Secrets default
      3) demo
    """
    m = st.session_state.get("shape_data_mode", None)
    if isinstance(m, str) and m.strip().lower() in {"demo", "production"}:
        return m.strip().lower()

    try:
        cfg = st.secrets.get("data", {}) or {}
        mm = str(cfg.get("mode", "demo")).strip().lower()
        if mm in {"demo", "production"}:
            return mm
    except Exception:
        pass

    return "demo"


def _get_production_url() -> str:
    """
    Order:
      1) Session override (optional)
      2) Secrets
    """
    u = st.session_state.get("shape_production_url", None)
    if isinstance(u, str) and u.strip():
        return u.strip()

    try:
        cfg = st.secrets.get("data", {}) or {}
        u2 = str(cfg.get("production_url", "")).strip()
        return u2
    except Exception:
        return ""


def _get_default_workbook_bytes() -> Tuple[Optional[bytes], str]:
    """
    Returns (bytes, source_type)
    source_type examples:
      - production_url
      - demo_file:data/shape_raw_data.xlsx
      - local_file:C:\\...
      - none
    """
    mode = _get_data_mode()

    # 1) Production (URL in secrets)
    if mode == "production":
        prod_url = _get_production_url()
        if prod_url:
            b = _download_bytes_from_url(prod_url)
            if b:
                return b, "production_url"

    # 2) Demo / repo-bundled file (recommended for public repos)
    for p in DEFAULT_REPO_WORKBOOK_PATHS:
        b = _safe_read_file_bytes(p)
        if b:
            return b, f"demo_file:{p}"

    # 3) Local dev path (your Windows machine)
    b = _safe_read_file_bytes(DEFAULT_LOCAL_WORKBOOK_PATH)
    if b:
        return b, f"local_file:{DEFAULT_LOCAL_WORKBOOK_PATH}"

    return None, "none"


# -------------------------------------------------------------------
# Enterprise-grade loader (single source of truth)
# -------------------------------------------------------------------
def load_and_prepare_data(uploaded_file=None) -> DataPackage:
    """
    Enterprise-grade loader priority:
      1) If user uploaded a workbook: use it
      2) Else if a previous upload exists in session_state: reuse it (multipage friendly)
      3) Else auto-load default:
          - production_url (if admin set mode=production + secrets url exists)
          - demo_file from repo paths
          - local_file Windows path (local dev only)
    """
    xlsx_bytes: Optional[bytes] = None
    source_type: str = "unknown"

    if uploaded_file is not None:
        xlsx_bytes = uploaded_file.getvalue()
        set_uploaded_workbook_bytes_in_session(xlsx_bytes)
        source_type = "upload"
    else:
        xlsx_bytes = get_uploaded_workbook_bytes_from_session()
        if xlsx_bytes is not None:
            source_type = "session_upload"

    if xlsx_bytes is None:
        xlsx_bytes, source_type = _get_default_workbook_bytes()

    if xlsx_bytes is None:
        return DataPackage(
            source_type="none",
            baseline_df=pd.DataFrame(),
            sheets={},
            report={
                "rows": 0,
                "missing_required": ["exporter"],
                "coverage": [],
                "mapping_used": {},
                "flags": ["No file loaded (no upload, no default file, and no production URL)."],
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

    # Add loader meta
    report = dict(report)
    report["source_type"] = source_type
    report["data_mode"] = _get_data_mode()

    return DataPackage(
        source_type=str(source_type),
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
    if df is None:
        df = load_farmer_data(uploaded_file=uploaded_file)

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(
            {
                "Field": ["(no data)"],
                "Detected Column": [""],
                "Completeness (%)": [0.0],
                "Missing (%)": [100.0],
                "Notes": ["Upload a workbook or configure a default dataset to view column coverage."],
            }
        )

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
        "GACC approval": ["gacc_approval", "gacc_status", "1.26 General Administration of Customs of the Peoples Republic of China (GACC ) Approval Status"],
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
        missing = s.isna() | (s.astype("string").str.strip() == "")
        miss_rate = float(missing.mean() * 100.0) if n else 100.0
        comp_rate = 100.0 - miss_rate

        note = ""
        if logical in ("Latitude", "Longitude"):
            sn = pd.to_numeric(s, errors="coerce")
            ok = sn.notna().mean() * 100.0
            note = f"Numeric coverage: {ok:.0f}%"
        elif any(x in logical.lower() for x in ["price", "area", "trees", "income"]):
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

    out = pd.DataFrame(rows).sort_values(["Missing (%)", "Field"], ascending=[False, True]).reset_index(drop=True)
    return out
