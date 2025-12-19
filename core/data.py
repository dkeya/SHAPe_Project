# core/data.py
from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from pathlib import Path

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


def _repo_root() -> Path:
    """
    Best-effort repo root locator.
    core/data.py is usually: <repo_root>/core/data.py
    """
    return Path(__file__).resolve().parents[1]

def _find_default_workbook_path() -> Optional[Path]:
    """
    Looks for the default workbook bundled with the app (in the repo).
    We ONLY support shape_data.xlsx as the auto-load default.
    """
    root = _repo_root()

    candidates = [
        root / "shape_data.xlsx",
        root / "data" / "shape_data.xlsx",
    ]

    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None

def _read_bytes_from_path(p: Path) -> bytes:
    with p.open("rb") as f:
        return f.read()

# -------------------------------------------------------------------
# Enterprise-grade loader (single source of truth)
# -------------------------------------------------------------------
def load_and_prepare_data(uploaded_file=None) -> DataPackage:
    """
    Loader priority (emulates old behavior, but keeps upload option):

    1) If user uploaded a workbook: use it
    2) Else if a previous upload exists in session_state: reuse it (multipage friendly)
    3) Else try a default workbook packaged inside the repo (shape_data.xlsx etc.)
    4) Else return empty package + friendly diagnostics
    """
    xlsx_bytes: Optional[bytes] = None
    source_type: str = "unknown"

    # 1) Upload wins
    if uploaded_file is not None:
        xlsx_bytes = uploaded_file.getvalue()
        set_uploaded_workbook_bytes_in_session(xlsx_bytes)
        source_type = "upload"

    # 2) Session cached
    if xlsx_bytes is None:
        xlsx_bytes = get_uploaded_workbook_bytes_from_session()
        if xlsx_bytes is not None:
            source_type = "session"

    # 3) Bundled default file in repo (old behavior)
    default_path = None
    if xlsx_bytes is None:
        default_path = _find_default_workbook_path()
        if default_path is not None:
            xlsx_bytes = _read_bytes_from_path(default_path)
            source_type = f"bundled:{default_path.name}"

    # 4) Still none → return empty + diagnostics
    if xlsx_bytes is None:
        looked_in = []
        root = _repo_root()
        looked_in.extend(
            [
                str(root / "shape_data.xlsx"),
                str(root / "data" / "shape_data.xlsx"),
                str(root / "shape_raw_data.xlsx"),
                str(root / "data" / "shape_raw_data.xlsx"),
            ]
        )

        return DataPackage(
            source_type="none",
            baseline_df=pd.DataFrame(),
            sheets={},
            report={
                "rows": 0,
                "missing_required": ["exporter"],
                "coverage": [],
                "mapping_used": {},
                "flags": [
                    "No file uploaded and no bundled default workbook found.",
                    "To enable auto-load on Streamlit Cloud, commit one of these files into the GitHub repo:",
                    *[f" - {p}" for p in looked_in],
                ],
            },
            workbook_bytes=b"",
        )

    # Transform
    h = _bytes_hash(xlsx_bytes)
    payload = _prepare_cached(xlsx_bytes, h)

    baseline_df = _ensure_df(payload.get("baseline_df"))
    derived = payload.get("derived_sheets")
    derived_sheets = derived if isinstance(derived, dict) else {}

    report = payload.get("report")
    report = report if isinstance(report, dict) else {}
    report.setdefault("rows", int(len(baseline_df)) if isinstance(baseline_df, pd.DataFrame) else 0)

    workbook_bytes = payload.get("export_workbook_bytes")
    workbook_bytes = workbook_bytes if isinstance(workbook_bytes, (bytes, bytearray)) else b""

    return DataPackage(
        source_type=str(payload.get("source_type", source_type)),
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
                "Notes": ["Upload a workbook to view column coverage."],
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
                {"Field": logical, "Detected Column": "", "Completeness (%)": 0.0, "Missing (%)": 100.0, "Notes": "Not found"}
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
        elif any(k in logical.lower() for k in ["price", "area", "trees", "income"]):
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
    out = out.sort_values(["Missing (%)", "Field"], ascending=[False, True]).reset_index(drop=True)
    return out
