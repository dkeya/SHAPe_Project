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
    # keep slashes for multi-select labels, but normalize spacing/punct
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
    """
    Determine if user uploaded:
    - "processed": workbook already has Baseline/Metrics etc
    - "raw": workbook is a raw survey export (often 1 big sheet)
    """
    if not sheets:
        return "unknown"

    sheet_names = {str(s).strip().lower() for s in sheets.keys()}
    if "baseline" in sheet_names:
        return "processed"

    # If there are many sheets but none baseline, we still treat as raw export
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
    """
    Normalize messy yes/no or 0/1 checkbox fields into boolean.
    """
    if s is None:
        return pd.Series(dtype="bool")
    # numeric checkbox
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.60:
        return (num.fillna(0) > 0).astype(bool)

    # strings
    st = s.astype("string").fillna("").str.strip().str.lower()
    return st.isin(["yes", "y", "true", "1", "approved", "compliant", "available"])


def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _to_str_series(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.strip()


def _to_datetime_series(s: pd.Series) -> pd.Series:
    # Try parsing; many survey exports have mixed formats
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)


# ----------------------------------------------------------
# Extra fields (robust fallback, even if schema not updated)
# ----------------------------------------------------------
EXTRA_FIELDS = [
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
]


def _add_extra_field(
    raw_df: pd.DataFrame,
    out: pd.DataFrame,
    used: Dict[str, str],
    name: str,
    dtype: str,
    synonyms: List[str],
) -> None:
    """
    Adds/overrides a canonical field from raw_df using synonyms, only if:
      - field is missing from out, OR
      - field exists but is all-NaN
    """
    if name in out.columns and out[name].notna().any():
        return

    src = _find_matching_col(list(raw_df.columns), synonyms or [])
    if src is None:
        if name not in out.columns:
            out[name] = np.nan
        return

    used[name] = src
    s = raw_df[src]
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


def map_to_canonical(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Produce canonical baseline dataframe from a raw dataframe.
    Returns (canonical_df, used_mapping) where used_mapping is {canonical_field: raw_column_used}.
    """
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

    # 3) Derived stability fields
    out["trees_per_acre"] = np.nan
    if "trees_total" in out.columns and "area_acres" in out.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["trees_per_acre"] = out["trees_total"] / out["area_acres"]

    # Income per acre
    out["income_per_acre"] = np.nan
    if "income_ksh_last_season" in out.columns and "area_acres" in out.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["income_per_acre"] = out["income_ksh_last_season"] / out["area_acres"]

    # Yield per acre (NEW)
    out["yield_per_acre"] = np.nan
    if "harvest_kg" in out.columns and "area_acres" in out.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["yield_per_acre"] = out["harvest_kg"] / out["area_acres"]

    # Dominant age group
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

    # Clean obvious infinities
    out = out.replace([np.inf, -np.inf], np.nan)

    return out, used


def extract_training_onehots(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Detect raw one-hot training need columns like:
      '8.6 What are your most pressing training/extension needs/<Option>'
    Convert to canonical columns:
      training_need__<slug(option)>
    """
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
    """
    Create derived sheets from the canonical baseline.
    """
    d = canonical_df.copy()
    if d is None or d.empty:
        return {
            "Baseline": pd.DataFrame(),
            "Metrics": pd.DataFrame(),
            "Certifications": pd.DataFrame(),
            "Training Needs": pd.DataFrame(),
            "Market Summary": pd.DataFrame(),
        }

    # Metrics (make it resilient to optional fields)
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

    metrics = (
        d.groupby("exporter", dropna=False)
        .agg(**agg_spec)
        .reset_index()
        .rename(columns={"exporter": "Exporter"})
    )

    # Certifications (farm-level view)
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
    ]
    cert = d[[c for c in cert_cols if c in d.columns]].copy()

    # Training Needs summary: top need per exporter if onehots exist
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

    # Market Summary
    market_rows = []
    if "exporter" in d.columns:
        for exp, g in d.groupby("exporter"):
            outlet = g.get("market_outlet", pd.Series(dtype="string"))
            outlet = outlet.astype("string").str.strip()
            outlet = outlet[outlet != ""]
            top_outlet = outlet.value_counts().index[0] if len(outlet) else ""

            prices = pd.to_numeric(g.get("hass_price_ksh_per_kg", pd.Series(dtype="float")), errors="coerce").dropna()
            incomes = pd.to_numeric(g.get("income_ksh_last_season", pd.Series(dtype="float")), errors="coerce").dropna()

            market_rows.append(
                {
                    "Exporter": exp,
                    "Top_Outlet": top_outlet,
                    "Median_Hass_Price_KSh_per_Kg": float(prices.median()) if len(prices) else np.nan,
                    "Median_Income_KSh_Last_Season": float(incomes.median()) if len(incomes) else np.nan,
                }
            )
    market = pd.DataFrame(market_rows)

    return {
        "Baseline": d,
        "Metrics": metrics,
        "Certifications": cert,
        "Training Needs": training,
        "Market Summary": market,
    }


def coverage_diagnostics(canonical_df: pd.DataFrame, used_mapping: Dict[str, str]) -> Dict[str, Any]:
    """
    Return a diagnostics report used to drive data coverage messaging and exec insights.
    """
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

    # Required checks
    missing_required = []
    for f in REQUIRED_FOR_STABILITY:
        if f not in d.columns or d[f].isna().all():
            missing_required.append(f)

    # Coverage table
    recommended_fields = list(dict.fromkeys(list(RECOMMENDED_FOR_EXEC) + ["harvest_kg", "yield_per_acre"]))
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

    # High-level flags
    if "lat" in d.columns and "lon" in d.columns:
        if int(d[["lat", "lon"]].dropna().shape[0]) == 0:
            flags.append("No valid GPS coordinates detected (lat/lon missing).")

    if "market_outlet" in d.columns and d["market_outlet"].astype("string").str.strip().replace("", np.nan).isna().mean() > 0.75:
        flags.append("Market outlet is mostly missing (outlet coverage is low).")

    if "submit_date" in d.columns and d["submit_date"].notna().sum() == 0:
        flags.append("No submission/interview dates detected (trend analysis will be limited).")

    # Harvest/yield warning (NEW)
    if "harvest_kg" not in d.columns or d["harvest_kg"].notna().sum() == 0:
        flags.append("Harvest (kg) is missing/empty (yield analytics will be limited).")

    return {
        "rows": rows,
        "missing_required": missing_required,
        "coverage": cov,
        "mapping_used": used_mapping,
        "flags": flags,
    }


def build_export_workbook_bytes(sheets: Dict[str, pd.DataFrame], report: Dict[str, Any]) -> bytes:
    """
    Create a canonical Excel workbook in memory:
      - Baseline (canonical)
      - Metrics
      - Certifications
      - Training Needs
      - Market Summary
      - Data_Quality_Report
      - Mapping_Used
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Ensure consistent ordering
        for name in ["Baseline", "Metrics", "Certifications", "Training Needs", "Market Summary"]:
            df = sheets.get(name, pd.DataFrame())
            df.to_excel(writer, sheet_name=name[:31], index=False)

        # Data quality report (coverage table)
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
    """
    Main pipeline:
      bytes -> read sheets -> detect type -> canonicalize baseline -> derive sheets -> diagnostics -> export bytes
    Returns a dict payload.
    """
    sheets = _read_excel_bytes(xlsx_bytes)
    source_type = detect_source_type(sheets)

    if source_type == "processed":
        # Prefer Baseline sheet, but still canonicalize it
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

    # raw export
    raw = _first_sheet(sheets)
    raw = clean_column_names(raw)

    canonical_df, used = map_to_canonical(raw)

    # training onehots from raw: attach to canonical baseline
    train_oh, created_cols = extract_training_onehots(raw)
    if not train_oh.empty:
        canonical_df = canonical_df.join(train_oh)

    # Derive sheets
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
