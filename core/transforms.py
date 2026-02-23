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


# ----------------------------------------------------------
# NEW: Derive sub_county from raw one-hot columns (1.4* SubCounty ...)
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
        # Examples:
        #  "1.4P SubCounty Baringo/Baringo Central" -> "Baringo Central"
        #  "1.4A Subcounty Meru" -> "Meru"
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
        # If multiple selected (rare), keep them all (stable + transparent)
        # but dedupe while preserving order
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

    # 3) NEW: sub_county derived from raw one-hots if sub_county is missing/empty
    if ("sub_county" not in out.columns) or (out["sub_county"].astype("string").str.strip().replace("", np.nan).isna().all()):
        derived_sub = _derive_sub_county_from_onehots(raw_df)
        if isinstance(derived_sub, pd.Series) and len(derived_sub) == len(out):
            out["sub_county"] = derived_sub
            # mark mapping as derived (helps diagnostics)
            if "sub_county" not in used:
                used["sub_county"] = "derived:1.4*_SubCounty_onehots"

    # 4) Derived stability fields
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

    if "lat" in d.columns and "lon" in d.columns:
        if int(d[["lat", "lon"]].dropna().shape[0]) == 0:
            flags.append("No valid GPS coordinates detected (lat/lon missing).")

    if "market_outlet" in d.columns and d["market_outlet"].astype("string").str.strip().replace("", np.nan).isna().mean() > 0.75:
        flags.append("Market outlet is mostly missing (outlet coverage is low).")

    if "submit_date" in d.columns and d["submit_date"].notna().sum() == 0:
        flags.append("No submission/interview dates detected (trend analysis will be limited).")

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
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name in ["Baseline", "Metrics", "Certifications", "Training Needs", "Market Summary"]:
            df = sheets.get(name, pd.DataFrame())
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