# pages/01_Monitoring_Dashboard.py
import streamlit as st
import pandas as pd

from core.auth import require_auth, logout_button, enforce_exporter_scope, permissions
from core.data import load_and_prepare_data
from core.ui import set_global_style
from core.sections_monitoring import (
    show_overview,
    show_geospatial,
    show_certification,
    show_production_metrics,
    show_market_analysis,
    show_training_needs,
    show_investor_income_view,
    show_income_potential_forecast,
)

st.set_page_config(page_title="SHAPe Avocado | Monitoring", page_icon="🥑", layout="wide")
set_global_style()


def _pick_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None or df.empty:
        return None
    cols = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in cols:
            return cols[key]
    # also allow direct exact match
    for cand in candidates:
        if cand in df.columns:
            return cand
    return None


@st.cache_data(show_spinner=False)
def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _apply_geo_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional operational filters (only when columns exist).
    Works for canonical schema OR legacy survey columns.
    """
    geo_cols = {
        "county": ["county", "1.2 County", "County", "county_name"],
        "subcounty": ["sub_county", "Sub_County", "Sub County", "1.3 Sub County", "subcounty"],
        "ward": ["ward", "1.4 Ward", "Ward"],
    }

    county_col = _pick_first_col(df, geo_cols["county"])
    subcounty_col = _pick_first_col(df, geo_cols["subcounty"])
    ward_col = _pick_first_col(df, geo_cols["ward"])

    st.sidebar.markdown("### Location filters")

    if county_col:
        counties = sorted([c for c in df[county_col].dropna().unique().tolist() if str(c).strip()])
        sel_counties = st.sidebar.multiselect("County", options=counties, default=[])
        if sel_counties:
            df = df[df[county_col].isin(sel_counties)]

    if subcounty_col:
        subcounties = sorted([c for c in df[subcounty_col].dropna().unique().tolist() if str(c).strip()])
        sel_subcounties = st.sidebar.multiselect("Sub-county", options=subcounties, default=[])
        if sel_subcounties:
            df = df[df[subcounty_col].isin(sel_subcounties)]

    if ward_col:
        wards = sorted([c for c in df[ward_col].dropna().unique().tolist() if str(c).strip()])
        sel_wards = st.sidebar.multiselect("Ward", options=wards, default=[])
        if sel_wards:
            df = df[df[ward_col].isin(sel_wards)]

    return df


def main():
    user = require_auth()
    logout_button()
    perms = permissions(user)

    st.title("🥑 SHAPe Avocado Dashboard")
    st.caption("Monitoring Kenya's avocado value chain for export excellence")

    # -----------------------------
    # Data intake (Smart Loader)
    # -----------------------------
    with st.sidebar:
        st.markdown("## Data intake")
        uploaded = st.file_uploader(
            "Upload raw survey export or processed workbook (.xlsx)",
            type=["xlsx", "xls"],
            help="Upload raw export (usually one big sheet) OR a processed workbook. The app will canonicalize internally.",
        )

    pkg = load_and_prepare_data(uploaded_file=uploaded)

    farmer_df = pkg.baseline_df
    if farmer_df is None or farmer_df.empty:
        st.warning("No data loaded yet. Upload an Excel workbook to begin.")
        return

    # ✅ FIX: do NOT use a DataFrame in boolean `or`
    metrics_df = pkg.sheets.get("Metrics")
    if metrics_df is None or not isinstance(metrics_df, pd.DataFrame):
        metrics_df = pd.DataFrame()

    # Download regenerated workbook (in-memory)
    with st.sidebar:
        st.download_button(
            "⬇️ Download regenerated shape_data.xlsx",
            data=pkg.workbook_bytes,
            file_name="shape_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        with st.expander("Coverage diagnostics"):
            st.caption(f"Rows loaded: {pkg.report.get('rows', 0)} | Source: {pkg.source_type}")
            missing_required = pkg.report.get("missing_required", []) or []
            if missing_required:
                st.error(f"Missing required fields: {', '.join(missing_required)}")
            for f in (pkg.report.get("flags", []) or []):
                st.warning(f)

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Filters")

    # ---------------------------------
    # Exporter scope enforcement
    # ---------------------------------
    exporter_col = _pick_first_col(farmer_df, ["exporter", "1.1 Company Name", "Exporter", "Company"])
    if not exporter_col:
        st.error("Exporter field not detected (expected canonical `exporter` or raw `1.1 Company Name`).")
        return

    if user.get("role") == "admin":
        pre_selected = "All"
    else:
        allowed = user.get("exporters") or []
        pre_selected = allowed[0] if (isinstance(allowed, list) and len(allowed) and allowed[0] != "*") else "All"

    scoped_df, selected_company, exporter_options = enforce_exporter_scope(
        farmer_df, user, pre_selected, exporter_col=exporter_col
    )

    if user.get("role") == "admin":
        selected_company = st.sidebar.radio(
            "Exporter",
            options=exporter_options,
            index=exporter_options.index(selected_company) if selected_company in exporter_options else 0,
            key="exporter_radio_secure",
        )
        scoped_df, selected_company, _ = enforce_exporter_scope(
            farmer_df, user, selected_company, exporter_col=exporter_col
        )
    else:
        st.sidebar.markdown("### Exporter")
        st.sidebar.info(f"{selected_company}")

    # -----------------
    # Date range filter
    # -----------------
    st.sidebar.markdown("### Date range")

    filtered_df = scoped_df.copy()

    # Support canonical submit_date OR legacy submitdate already created
    submit_col = _pick_first_col(filtered_df, ["submitdate", "submit_date", "data_time", "SubmissionDate", "Date of interview:"])
    if submit_col and submit_col in filtered_df.columns:
        dt = pd.to_datetime(filtered_df[submit_col], errors="coerce")
        filtered_df["submitdate"] = dt  # normalize for downstream code
    else:
        filtered_df["submitdate"] = pd.NaT

    if filtered_df["submitdate"].notna().any():
        min_date = filtered_df["submitdate"].min().date()
        max_date = filtered_df["submitdate"].max().date()

        date_range = st.sidebar.date_input(
            "Submission/interview date",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date,
            key="date_range",
        )

        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df["submitdate"].dt.date >= date_range[0]) &
                (filtered_df["submitdate"].dt.date <= date_range[1])
            ]
    else:
        st.sidebar.caption("No submission dates detected.")

    # -----------------
    # Geo filters (ops)
    # -----------------
    filtered_df = _apply_geo_filters(filtered_df)

    if filtered_df.empty:
        st.warning("No records match the selected filters.")
        return

    # -----------------
    # Monitoring sections
    # -----------------
    show_overview(filtered_df, metrics_df)
    show_geospatial(filtered_df)
    show_certification(filtered_df)
    show_production_metrics(filtered_df)

    # Investor-grade sections
    show_investor_income_view(filtered_df)
    show_income_potential_forecast(filtered_df)

    show_market_analysis(filtered_df)
    show_training_needs(filtered_df)

    # -----------------
    # Data explorer (admin only)
    # -----------------
    st.subheader("Data Explorer")
    if perms.get("can_view_raw_data"):
        if st.checkbox("Show raw data"):
            st.dataframe(filtered_df, use_container_width=True)
    else:
        st.caption("Raw data is restricted to admin users.")

    # -----------------
    # Export (filtered CSV)
    # -----------------
    st.download_button(
        "Download filtered data (CSV)",
        data=_to_csv_bytes(filtered_df),
        file_name="shape_filtered_data.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
