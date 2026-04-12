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
    show_sustainability,
    show_geo_spatial_aez,
    show_farmer_profile,
    show_farm_characteristics,
    show_productivity_efficiency,
    show_enhanced_compliance,
    show_ipm_measures,
    show_grade_completeness,
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
    for cand in candidates:
        if cand in df.columns:
            return cand
    return None


@st.cache_data(show_spinner=False)
def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _apply_geo_filters(df: pd.DataFrame) -> pd.DataFrame:
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

    st.markdown("""
        <style>
            .main .block-container {
                padding-top: 0rem !important;
            }
            
            .fixed-header-space {
                height: 120px;
            }
            
            .main .block-container {
                padding-top: 0rem !important;
            }
            
            header[data-testid="stHeader"] {
                display: none;
            }
            
            .fixed-header-space {
                height: 20px;
            }
            
            .fixed-header {
                position: fixed;
                top: 0;
                left: 240px;
                right: 0;
                background-color: white;
                z-index: 9999;
                padding: 15px 20px 15px 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                border-bottom: 2px solid #e0e0e0;
            }
            
            .fixed-header h1 {
                margin: 0 0 8px 0;
                padding: 0;
                font-size: 2rem;
                line-height: 1.2;
            }
        </style>
    """, unsafe_allow_html=True)
    
    subtitle_text = "📊 Tracking Kenya's Avocado Export Performance | GACC Approved: Tracking | Market Prices: Live | Farm Compliance: Monitoring | Yield Trends: Analyzing"
    
    st.markdown(f"""
        <div class="fixed-header">
            <h1>🥑 SHAPe Avocado Dashboard</h1>
            <div style="
                background-color: #f0f2f6;
                padding: 8px;
                border-radius: 5px;
                overflow: hidden;
                white-space: nowrap;
                font-family: monospace;
                font-size: 14px;
                color: #1f77b4;
            ">
                <div style="
                    display: inline-block;
                    animation: scroll 25s linear infinite;
                    padding-left: 100%;
                ">
                    {subtitle_text} &nbsp;&nbsp;|&nbsp;&nbsp; 
                    📈 Export Excellence | 🌍 Global Market Access | 🥑 Quality Assured
                </div>
            </div>
        </div>
        
        <div class="fixed-header-space"></div>
        
        <style>
        @keyframes scroll {{
            0% {{ transform: translate(0, 0); }}
            100% {{ transform: translate(-100%, 0); }}
        }}
        </style>
    """, unsafe_allow_html=True)

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

    metrics_df = pkg.sheets.get("Metrics")
    if metrics_df is None or not isinstance(metrics_df, pd.DataFrame):
        metrics_df = pd.DataFrame()

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

    st.sidebar.markdown("### Date range")

    filtered_df = scoped_df.copy()

    submit_col = _pick_first_col(filtered_df, ["submitdate", "submit_date", "data_time", "SubmissionDate", "Date of interview:"])
    if submit_col and submit_col in filtered_df.columns:
        dt = pd.to_datetime(filtered_df[submit_col], errors="coerce")
        filtered_df["submitdate"] = dt
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

    filtered_df = _apply_geo_filters(filtered_df)

    if filtered_df.empty:
        st.warning("No records match the selected filters.")
        return

    show_overview(filtered_df, metrics_df)
    
    with st.expander("🌍 Geo-Spatial & Agro-Ecological Zones", expanded=False):
        show_geo_spatial_aez(filtered_df)
    
    show_farmer_profile(filtered_df)
    
    with st.expander("🌱 Farm Characteristics", expanded=False):
        show_farm_characteristics(filtered_df)
    
    with st.expander("📈 Productivity & Efficiency", expanded=False):
        show_productivity_efficiency(filtered_df)
    
    show_production_metrics(filtered_df)
    
    show_certification(filtered_df)
    
    with st.expander("✅ Enhanced Compliance Metrics", expanded=False):
        show_enhanced_compliance(filtered_df)
    
    with st.expander("🪰 Integrated Pest Management (IPM) Measures", expanded=False):
        show_ipm_measures(filtered_df)
    
    with st.expander("📊 Grade Completeness Analysis", expanded=False):
        show_grade_completeness(filtered_df)
    
    show_market_analysis(filtered_df)
    
    show_sustainability(filtered_df)
    
    show_training_needs(filtered_df)

    st.subheader("Data Explorer")
    if perms.get("can_view_raw_data"):
        if st.checkbox("Show raw data"):
            st.dataframe(filtered_df, use_container_width=True)
    else:
        st.caption("Raw data is restricted to admin users.")

    st.download_button(
        "Download filtered data (CSV)",
        data=_to_csv_bytes(filtered_df),
        file_name="shape_filtered_data.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()