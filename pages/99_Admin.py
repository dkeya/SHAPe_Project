# pages/99_Admin.py
import streamlit as st
import pandas as pd

from core.auth import require_auth, logout_button, permissions
from core.data import load_and_prepare_data, key_columns_summary
from core.ui import set_global_style

st.set_page_config(page_title="SHAPe Avocado | Admin", page_icon="🥑", layout="wide")
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


def main():
    user = require_auth()
    logout_button()
    perms = permissions(user)

    if not perms.get("can_view_admin", False):
        st.error("You do not have permission to view the Admin page.")
        st.stop()

    st.title("🛠️ Admin Console")
    st.caption("Data health + access control guidance (no manual transformations required).")

    # -----------------------------
    # Data intake (Smart Loader)
    # -----------------------------
    with st.sidebar:
        st.markdown("## Data intake")
        uploaded = st.file_uploader(
            "Upload raw survey export or processed workbook (.xlsx)",
            type=["xlsx", "xls"],
            help="Upload raw export (usually one big sheet) OR a processed workbook. The system canonicalizes internally.",
        )

    pkg = load_and_prepare_data(uploaded_file=uploaded)
    df = pkg.baseline_df
    metrics = pkg.sheets.get("Metrics")
    if not isinstance(metrics, pd.DataFrame):
        metrics = pd.DataFrame()

    # Sidebar: export regenerated workbook + diagnostics
    with st.sidebar:
        st.download_button(
            "⬇️ Download regenerated shape_data.xlsx",
            data=pkg.workbook_bytes or b"",
            file_name="shape_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            disabled=not bool(pkg.workbook_bytes),
        )

        with st.expander("Coverage diagnostics", expanded=True):
            st.caption(f"Source: **{pkg.source_type}**")
            st.caption(f"Rows loaded: **{pkg.report.get('rows', 0)}**")

            missing_required = pkg.report.get("missing_required", []) or []
            if missing_required:
                st.error("Missing required fields: " + ", ".join(missing_required))
            else:
                st.success("Required fields look OK.")

            flags = pkg.report.get("flags", []) or []
            for f in flags:
                st.warning(f)

    # If no data
    if df is None or df.empty:
        st.warning("No data loaded yet. Upload a workbook to view Admin diagnostics.")
        return

    # -----------------------------
    # Tabs
    # -----------------------------
    tab_health, tab_metrics, tab_mapping, tab_access = st.tabs(
        ["Data Health", "Metrics", "Mapping & Schema", "Access Control"]
    )

    # -----------------------------
    # Data Health
    # -----------------------------
    with tab_health:
        st.subheader("Data Health Summary")
        st.dataframe(key_columns_summary(df), use_container_width=True)

        with st.expander("Row/Column quick stats"):
            exporter_col = _pick_first_col(df, ["exporter", "1.1 Company Name", "Exporter", "Company"])
            exporters = []
            if exporter_col and exporter_col in df.columns:
                exporters = sorted([x for x in df[exporter_col].dropna().unique().tolist() if str(x).strip()])

            st.write(
                {
                    "rows": int(df.shape[0]),
                    "columns": int(df.shape[1]),
                    "exporter_col_detected": exporter_col or "",
                    "exporters_detected": exporters,
                }
            )

        # Optional: show the full baseline to admin
        if perms.get("can_view_raw_data", False):
            with st.expander("Baseline preview (admin only)"):
                st.dataframe(df.head(2000), use_container_width=True)

    # -----------------------------
    # Metrics sheet check
    # -----------------------------
    with tab_metrics:
        st.subheader("Metrics Sheet Check")
        if metrics.empty:
            st.warning("Metrics sheet not found/empty in the derived outputs.")
        else:
            st.dataframe(metrics, use_container_width=True, hide_index=True)

    # -----------------------------
    # Mapping & Schema
    # -----------------------------
    with tab_mapping:
        st.subheader("Canonicalization Report")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Missing required fields**")
            mr = pkg.report.get("missing_required", []) or []
            if mr:
                st.error(", ".join(mr))
            else:
                st.success("None")

        with c2:
            st.markdown("**Derived sheets produced**")
            st.write(sorted(list((pkg.sheets or {}).keys())))

        st.markdown("**Mapping used**")
        mapping = pkg.report.get("mapping_used", {}) or {}
        if mapping:
            # mapping_used can be large; show in a clean way
            st.json(mapping)
        else:
            st.caption("No mapping metadata available (transform did not return mapping_used).")

        with st.expander("Full report payload (debug)"):
            st.json(pkg.report or {})

    # -----------------------------
    # Access Control Setup (Streamlit Secrets)
    # -----------------------------
    with tab_access:
        st.subheader("Access Control Setup (Streamlit Secrets)")
        st.caption("Keep secrets in `.streamlit/secrets.toml` and DO NOT commit it to GitHub.")

        st.code(
            """
# Create: .streamlit/secrets.toml  (DO NOT commit it)
[auth.users.admin]
name = "Admin"
role = "admin"
exporters = ["*"]
password_hash = "pbkdf2_sha256$200000$<salt_b64>$<hash_b64>"

[auth.users.mavuno_user]
name = "Mavuno Ops"
role = "exporter"
exporters = ["Mavuno"]
password_hash = "pbkdf2_sha256$200000$<salt_b64>$<hash_b64>"
            """.strip(),
            language="toml",
        )


if __name__ == "__main__":
    main()
