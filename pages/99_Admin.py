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
    st.caption("Data health + access control guidance + admin-only data mode controls.")

    # -----------------------------
    # Admin-only data controls
    # -----------------------------
    with st.sidebar:
        st.markdown("## Admin controls")

        # Data mode switch (session override)
        current_mode = st.session_state.get("shape_data_mode", None)
        if current_mode not in {"demo", "production"}:
            # No session override yet; show what secrets says (if any) as a hint
            try:
                cfg = st.secrets.get("data", {}) or {}
                hinted = str(cfg.get("mode", "demo")).strip().lower()
                if hinted not in {"demo", "production"}:
                    hinted = "demo"
            except Exception:
                hinted = "demo"
        else:
            hinted = current_mode

        mode = st.radio(
            "Default data mode (this session)",
            options=["demo", "production"],
            index=0 if hinted == "demo" else 1,
            help=(
                "demo = load repo-bundled demo workbook if no upload exists.\n"
                "production = load from secrets URL if no upload exists."
            ),
            key="shape_data_mode_radio",
        )
        st.session_state["shape_data_mode"] = mode

        # Optional: production URL override for this session (if you want)
        with st.expander("Production URL override (optional)"):
            st.caption("Prefer setting this in Streamlit Secrets. This is session-only.")
            st.text_input(
                "production_url (session)",
                value=st.session_state.get("shape_production_url", ""),
                key="shape_production_url",
                placeholder="https://.../shape_raw_data.xlsx?<signed_token>",
            )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Clear uploaded file", use_container_width=True):
                if "shape_workbook_bytes" in st.session_state:
                    del st.session_state["shape_workbook_bytes"]
                st.success("Cleared session upload. Default auto-load will be used.")
                st.rerun()

        with c2:
            if st.button("Clear cache", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache cleared.")
                st.rerun()

    # -----------------------------
    # Data intake (still available)
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
            st.caption(f"Loaded source: **{pkg.source_type}**")
            st.caption(f"Mode: **{pkg.report.get('data_mode', '—')}**")
            st.caption(f"Rows loaded: **{pkg.report.get('rows', 0)}**")

            missing_required = pkg.report.get("missing_required", []) or []
            if missing_required:
                st.error("Missing required fields: " + ", ".join(missing_required))
            else:
                st.success("Required fields look OK.")

            for f in (pkg.report.get("flags", []) or []):
                st.warning(f)

    # If no data
    if df is None or df.empty:
        st.warning("No data loaded yet. Upload a workbook, or configure a default dataset.")
        st.info(
            "Tip: For deployed app, commit a demo file at `data/shape_raw_data.xlsx` "
            "OR set `[data].production_url` in Streamlit Secrets."
        )
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

# Optional: production data config (recommended for real farmer data)
[data]
mode = "production"   # demo | production
production_url = "https://<private-storage>/shape_raw_data.xlsx?<signed_token>"
            """.strip(),
            language="toml",
        )


if __name__ == "__main__":
    main()
