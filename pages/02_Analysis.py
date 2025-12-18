# pages/02_Analysis.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score

from core.auth import require_auth, logout_button, permissions, enforce_exporter_scope
from core.data import load_and_prepare_data
from core.ui import set_global_style

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="SHAPe Avocado | Analytics", page_icon="📈", layout="wide")
set_global_style()

# -----------------------------
# Utilities
# -----------------------------
def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def _format_ksh(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"KSh {x:,.0f}"


def _pct(a, b):
    if b in (0, 0.0, None) or (isinstance(b, float) and np.isnan(b)):
        return None
    return 100.0 * (a / b)


def _clip_price(series, upper=120.0):
    s = _to_num(series)
    return s.where(s <= float(upper), np.nan)


# -----------------------------
# Canonical columns (contract)
# -----------------------------
CAN = {
    "exporter": "exporter",
    "county": "county",
    "sub_county": "sub_county",
    "ward": "ward",
    "submit_date": "submit_date",
    "lat": "lat",
    "lon": "lon",
    "area_acres": "area_acres",
    "trees_total": "trees_total",
    "harvest_kg": "harvest_kg",
    "income_ksh_last_season": "income_ksh_last_season",
    "hass_price_ksh_per_kg": "hass_price_ksh_per_kg",
    "gacc_status": "gacc_status",
    "trees_per_acre": "trees_per_acre",
    "yield_per_acre": "yield_per_acre",
    "income_per_acre": "income_per_acre",
    "trees_0_3": "trees_0_3",
    "trees_4_7": "trees_4_7",
    "trees_8_plus": "trees_8_plus",
    "dominant_age_group": "dominant_age_group",
}


def _ensure_analysis_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create analysis-friendly aliases while keeping canonical source-of-truth intact.
    Safe even when some canonical fields are missing.
    """
    d = df.copy()

    # Core numeric aliases
    d["avocado_area_acres"] = _to_num(d.get(CAN["area_acres"], np.nan))
    d["trees_planted"] = _to_num(d.get(CAN["trees_total"], np.nan))
    d["harvest_kg"] = _to_num(d.get(CAN["harvest_kg"], np.nan))
    d["income_ksh"] = _to_num(d.get(CAN["income_ksh_last_season"], np.nan))
    d["price_ksh"] = _clip_price(d.get(CAN["hass_price_ksh_per_kg"], np.nan), upper=120.0)

    # GACC boolean
    gacc_raw = d.get(CAN["gacc_status"], False)
    if isinstance(gacc_raw, pd.Series):
        if gacc_raw.dtype == bool:
            d["gacc_yes"] = gacc_raw.fillna(False).astype(int)
        else:
            s = gacc_raw.astype("string").fillna("").str.strip().str.lower()
            d["gacc_yes"] = s.isin(["yes", "y", "true", "1", "approved", "compliant"]).astype(int)
    else:
        d["gacc_yes"] = 0

    # Derived ratios (prefer canonical if present, else compute)
    if CAN["trees_per_acre"] in d.columns:
        d["trees_per_acre"] = _to_num(d[CAN["trees_per_acre"]])
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            d["trees_per_acre"] = d["trees_planted"] / d["avocado_area_acres"]

    if CAN["yield_per_acre"] in d.columns:
        d["yield_per_acre"] = _to_num(d[CAN["yield_per_acre"]])
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            d["yield_per_acre"] = d["harvest_kg"] / d["avocado_area_acres"]

    if CAN["income_per_acre"] in d.columns:
        d["income_per_acre"] = _to_num(d[CAN["income_per_acre"]])
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            d["income_per_acre"] = d["income_ksh"] / d["avocado_area_acres"]

    # Tree age portfolio (optional)
    for src, out in [
        (CAN["trees_0_3"], "trees_0_3"),
        (CAN["trees_4_7"], "trees_4_7"),
        (CAN["trees_8_plus"], "trees_8_plus"),
    ]:
        d[out] = _to_num(d.get(src, 0)).fillna(0)

    # Dominant age group (prefer canonical if present)
    if CAN["dominant_age_group"] in d.columns:
        s = d[CAN["dominant_age_group"]].astype("string").fillna("").str.strip()
        d["dominant_age_group"] = s
    else:
        age_counts = d[["trees_0_3", "trees_4_7", "trees_8_plus"]].copy()
        dom = age_counts.idxmax(axis=1).fillna("trees_0_3")
        d["dominant_age_group"] = dom.map(
            {"trees_0_3": "0-3 years", "trees_4_7": "4-7 years", "trees_8_plus": "8+ years"}
        )

    # Submit date alias for filtering/trends
    if CAN["submit_date"] in d.columns:
        d["submitdate"] = pd.to_datetime(d[CAN["submit_date"]], errors="coerce")
    else:
        d["submitdate"] = pd.NaT

    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    return d


def _apply_geo_filters(d: pd.DataFrame) -> pd.DataFrame:
    """
    Optional operational filters (only when canonical columns exist).
    """
    st.sidebar.markdown("### Location filters")

    if CAN["county"] in d.columns:
        counties = sorted([c for c in d[CAN["county"]].astype("string").dropna().unique().tolist() if str(c).strip()])
        sel = st.sidebar.multiselect("County", counties, default=[], key="analysis_county_filter")
        if sel:
            d = d[d[CAN["county"]].isin(sel)]

    if CAN["sub_county"] in d.columns:
        subs = sorted([c for c in d[CAN["sub_county"]].astype("string").dropna().unique().tolist() if str(c).strip()])
        sel = st.sidebar.multiselect("Sub-county", subs, default=[], key="analysis_subcounty_filter")
        if sel:
            d = d[d[CAN["sub_county"]].isin(sel)]

    if CAN["ward"] in d.columns:
        wards = sorted([c for c in d[CAN["ward"]].astype("string").dropna().unique().tolist() if str(c).strip()])
        sel = st.sidebar.multiselect("Ward", wards, default=[], key="analysis_ward_filter")
        if sel:
            d = d[d[CAN["ward"]].isin(sel)]

    return d


# -----------------------------
# Advanced analytics (guardrailed)
# -----------------------------
def run_segmentation(df: pd.DataFrame):
    features = [
        "avocado_area_acres",
        "trees_planted",
        "harvest_kg",
        "income_ksh",
        "trees_per_acre",
        "yield_per_acre",
        "income_per_acre",
    ]
    X = df[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    if len(X) < 25:
        return None

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # IMPORTANT: n_init="auto" can fail on older sklearn → keep numeric.
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = km.fit_predict(Xs)

    out = df.copy()
    out["cluster"] = clusters

    profile = out.groupby("cluster")[features].median(numeric_only=True).reset_index()
    sizes = out["cluster"].value_counts().sort_index()

    profile["label"] = "Segment"
    if "income_per_acre" in profile.columns:
        rank_income = profile["income_per_acre"].rank(method="dense")
        for i in range(len(profile)):
            if rank_income.iloc[i] == rank_income.max():
                profile.loc[i, "label"] = "High-Performance"
            elif rank_income.iloc[i] == 1:
                profile.loc[i, "label"] = "Low-Return"
            else:
                profile.loc[i, "label"] = "Emerging"

    return {"df": out, "profile": profile, "sizes": sizes}


def yield_driver_model(df: pd.DataFrame):
    target = "harvest_kg"
    feats = ["avocado_area_acres", "trees_planted", "trees_per_acre", "price_ksh"]

    if df[target].dropna().nunique() < 5 or len(df) < 30:
        return None

    X = df[feats].copy().replace([np.inf, -np.inf], np.nan)
    y = df[target].copy()

    X = X.fillna(X.median(numeric_only=True))
    y = y.fillna(y.median())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    return {"r2": float(r2_score(y_test, pred)), "importance": dict(zip(feats, model.feature_importances_))}


def certification_driver_model(df: pd.DataFrame):
    if "gacc_yes" not in df.columns:
        return None

    feats = ["avocado_area_acres", "trees_planted", "trees_per_acre", "yield_per_acre"]
    X = df[feats].copy().replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    y = df["gacc_yes"].fillna(0).astype(int)

    if len(X) < 30 or y.nunique() < 2:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    acc = float(accuracy_score(y_test, model.predict(X_test)))
    return {
        "accuracy": acc,
        "importance": dict(zip(feats, model.feature_importances_)),
        "class_balance": {"yes": int(y.sum()), "no": int((y == 0).sum())},
    }


# -----------------------------
# Auth
# -----------------------------
user = require_auth()
logout_button()
perm = permissions(user)

# -----------------------------
# Page header
# -----------------------------
st.title("📈 SHAPe Avocado — Analytics")
st.caption("Decision intelligence for program monitoring, performance improvement, and export readiness.")

# -----------------------------
# Data intake (same loader as Monitoring/Admin)
# -----------------------------
with st.sidebar:
    st.markdown("## Data intake")
    uploaded = st.file_uploader(
        "Upload raw survey export or processed workbook (.xlsx)",
        type=["xlsx", "xls"],
        help="Upload the raw export (usually 1 big sheet) or a processed workbook. The app will canonicalize internally.",
    )

pkg = load_and_prepare_data(uploaded_file=uploaded)
base = pkg.baseline_df

if base is None or base.empty:
    st.warning("No data loaded yet. Upload a workbook to begin.")
    st.stop()

with st.sidebar:
    st.download_button(
        "⬇️ Download regenerated shape_data.xlsx",
        data=pkg.workbook_bytes or b"",
        file_name="shape_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        disabled=not bool(pkg.workbook_bytes),
    )
    with st.expander("Coverage diagnostics", expanded=False):
        st.caption(f"Rows loaded: {pkg.report.get('rows', 0)} | Source: {pkg.source_type}")
        missing_required = pkg.report.get("missing_required", []) or []
        if missing_required:
            st.error(f"Missing required fields: {', '.join(missing_required)}")
        for f in (pkg.report.get("flags", []) or []):
            st.warning(f)

# -----------------------------
# Exporter scope enforcement (canonical)
# -----------------------------
exporter_col = CAN["exporter"]

# If canonical exporter column missing, do not crash — just run unscoped
if exporter_col not in base.columns:
    scoped_df = base.copy()
    selected_company = "All"
    exporter_options = ["All"]
    with st.sidebar:
        st.warning("Exporter field not detected; running Analytics without exporter scoping.")
else:
    if user.get("role") == "admin":
        pre_selected = "All"
    else:
        allowed = user.get("exporters") or []
        pre_selected = allowed[0] if (isinstance(allowed, list) and len(allowed) and allowed[0] != "*") else "All"

    scoped_df, selected_company, exporter_options = enforce_exporter_scope(
        base, user, pre_selected, exporter_col=exporter_col
    )

    with st.sidebar:
        st.markdown("---")
        st.markdown("## Filters")

        if user.get("role") == "admin":
            selected_company = st.radio(
                "Exporter",
                options=exporter_options,
                index=exporter_options.index(selected_company) if selected_company in exporter_options else 0,
                key="exporter_radio_secure_analysis",
            )
            scoped_df, selected_company, _ = enforce_exporter_scope(
                base, user, selected_company, exporter_col=exporter_col
            )
        else:
            st.markdown("### Exporter")
            st.info(f"{selected_company}")

# Build analysis fields
df = _ensure_analysis_fields(scoped_df)

# -----------------------------
# Filters (date + geography canonical)
# -----------------------------
with st.sidebar:
    st.markdown("### Date range")
    if df["submitdate"].notna().any():
        min_d = df["submitdate"].min().date()
        max_d = df["submitdate"].max().date()
        dr = st.date_input(
            "Submission/interview date",
            value=[min_d, max_d],
            min_value=min_d,
            max_value=max_d,
            key="analysis_date_range",
        )
        if isinstance(dr, (list, tuple)) and len(dr) == 2:
            df = df[(df["submitdate"].dt.date >= dr[0]) & (df["submitdate"].dt.date <= dr[1])]
    else:
        st.caption("No submission dates detected.")

df = _apply_geo_filters(df)

if df.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# -----------------------------
# Executive summary KPIs
# -----------------------------
k1, k2, k3, k4, k5 = st.columns(5)

farmers = len(df)
area = float(_to_num(df["avocado_area_acres"]).fillna(0).sum())
trees = float(_to_num(df["trees_planted"]).fillna(0).sum())
harvest = float(_to_num(df["harvest_kg"]).fillna(0).sum())
gacc = int(df["gacc_yes"].sum()) if "gacc_yes" in df.columns else 0

k1.metric("Farmers", f"{farmers:,}")
k2.metric("Avocado area (acres)", f"{area:,.1f}")
k3.metric("Trees", f"{trees:,.0f}")
k4.metric("Harvest (kg)", f"{harvest:,.0f}" if harvest > 0 else "—")
k5.metric("China-approved farms", f"{gacc:,}")

# -----------------------------
# Auto-insights
# -----------------------------
def build_insights(d: pd.DataFrame) -> list[str]:
    out = []

    if len(d) > 0 and "gacc_yes" in d.columns:
        rate = _pct(int(d["gacc_yes"].sum()), len(d))
        if rate is not None:
            out.append(f"China approval rate is **{rate:.1f}%** ({int(d['gacc_yes'].sum())}/{len(d)} farms).")

    med_tpa = np.nanmedian(d["trees_per_acre"]) if "trees_per_acre" in d.columns else np.nan
    med_inc = np.nanmedian(d["income_per_acre"]) if "income_per_acre" in d.columns else np.nan
    med_yld = np.nanmedian(d["yield_per_acre"]) if "yield_per_acre" in d.columns else np.nan

    if np.isfinite(med_tpa):
        out.append(f"Median orchard density is **{med_tpa:.0f} trees/acre**.")
    if np.isfinite(med_yld):
        out.append(f"Median productivity is **{med_yld:,.0f} kg/acre**.")
    if np.isfinite(med_inc):
        out.append(f"Median income intensity is **{_format_ksh(med_inc)} per acre**.")

    if CAN["exporter"] in d.columns:
        top_exp = d[CAN["exporter"]].astype("string").value_counts(dropna=True).head(1)
        if len(top_exp):
            out.append(f"Highest coverage exporter by records: **{top_exp.index[0]}** ({int(top_exp.iloc[0])} rows).")
    if CAN["county"] in d.columns:
        top_c = d[CAN["county"]].astype("string").value_counts(dropna=True).head(1)
        if len(top_c):
            out.append(f"Highest coverage county by records: **{top_c.index[0]}** ({int(top_c.iloc[0])} rows).")

    return out[:6]


ins = build_insights(df)
if ins:
    with st.container(border=True):
        st.markdown("### What the data is saying")
        for i in ins:
            st.markdown(f"- {i}")

st.divider()

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_prod, tab_cert, tab_market, tab_advanced, tab_quality = st.tabs(
    ["Overview", "Production & Yield", "Certification", "Market & Income", "Segments & Drivers", "Data Quality"]
)

# ---- Overview ----
with tab_overview:
    c1, c2 = st.columns([1.1, 1])

    with c1:
        st.subheader("Coverage by exporter")
        if CAN["exporter"] in df.columns:
            tmp = df.groupby(CAN["exporter"]).size().reset_index(name="Records").sort_values("Records", ascending=False)
            st.plotly_chart(px.bar(tmp, x=CAN["exporter"], y="Records"), use_container_width=True)
        else:
            st.info("Exporter field not available.")

    with c2:
        st.subheader("Coverage by county")
        if CAN["county"] in df.columns:
            tmp = df.groupby(CAN["county"]).size().reset_index(name="Records").sort_values("Records", ascending=False)
            st.plotly_chart(px.bar(tmp, x=CAN["county"], y="Records"), use_container_width=True)
        else:
            st.info("County field not available.")

    st.subheader("Submission trend")
    if df["submitdate"].notna().any():
        by_m = (
            df.dropna(subset=["submitdate"])
            .assign(month=lambda x: x["submitdate"].dt.to_period("M").astype(str))
            .groupby("month")
            .size()
            .reset_index(name="Records")
        )
        st.plotly_chart(px.line(by_m, x="month", y="Records", markers=True), use_container_width=True)
    else:
        st.info("No submission dates available for trends.")

# ---- Production & Yield ----
with tab_prod:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Trees per acre")
        s = df["trees_per_acre"].dropna()
        if len(s):
            st.plotly_chart(px.histogram(s, nbins=30, labels={"value": "Trees/acre"}), use_container_width=True)
        else:
            st.info("Trees/acre cannot be computed (need trees + avocado area).")

    with c2:
        st.subheader("Yield per acre (kg/acre)")
        s = df["yield_per_acre"].dropna()
        if len(s):
            st.plotly_chart(px.histogram(s, nbins=30, labels={"value": "kg/acre"}), use_container_width=True)
        else:
            st.info("Yield per acre cannot be computed (need harvest + avocado area).")

    st.subheader("Age portfolio (tree counts)")
    age_totals = pd.DataFrame(
        {"Age group": ["0-3 years", "4-7 years", "8+ years"],
         "Trees": [df["trees_0_3"].sum(), df["trees_4_7"].sum(), df["trees_8_plus"].sum()]}
    )
    age_totals = age_totals[age_totals["Trees"] > 0]
    if not age_totals.empty:
        st.plotly_chart(px.pie(age_totals, names="Age group", values="Trees"), use_container_width=True)
    else:
        st.info("Age-class tree counts not available.")

# ---- Certification ----
with tab_cert:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("GACC approval")
        yes = int(df["gacc_yes"].sum()) if "gacc_yes" in df.columns else 0
        no = int(len(df) - yes)
        cert_df = pd.DataFrame({"Status": ["Yes", "No"], "Count": [yes, no]})
        st.plotly_chart(px.pie(cert_df, names="Status", values="Count"), use_container_width=True)

    with c2:
        st.subheader("Approval rate by exporter")
        if CAN["exporter"] in df.columns and "gacc_yes" in df.columns:
            tmp = df.groupby(CAN["exporter"])["gacc_yes"].mean().reset_index(name="ApprovalRate")
            tmp["ApprovalRate"] = tmp["ApprovalRate"] * 100.0
            fig = px.bar(
                tmp.sort_values("ApprovalRate", ascending=False),
                x=CAN["exporter"],
                y="ApprovalRate",
                labels={"ApprovalRate": "Approval rate (%)"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Exporter and/or GACC field not available to compute approval rate.")

# ---- Market & Income ----
with tab_market:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Hass price distribution")
        s = df["price_ksh"].dropna()
        if len(s):
            st.plotly_chart(px.histogram(s, nbins=30, labels={"value": "KSh/kg"}), use_container_width=True)
            st.caption(f"Median reference price: {_format_ksh(float(s.median()))}/kg")
        else:
            st.info("No Hass price values detected.")

    with c2:
        st.subheader("Income per acre")
        s = df["income_per_acre"].dropna().replace([np.inf, -np.inf], np.nan).dropna()
        if len(s):
            lo, hi = s.quantile(0.01), s.quantile(0.99)
            s2 = s[(s >= lo) & (s <= hi)]
            st.plotly_chart(px.histogram(s2, nbins=30, labels={"value": "KSh/acre"}), use_container_width=True)
            st.caption(f"Median income intensity: {_format_ksh(float(s2.median()))}/acre")
        else:
            st.info("Income/acre cannot be computed (need income + avocado area).")

# ---- Segments & Drivers ----
with tab_advanced:
    st.subheader("Farmer segmentation (KMeans)")
    seg = run_segmentation(df)
    if seg is None:
        st.info("Not enough clean numeric data for segmentation (need ~25+ rows with area/trees/harvest/income).")
    else:
        prof = seg["profile"]
        sizes = seg["sizes"]

        c1, c2 = st.columns([1, 1])
        with c1:
            donut = pd.DataFrame({"Cluster": sizes.index.astype(str), "Count": sizes.values})
            st.plotly_chart(px.pie(donut, names="Cluster", values="Count", hole=0.5), use_container_width=True)
        with c2:
            st.dataframe(prof, use_container_width=True, hide_index=True)

        st.caption(
            "Segmentation is descriptive (not a judgement). Use it to target extension, certification support, and orchard densification coaching."
        )

    st.divider()
    st.subheader("Drivers (models with guardrails)")

    c1, c2 = st.columns(2)
    with c1:
        yd = yield_driver_model(df)
        if yd is None:
            st.info("Yield driver model needs more data (harvest + core numeric fields).")
        else:
            st.metric("Yield model fit (R²)", f"{yd['r2']:.3f}")
            imp = (
                pd.DataFrame({"Feature": list(yd["importance"].keys()), "Importance": list(yd["importance"].values())})
                .sort_values("Importance", ascending=False)
            )
            st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h"), use_container_width=True)

    with c2:
        cd = certification_driver_model(df)
        if cd is None:
            st.info("Certification driver model needs both Yes and No cases + enough rows.")
        else:
            st.metric("Certification model accuracy", f"{cd['accuracy']:.3f}")
            imp = (
                pd.DataFrame({"Feature": list(cd["importance"].keys()), "Importance": list(cd["importance"].values())})
                .sort_values("Importance", ascending=False)
            )
            st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h"), use_container_width=True)

# ---- Data Quality ----
with tab_quality:
    st.subheader("Completeness scan (top 25 fields)")
    completeness = (df.notna().mean() * 100.0).sort_values(ascending=True).head(25).reset_index()
    completeness.columns = ["Field", "Completeness (%)"]
    st.plotly_chart(px.bar(completeness, x="Completeness (%)", y="Field", orientation="h"), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        gps_ok = 0
        if CAN["lat"] in df.columns and CAN["lon"] in df.columns:
            gps_ok = int(df.dropna(subset=[CAN["lat"], CAN["lon"]]).shape[0])
        st.metric("GPS populated", f"{gps_ok:,}")

    with c2:
        area_ok = int(df["avocado_area_acres"].dropna().shape[0])
        st.metric("Avocado area populated", f"{area_ok:,}")

    with c3:
        trees_ok = int(df["trees_planted"].dropna().shape[0])
        st.metric("Trees populated", f"{trees_ok:,}")

    if perm.get("can_view_raw_data"):
        st.divider()
        with st.expander("Preview (admin only)"):
            st.dataframe(df.head(2000), use_container_width=True)
