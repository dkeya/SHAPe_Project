# core/sections_monitoring.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
from wordcloud import WordCloud

try:
    from core.data import YIELD_REFERENCE as YIELD_REFERENCE_EXTERNAL
except Exception:
    YIELD_REFERENCE_EXTERNAL = None


# =========================
# Canonical column contract
# =========================
CAN = {
    "exporter": "exporter",
    "farmer": "farmer_name",
    "orchard": "orchard_name",
    "county": "county",
    "sub_county": "sub_county",
    "ward": "ward",
    "lat": "lat",
    "lon": "lon",
    "area": "area_acres",
    "trees": "trees_total",
    "trees_per_acre": "trees_per_acre",
    "gacc": "gacc_status",
    "kephis": "kephis_status",
    "pest": "pest_monitoring",
    "sanitation": "sanitation_record",
    "approved_pesticides": "approved_pesticides",
    "cert_globalgap": "cert_globalgap",
    "cert_organic": "cert_organic",
    "cert_fairtrade": "cert_fairtrade",
    "market_outlet": "market_outlet",
    "price_hass": "hass_price_ksh_per_kg",
    "price_hass_current": "hass_price_current_ksh_per_kg",  # NEW
    "income_last": "income_ksh_last_season",
    "income_per_acre": "income_per_acre",
    "grade1_share_last": "grade1_share_last_season",  # NEW
    "grade1_share_current": "grade1_share_current_season",  # NEW
    "trees_0_3": "trees_0_3",
    "trees_4_7": "trees_4_7",
    "trees_8_plus": "trees_8_plus",
    "fruits_0_3": "fruits_per_tree_0_3",
    "fruits_4_7": "fruits_per_tree_4_7",
    "fruits_8_plus": "fruits_per_tree_8_plus",
    "dominant_age": "dominant_age_group",
    # NEW: Farmer profile
    "gender": "gender",
    "age": "age",
    "education": "education_level",
    "experience": "farming_experience_years",
    # NEW: Legal registration
    "orchard_group": "orchard_group_registered",
    # NEW: Production practices
    "fertilizer_organic": "fertilizer_organic",
    "fertilizer_inorganic": "fertilizer_inorganic",
    "fertilizer_quantity": "fertilizer_quantity_kg_per_tree",
    "soil_conservation": "soil_conservation_practices",
    "irrigation_drip": "irrigation_drip",
    "irrigation_sprinkler": "irrigation_sprinkler",
    "irrigation_rainfed": "irrigation_rainfed",
    # NEW: Compliance
    "sps_training": "sps_training_received",
    "training_last_year": "training_last_year",
    "training_last_6months": "training_last_6months",
    "training_provider": "training_provider",
    "record_keeping": "record_keeping_practices",
    "ipm_use": "ipm_implemented",
    "pesticide_use": "pesticide_use_details",
    "biological_control": "biological_control_used",
    # NEW: Sustainability
    "water_source": "irrigation_water_source",
    "waste_management": "waste_management_practice",
    "biodiversity": "biodiversity_practices",
    "additional_trees": "additional_trees_planted",
    "other_value_chains": "other_value_chains",
    # NEW: Extension
    "extension_access": "extension_services_access",
    "training_needs_text": "training_needs_text",
}


def _safe_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _safe_str(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series("", index=df.index, dtype="string")
    return df[col].astype("string").fillna("").str.strip()


def _safe_bool(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index, dtype="bool")
    s = df[col]
    if s.dtype == bool:
        return s.fillna(False)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.60:
        return (num.fillna(0) > 0).astype(bool)
    stx = s.astype("string").fillna("").str.strip().str.lower()
    return stx.isin(["yes", "y", "true", "1", "approved", "compliant"])


def _clip_inf(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan)


def classify_farm_size(acres: float) -> str:
    """Classify farm size based on acres."""
    if pd.isna(acres):
        return "Unknown"
    if acres <= 3:
        return "Micro-Small"
    elif acres <= 10:
        return "Medium"
    else:
        return "Large"


def _yield_reference():
    if isinstance(YIELD_REFERENCE_EXTERNAL, dict) and len(YIELD_REFERENCE_EXTERNAL) > 0:
        ref = {}
        for k, v in YIELD_REFERENCE_EXTERNAL.items():
            ref[k] = dict(v)
            if "kg" not in ref[k]:
                ref[k]["kg"] = np.nan
        return ref

    return {
        "0-3": {"fruits": 275, "kg": 45.8},
        "4-7": {"fruits": 350, "kg": 58.3},
        "8+": {"fruits": 900, "kg": 150.0},
    }


def _ensure_derived(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # numeric coercions for key fields
    for c in [CAN["lat"], CAN["lon"], CAN["area"], CAN["trees"], CAN["trees_per_acre"], CAN["income_last"], CAN["income_per_acre"],
              CAN["trees_0_3"], CAN["trees_4_7"], CAN["trees_8_plus"], CAN["fruits_0_3"], CAN["fruits_4_7"], CAN["fruits_8_plus"], 
              CAN["price_hass"], CAN["price_hass_current"], CAN["grade1_share_last"], CAN["grade1_share_current"],
              CAN["age"], CAN["experience"]]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # Compute trees_per_acre if missing
    if CAN["trees_per_acre"] not in d.columns:
        d[CAN["trees_per_acre"]] = np.nan

    if (d[CAN["trees_per_acre"]].isna().all()) and (CAN["trees"] in d.columns) and (CAN["area"] in d.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            d[CAN["trees_per_acre"]] = d[CAN["trees"]] / d[CAN["area"]]

    # Income per acre
    if CAN["income_per_acre"] not in d.columns:
        d[CAN["income_per_acre"]] = np.nan
    if (d[CAN["income_per_acre"]].isna().all()) and (CAN["income_last"] in d.columns) and (CAN["area"] in d.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            d[CAN["income_per_acre"]] = d[CAN["income_last"]] / d[CAN["area"]]

    # Dominant age group (if missing)
    if CAN["dominant_age"] not in d.columns:
        d[CAN["dominant_age"]] = ""

    if d[CAN["dominant_age"]].astype("string").fillna("").str.strip().eq("").mean() > 0.90:
        if all(x in d.columns for x in [CAN["trees_0_3"], CAN["trees_4_7"], CAN["trees_8_plus"]]):
            age_counts = d[[CAN["trees_0_3"], CAN["trees_4_7"], CAN["trees_8_plus"]]].fillna(0)
            dom = age_counts.idxmax(axis=1)
            d[CAN["dominant_age"]] = dom.map(
                {
                    CAN["trees_0_3"]: "0-3 years",
                    CAN["trees_4_7"]: "4-7 years",
                    CAN["trees_8_plus"]: "8+ years",
                }
            ).fillna("")

    return _clip_inf(d)


# ==========================================================
# Map + core charts
# ==========================================================
def create_farm_map(df: pd.DataFrame):
    d = _ensure_derived(df)
    if CAN["lat"] not in d.columns or CAN["lon"] not in d.columns:
        return None

    map_df = d.dropna(subset=[CAN["lat"], CAN["lon"]])
    if map_df.empty:
        return None

    m = folium.Map(location=[0.0236, 37.9062], zoom_start=6, control_scale=True)
    cluster = MarkerCluster().add_to(m)

    for _, row in map_df.iterrows():
        popup_text = (
            f"<b>Exporter:</b> {row.get(CAN['exporter'], 'N/A')}<br>"
            f"<b>Farm:</b> {row.get(CAN['orchard'], 'N/A')}<br>"
            f"<b>Farmer:</b> {row.get(CAN['farmer'], 'N/A')}<br>"
            f"<b>Trees:</b> {row.get(CAN['trees'], 'N/A')}"
        )

        folium.Marker(
            location=[row[CAN["lat"]], row[CAN["lon"]]],
            popup=folium.Popup(popup_text, max_width=280),
            icon=folium.Icon(color="green", icon="leaf", prefix="fa"),
        ).add_to(cluster)

    return m


def create_certification_chart(df: pd.DataFrame):
    d = _ensure_derived(df)
    cert_data = []

    for label, col in [
        ("GlobalGAP", CAN["cert_globalgap"]),
        ("Organic", CAN["cert_organic"]),
        ("FairTrade", CAN["cert_fairtrade"]),
    ]:
        if col in d.columns:
            count = int(_safe_bool(d, col).sum())
            cert_data.append({"Certification": label, "Count": count})

    if CAN["gacc"] in d.columns:
        cert_data.append({"Certification": "China (GACC)", "Count": int(_safe_bool(d, CAN["gacc"]).sum())})

    if not cert_data:
        return None

    cert_df = pd.DataFrame(cert_data)
    return (
        alt.Chart(cert_df)
        .mark_bar()
        .encode(
            x=alt.X("Certification:N", title=""),
            y=alt.Y("Count:Q", title="Farms"),
            tooltip=["Certification", "Count"],
        )
        .properties(title="Farm Certification & Market Compliance", height=380)
    )


def create_yield_comparison_chart(df: pd.DataFrame):
    d = _ensure_derived(df)
    ref = _yield_reference()
    rows = []

    for age_key, label, col in [
        ("0-3", "0-3 years", CAN["fruits_0_3"]),
        ("4-7", "4-7 years", CAN["fruits_4_7"]),
        ("8+", "8+ years", CAN["fruits_8_plus"]),
    ]:
        if col in d.columns:
            actual = float(pd.to_numeric(d[col], errors="coerce").mean())
            expected = float(ref[age_key]["fruits"])
            if np.isfinite(actual) and expected > 0:
                rows += [
                    {"Age Group": label, "Category": "Actual", "Value": actual, "Expected": expected},
                    {"Age Group": label, "Category": "Expected", "Value": expected, "Expected": expected},
                ]

    if not rows:
        return None

    data = pd.DataFrame(rows)
    data["PctOfExpected"] = np.where(
        (data["Category"] == "Actual") & (data["Expected"] > 0),
        (data["Value"] / data["Expected"]) * 100.0,
        np.nan,
    )
    data["PctLabel"] = data["PctOfExpected"].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "")

    base = alt.Chart(data)
    bar = base.mark_bar(opacity=0.85).encode(
        x=alt.X("Age Group:N", axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("Category:N"),
        y=alt.Y("Value:Q", axis=alt.Axis(format=",")),
        color="Category:N",
        tooltip=["Age Group", "Category", alt.Tooltip("Value:Q", format=",.0f")],
    )
    labels = base.mark_text(dy=-6, fontWeight="bold").encode(
        x="Age Group:N",
        xOffset="Category:N",
        y="Value:Q",
        text=alt.Text("Value:Q", format=",.0f"),
    )
    pct = (
        base.transform_filter(alt.datum.Category == "Actual")
        .mark_text(dy=-22)
        .encode(x="Age Group:N", xOffset="Category:N", y="Value:Q", text="PctLabel:N")
    )

    return (bar + labels + pct).properties(title="Average Fruits per Tree (Actual vs Expected)", height=380)


def create_wordcloud(text: str, title: str):
    wc = WordCloud(width=900, height=450, background_color="white", colormap="viridis").generate(text or "")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(title, fontsize=16, pad=20)
    ax.axis("off")
    return fig


# ==========================================================
# Executive auto-insights
# ==========================================================
def executive_insights(df: pd.DataFrame) -> dict:
    d = _ensure_derived(df)

    insights = []
    risks = []
    actions = []

    n = len(d)

    # Density
    if CAN["trees_per_acre"] in d.columns:
        tpa = pd.to_numeric(d[CAN["trees_per_acre"]], errors="coerce").dropna()
        if len(tpa) >= 10:
            med = float(tpa.median())
            insights.append(f"Median orchard density is ~{med:,.0f} trees/acre (based on {len(tpa)} farms with area+trees).")
        else:
            risks.append("Orchard density insights are limited (missing area/trees).")
            actions.append("Improve capture of acreage and total tree counts during data collection.")

    # Geo coverage
    geo_ok = int(d[[CAN["lat"], CAN["lon"]]].dropna().shape[0]) if all(x in d.columns for x in [CAN["lat"], CAN["lon"]]) else 0
    if geo_ok > 0:
        insights.append(f"GPS coverage: {geo_ok}/{n} farms have valid coordinates for mapping.")
    else:
        risks.append("GPS mapping is unavailable (lat/lon missing).")
        actions.append("Ensure GPS capture is enabled and consistently collected in field surveys.")

    # Gender distribution
    if CAN["gender"] in d.columns:
        female = int(_safe_str(d, CAN["gender"]).str.lower().eq("female").sum())
        if female > 0:
            insights.append(f"Female farmers represent {female}/{n} ({female/n*100:.0f}%) of participants.")

    # Market
    if CAN["market_outlet"] in d.columns:
        outlet = _safe_str(d, CAN["market_outlet"])
        outlet = outlet[outlet != ""]
        if len(outlet):
            top = outlet.value_counts().index[0]
            insights.append(f"Top reported market outlet is **{top}** (from {len(outlet)} records).")
        else:
            risks.append("Market outlet is present but empty in most rows.")
            actions.append("Validate market section skip logic and enumerator training for market questions.")

    # Price anomaly flag
    if CAN["price_hass"] in d.columns:
        prices = pd.to_numeric(d[CAN["price_hass"]], errors="coerce").dropna()
        if len(prices) >= 20:
            q1, q99 = float(prices.quantile(0.01)), float(prices.quantile(0.99))
            outliers = prices[(prices < q1) | (prices > q99)]
            if len(outliers) > 0:
                risks.append(f"Price outliers detected ({len(outliers)} records outside ~1st–99th percentile).")
                actions.append("Review extreme price entries and confirm unit consistency (KSh/kg).")

    if not actions:
        actions.append("Continue periodic data quality checks and expand coverage for income, age portfolio and compliance fields.")

    return {
        "what_it_means": " ".join(insights) if insights else "The dashboard is ready; insights will strengthen as coverage improves.",
        "risks_gaps": list(dict.fromkeys(risks))[:6],
        "recommended_actions": list(dict.fromkeys(actions))[:6],
    }


# ==========================================================
# Dashboard sections
# ==========================================================
def show_overview(df: pd.DataFrame, metrics_df: pd.DataFrame | None = None):
    """Enhanced overview with farmer profile metrics."""
    st.subheader("Program Overview")
    d = _ensure_derived(df)

    # First row: core metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Farmers", int(len(d)))

    total_area = float(_safe_num(d, CAN["area"]).fillna(0).sum()) if CAN["area"] in d.columns else 0.0
    c2.metric("Total Area (Acres)", f"{total_area:,.2f}")

    total_trees = float(_safe_num(d, CAN["trees"]).fillna(0).sum()) if CAN["trees"] in d.columns else 0.0
    c3.metric("Total Trees", f"{total_trees:,.0f}")

    china_ok = int(_safe_bool(d, CAN["gacc"]).sum()) if CAN["gacc"] in d.columns else 0
    c4.metric("China-Approved Farms (GACC)", china_ok)

    # Second row: farmer profile metrics
    st.markdown("---")
    st.markdown("**👥 Farmer Profile**")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if CAN["gender"] in d.columns:
            gender_series = _safe_str(d, CAN["gender"]).str.lower()
            female = int(gender_series.eq("female").sum())
            male = int(gender_series.eq("male").sum())
            st.metric("👩 Female Farmers", f"{female} ({female/len(d)*100:.0f}%)" if len(d) > 0 else "N/A")
        else:
            st.caption("Gender data not available")

    with col2:
        if CAN["age"] in d.columns:
            ages = _safe_num(d, CAN["age"]).dropna()
            avg_age = ages.mean()
            st.metric("📊 Avg. Farmer Age", f"{avg_age:.0f} years" if not pd.isna(avg_age) else "N/A")
        else:
            st.caption("Age data not available")

    with col3:
        if CAN["education"] in d.columns:
            edu = _safe_str(d, CAN["education"])
            edu_counts = edu[edu != ""].value_counts()
            if not edu_counts.empty:
                top_edu = edu_counts.index[0]
                st.metric("🎓 Most Common Education", top_edu[:20])
            else:
                st.caption("Education data not available")
        else:
            st.caption("Education data not available")

    with col4:
        if CAN["experience"] in d.columns:
            exp_years = _safe_num(d, CAN["experience"]).dropna()
            avg_exp = exp_years.mean()
            st.metric("🌱 Avg. Experience", f"{avg_exp:.1f} years" if not pd.isna(avg_exp) else "N/A")
        else:
            st.caption("Experience data not available")

    # Third row: farm size classification
    st.markdown("---")
    st.markdown("**🏠 Farm Size Classification**")
    col1, col2, col3, col4 = st.columns(4)

    if CAN["area"] in d.columns:
        acres = _safe_num(d, CAN["area"])
        micro_small = sum(acres <= 3)
        medium = sum((acres > 3) & (acres <= 10))
        large = sum(acres > 10)
        
        col1.metric("🏡 Micro-Small (≤3 acres)", f"{micro_small} ({micro_small/len(d)*100:.0f}%)")
        col2.metric("🏠 Medium (4-10 acres)", f"{medium} ({medium/len(d)*100:.0f}%)")
        col3.metric("🏢 Large (>10 acres)", f"{large} ({large/len(d)*100:.0f}%)")
        
        # Orchard group registration
        if CAN["orchard_group"] in d.columns:
            group_reg = int(_safe_bool(d, CAN["orchard_group"]).sum())
            col4.metric("🤝 Registered in Farmer Group", f"{group_reg} ({group_reg/len(d)*100:.0f}%)")
    else:
        st.caption("Farm size data not available for classification")


def show_geospatial(df: pd.DataFrame):
    st.subheader("Farm Locations")
    m = create_farm_map(df)
    if m:
        folium_static(m, width=1100, height=620)
    else:
        st.warning("No valid geographic coordinates found in the data (latitude/longitude missing or empty).")


def show_certification(df: pd.DataFrame):
    """Enhanced certification with compliance details."""
    st.subheader("Certification & Compliance Status")

    col1, col2 = st.columns(2)

    with col1:
        chart = create_certification_chart(df)
        if chart:
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("No certification/compliance data available in this dataset.")

    with col2:
        st.markdown("**🇨🇳 China Market Requirements Checklist**")

        checks = [
            ("KEPHIS registration", CAN["kephis"]),
            ("GACC approval", CAN["gacc"]),
            ("Pest monitoring", CAN["pest"]),
            ("Sanitation records", CAN["sanitation"]),
            ("Approved pesticide use", CAN["approved_pesticides"]),
        ]

        d = _ensure_derived(df)
        n = max(len(d), 1)

        for label, col in checks:
            if col not in d.columns:
                st.caption(f"• {label}: (data not available)")
                continue

            compliant = int(_safe_bool(d, col).sum())
            st.progress(compliant / n, text=f"{label}: {compliant}/{n} farms")

    # Expandable compliance details
    with st.expander("📋 Detailed Compliance Metrics"):
        d = _ensure_derived(df)
        
        # SPS Training
        if CAN["sps_training"] in d.columns:
            sps_trained = int(_safe_bool(d, CAN["sps_training"]).sum())
            st.metric("SPS Training Received", f"{sps_trained}/{len(d)} farms ({sps_trained/len(d)*100:.0f}%)")
        
        # Training in last year
        if CAN["training_last_year"] in d.columns:
            trained_last_year = int(_safe_bool(d, CAN["training_last_year"]).sum())
            st.metric("Trained in Last Year", f"{trained_last_year} farms")
        
        # Training provider
        if CAN["training_provider"] in d.columns:
            provider = _safe_str(d, CAN["training_provider"])
            provider = provider[provider != ""]
            if not provider.empty:
                st.markdown("**Training Providers:**")
                provider_counts = provider.value_counts().head(3)
                for prov, count in provider_counts.items():
                    st.write(f"- {prov}: {count} farms")
        
        # Record keeping
        if CAN["record_keeping"] in d.columns:
            record_keepers = int(_safe_bool(d, CAN["record_keeping"]).sum())
            st.metric("Maintain Farm Records", f"{record_keepers} farms")
        
        # IPM implementation
        if CAN["ipm_use"] in d.columns:
            ipm_users = int(_safe_bool(d, CAN["ipm_use"]).sum())
            st.metric("IPM Implemented", f"{ipm_users}/{len(d)} farms ({ipm_users/len(d)*100:.0f}%)")
        
        # Biological control
        if CAN["biological_control"] in d.columns:
            bio_users = int(_safe_bool(d, CAN["biological_control"]).sum())
            st.metric("Biological Control Used", f"{bio_users} farms")


def show_production_metrics(df: pd.DataFrame):
    """Enhanced production metrics with detailed inputs."""
    st.subheader("Production Metrics")

    tab1, tab2, tab3, tab4 = st.tabs(["Yields", "Inputs", "Losses", "Productivity Analysis"])

    with tab1:
        chart = create_yield_comparison_chart(df)
        if chart:
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("No fruits-per-tree yield fields detected in the canonical schema.")

    with tab2:
        d = _ensure_derived(df)
        st.markdown("**Fertilizer Usage**")
        
        col1, col2 = st.columns(2)
        with col1:
            if CAN["fertilizer_organic"] in d.columns:
                organic = int(_safe_bool(d, CAN["fertilizer_organic"]).sum())
                st.metric("Organic Fertilizer Users", f"{organic} farms")
            else:
                st.caption("Organic fertilizer data not available")
        
        with col2:
            if CAN["fertilizer_inorganic"] in d.columns:
                inorganic = int(_safe_bool(d, CAN["fertilizer_inorganic"]).sum())
                st.metric("Inorganic Fertilizer Users", f"{inorganic} farms")
        
        # Fertilizer quantity
        if CAN["fertilizer_quantity"] in d.columns:
            fert_qty = _safe_num(d, CAN["fertilizer_quantity"]).dropna()
            if not fert_qty.empty:
                st.metric("Avg. Fertilizer (kg/tree/year)", f"{fert_qty.mean():.1f}")
        
        st.markdown("---")
        st.markdown("**Soil Conservation Measures**")
        
        if CAN["soil_conservation"] in d.columns:
            soil_practices = _safe_str(d, CAN["soil_conservation"])
            soil_practices = soil_practices[soil_practices != ""]
            if not soil_practices.empty:
                practices = soil_practices.str.split(',').explode().str.strip()
                practices = practices[practices != ""]
                practice_counts = practices.value_counts()
                for practice, count in practice_counts.head(4).items():
                    st.write(f"- {practice}: {count} farms")
        else:
            st.caption("Soil conservation data not available")
        
        st.markdown("---")
        st.markdown("**Irrigation Methods**")
        
        irrigation_data = []
        if CAN["irrigation_drip"] in d.columns:
            irrigation_data.append(("Drip", int(_safe_bool(d, CAN["irrigation_drip"]).sum())))
        if CAN["irrigation_sprinkler"] in d.columns:
            irrigation_data.append(("Sprinkler", int(_safe_bool(d, CAN["irrigation_sprinkler"]).sum())))
        if CAN["irrigation_rainfed"] in d.columns:
            irrigation_data.append(("Rainfed", int(_safe_bool(d, CAN["irrigation_rainfed"]).sum())))
        
        if irrigation_data:
            irr_df = pd.DataFrame(irrigation_data, columns=["Method", "Count"])
            chart = alt.Chart(irr_df).mark_bar().encode(
                x="Method:N", y="Count:Q", color="Method:N"
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.caption("Irrigation data not available")

    with tab3:
        st.caption("Loss analysis can be expanded once loss fields are mapped into the canonical schema (post-harvest loss %, key reasons, etc.).")

    with tab4:
        d = _ensure_derived(df)
        st.markdown("**📈 Productivity Analysis**")
        
        # Hass variety adoption
        if "variety_hass" in d.columns:
            hass_count = int(_safe_bool(d, "variety_hass").sum())
            st.metric("Hass Variety Farmers", f"{hass_count}/{len(d)} ({hass_count/len(d)*100:.0f}%)")
        
        # Total harvest
        if CAN["income_last"] in d.columns:
            total_income = _safe_num(d, CAN["income_last"]).sum()
            avg_income = _safe_num(d, CAN["income_last"]).mean()
            st.metric("Total Avocado Income (Last Season)", f"KSh {total_income:,.0f}")
            st.metric("Avg Income per Farm", f"KSh {avg_income:,.0f}")


def show_market_analysis(df: pd.DataFrame):
    """Enhanced market analysis with grade share and current prices."""
    st.subheader("Market Analysis")
    d = _ensure_derived(df)

    col1, col2 = st.columns(2)

    with col1:
        if CAN["market_outlet"] in d.columns:
            counts = _safe_str(d, CAN["market_outlet"])
            counts = counts[counts != ""]
            if not counts.empty:
                market_counts = counts.value_counts().reset_index()
                market_counts.columns = ["Outlet", "Count"]

                chart = (
                    alt.Chart(market_counts)
                    .mark_bar()
                    .encode(
                        x=alt.X("Count:Q", title="Count"),
                        y=alt.Y("Outlet:N", sort="-x", title="Outlet"),
                        color="Outlet:N",
                        tooltip=["Outlet", "Count"],
                    )
                    .properties(title="Main Market Outlets", height=380)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.warning("Market outlet exists, but values appear empty.")
        else:
            st.warning("Market outlet field is not available in this upload.")

    with col2:
        st.markdown("**Grade 1 Share**")
        
        if CAN["grade1_share_last"] in d.columns:
            grade_share = _safe_num(d, CAN["grade1_share_last"]).dropna()
            if not grade_share.empty:
                st.metric("Avg. Grade 1 Share (Last Season)", f"{grade_share.mean():.1f}%")
                
                # Grade distribution chart
                grade_data = grade_share
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(grade_data, bins=20, color='#2ecc71', edgecolor='black')
                ax.set_xlabel('Grade 1 Share (%)')
                ax.set_ylabel('Number of Farmers')
                ax.set_title('Distribution of Grade 1 Share')
                st.pyplot(fig)
        else:
            st.caption("Grade 1 share data not available")
    
    st.markdown("---")
    
    # Price comparison: last season vs current season
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**💰 Hass Prices (Last Season)**")
        if CAN["price_hass"] in d.columns:
            prices = _safe_num(d, CAN["price_hass"]).dropna()
            prices = prices[prices <= 120]  # Remove outliers
            if not prices.empty:
                st.metric("Average Price", f"KSh {prices.mean():.2f}/kg")
                st.metric("Median Price", f"KSh {prices.median():.2f}/kg")
            else:
                st.caption("Price data not available")
        else:
            st.caption("Price data not available")
    
    with col4:
        st.markdown("**💰 Hass Prices (Current Season)**")
        if CAN["price_hass_current"] in d.columns:
            current_prices = _safe_num(d, CAN["price_hass_current"]).dropna()
            current_prices = current_prices[current_prices <= 120]
            if not current_prices.empty:
                st.metric("Average Price", f"KSh {current_prices.mean():.2f}/kg")
                st.metric("Median Price", f"KSh {current_prices.median():.2f}/kg")
            else:
                st.caption("Current season price data not available")
        else:
            st.caption("Current season price data not available")


def show_training_needs(df: pd.DataFrame):
    """Enhanced training needs with extension access."""
    st.subheader("Training & Extension Needs")
    d = _ensure_derived(df)

    col1, col2 = st.columns(2)

    with col1:
        # Option A: already summarized (canonical export sheet may carry this)
        if "Top_Training_Need" in d.columns:
            s = _safe_str(d, "Top_Training_Need")
            s = s[s != ""]
            if not s.empty:
                c = s.value_counts().reset_index()
                c.columns = ["Need", "Count"]
                chart = (
                    alt.Chart(c)
                    .mark_bar()
                    .encode(
                        x=alt.X("Count:Q", title="Count"),
                        y=alt.Y("Need:N", sort="-x"),
                        color="Need:N",
                        tooltip=["Need", "Count"],
                    )
                    .properties(title="Most Pressing Training Needs", height=380)
                )
                st.altair_chart(chart, use_container_width=True)
                return

        # Option B: canonical one-hots derived from raw
        onehot_cols = [c for c in d.columns if str(c).startswith("training_need__")]
        if not onehot_cols:
            st.warning("Training needs fields are not available in this upload.")
        else:
            rows = []
            for c in onehot_cols:
                label = str(c).replace("training_need__", "").replace("_", " ").title()
                cnt = int(pd.to_numeric(d[c], errors="coerce").fillna(0).sum())
                if cnt > 0:
                    rows.append({"Need": label, "Count": cnt})

            if rows:
                out = pd.DataFrame(rows).sort_values("Count", ascending=False)
                chart = (
                    alt.Chart(out)
                    .mark_bar()
                    .encode(
                        x=alt.X("Count:Q", title="Count"),
                        y=alt.Y("Need:N", sort="-x"),
                        color="Need:N",
                        tooltip=["Need", "Count"],
                    )
                    .properties(title="Most Pressing Training Needs", height=380)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Training needs fields exist, but no selections were detected (all zeros).")

    with col2:
        st.markdown("**Extension Services Access**")
        
        if CAN["extension_access"] in d.columns:
            ext_access = _safe_str(d, CAN["extension_access"])
            ext_access = ext_access[ext_access != ""]
            if not ext_access.empty:
                ext_counts = ext_access.value_counts().reset_index()
                ext_counts.columns = ["Source", "Count"]
                chart = (
                    alt.Chart(ext_counts)
                    .mark_arc()
                    .encode(
                        theta=alt.Theta("Count:Q", stack=True),
                        color=alt.Color("Source:N"),
                        tooltip=["Source", "Count"],
                    )
                    .properties(title="Extension Services Access", height=300)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.caption("Extension access data not available")
        else:
            st.caption("Extension access data not available")
        
        st.markdown("---")
        st.markdown("**Training Providers**")
        
        if CAN["training_provider"] in d.columns:
            provider = _safe_str(d, CAN["training_provider"])
            provider = provider[provider != ""]
            if not provider.empty:
                prov_counts = provider.value_counts().reset_index()
                prov_counts.columns = ["Provider", "Count"]
                chart = (
                    alt.Chart(prov_counts)
                    .mark_bar()
                    .encode(
                        x="Provider:N", y="Count:Q", color="Provider:N"
                    )
                    .properties(height=250)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.caption("Training provider data not available")
        else:
            st.caption("Training provider data not available")


def show_sustainability(df: pd.DataFrame):
    """NEW: Sustainability practices section."""
    st.subheader("🌱 Sustainability Practices")
    d = _ensure_derived(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💧 Water Source**")
        if CAN["water_source"] in d.columns:
            water = _safe_str(d, CAN["water_source"])
            water = water[water != ""]
            if not water.empty:
                water_counts = water.value_counts()
                for source, count in water_counts.items():
                    st.write(f"- {source}: {count} farms")
            else:
                st.caption("Water source data not available")
        else:
            st.caption("Water source data not available")
        
        st.markdown("---")
        st.markdown("**🗑️ Waste Management**")
        if CAN["waste_management"] in d.columns:
            waste = _safe_str(d, CAN["waste_management"])
            waste = waste[waste != ""]
            if not waste.empty:
                waste_counts = waste.value_counts()
                for method, count in waste_counts.items():
                    st.write(f"- {method}: {count} farms")
            else:
                st.caption("Waste management data not available")
        else:
            st.caption("Waste management data not available")
    
    with col2:
        st.markdown("**🌳 Biodiversity Practices**")
        if CAN["biodiversity"] in d.columns:
            bio = _safe_str(d, CAN["biodiversity"])
            bio = bio[bio != ""]
            if not bio.empty:
                bio_counts = bio.value_counts()
                for practice, count in bio_counts.items():
                    st.write(f"- {practice}: {count} farms")
            else:
                st.caption("Biodiversity data not available")
        else:
            st.caption("Biodiversity data not available")
        
        st.markdown("---")
        st.markdown("**🌾 Other Trees Planted**")
        if CAN["additional_trees"] in d.columns:
            trees = _safe_str(d, CAN["additional_trees"])
            trees = trees[trees != ""]
            if not trees.empty:
                tree_counts = trees.value_counts()
                for tree, count in tree_counts.head(5).items():
                    st.write(f"- {tree}: {count} farms")
            else:
                st.caption("Additional trees data not available")
        else:
            st.caption("Additional trees data not available")
    
    st.markdown("---")
    st.markdown("**🐄 Other Value Chains**")
    if CAN["other_value_chains"] in d.columns:
        chains = _safe_str(d, CAN["other_value_chains"])
        chains = chains[chains != ""]
        if not chains.empty:
            chain_counts = chains.value_counts()
            for chain, count in chain_counts.items():
                st.write(f"- {chain}: {count} farms")
        else:
            st.caption("Other value chains data not available")
    else:
        st.caption("Other value chains data not available")


# ==========================================================
# Investor-grade sections (canonical - preserved)
# ==========================================================
def _prep_age_density_fields(df: pd.DataFrame, density_threshold: float) -> pd.DataFrame:
    d = _ensure_derived(df)

    if CAN["trees_0_3"] in d.columns:
        d["trees_0_3__derived"] = _safe_num(d, CAN["trees_0_3"]).fillna(0)
    else:
        d["trees_0_3__derived"] = 0.0

    if CAN["trees_4_7"] in d.columns:
        d["trees_4_7__derived"] = _safe_num(d, CAN["trees_4_7"]).fillna(0)
    else:
        d["trees_4_7__derived"] = 0.0

    if CAN["trees_8_plus"] in d.columns:
        d["trees_8_plus__derived"] = _safe_num(d, CAN["trees_8_plus"]).fillna(0)
    else:
        d["trees_8_plus__derived"] = 0.0

    d["density_bin__derived"] = np.where(
        pd.to_numeric(d.get(CAN["trees_per_acre"], np.nan), errors="coerce") > float(density_threshold),
        f">{int(density_threshold)}",
        f"≤{int(density_threshold)}",
    )

    if CAN["dominant_age"] in d.columns:
        d["dominant_age_group__derived"] = _safe_str(d, CAN["dominant_age"])
    else:
        d["dominant_age_group__derived"] = ""

    if d["dominant_age_group__derived"].astype("string").str.strip().eq("").mean() > 0.80:
        age_counts = d[["trees_0_3__derived", "trees_4_7__derived", "trees_8_plus__derived"]].copy()
        dom = age_counts.idxmax(axis=1).fillna("trees_0_3__derived")
        d["dominant_age_group__derived"] = dom.map(
            {
                "trees_0_3__derived": "0-3 years",
                "trees_4_7__derived": "4-7 years",
                "trees_8_plus__derived": "8+ years",
            }
        ).fillna("0-3 years")

    d["income_per_acre__derived"] = pd.to_numeric(d.get(CAN["income_per_acre"], np.nan), errors="coerce")

    return _clip_inf(d)


def show_investor_income_view(df: pd.DataFrame):
    st.subheader("💰 Income by Tree Age & Density")

    if df is None or len(df) == 0:
        st.info("No data available.")
        return

    d0 = _ensure_derived(df)
    observed = pd.to_numeric(d0.get(CAN["trees_per_acre"], np.nan), errors="coerce").dropna()
    default_threshold = float(np.nanpercentile(observed, 50)) if len(observed) else 90.0
    if not np.isfinite(default_threshold) or default_threshold <= 0:
        default_threshold = 90.0

    density_threshold = st.number_input(
        "Density threshold (trees/acre)",
        min_value=1.0,
        value=float(default_threshold),
        help="Farms at/below are low density; above are high density.",
    )

    d = _prep_age_density_fields(df, density_threshold)
    d = d.dropna(subset=["income_per_acre__derived", "dominant_age_group__derived", "density_bin__derived"])
    d = d[d["income_per_acre__derived"].notna()]

    if len(d) < 10:
        st.info("Not enough complete records for investor-grade income insights (need income + area + age-portfolio counts).")
        return

    grp = (
        d.groupby(["dominant_age_group__derived", "density_bin__derived"])["income_per_acre__derived"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .rename(
            columns={
                "dominant_age_group__derived": "Dominant age",
                "density_bin__derived": "Density bin",
                "mean": "Avg income/acre",
                "median": "Median income/acre",
            }
        )
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Income per acre by dominant tree age & density**")
        chart1 = (
            alt.Chart(grp)
            .mark_bar()
            .encode(
                x=alt.X("Dominant age:N", title="Dominant tree age"),
                y=alt.Y("Avg income/acre:Q", title="Average income/acre (KSh)"),
                color=alt.Color("Density bin:N", title="Density"),
                tooltip=[
                    "Dominant age",
                    "Density bin",
                    "count",
                    alt.Tooltip("Avg income/acre:Q", format=",.0f"),
                    alt.Tooltip("Median income/acre:Q", format=",.0f"),
                ],
            )
            .properties(height=380)
        )
        st.altair_chart(chart1, use_container_width=True)

    with col2:
        st.markdown("**Tree portfolio by age category (share)**")
        total_trees = pd.DataFrame(
            {
                "Age Group": ["0-3 years", "4-7 years", "8+ years"],
                "Trees": [
                    d["trees_0_3__derived"].sum(),
                    d["trees_4_7__derived"].sum(),
                    d["trees_8_plus__derived"].sum(),
                ],
            }
        )
        total_trees = total_trees[total_trees["Trees"] > 0]
        if total_trees.empty:
            st.caption("Age-portfolio tree counts appear missing/zero.")
        else:
            chart2 = (
                alt.Chart(total_trees)
                .mark_arc()
                .encode(
                    theta=alt.Theta("Trees:Q", stack=True),
                    color=alt.Color("Age Group:N"),
                    tooltip=["Age Group", alt.Tooltip("Trees:Q", format=",.0f")],
                )
                .properties(height=380)
            )
            st.altair_chart(chart2, use_container_width=True)

    with st.expander("Show summary table"):
        st.dataframe(grp.sort_values(["Dominant age", "Density bin"]), use_container_width=True)


def show_income_potential_forecast(df: pd.DataFrame):
    st.subheader("📈 Income Potential Forecast")

    if df is None or len(df) == 0:
        st.info("No data available.")
        return

    ref = _yield_reference()
    kg03 = float(ref["0-3"].get("kg", np.nan))
    kg47 = float(ref["4-7"].get("kg", np.nan))
    kg8p = float(ref["8+"].get("kg", np.nan))

    if not np.isfinite(kg03) or not np.isfinite(kg47) or not np.isfinite(kg8p):
        st.warning("Yield reference (kg/tree) is not configured. Add kg/tree to core.data.YIELD_REFERENCE.")
        return

    d0 = _ensure_derived(df)

    # Price: use median if available; else allow manual entry
    prices = pd.to_numeric(d0.get(CAN["price_hass"], np.nan), errors="coerce").dropna()
    if len(prices) > 20:
        lo = float(prices.quantile(0.01))
        hi = float(prices.quantile(0.99))
        prices = prices[(prices >= lo) & (prices <= hi)]

    median_price = float(prices.median()) if len(prices) else 0.0
    if median_price <= 0:
        median_price = st.number_input("Reference price (KSh/kg)", min_value=0.0, value=0.0)

    # Density threshold for grouping
    observed = pd.to_numeric(d0.get(CAN["trees_per_acre"], np.nan), errors="coerce").dropna()
    default_threshold = float(np.nanpercentile(observed, 50)) if len(observed) else 90.0
    if not np.isfinite(default_threshold) or default_threshold <= 0:
        default_threshold = 90.0

    density_threshold = st.number_input(
        "Density threshold (trees/acre)",
        min_value=1.0,
        value=float(default_threshold),
        help="Used to label farms into low/high density buckets.",
    )

    d = _prep_age_density_fields(df, density_threshold)

    if CAN["area"] not in d.columns or d[CAN["area"]].isna().all():
        st.warning("Cannot compute forecast: `area_acres` is missing.")
        return

    d = d.dropna(subset=[CAN["area"]])
    d = d[d[CAN["area"]] > 0]

    # Need age-class counts
    if (d["trees_0_3__derived"].sum() + d["trees_4_7__derived"].sum() + d["trees_8_plus__derived"].sum()) <= 0:
        st.warning("Cannot compute forecast: age portfolio counts (trees_0_3/trees_4_7/trees_8_plus) are missing.")
        return

    # Trees per acre by age
    d["tpa_0_3"] = d["trees_0_3__derived"] / d[CAN["area"]]
    d["tpa_4_7"] = d["trees_4_7__derived"] / d[CAN["area"]]
    d["tpa_8p"] = d["trees_8_plus__derived"] / d[CAN["area"]]

    d["forecast_income_per_acre"] = (d["tpa_0_3"] * kg03 + d["tpa_4_7"] * kg47 + d["tpa_8p"] * kg8p) * float(median_price)
    d = _clip_inf(d).dropna(subset=["forecast_income_per_acre", "density_bin__derived", "dominant_age_group__derived"])

    if len(d) < 10:
        st.info("Not enough complete records to compute the forecast.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Baseline distribution** (price ≈ KSh {median_price:,.0f}/kg)")
        hist = pd.DataFrame({"Forecast income/acre (KSh)": d["forecast_income_per_acre"]})
        st.altair_chart(
            alt.Chart(hist)
            .mark_bar()
            .encode(x=alt.X("Forecast income/acre (KSh):Q", bin=alt.Bin(maxbins=30)), y="count()")
            .properties(height=380),
            use_container_width=True,
        )

    by_cut = (
        d.groupby(["dominant_age_group__derived", "density_bin__derived"])["forecast_income_per_acre"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .rename(
            columns={
                "dominant_age_group__derived": "Dominant age",
                "density_bin__derived": "Density bin",
                "mean": "Avg forecast/acre",
                "median": "Median forecast/acre",
            }
        )
    )

    with col2:
        st.markdown("**Baseline by dominant age & density**")
        st.altair_chart(
            alt.Chart(by_cut)
            .mark_bar()
            .encode(
                x="Dominant age:N",
                y=alt.Y("Avg forecast/acre:Q", title="Avg forecast income/acre (KSh)"),
                color=alt.Color("Density bin:N"),
                tooltip=[
                    "Dominant age",
                    "Density bin",
                    "count",
                    alt.Tooltip("Avg forecast/acre:Q", format=",.0f"),
                    alt.Tooltip("Median forecast/acre:Q", format=",.0f"),
                ],
            )
            .properties(height=380),
            use_container_width=True,
        )

    k1, k2, k3 = st.columns(3)
    k1.metric("Median forecast (KSh/acre)", f"{d['forecast_income_per_acre'].median():,.0f}")
    k2.metric("P75 forecast (KSh/acre)", f"{d['forecast_income_per_acre'].quantile(0.75):,.0f}")
    k3.metric("Top decile (KSh/acre)", f"{d['forecast_income_per_acre'].quantile(0.90):,.0f}")


def show_loss_wordcloud(df: pd.DataFrame):
    st.subheader("🗣️ Loss Reasons Word Cloud")
    if "loss_reasons_text" not in df.columns:
        st.info("Provide/compute `loss_reasons_text` to display a word cloud.")
        return
    txt = " ".join(df["loss_reasons_text"].astype(str).tolist())
    fig = create_wordcloud(txt, "Loss Reasons")
    st.pyplot(fig)