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
    "price_hass_current": "hass_price_current_ksh_per_kg",
    "income_last": "income_ksh_last_season",
    "income_per_acre": "income_per_acre",
    "income_per_tree": "income_per_tree",
    "yield_per_acre": "yield_per_acre",
    "grade1_share_last": "grade1_share_last",
    "grade1_share_current": "grade1_share_current",
    "grade2_share_last": "grade2_share_last",
    "grade2_share_current": "grade2_share_current",
    "market_constraints": "market_constraints",  # ADDED - fixes KeyError
    "trees_0_3": "trees_0_3",
    "trees_4_7": "trees_4_7",
    "trees_8_plus": "trees_8_plus",
    "fruits_0_3": "fruits_per_tree_0_3",
    "fruits_4_7": "fruits_per_tree_4_7",
    "fruits_8_plus": "fruits_per_tree_8_plus",
    "dominant_age": "dominant_age_group",
    # Farmer profile
    "gender": "gender",
    "age": "age",
    "education": "education",
    "experience": "experience",
    # Legal registration
    "orchard_group": "orchard_group",
    # Production practices
    "fertilizer_organic": "fertilizer_organic",
    "fertilizer_inorganic": "fertilizer_inorganic",
    "fertilizer_quantity": "fertilizer_quantity",
    "soil_conservation": "soil_conservation",
    "irrigation_drip": "irrigation_drip",
    "irrigation_sprinkler": "irrigation_sprinkler",
    "irrigation_rainfed": "irrigation_rainfed",
    # Production losses
    "harvest_kg": "harvest_kg",
    "harvest_losses": "harvest_losses_kg",
    "loss_causes": "loss_causes",
    # NEW: Enhanced Production Practices
    "dropped_fruits_elimination": "dropped_fruits_elimination",
    "weed_management": "weed_management",
    "pruning_practices": "pruning_practices",
    # NEW: Detailed IPM Measures
    "ipm_traps": "ipm_traps",
    "ipm_chemical": "ipm_chemical",
    "ipm_biological": "ipm_biological",
    "ipm_mating_disruption": "ipm_mating_disruption",
    # NEW: Certification Compliance Training
    "cert_compliance_training": "cert_compliance_training",
    # Compliance
    "sps_training": "sps_training",
    "training_last_year": "training_last_year",
    "training_last_6months": "training_last_6months",
    "training_provider": "training_provider",
    "record_keeping": "record_keeping",
    "ipm_use": "ipm_use",
    "pesticide_use": "pesticide_use",
    "biological_control": "biological_control",
    # Sustainability
    "water_source": "water_source",
    "waste_management": "waste_management",
    "biodiversity": "biodiversity",
    "additional_trees": "additional_trees",
    "other_value_chains": "other_value_chains",
    # Extension
    "extension_access": "extension_access",
    "training_needs_text": "training_needs_text",
    # NEW: Derived metrics
    "organic_alignment": "organic_alignment",
    "grade_total_completeness": "grade_total_completeness",
    "ipm_adoption_score": "ipm_adoption_score",
    "farm_size_classification": "farm_size_classification",
    "aez_zone": "aez_zone",
    "altitude": "altitude",
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
              CAN["grade2_share_last"], CAN["grade2_share_current"], CAN["yield_per_acre"],
              CAN["age"], CAN["experience"], CAN["altitude"], CAN["harvest_kg"], CAN["harvest_losses"]]:
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

    # Income per tree (NEW)
    if CAN["income_per_tree"] not in d.columns:
        d[CAN["income_per_tree"]] = np.nan
    if (d[CAN["income_per_tree"]].isna().all()) and (CAN["income_last"] in d.columns) and (CAN["trees"] in d.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            d[CAN["income_per_tree"]] = d[CAN["income_last"]] / d[CAN["trees"]]

    # Yield per acre (NEW)
    if CAN["yield_per_acre"] not in d.columns and CAN["harvest_kg"] in d.columns:
        if CAN["area"] in d.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                d[CAN["yield_per_acre"]] = d[CAN["harvest_kg"]] / d[CAN["area"]]

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

    # Farm size classification (NEW)
    if CAN["farm_size_classification"] not in d.columns and CAN["area"] in d.columns:
        d[CAN["farm_size_classification"]] = d[CAN["area"]].apply(classify_farm_size)

    # IPM adoption score (NEW)
    if CAN["ipm_adoption_score"] not in d.columns:
        ipm_fields = [CAN.get(f) for f in ["ipm_traps", "ipm_chemical", "ipm_biological", "ipm_mating_disruption"]]
        ipm_fields = [f for f in ipm_fields if f and f in d.columns]
        if ipm_fields:
            ipm_df = d[ipm_fields].fillna(False).astype(bool)
            d[CAN["ipm_adoption_score"]] = ipm_df.mean(axis=1) * 100

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
# NEW: Enhanced visualization functions
# ==========================================================

def show_geo_spatial_aez(df: pd.DataFrame):
    """NEW: Display geo-spatial distribution and AEZ mapping."""
    st.subheader("🌍 Geo-Spatial & Agro-Ecological Zones")
    d = _ensure_derived(df)

    # Check if we have AEZ data
    if CAN["aez_zone"] in d.columns:
        col1, col2 = st.columns(2)

        with col1:
            # AEZ Distribution
            aez_counts = d[CAN["aez_zone"]].value_counts().reset_index()
            aez_counts.columns = ["AEZ Zone", "Count"]

            if not aez_counts.empty:
                chart = alt.Chart(aez_counts).mark_bar().encode(
                    x=alt.X("AEZ Zone:N", sort="-y"),
                    y=alt.Y("Count:Q", title="Number of Farms"),
                    color="AEZ Zone:N",
                    tooltip=["AEZ Zone", "Count"]
                ).properties(title="Farmer Distribution by Agro-Ecological Zone", height=380)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("AEZ distribution data not available")

        with col2:
            # Altitude distribution by AEZ
            if CAN["altitude"] in d.columns and CAN["aez_zone"] in d.columns:
                alt_data = d[[CAN["aez_zone"], CAN["altitude"]]].dropna()
                if not alt_data.empty:
                    chart = alt.Chart(alt_data).mark_boxplot().encode(
                        x=CAN["aez_zone"],
                        y=alt.Y(CAN["altitude"], title="Altitude (m)"),
                        color=CAN["aez_zone"]
                    ).properties(title="Altitude Distribution by AEZ", height=380)
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("Altitude data not available for AEZ analysis")
            else:
                st.info("Altitude data not available")

        # Show AEZ summary table - using st.markdown with separator instead of expander
        st.markdown("---")
        st.markdown("**📊 AEZ Distribution Details**")
        aez_summary = d.groupby(CAN["aez_zone"]).agg(
            Farm_Count=(CAN["aez_zone"], "count"),
            Avg_Altitude=(CAN["altitude"], "mean") if CAN["altitude"] in d.columns else None,
            Min_Altitude=(CAN["altitude"], "min") if CAN["altitude"] in d.columns else None,
            Max_Altitude=(CAN["altitude"], "max") if CAN["altitude"] in d.columns else None,
        ).reset_index()

        if "Avg_Altitude" in aez_summary.columns:
            aez_summary["Avg_Altitude"] = aez_summary["Avg_Altitude"].round(1)
            aez_summary["Min_Altitude"] = aez_summary["Min_Altitude"].round(1)
            aez_summary["Max_Altitude"] = aez_summary["Max_Altitude"].round(1)

        st.dataframe(aez_summary, use_container_width=True)
    else:
        st.info("AEZ data not available. Ensure altitude data is present for AEZ classification.")

def show_farmer_profile(df: pd.DataFrame):
    """NEW: Display farmer profile metrics (gender, age, education, experience)."""
    st.subheader("👥 Farmer Profile")
    d = _ensure_derived(df)

    col1, col2 = st.columns(2)

    with col1:
        # Gender distribution
        if CAN["gender"] in d.columns:
            gender_series = _safe_str(d, CAN["gender"]).str.lower()
            gender_counts = gender_series.value_counts().reset_index()
            gender_counts.columns = ["Gender", "Count"]
            
            chart = alt.Chart(gender_counts).mark_bar().encode(
                x="Gender:N",
                y="Count:Q",
                color="Gender:N",
                tooltip=["Gender", "Count"]
            ).properties(title="Gender Distribution", height=300)
            st.altair_chart(chart, use_container_width=True)

    with col2:
        # Age distribution
        if CAN["age"] in d.columns:
            ages = _safe_num(d, CAN["age"]).dropna()
            if not ages.empty:
                age_data = pd.DataFrame({"Age": ages})
                chart = alt.Chart(age_data).mark_bar().encode(
                    x=alt.X("Age:Q", bin=alt.Bin(maxbins=30)),
                    y="count()",
                    tooltip=["count()"]
                ).properties(title="Age Distribution", height=300)
                st.altair_chart(chart, use_container_width=True)

    # Education and Experience
    col3, col4 = st.columns(2)

    with col3:
        # Education level
        if CAN["education"] in d.columns:
            edu_series = _safe_str(d, CAN["education"])
            edu_clean = edu_series.str.lower().str.strip()
            
            # Map education levels
            edu_map = {
                "none": "None",
                "primary": "Primary",
                "secondary": "Secondary",
                "tertiary": "Tertiary",
                "graduate": "Graduate",
                "post-graduate": "Post-Graduate",
                "postgraduate": "Post-Graduate"
            }
            edu_clean = edu_clean.map(lambda x: edu_map.get(x, x.title() if x else "Unknown"))
            edu_clean = edu_clean[edu_clean != ""]
            
            if not edu_clean.empty:
                edu_counts = edu_clean.value_counts().reset_index()
                edu_counts.columns = ["Education Level", "Count"]
                
                chart = alt.Chart(edu_counts).mark_bar().encode(
                    x="Education Level:N",
                    y="Count:Q",
                    color="Education Level:N",
                    tooltip=["Education Level", "Count"]
                ).properties(title="Education Level Distribution", height=300)
                st.altair_chart(chart, use_container_width=True)
                
                # Show data availability
                edu_coverage = (d[CAN["education"]].notna().sum() / len(d) * 100)
                if edu_coverage < 80:
                    st.caption(f"⚠️ Education data available for {edu_coverage:.1f}% of farmers")
            else:
                st.info("Education data not available")
        else:
            st.info("Education data not available")

    with col4:
        # Experience distribution
        if CAN["experience"] in d.columns:
            exp_years = _safe_num(d, CAN["experience"]).dropna()
            if not exp_years.empty:
                exp_data = pd.DataFrame({"Experience (years)": exp_years})
                chart = alt.Chart(exp_data).mark_bar().encode(
                    x=alt.X("Experience (years):Q", bin=alt.Bin(maxbins=20)),
                    y="count()",
                    tooltip=["count()"]
                ).properties(title="Farming Experience Distribution", height=300)
                st.altair_chart(chart, use_container_width=True)
                
                # Experience statistics
                avg_exp = exp_years.mean()
                st.metric("Average Experience", f"{avg_exp:.1f} years")
                
                # Show data availability
                exp_coverage = (d[CAN["experience"]].notna().sum() / len(d) * 100)
                if exp_coverage < 80:
                    st.caption(f"⚠️ Experience data available for {exp_coverage:.1f}% of farmers")
            else:
                st.info("Experience data not available")
        else:
            st.info("Experience data not available")


def show_farm_characteristics(df: pd.DataFrame):
    """NEW: Display farm characteristics (size, varieties, tree age distribution)."""
    st.subheader("🌱 Farm Characteristics")
    d = _ensure_derived(df)

    col1, col2 = st.columns(2)

    with col1:
        # Farm size classification
        if CAN["farm_size_classification"] in d.columns:
            size_counts = d[CAN["farm_size_classification"]].value_counts().reset_index()
            size_counts.columns = ["Farm Size", "Count"]
            
            chart = alt.Chart(size_counts).mark_bar().encode(
                x="Farm Size:N",
                y="Count:Q",
                color="Farm Size:N",
                tooltip=["Farm Size", "Count"]
            ).properties(title="Farm Size Distribution", height=300)
            st.altair_chart(chart, use_container_width=True)

    with col2:
        # Variety distribution (Hass adoption)
        if "variety_hass" in d.columns:
            hass_count = int(_safe_bool(d, "variety_hass").sum())
            st.metric("Hass Variety Adoption", f"{hass_count} ({hass_count/len(d)*100:.0f}%)")

    # Tree age distribution
    st.markdown("### 🌳 Tree Age Distribution")
    if all(c in d.columns for c in [CAN["trees_0_3"], CAN["trees_4_7"], CAN["trees_8_plus"]]):
        age_data = {
            "Age Group": ["0-3 years", "4-7 years", "8+ years"],
            "Tree Count": [
                d[CAN["trees_0_3"]].sum(),
                d[CAN["trees_4_7"]].sum(),
                d[CAN["trees_8_plus"]].sum(),
            ],
        }
        age_df = pd.DataFrame(age_data)

        chart = alt.Chart(age_df).mark_bar().encode(
            x="Age Group:N",
            y="Tree Count:Q",
            color="Age Group:N",
            tooltip=["Age Group", "Tree Count"]
        ).properties(title="Total Trees by Age Group", height=380)
        st.altair_chart(chart, use_container_width=True)

        # Dominant age group
        if CAN["dominant_age"] in d.columns:
            dom_counts = d[CAN["dominant_age"]].value_counts().reset_index()
            dom_counts.columns = ["Dominant Age Group", "Count"]
            dom_counts = dom_counts[dom_counts["Dominant Age Group"] != ""]
            
            if not dom_counts.empty:
                chart = alt.Chart(dom_counts).mark_bar().encode(
                    x="Dominant Age Group:N",
                    y="Count:Q",
                    color="Dominant Age Group:N"
                ).properties(title="Dominant Tree Age Group by Farm", height=300)
                st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Tree age distribution data not available")


def show_productivity_efficiency(df: pd.DataFrame):
    """NEW: Show productivity and efficiency metrics (Yield/Acre, Income/Acre, Income/Tree)."""
    st.subheader("📊 Productivity & Efficiency")
    d = _ensure_derived(df)

    col1, col2, col3 = st.columns(3)

    with col1:
        # Yield per Acre
        if CAN["yield_per_acre"] in d.columns:
            yield_acre = _safe_num(d, CAN["yield_per_acre"]).dropna()
            # Remove extreme outliers for better display
            if len(yield_acre) > 10:
                q99 = yield_acre.quantile(0.99)
                yield_acre = yield_acre[yield_acre <= q99]
            if not yield_acre.empty:
                st.metric("Avg. Yield per Acre", f"{yield_acre.mean():,.0f} kg/acre")
                st.metric("Median Yield per Acre", f"{yield_acre.median():,.0f} kg/acre")
            else:
                st.caption("Yield per acre data not available")
        else:
            st.caption("Yield per acre data not available")

    with col2:
        # Income per Acre
        if CAN["income_per_acre"] in d.columns:
            income_acre = _safe_num(d, CAN["income_per_acre"]).dropna()
            # Remove extreme outliers
            if len(income_acre) > 10:
                q99 = income_acre.quantile(0.99)
                income_acre = income_acre[income_acre <= q99]
            if not income_acre.empty:
                st.metric("Avg. Income per Acre", f"KSh {income_acre.mean():,.0f}")
                st.metric("Median Income per Acre", f"KSh {income_acre.median():,.0f}")
            else:
                st.caption("Income per acre data not available")
        else:
            st.caption("Income per acre data not available")

    with col3:
        # Income per Tree
        if CAN["income_per_tree"] in d.columns:
            income_tree = _safe_num(d, CAN["income_per_tree"]).dropna()
            # Remove extreme outliers
            if len(income_tree) > 10:
                q99 = income_tree.quantile(0.99)
                income_tree = income_tree[income_tree <= q99]
            if not income_tree.empty:
                st.metric("Avg. Income per Tree", f"KSh {income_tree.mean():,.0f}")
                st.metric("Median Income per Tree", f"KSh {income_tree.median():,.0f}")
            else:
                st.caption("Income per tree data not available")
        else:
            st.caption("Income per tree data not available")

    # Income per Tree by Age Group
    st.markdown("---")
    st.markdown("### 💰 Income per Tree by Age Group")
    income_age_data = []

    for age_group, col_name in [
        ("0-3 years", "income_per_tree_0_3"),
        ("4-7 years", "income_per_tree_4_7"),
        ("8+ years", "income_per_tree_8_plus"),
    ]:
        if col_name in d.columns:
            valid_data = _safe_num(d, col_name).dropna()
            # Remove extreme outliers
            if len(valid_data) > 10:
                q99 = valid_data.quantile(0.99)
                valid_data = valid_data[valid_data <= q99]
            if not valid_data.empty:
                income_age_data.append({
                    "Age Group": age_group,
                    "Median Income (KSh)": valid_data.median(),
                    "Mean Income (KSh)": valid_data.mean(),
                })

    if income_age_data:
        income_age_df = pd.DataFrame(income_age_data)
        chart = alt.Chart(income_age_df).mark_bar().encode(
            x="Age Group:N",
            y="Median Income (KSh):Q",
            color="Age Group:N",
            tooltip=["Age Group", "Median Income (KSh)", "Mean Income (KSh)"]
        ).properties(title="Median Income per Tree by Age Group", height=380)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Income per tree by age group data not available")

    # Grade distribution
    st.markdown("---")
    st.markdown("### 📊 Grade Distribution")
    if "grade1_share_last" in d.columns:
        grade1_vals = _safe_num(d, "grade1_share_last")
        grade2_vals = _safe_num(d, "grade2_share_last") if "grade2_share_last" in d.columns else pd.Series([0] * len(d))
        
        # Handle decimal vs percentage format
        if not grade1_vals.empty and grade1_vals.max() <= 1:
            grade1_vals = grade1_vals * 100
        if not grade2_vals.empty and grade2_vals.max() <= 1:
            grade2_vals = grade2_vals * 100
        
        grade1_mean = grade1_vals.mean()
        grade2_mean = grade2_vals.mean()
        
        grade_data = {
            "Grade": ["Grade 1", "Grade 2 (Own Use)"],
            "Percentage": [grade1_mean, grade2_mean],
        }
        grade_df = pd.DataFrame(grade_data)

        chart = alt.Chart(grade_df).mark_bar().encode(
            x="Grade:N",
            y="Percentage:Q",
            color="Grade:N",
            tooltip=["Grade", "Percentage"]
        ).properties(title="Average Grade Distribution (Last Season)", height=300)
        st.altair_chart(chart, use_container_width=True)

        # Grade completeness validation
        completeness = grade1_mean + grade2_mean
        if abs(completeness - 100) > 10:
            st.warning(f"⚠️ Grade data may be inconsistent (Grade 1 + Grade 2 = {completeness:.1f}%)")
    else:
        st.info("Grade distribution data not available")


def show_enhanced_compliance(df: pd.DataFrame):
    """NEW: Show enhanced compliance metrics (dropped fruits, weed mgmt, pruning)."""
    st.subheader("📋 Enhanced Compliance Practices")
    d = _ensure_derived(df)

    col1, col2, col3 = st.columns(3)

    with col1:
        if CAN["dropped_fruits_elimination"] in d.columns:
            dropped = int(_safe_bool(d, CAN["dropped_fruits_elimination"]).sum())
            st.metric("🍂 Prompt Dropped Fruits Elimination", f"{dropped} ({dropped/len(d)*100:.0f}%)")
        else:
            st.caption("Dropped fruits data not available")

    with col2:
        if CAN["weed_management"] in d.columns:
            weed = int(_safe_bool(d, CAN["weed_management"]).sum())
            st.metric("🌿 Weed Management Practices", f"{weed} ({weed/len(d)*100:.0f}%)")
        else:
            st.caption("Weed management data not available")

    with col3:
        if CAN["pruning_practices"] in d.columns:
            pruning = int(_safe_bool(d, CAN["pruning_practices"]).sum())
            st.metric("✂️ Pruning Practices", f"{pruning} ({pruning/len(d)*100:.0f}%)")
        else:
            st.caption("Pruning data not available")

    # Certification compliance training
    st.markdown("---")
    if CAN["cert_compliance_training"] in d.columns:
        cert_training = int(_safe_bool(d, CAN["cert_compliance_training"]).sum())
        st.metric("📚 Certification Compliance Training", f"{cert_training} ({cert_training/len(d)*100:.0f}%)")
    else:
        st.caption("Certification compliance training data not available")


def show_ipm_measures(df: pd.DataFrame):
    """NEW: Show detailed IPM measures adoption."""
    st.subheader("🪰 Integrated Pest Management (IPM) Measures")
    d = _ensure_derived(df)

    ipm_data = []

    if CAN["ipm_traps"] in d.columns:
        ipm_data.append(("Fly Traps", int(_safe_bool(d, CAN["ipm_traps"]).sum())))
    if CAN["ipm_chemical"] in d.columns:
        ipm_data.append(("Chemical Control", int(_safe_bool(d, CAN["ipm_chemical"]).sum())))
    if CAN["ipm_biological"] in d.columns:
        ipm_data.append(("Biological Control", int(_safe_bool(d, CAN["ipm_biological"]).sum())))
    if CAN["ipm_mating_disruption"] in d.columns:
        ipm_data.append(("Mating Disruption", int(_safe_bool(d, CAN["ipm_mating_disruption"]).sum())))

    if ipm_data:
        ipm_df = pd.DataFrame(ipm_data, columns=["Measure", "Count"])

        col1, col2 = st.columns(2)

        with col1:
            chart = alt.Chart(ipm_df).mark_bar().encode(
                x=alt.X("Count:Q", title="Number of Farms"),
                y=alt.Y("Measure:N", sort="-x", title="IPM Measure"),
                color="Measure:N",
                tooltip=["Measure", "Count"]
            ).properties(title="IPM Measures Adoption", height=300)
            st.altair_chart(chart, use_container_width=True)

        with col2:
            # Overall IPM adoption
            if CAN["ipm_use"] in d.columns:
                ipm_overall = int(_safe_bool(d, CAN["ipm_use"]).sum())
                st.metric("Overall IPM Implementation", f"{ipm_overall} ({ipm_overall/len(d)*100:.0f}%)")

            # IPM adoption score
            if CAN["ipm_adoption_score"] in d.columns:
                scores = _safe_num(d, CAN["ipm_adoption_score"]).dropna()
                if not scores.empty:
                    st.metric("Avg. IPM Adoption Score", f"{scores.mean():.1f}%")
                    
                    # Score distribution
                    score_data = pd.DataFrame({"IPM Score (%)": scores})
                    chart = alt.Chart(score_data).mark_bar().encode(
                        x=alt.X("IPM Score (%):Q", bin=alt.Bin(maxbins=20)),
                        y="count()"
                    ).properties(title="IPM Score Distribution", height=250)
                    st.altair_chart(chart, use_container_width=True)

            # Multiple measures adoption
            measures_count = sum(1 for _, count in ipm_data if count > 0)
            st.metric("IPM Measures Used", f"{measures_count} of {len(ipm_data)}")
    else:
        st.info("IPM measures data not available")

def show_gacc_progress(df: pd.DataFrame):
    """NEW: Show GACC progress with KEPHIS reference data."""
    st.subheader("🇨🇳 China Market (GACC) Progress")
    d = _ensure_derived(df)

    # Get GACC approved farms
    gacc_approved = int(_safe_bool(d, CAN["gacc"]).sum()) if CAN["gacc"] in d.columns else 0

    # KEPHIS reference data (target: 270 total coded farms)
    kephis_total = 270  # Total KEPHIS-China coded farms in Kenya
    new_inspected = 41  # Newly inspected farms (from document)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("SHAPe GACC Approved Farms", f"{gacc_approved}", help="Farms in SHAPe program with GACC approval")

    with col2:
        st.metric("KEPHIS-China Coded Farms (Kenya)", f"{kephis_total}", help="Total KEPHIS-China coded farms in Kenya (reference)")

    with col3:
        st.metric("Newly Inspected Farms", f"{new_inspected}", help="Newly inspected farms for GACC compliance")

    with col4:
        pct_of_total = (gacc_approved / kephis_total * 100) if kephis_total > 0 else 0
        st.metric("SHAPe % of Total", f"{pct_of_total:.1f}%", help="SHAPe farms as percentage of all KEPHIS-China coded farms")

    # Progress bar
    st.markdown("---")
    st.markdown("**GACC Approval Progress**")
    st.progress(gacc_approved / kephis_total if kephis_total > 0 else 0, 
                text=f"{gacc_approved} of {kephis_total} KEPHIS-China coded farms")

    # Newly inspected farms info
    st.markdown("---")
    st.info(f"📋 **Newly Inspected Farms:** {new_inspected} farms have been newly inspected for GACC compliance. These will be added to the registry upon final approval.")

    # GACC by exporter (if multiple exporters) - using st.markdown and st.dataframe instead of expander
    if CAN["exporter"] in d.columns and len(d[CAN["exporter"]].unique()) > 1:
        st.markdown("---")
        st.markdown("**GACC Approval by Exporter**")
        gacc_by_exporter = d.groupby(CAN["exporter"])[CAN["gacc"]].apply(
            lambda x: (x.fillna(False).astype(bool).sum(), len(x))
        ).reset_index()
        
        gacc_by_exporter["Approved"] = gacc_by_exporter[CAN["gacc"]].apply(lambda x: x[0])
        gacc_by_exporter["Total"] = gacc_by_exporter[CAN["gacc"]].apply(lambda x: x[1])
        gacc_by_exporter["Percentage"] = (gacc_by_exporter["Approved"] / gacc_by_exporter["Total"] * 100).round(1)
        gacc_by_exporter = gacc_by_exporter.drop(columns=[CAN["gacc"]])
        gacc_by_exporter.columns = ["Exporter", "GACC Approved", "Total Farms", "Approval Rate (%)"]
        
        st.dataframe(gacc_by_exporter, use_container_width=True)


def show_organic_alignment(df: pd.DataFrame):
    """NEW: Show organic users vs organic certified alignment."""
    st.subheader("🌱 Organic Practices Alignment")
    d = _ensure_derived(df)

    col1, col2 = st.columns(2)

    with col1:
        if CAN["fertilizer_organic"] in d.columns:
            organic_users = int(_safe_bool(d, CAN["fertilizer_organic"]).sum())
            st.metric("Farmers Using Organic Fertilizer", f"{organic_users} ({organic_users/len(d)*100:.0f}%)")
        else:
            st.caption("Organic fertilizer data not available")

    with col2:
        if CAN["cert_organic"] in d.columns:
            organic_certified = int(_safe_bool(d, CAN["cert_organic"]).sum())
            st.metric("Organic Certified Farms", f"{organic_certified} ({organic_certified/len(d)*100:.0f}%)")
        else:
            st.caption("Organic certification data not available")

    # Alignment analysis
    if CAN["organic_alignment"] in d.columns:
        misaligned = int(d[CAN["organic_alignment"]].sum())
        st.markdown("---")
        st.markdown("**Alignment Analysis**")
        st.metric("⚖️ Organic Users WITHOUT Organic Certification", f"{misaligned} farms")
        if misaligned > 0:
            st.info(f"{misaligned} farms use organic fertilizer but are not organically certified. This represents an opportunity for certification support.")


def show_grade_completeness(df: pd.DataFrame):
    """NEW: Show Grade 1 vs Grade 2 completeness."""
    st.subheader("📦 Grade Distribution Analysis")
    d = _ensure_derived(df)

    col1, col2 = st.columns(2)

    with col1:
        if "grade1_share_last" in d.columns:
            grade1 = _safe_num(d, "grade1_share_last").dropna()
            # Handle decimal vs percentage format
            if not grade1.empty and grade1.max() <= 1:
                grade1 = grade1 * 100
            if not grade1.empty:
                st.metric("Avg. Grade 1 Share (Last Season)", f"{grade1.mean():.1f}%")
            else:
                st.caption("Grade 1 data not available")
        else:
            st.caption("Grade 1 data not available")

    with col2:
        if "grade2_share_last" in d.columns:
            grade2 = _safe_num(d, "grade2_share_last").dropna()
            # Handle decimal vs percentage format
            if not grade2.empty and grade2.max() <= 1:
                grade2 = grade2 * 100
            if not grade2.empty:
                st.metric("Avg. Grade 2 / Own Use Share", f"{grade2.mean():.1f}%")
            else:
                st.caption("Grade 2 data not available")
        else:
            st.caption("Grade 2 data not available")

    # Completeness validation
    if "grade1_share_last" in d.columns and "grade2_share_last" in d.columns:
        grade1_vals = _safe_num(d, "grade1_share_last")
        grade2_vals = _safe_num(d, "grade2_share_last")
        
        # Handle decimal vs percentage format
        if not grade1_vals.empty and grade1_vals.max() <= 1:
            grade1_vals = grade1_vals * 100
        if not grade2_vals.empty and grade2_vals.max() <= 1:
            grade2_vals = grade2_vals * 100
            
        completeness = grade1_vals + grade2_vals
        completeness = completeness.dropna()
        
        if not completeness.empty:
            valid = completeness[completeness.between(95, 105)]
            st.markdown("---")
            st.metric("✅ Grade Share Completeness", f"{len(valid)}/{len(completeness)} farms (Grade1+Grade2 ≈ 100%)")
            
            # Distribution of completeness
            comp_data = pd.DataFrame({"Total Percentage": completeness})
            chart = alt.Chart(comp_data).mark_bar().encode(
                x=alt.X("Total Percentage:Q", bin=alt.Bin(maxbins=30)),
                y="count()",
                tooltip=["count()"]
            ).properties(title="Grade Completeness Distribution", height=300)
            st.altair_chart(chart, use_container_width=True)
            
            if len(completeness) - len(valid) > 0:
                st.warning(f"{len(completeness) - len(valid)} farms have grade shares that don't sum to 100%")
    else:
        st.info("Grade completeness data not available (requires both Grade 1 and Grade 2 data)")


# ==========================================================
# Executive auto-insights (enhanced)
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

    # Education and Experience
    if CAN["education"] in d.columns:
        edu_coverage = d[CAN["education"]].notna().sum()
        if edu_coverage > 0:
            insights.append(f"Education data available for {edu_coverage}/{n} farms.")
        else:
            risks.append("Education data missing - check column mapping for '1.13 Formal education level'")
            actions.append("Verify column mapping for education field.")
    
    if CAN["experience"] in d.columns:
        exp_coverage = d[CAN["experience"]].notna().sum()
        if exp_coverage > 0:
            avg_exp = d[CAN["experience"]].mean()
            insights.append(f"Average farming experience: {avg_exp:.1f} years.")
        else:
            risks.append("Experience data missing - check column mapping for '1.14 Experience in Avocado farming in years'")
            actions.append("Verify column mapping for experience field.")

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

    # IPM adoption insight
    if CAN["ipm_use"] in d.columns:
        ipm_adopters = int(_safe_bool(d, CAN["ipm_use"]).sum())
        if ipm_adopters > 0:
            insights.append(f"IPM implemented on {ipm_adopters}/{n} farms ({ipm_adopters/n*100:.0f}%).")

    # GACC progress insight
    if CAN["gacc"] in d.columns:
        gacc_count = int(_safe_bool(d, CAN["gacc"]).sum())
        insights.append(f"GACC approved farms: {gacc_count} (target: 270 nationally, {gacc_count/270*100:.1f}% of national total).")

    # AEZ insights
    if CAN["aez_zone"] in d.columns:
        top_aez = d[CAN["aez_zone"]].value_counts().index[0] if not d[CAN["aez_zone"]].empty else None
        if top_aez:
            insights.append(f"Most common Agro-Ecological Zone: {top_aez}.")

    if not actions:
        actions.append("Continue periodic data quality checks and expand coverage for income, age portfolio and compliance fields.")

    return {
        "what_it_means": " ".join(insights) if insights else "The dashboard is ready; insights will strengthen as coverage improves.",
        "risks_gaps": list(dict.fromkeys(risks))[:6],
        "recommended_actions": list(dict.fromkeys(actions))[:6],
    }


# ==========================================================
# Dashboard sections (preserved and enhanced)
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
        d = _ensure_derived(df)
        st.markdown("**Loss Analysis**")
        
        # FIXED: Use the correct CAN key "harvest_losses"
        if CAN["harvest_losses"] in d.columns and CAN["harvest_kg"] in d.columns:
            losses = _safe_num(d, CAN["harvest_losses"]).sum()
            harvest = _safe_num(d, CAN["harvest_kg"]).sum()
            if harvest > 0:
                loss_pct = (losses / harvest) * 100
                st.metric("Total Loss Rate", f"{loss_pct:.1f}%")
        else:
            st.caption("Loss data not available")
        
        if CAN["loss_causes"] in d.columns:
            causes = _safe_str(d, CAN["loss_causes"])
            causes = causes[causes != ""]
            if not causes.empty:
                cause_counts = causes.value_counts().reset_index()
                cause_counts.columns = ["Cause", "Count"]
                chart = alt.Chart(cause_counts).mark_bar().encode(
                    x="Count:Q", y="Cause:N", color="Cause:N"
                ).properties(title="Primary Causes of Loss", height=300)
                st.altair_chart(chart, use_container_width=True)
        else:
            st.caption("Loss causes data not available")

    with tab4:
        d = _ensure_derived(df)
        st.markdown("**📈 Productivity Analysis**")
        
        # Hass variety adoption
        if "variety_hass" in d.columns:
            hass_count = int(_safe_bool(d, "variety_hass").sum())
            st.metric("Hass Variety Farmers", f"{hass_count}/{len(d)} ({hass_count/len(d)*100:.0f}%)")
        
        # Total harvest
        if CAN["harvest_kg"] in d.columns:
            total_harvest = _safe_num(d, CAN["harvest_kg"]).sum()
            avg_harvest = _safe_num(d, CAN["harvest_kg"]).mean()
            st.metric("Total Harvest (Last Season)", f"{total_harvest:,.0f} kg")
            st.metric("Avg Harvest per Farm", f"{avg_harvest:,.0f} kg")
        
        # Yield per tree by age group
        col1, col2, col3 = st.columns(3)
        with col1:
            if CAN["fruits_0_3"] in d.columns:
                fruits_0_3 = _safe_num(d, CAN["fruits_0_3"]).mean()
                st.metric("Fruits/Tree (0-3 years)", f"{fruits_0_3:.0f}" if not pd.isna(fruits_0_3) else "N/A")
        with col2:
            if CAN["fruits_4_7"] in d.columns:
                fruits_4_7 = _safe_num(d, CAN["fruits_4_7"]).mean()
                st.metric("Fruits/Tree (4-7 years)", f"{fruits_4_7:.0f}" if not pd.isna(fruits_4_7) else "N/A")
        with col3:
            if CAN["fruits_8_plus"] in d.columns:
                fruits_8_plus = _safe_num(d, CAN["fruits_8_plus"]).mean()
                st.metric("Fruits/Tree (8+ years)", f"{fruits_8_plus:.0f}" if not pd.isna(fruits_8_plus) else "N/A")


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
        
        if "grade1_share_last" in d.columns:
            grade_share = _safe_num(d, "grade1_share_last").dropna()
            if not grade_share.empty:
                # Check the range to determine format
                max_val = grade_share.max()
                min_val = grade_share.min()
                
                # If values are between 0 and 1, it's decimal format
                if max_val <= 1:
                    grade_share = grade_share * 100
                    st.caption("ℹ️ Values converted from decimal (0-1) to percentage")
                # If values are > 100, they might be already in percentage but with outliers
                elif max_val > 100:
                    st.caption("⚠️ Some values exceed 100% - possible data quality issues")
                
                # Remove extreme outliers for better display
                if len(grade_share) > 10:
                    q99 = grade_share.quantile(0.99)
                    grade_share = grade_share[grade_share <= q99]
                
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
        else:
            st.caption("Grade 1 share data not available")
    
    st.markdown("---")
    
    # Price comparison: last season vs current season
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**💰 Hass Prices (Last Season)**")
        if CAN["price_hass"] in d.columns:
            prices = _safe_num(d, CAN["price_hass"]).dropna()
            # Remove extreme outliers for better display
            if len(prices) > 10:
                q99 = prices.quantile(0.99)
                prices = prices[prices <= q99]
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
            # Remove extreme outliers for better display
            if len(current_prices) > 10:
                q99 = current_prices.quantile(0.99)
                current_prices = current_prices[current_prices <= q99]
            if not current_prices.empty:
                st.metric("Average Price", f"KSh {current_prices.mean():.2f}/kg")
                st.metric("Median Price", f"KSh {current_prices.median():.2f}/kg")
            else:
                st.caption("Current season price data not available")
        else:
            st.caption("Current season price data not available")
    
    # Market constraints expandable
    if CAN["market_constraints"] in d.columns:
        with st.expander("🚧 Market Access Challenges"):
            constraints = _safe_str(d, CAN["market_constraints"])
            constraints = constraints[constraints != ""]
            if not constraints.empty:
                constraint_counts = constraints.value_counts().reset_index()
                constraint_counts.columns = ["Constraint", "Count"]
                chart = alt.Chart(constraint_counts).mark_bar().encode(
                    x="Count:Q", y="Constraint:N", color="Constraint:N"
                ).properties(height=250)
                st.altair_chart(chart, use_container_width=True)


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
    """Sustainability practices section with collapsible details."""
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
                # Show top 5 most common
                st.markdown("**Top 5 Water Sources:**")
                for source, count in water_counts.head(5).items():
                    st.write(f"- {source}: {count} farms")
                
                # Show expander for all sources
                with st.expander(f"View all {len(water_counts)} water sources"):
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
                # Show top 5 most common
                st.markdown("**Top 5 Waste Management Practices:**")
                for method, count in waste_counts.head(5).items():
                    st.write(f"- {method}: {count} farms")
                
                # Show expander for all practices
                with st.expander(f"View all {len(waste_counts)} waste management practices"):
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
                # Show top 5 most common
                st.markdown("**Top 5 Biodiversity Practices:**")
                for practice, count in bio_counts.head(5).items():
                    st.write(f"- {practice}: {count} farms")
                
                # Show expander for all practices
                with st.expander(f"View all {len(bio_counts)} biodiversity practices"):
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
                # Show top 5 most common
                st.markdown("**Top 5 Tree Species:**")
                for tree, count in tree_counts.head(5).items():
                    st.write(f"- {tree}: {count} farms")
                
                # Show expander for all tree species
                with st.expander(f"View all {len(tree_counts)} tree species"):
                    for tree, count in tree_counts.items():
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
            # Show top 5 most common
            st.markdown("**Top 5 Value Chains:**")
            for chain, count in chain_counts.head(5).items():
                st.write(f"- {chain}: {count} farms")
            
            # Show expander for all value chains
            with st.expander(f"View all {len(chain_counts)} value chains"):
                for chain, count in chain_counts.items():
                    st.write(f"- {chain}: {count} farms")
        else:
            st.caption("Other value chains data not available")
    else:
        st.caption("Other value chains data not available")


# ==========================================================
# Investor-grade sections (preserved)
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