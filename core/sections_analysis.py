# core/sections_analysis.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

class ShapeAdvancedAnalytics:
    def __init__(self, df):
        self.df = df.copy()

    def analyze_yield_drivers(self):
        ycol = '4.2 Total Harvest Last Season (kg)'
        if self.df.empty or ycol not in self.df.columns:
            return None
        features = [c for c in ['2.1 Total Farm Size (Acres)', '2.3 Number of Avocado Trees Planted', '1.14 Experience in Avocado farming in years'] if c in self.df.columns]
        if len(features) < 2:
            return None
        X = self.df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
        y = pd.to_numeric(self.df[ycol], errors='coerce').fillna(0)
        if len(X) < 20:
            return None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        imp = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
        return {"r2": r2, "importance": imp}

    def predict_certification_success(self):
        target = '1.26 General Administration of Customs of the Peoples Republic of China (GACC ) Approval Status'
        if self.df.empty or target not in self.df.columns:
            return None
        features = [c for c in ['2.1 Total Farm Size (Acres)', '2.3 Number of Avocado Trees Planted', '1.14 Experience in Avocado farming in years'] if c in self.df.columns]
        if len(features) < 2:
            return None
        X = self.df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
        y = self.df[target].map({'Yes': 1, 'No': 0, 'Y': 1, 'N': 0}).fillna(0)
        if len(X) < 20:
            return None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        imp = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
        proba = model.predict_proba(X_test)[:, 1]
        return {"accuracy": acc, "importance": imp, "proba": proba}

def show_analysis_page(df):
    st.header("Analysis")
    if df.empty:
        st.warning("No data available.")
        return

    analytics = ShapeAdvancedAnalytics(df)
    tab1, tab2 = st.tabs(["Yield Drivers", "Certification Prediction"])

    with tab1:
        res = analytics.analyze_yield_drivers()
        if not res:
            st.info("Not enough data/columns for yield driver modeling.")
        else:
            st.metric("Yield model R²", f"{res['r2']:.3f}")
            st.plotly_chart(px.bar(res['importance'].head(12), x="Importance", y="Feature", orientation="h"), use_container_width=True)

    with tab2:
        res = analytics.predict_certification_success()
        if not res:
            st.info("Not enough data/columns for certification modeling.")
        else:
            st.metric("Certification model accuracy", f"{res['accuracy']:.3f}")
            st.plotly_chart(px.bar(res['importance'].head(12), x="Importance", y="Feature", orientation="h"), use_container_width=True)
            st.plotly_chart(px.histogram(res["proba"], nbins=20, title="Predicted success probability"), use_container_width=True)
