# app.py
import streamlit as st
from core.auth import require_auth, logout_button
from core.ui import set_global_style
from core.data import set_uploaded_workbook_bytes_in_session

st.set_page_config(page_title="SHAPe Avocado Dashboard", page_icon="🥑", layout="wide")
set_global_style()


def main():
    user = require_auth()
    logout_button()

    st.title("🥑 SHAPe Avocado Program")

    st.markdown(
        """
Welcome. Use the left sidebar to navigate pages.

### ✅ How you update data (enterprise workflow)
1) Go to **Monitoring Dashboard**
2) Upload the **raw survey export** workbook (often 1 big sheet with many columns)  
   *(or upload an already-processed workbook if you have one)*
3) The app automatically:
   - standardizes columns into a **canonical Monitoring schema**
   - generates derived sheets (Metrics / Certifications / Training / Market summary)
   - enables stable dashboards without manual restructuring
4) You can download a regenerated **shape_data.xlsx** directly from the app

### Pages
- Monitoring Dashboard
- Analysis
- Admin (admin)
        """
    )

    with st.sidebar:
        st.markdown("## Optional: Upload once for all pages")
        uploaded = st.file_uploader(
            "Upload workbook (.xlsx)",
            type=["xlsx", "xls"],
            help="If you upload here, Monitoring/Analysis/Admin pages can reuse the same upload via session state.",
        )
        if uploaded is not None:
            set_uploaded_workbook_bytes_in_session(uploaded.getvalue())
            st.success("Workbook stored for this session. Open Monitoring Dashboard to proceed.")


if __name__ == "__main__":
    main()
