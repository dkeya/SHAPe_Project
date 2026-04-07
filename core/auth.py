# core/auth.py
import os
import hmac
import json
import base64
import hashlib
import streamlit as st

# PBKDF2 storage format:
#   pbkdf2_sha256$<iterations>$<salt_b64_urlsafe_no_padding>$<hash_b64_urlsafe_no_padding>
PBKDF2_ITERATIONS_DEFAULT = 260_000

def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("utf-8").rstrip("=")

def _b64d(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("utf-8"))

def hash_password(password: str, *, iterations: int = PBKDF2_ITERATIONS_DEFAULT, salt: bytes | None = None) -> str:
    if not password:
        raise ValueError("password is required")
    if salt is None:
        salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, dklen=32)
    return f"pbkdf2_sha256${iterations}${_b64e(salt)}${_b64e(dk)}"

def verify_password(password: str, stored: str) -> bool:
    """
    stored format: pbkdf2_sha256$<iterations>$<salt_b64>$<hash_b64>
    salt_b64/hash_b64 are URL-safe Base64 (usually without '=' padding).
    """
    try:
        algo, iters, salt_b64, hash_b64 = stored.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        iterations = int(iters)
        salt = _b64d(salt_b64)
        expected = _b64d(hash_b64)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, dklen=len(expected))
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False

def load_users_registry() -> dict:
    """
    Priority:
      1) Streamlit secrets: st.secrets["auth"]["users"]
      2) ENV fallback: SHAPE_AUTH_JSON  (JSON string)
      3) Inline fallback (demo only)
    """
    try:
        if "auth" in st.secrets and "users" in st.secrets["auth"]:
            return dict(st.secrets["auth"]["users"])
    except Exception:
        pass

    auth_json = os.getenv("SHAPE_AUTH_JSON", "").strip()
    if auth_json:
        try:
            return json.loads(auth_json)
        except Exception:
            pass

    # Demo fallback (ONLY if secrets/env missing)
    return {
        "admin": {
            "name": "Admin",
            "role": "admin",
            "exporters": ["*"],
            "password_hash": hash_password("admin123*"),
        }
    }

def require_auth():
    """Require authentication - login form shown in main area, not sidebar."""
    # Check if already authenticated
    if st.session_state.get("auth_user"):
        return st.session_state["auth_user"]

    users = load_users_registry()

    # Show login form in the main area (not sidebar) to avoid collapse issues
    st.markdown("## 🔐 Please sign in to continue")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Login")
        username = st.text_input("Username", key="auth_username_main")
        password = st.text_input("Password", type="password", key="auth_password_main")
        
        col_a, col_b = st.columns(2)
        do_login = col_a.button("Login", type="primary", use_container_width=True)
        do_clear = col_b.button("Clear", use_container_width=True)
        
        if do_clear:
            st.session_state.pop("auth_username_main", None)
            st.session_state.pop("auth_password_main", None)
            st.rerun()
        
        if do_login:
            if not username or not password:
                st.error("Please enter both username and password.")
                st.stop()
            
            u = users.get(username)
            if (not u) or (not verify_password(password, u.get("password_hash", ""))):
                st.error("Invalid username or password.")
                st.stop()
            
            user = {
                "username": username,
                "name": u.get("name", username),
                "role": u.get("role", "exporter"),
                "exporters": u.get("exporters", []),
            }
            st.session_state["auth_user"] = user
            st.session_state.pop("auth_username_main", None)
            st.session_state.pop("auth_password_main", None)
            st.rerun()
    
    with col2:
        st.info("### Demo Credentials\n\n**Username:** admin\n**Password:** admin123*")
    
    st.stop()

def logout_button():
    """Logout button in sidebar."""
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.pop("auth_user", None)
        st.rerun()

def normalize_exporter(x):
    if x is None:
        return None
    return str(x).strip()

def get_allowed_exporters(user: dict, df, exporter_col: str):
    if user.get("role") == "admin" and "*" in user.get("exporters", []):
        if exporter_col in df.columns:
            vals = [normalize_exporter(v) for v in df[exporter_col].dropna().unique()]
            return sorted([v for v in vals if v])
        return []
    vals = [normalize_exporter(v) for v in user.get("exporters", [])]
    return sorted([v for v in vals if v])

def enforce_exporter_scope(df, user: dict, selected: str, exporter_col: str):
    """
    Returns (scoped_df, effective_selected, exporter_options)
    """
    if df.empty or exporter_col not in df.columns:
        return df, selected, ["All"] if user.get("role") == "admin" else []

    allowed = get_allowed_exporters(user, df, exporter_col)

    if user.get("role") == "admin":
        options = ["All"] + allowed
        if selected == "All":
            return df, "All", options
        if selected in allowed:
            return df[df[exporter_col] == selected], selected, options
        return df, "All", options

    if not allowed:
        st.error("Your account has no exporter access assigned. Contact the admin.")
        st.stop()

    options = allowed  # no "All"
    if (selected == "All") or (selected not in allowed):
        selected = allowed[0]

    return df[df[exporter_col] == selected], selected, options

def permissions(user: dict):
    role = (user or {}).get("role", "exporter")

    return {
        # ✅ Analysis should be accessible to both admin + exporters
        "can_view_analysis": True,

        # ✅ Raw data / downloads should remain admin-only (enterprise governance)
        "can_view_raw_data": role == "admin",

        # ✅ Admin page remains admin-only
        "can_view_admin": role == "admin",
    }