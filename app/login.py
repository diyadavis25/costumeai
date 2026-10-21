"""Login / sign-up screen. app.py calls require_login() before showing anything else."""
import streamlit as st

from src import auth


def _hide_sidebar():
    st.markdown("<style>[data-testid='stSidebar'],[data-testid='collapsedControl'],"
                "[data-testid='stSidebarCollapsedControl']{display:none}</style>", unsafe_allow_html=True)


def require_login() -> bool:
    """Show the login screen unless signed in. Returns True when the user may continue."""
    if st.session_state.get("user"):
        return True
    _hide_sidebar()
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown("## 👗 Outfit AI")
        st.caption("Smart mirror for outfit compatibility. Log in to continue.")
        tab_in, tab_up = st.tabs(["Log in", "Create account"])

        with tab_in:
            with st.form("login_form"):
                u = st.text_input("Username", key="li_user")
                p = st.text_input("Password", type="password", key="li_pass")
                go = st.form_submit_button("Log in", type="primary", use_container_width=True)
            if go:
                ok, msg, name = auth.login(u, p)
                if ok:
                    st.session_state.user = name
                    st.rerun()
                st.error(msg)

        with tab_up:
            with st.form("signup_form"):
                nu = st.text_input("Choose a username", key="su_user", help="3-20 letters, numbers or _, starting with a letter")
                np1 = st.text_input("Choose a password", type="password", key="su_p1",
                                    help="8+ characters with upper and lower case, a number and a symbol")
                np2 = st.text_input("Confirm password", type="password", key="su_p2")
                create = st.form_submit_button("Create account", type="primary", use_container_width=True)
            if create:
                ok, msg = auth.register(nu, np1, np2)
                if ok:
                    st.success(msg + " Switch to the Log in tab.")
                else:
                    st.error(msg)
                if np1:
                    st.caption(f"Password strength: {auth.password_strength(np1)}")
    return False


def logout_button():
    st.caption(f"Signed in as **{st.session_state.get('user')}**")
    if st.button("Log out", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()