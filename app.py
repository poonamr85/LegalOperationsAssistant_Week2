import streamlit as st

st.set_page_config(page_title="ExampleCo | Legal workspace", page_icon=":material/balance:", layout="wide")
st.session_state.setdefault("messages", [])

with st.sidebar:
    st.title(":material/balance: ExampleCo")
    st.caption("LEGAL OPERATIONS")

page = st.navigation([
    st.Page("legal_assistant/app_pages/home.py", title="Overview", icon=":material/grid_view:", default=True),
    st.Page("legal_assistant/app_pages/assistant.py", title="Legal assistant", icon=":material/chat:", url_path="assistant"),
], position="sidebar")

with st.sidebar:
    with st.expander("About this workspace", icon=":material/info:"):
        st.caption("A sample employee portal for finding legal policies and procedures. All documents are fictional.")
        st.caption("The assistant explains documented guidance. Decisions and approvals remain with your Legal team.")
    st.badge("Demo workspace", icon=":material/science:", color="orange")
    st.caption("Use fictional data only.")

with st.container(horizontal=True, vertical_alignment="center"):
    st.caption("EMPLOYEE SERVICES  /  LEGAL")
    st.badge("Fictional ExampleCo policies", color="gray")
page.run()
