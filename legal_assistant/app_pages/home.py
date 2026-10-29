import streamlit as st

st.title("Legal guidance. A clear next step.")
st.markdown("Your workspace for internal legal questions, policy guidance and review procedures.")

main, aside = st.columns([2.1, 1], gap="large")
with main.container(border=True, height="stretch"):
    st.badge("Legal assistant", icon=":material/auto_awesome:", color="blue")
    st.header("Get guidance before you get started")
    st.markdown("Find what you need to prepare a contract review, onboard a vendor or understand an internal policy. Ask in your own words and get an answer grounded in the sample knowledge base.")
    if st.button("Open Legal assistant", type="primary", icon=":material/arrow_forward:", key="open_assistant"):
        st.switch_page("legal_assistant/app_pages/assistant.py")
    st.caption("Cited guidance · Clear evidence gaps · Human review when needed")
with aside.container(border=True, height="stretch"):
    st.subheader("A useful starting point")
    st.markdown(":material/menu_book: **Policies & procedures**")
    st.caption("Understand documented requirements and prepare the right information.")
    st.markdown(":material/format_quote: **Sources you can reference**")
    st.caption("Answer citations identify supporting passages in the sample documents.")
    st.markdown(":material/person: **Legal decisions stay with people**")
    st.caption("Requests needing interpretation or approval are directed to Legal.")

st.subheader("What do you need help with?")
st.caption("Choose a topic to prepare a question in the assistant. Review it before sending.")
topics = [
    ("Contract review", "description", "Prepare an agreement for Legal review.", "How do I request a contract review?"),
    ("NDAs & confidentiality", "lock", "Understand the process before sharing information.", "Can I send an NDA to a prospective vendor?"),
    ("Privacy & data", "shield", "Find guidance on sharing employee information.", "Can I share employee information with a vendor?"),
    ("Vendor onboarding", "business_center", "Identify the legal steps for a new vendor.", "What legal steps are required to onboard a new vendor?"),
    ("Signing authority", "draw", "Check the documented contract signing process.", "Can I sign a contract just because I requested it?"),
    ("Legal holds", "inventory_2", "Understand document preservation requirements.", "Can I delete records under an active legal hold?"),
]
for start in (0, 3):
    for column, (title, icon, description, question) in zip(st.columns(3, gap="medium"), topics[start:start + 3]):
        with column.container(border=True, height="stretch"):
            st.subheader(title, icon=f":material/{icon}:")
            st.caption(description)
            st.space("stretch")
            if st.button("Ask about this", icon=":material/arrow_forward:", key=f"topic_{start}_{icon}", width="stretch"):
                st.session_state.pending_question = question
                st.switch_page("legal_assistant/app_pages/assistant.py")

with st.expander("How to get a useful answer", icon=":material/help:"):
    st.markdown("**Be specific.** Include the process you are asking about and the relevant fictional details.\n\n**Check the references.** Citations point to supporting passages, but do not independently verify legal accuracy.\n\n**Include context each time.** Earlier messages remain visible but are not used to interpret follow-up questions.")
    st.caption("This demonstration does not submit intake requests or contact Legal on your behalf.")
