import json
import streamlit as st
from legal_assistant.knowledge_base import ROOT
from legal_assistant.workflow import run_workflow
from legal_assistant.bot import LEGAL_CONTACT_MESSAGE, handoff_packet, pipeline_error_message

st.set_page_config(layout="centered")
heading, actions = st.columns([2, 1.5], vertical_alignment="center")
with heading:
    st.title("Legal assistant", icon=":material/balance:")
with actions.container(horizontal=True, vertical_alignment="center"):
    back_slot = st.empty()
    if st.button("Clear conversation", icon=":material/delete_sweep:",
                 help="Clear the chat messages in this session.", width="stretch"):
        st.session_state.messages = []
        st.session_state.pop("handoff", None)
        st.rerun()
st.caption("Ask about documented procedures and policies. Sources appear as citations in the answer.")
with st.sidebar:
    st.subheader("Conversation")
    manifest_path = ROOT / "storage/index.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    pairs = sorted({(d["metadata"]["organization"], d["metadata"]["policy_version"]) for d in manifest["documents"]}) if manifest else []
    selected = st.selectbox("Organization and policy edition", pairs, format_func=lambda p: f"{p[0]} {p[1]}") if pairs else None
    st.caption("Include the relevant details in every question. Earlier turns are displayed for your reference.")


def escalate(reason, passages):
    st.session_state.handoff = handoff_packet(st.session_state.messages, reason, passages)
    return LEGAL_CONTACT_MESSAGE

if not st.session_state.messages:
    with st.container(border=True):
        st.badge("Guidance from sample documents", icon=":material/menu_book:", color="blue")
        st.subheader("What would you like to understand?")
        st.markdown("Ask about a legal procedure or policy. I’ll use the sample documents, cite supporting sources and flag when Legal review is needed.")
        st.caption("For example: What information do I need for a contract review?")
    st.caption("Fictional data only · Answers are procedural guidance, not legal approval.")

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=":material/balance:" if message["role"] == "assistant" else ":material/person:"):
        st.markdown(message["content"])
if "pending_question" in st.session_state:
    st.session_state.legal_question = st.session_state.pop("pending_question")
question = st.chat_input("Ask about a legal policy or procedure…", key="legal_question", submit_mode="disable")
if question:
    # A previous handoff should not remain attached to a new, answered question.
    st.session_state.pop("handoff", None)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(question)
    passages = []
    with st.chat_message("assistant", avatar=":material/balance:"):
        with st.spinner("Checking the knowledge base…"):
            try:
                result = run_workflow(question, *(selected or ("", "")))
                passages = result.get("passages", [])
                if result["action"] in {"answer", "clarify", "index_needed"}:
                    text = result["answer"]
                else:
                    text = escalate(result["escalation_reason"], passages)
                if result.get("error_message"):
                    st.warning(result["error_message"])
            except Exception as exc:
                text = escalate(f"Pipeline unavailable: {type(exc).__name__}", passages)
                st.warning(pipeline_error_message(exc, "workflow"))
        st.markdown(text)
        st.session_state.messages.append({"role": "assistant", "content": text})

# Populate the header action after generation, so it appears with the first response.
if any(message["role"] == "assistant" for message in st.session_state.messages):
    with back_slot:
        if st.button("Back", icon=":material/arrow_back:", key="back_to_overview",
                     help="Return to Overview without clearing this conversation.", width="stretch"):
            st.switch_page("legal_assistant/app_pages/home.py")
