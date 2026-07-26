import streamlit as st
import pandas as pd
import os
import sys

# Hardcoded API key as requested for deployment
os.environ["GOOGLE_API_KEY"] = "AQ.Ab8RN6Kqo2zc-H7NPo5iEresc1DkS8_x7twksd7JtOcIbWw-fw"

# Ensure tools and agent directories are in import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'tools')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'agent')))

from test_agent import execute_agent_loop

# --------------------------------------------------
# Streamlit Page Setup
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation & Personalization Agent",
    page_icon="🏦",
    layout="wide"
)

# Dark theme styling override
st.markdown("""
<style>
    .reportview-container {
        background: #111;
    }
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    h1 {
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)



st.title("🏦 Personalization & Segmentation AI Agent")
st.write("Ask the agent about customer segments, segment selection criteria, and upgrade offers.")

# --------------------------------------------------
# Data Loading (Cached for Performance)
# --------------------------------------------------
@st.cache_data
def load_data():
    features_path = "data/customer_features.csv"
    if not os.path.exists(features_path):
        return None
    return pd.read_csv(features_path)

df_feats = load_data()

if df_feats is None:
    st.error("❌ Dataset features file not found under `data/customer_features.csv`. Please verify the dataset download and run test_tools.py.")
    st.stop()

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/bank.png", width=80)
    st.title("Banking Agent Hub")
    st.markdown("---")
    st.markdown("### Problem Statement 2")
    st.info("**Customer Segmentation & Personalization Agent**")
    
    st.markdown("### 📊 Dataset Overview")
    total_customers = len(df_feats)
    total_transactions = int(df_feats['Frequency'].sum()) if 'Frequency' in df_feats.columns else 1048567
    
    st.metric("Total Customers", f"{total_customers:,}")
    st.metric("Total Transactions", f"{total_transactions:,}")
    
    st.markdown("### 🧠 Models Used")
    st.markdown(
        "- **LLM**: Gemini 2.5 Flash\n"
        "- **Clustering**: KMeans\n"
        "- **Explainability**: Decision Tree\n"
        "- **Embeddings**: N/A (Rules + Clustering)"
    )
    
    st.markdown("---")
    if st.button("🔄 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent_context = {}
        st.session_state.processing = False
        st.rerun()

# --------------------------------------------------
# Chat Interface & Session State
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your retail banking personalization assistant. How can I help you analyze customer segments or draft product campaign offers today?"}
    ]
if "processing" not in st.session_state:
    st.session_state.processing = False
if "agent_context" not in st.session_state:
    # This context dict is passed directly to execute_agent_loop to persist state 
    # like pending_query for clarifications. Streamlit keeps it alive naturally.
    st.session_state.agent_context = {}

# Render all previous chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User query input (disabled while processing)
if st.session_state.processing:
    st.chat_input("Agent is processing, please wait...", disabled=True)
else:
    if prompt := st.chat_input("Enter your query..."):
        # Add message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.processing = True
        st.rerun()

# Execute agent logic (runs after rerun if processing is True)
if st.session_state.processing:
    user_prompt = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking & running tools..."):
            try:
                # execute_agent_loop handles routing, tool logic, synthesis, and state (agent_context)
                response = execute_agent_loop(user_prompt, df_feats, session_context=st.session_state.agent_context)
            except Exception as e:
                response = f"**Agent encountered an error:** {e}"
                
            st.markdown(response)
            
    # Save the agent's response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.processing = False
    st.rerun()
