import os
import requests
import streamlit as st

# Configure the Streamlit page
st.set_page_config(
    page_title="Financial Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
)

# API base URL: defaults to local server, can be overridden by environment variable
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")

# Color mapping for sentiment badges
COLOR_MAP = {
    "positive": "🟢",
    "neutral": "⚪",
    "negative": "🔴",
}


def check_backend_health():
    """Checks if the FastAPI backend is running and model weights are loaded."""
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if res.status_code == 200:
            return res.json().get("model_loaded", False)
        return False
    except requests.exceptions.RequestException:
        return False


# --- UI Header ---
st.title("📈 Financial Sentiment Intelligence")
st.markdown(
    "Transfer Learning classifier fine-tuned on financial news & earnings statements."
)

# Sidebar with backend status and controls
with st.sidebar:
    st.header("Service Health")
    is_healthy = check_backend_health()
    if is_healthy:
        st.success("Backend API: Online & Model Ready")
    else:
        st.error("Backend API: Offline / Model Loading")
        st.caption("Ensure your FastAPI server is running at port 8000.")

    st.divider()
    st.markdown("### Model Details")
    st.write("**Backbone:** `distilbert-base-uncased`")
    st.write("**Dataset:** `financial_phrasebank (allagree)`")
    st.write("**Classes:** Negative, Neutral, Positive")

# Main Interface Tabs
tab_single, tab_doc = st.tabs(["Headline Analysis", "Full Article / Earnings Release"])

# ==========================================
# Tab 1: Single Headline Analysis
# ==========================================
with tab_single:
    st.subheader("Analyze Single Sentence / Headline")
    default_text = "Operating profit rose 14% to EUR 5.1M in the second quarter."
    sentence_input = st.text_input("Enter financial headline:", value=default_text)

    if st.button("Classify Headline", type="primary", key="btn_single"):
        if not sentence_input.strip():
            st.warning("Please enter a valid sentence.")
        else:
            with st.spinner("Classifying sentiment..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/predict",
                        json={"sentence": sentence_input},
                        timeout=5,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        sentiment = data["sentiment"]
                        confidence = data["confidence"]
                        probs = data["probabilities"]

                        st.markdown("---")
                        col1, col2 = st.columns([1, 2])

                        with col1:
                            st.metric(
                                label="Predicted Sentiment",
                                value=f"{COLOR_MAP.get(sentiment, '')} {sentiment.upper()}",
                                delta=f"{confidence * 100:.1f}% Confidence",
                            )

                        with col2:
                            st.write("**Class Probability Distribution:**")
                            for label, prob in probs.items():
                                st.write(f"- **{label.capitalize()}**: `{prob * 100:.2f}%`")
                                st.progress(prob)
                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to communicate with API server: {e}")

# ==========================================
# Tab 2: Document / Long Text Analysis
# ==========================================
with tab_doc:
    st.subheader("Analyze Long Text (Up to 500+ Words)")
    default_doc = (
        "Operating profit rose by 14% to EUR 5.1M in the second quarter. "
        "However, net sales in Europe dipped 2% due to supply chain headwinds. "
        "Management reaffirmed its full-year guidance and expects strong recovery in Q4."
    )
    doc_input = st.text_area("Paste financial passage or report:", value=default_doc, height=180)

    if st.button("Analyze Document", type="primary", key="btn_doc"):
        if not doc_input.strip():
            st.warning("Please paste some text to analyze.")
        else:
            with st.spinner("Processing document sentences..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/predict-document",
                        json={"text": doc_input},
                        timeout=10,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        overall = data["overall_sentiment"]
                        count = data["sentence_count"]
                        dist = data["document_probabilities"]
                        breakdown = data["sentence_breakdown"]

                        st.markdown("---")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Overall Sentiment", f"{COLOR_MAP.get(overall, '')} {overall.upper()}")
                        c2.metric("Total Sentences", count)
                        c3.metric("Positive Count", dist.get("positive", 0))
                        c4.metric("Negative Count", dist.get("negative", 0))

                        st.markdown("### Sentence Breakdown")
                        for idx, item in enumerate(breakdown, start=1):
                            s_badge = COLOR_MAP.get(item["sentiment"], "")
                            with st.expander(
                                f"{idx}. [{s_badge} {item['sentiment'].upper()}] - {item['sentence'][:60]}..."
                            ):
                                st.write(f"**Full Text:** {item['sentence']}")
                                st.write(f"**Confidence:** `{item['confidence'] * 100:.1f}%`")
                                st.json(item["probabilities"])
                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to communicate with API server: {e}")