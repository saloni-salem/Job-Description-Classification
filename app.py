import re
import joblib
import streamlit as st

st.set_page_config(page_title="Job Description Classifier", page_icon="💼")


@st.cache_resource
def load_artifacts():
    """Load the saved model and TF-IDF vectorizer once."""
    model = joblib.load("best_job_classifier.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return model, tfidf


def clean_text(text):
    """Same cleaning function that was used during training."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # remove urls
    text = re.sub(r"[^a-z0-9+#\s]", " ", text)      # keep letters, digits, + and #
    text = re.sub(r"\s+", " ", text).strip()
    return text


model, tfidf = load_artifacts()

st.title("Job Description Classifier")
st.write("Educational demo: predicts the job category from the description text.")

job_text = st.text_area("Paste a job description", height=220)

if st.button("Predict", type="primary"):
    if not job_text.strip():
        st.warning("Please paste a job description first.")
    else:
        vec = tfidf.transform([clean_text(job_text)])
        label = model.predict(vec)[0]
        st.success(f"Predicted job category: {label}")
