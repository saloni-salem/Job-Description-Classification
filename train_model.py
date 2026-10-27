"""Creates best_job_classifier.pkl and tfidf_vectorizer.pkl from job_descriptions.csv.
Run:  python train_model.py   (in the same folder as job_descriptions.csv)
"""
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # remove urls
    text = re.sub(r"[^a-z0-9+#\s]", " ", text)      # keep letters, digits, + and #
    return re.sub(r"\s+", " ", text).strip()


df = pd.read_csv("job_descriptions.csv")
df = df.drop_duplicates().dropna(subset=["job_description", "category"]).reset_index(drop=True)
df["clean_description"] = df["job_description"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_description"], df["category"],
    test_size=0.2, random_state=42, stratify=df["category"])

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                        stop_words="english", sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LinearSVC()
model.fit(X_train_tfidf, y_train)
print("Test accuracy:", round(accuracy_score(y_test, model.predict(X_test_tfidf)), 4))

joblib.dump(model, "best_job_classifier.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
print("Saved: best_job_classifier.pkl and tfidf_vectorizer.pkl")
