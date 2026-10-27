# Job Description Classification

A machine learning project that reads the text of a job description and predicts its category
(Data Science, Software Development, Marketing, Human Resources, Finance, Sales or Design).
It includes data exploration, statistical analysis, five classification models, and a
**Streamlit** web app for live predictions.

**Best model:** Linear SVM, 98.57% test accuracy (weighted F1 = 0.9857, 5-fold CV accuracy = 97.86%).

## Features
- Text cleaning and TF-IDF feature extraction (unigrams + bigrams, 3,048 features)
- Exploratory data analysis (class balance, word counts, salary, correlation)
- Statistical analysis: sampling distributions, bootstrap confidence intervals,
  one-sample, Welch and paired t-tests, chi-square test
- Five models compared: Logistic Regression, Naive Bayes, Linear SVM, Decision Tree, Random Forest
- Evaluation with accuracy, precision, recall, F1-score, confusion matrix and cross-validation
- Trained model saved with `joblib` and used in a Streamlit app

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Linear SVM** | **0.9857** | **0.9860** | **0.9857** | **0.9857** |
| Logistic Regression | 0.9571 | 0.9574 | 0.9571 | 0.9572 |
| Naive Bayes | 0.9571 | 0.9574 | 0.9571 | 0.9572 |
| Random Forest | 0.9262 | 0.9273 | 0.9262 | 0.9264 |
| Decision Tree | 0.7833 | 0.7896 | 0.7833 | 0.7838 |

Evaluated on 420 test descriptions (80:20 stratified split, `random_state=42`).

## Dataset
`job_descriptions.csv` contains 2,100 rows (300 per category) with the columns
`job_description`, `category`, `experience_years` and `salary_lpa`.

> **Note:** the dataset is **synthetic**, generated with Python code for educational use.
> Scores on real job postings are expected to be lower.

## Project Structure
```
├── B22-18-Saloni-SI-Report-Job.ipynb   # full notebook (EDA, statistics, models)
├── app.py                              # Streamlit web app
├── train_model.py                      # creates the .pkl model files from the CSV
├── job_descriptions.csv                # dataset
├── requirements.txt                    # Python libraries
└── README.md
```

## Installation
```bash
git clone https://github.com/saloni-salem/-Job-Description-Classification.git
cd -Job-Description-Classification
pip install -r requirements.txt
```

## Usage

**1. Create the model files** (`best_job_classifier.pkl` and `tfidf_vectorizer.pkl`):
```bash
python train_model.py
```
(Running the notebook up to the "Save selected model" cell creates the same files.)

**2. Start the Streamlit app:**
```bash
streamlit run app.py
```
The app opens at `http://localhost:8501`. Paste a job description and click **Predict**.

**Example input:**
```
We are hiring a Data Scientist with 3+ years of experience. Responsibilities: build predictive
models, analyze large datasets. Required skills: python, statistics, sql, tensorflow.
```
Expected output: `Predicted job category: Data Science`

## Tech Stack
Python, Pandas, NumPy, SciPy, scikit-learn, Matplotlib, Seaborn, Joblib, Streamlit

## Limitations and Future Work
- The synthetic dataset is cleaner than real postings; test on a real job-postings dataset.
- Only 7 English-language categories and one label per description.
- Possible improvements: more categories, multi-label output, BERT or word embeddings,
  hyperparameter tuning, and probability scores in the app.

## Author
Saloni Salem (Roll No. B22-18), B.Tech CSE (AI & Data Science), Ramdeobaba University, Nagpur
