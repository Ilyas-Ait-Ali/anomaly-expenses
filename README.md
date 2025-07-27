# 💸 Anomaly Expenses

**Anomaly Expenses** is a machine learning project designed to detect unusual or potentially fraudulent expense transactions using real-world credit card data.

This project was built as part of my effort to deepen my understanding of ML workflows. It demonstrates the full pipeline. from raw data to usable fraud detection model, using common industry tools and clean coding practices.

The pipeline includes:
- Data loading and preprocessing
- Exploratory data analysis (EDA)
- Fraud detection using machine learning models
- A web app for interactive prediction

> 🚧 Project in progress

---

## 🔍 Dataset

- **Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- To run the notebooks, download the dataset and place the `creditcard.csv` file inside the `data/` folder.

---

## 🧪 Notebooks Overview

| Notebook | Description |
|----------|-------------|
| `01_data_cleaning.ipynb` | Loads and cleans raw data (deduplication, label conversion) |
| `02_exploratory_analysis.ipynb` | Visualizes feature distributions, fraud patterns, and correlations |
| `03_modeling_logistic_regression.ipynb` | Trains logistic regression with class weighting and evaluates performance |
| `04_modeling_random_forest.ipynb` | Trains a Random Forest classifier with better precision-recall balance |

---

## 📊 Model Comparison

| Model               | Precision | Recall | F1 Score | Notes |
|---------------------|-----------|--------|----------|-------|
| **Logistic Regression** | 0.06      | **0.87**   | 0.11     | High recall, too many false positives |
| **Random Forest**        | **0.97**  | 0.71   | **0.82** | Balanced, reliable — suitable for deployment ✅ |


---

## 🖥️ Streamlit App

An interactive frontend allows users to upload a CSV file and detect fraudulent transactions instantly.

### ▶️ Run locally

```bash
streamlit run streamlit_app.py
```

📤 Upload format
Must contain exactly these columns:
Time, Amount, and V1 to V28

Must not include the label column is_fraud (it will be dropped automatically if present)

Use this pre-validated **sample_input.csv** to test instantly.

---

🔍 Output
Summary of total vs. fraudulent predictions

A downloadable table of transactions flagged as fraud

Graceful error messages for format issues or upload problems

⚠️ Note: This model was trained on anonymized PCA features and should not be used in production without proper retraining and calibration.

---

## 📄 License

This project is released under the [MIT License](LICENSE).
