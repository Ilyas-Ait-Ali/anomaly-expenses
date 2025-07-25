# 💸 Anomaly Expenses

Anomaly Expenses is a machine learning project designed to detect unusual or potentially fraudulent expense transactions using real-world credit card data.

The goal is to build an end-to-end pipeline that:
- Loads and cleans transaction data
- Detects anomalies using machine learning
- Visualizes insights and alerts through a dashboard

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

## 📄 License

This project is released under the [MIT License](LICENSE).
