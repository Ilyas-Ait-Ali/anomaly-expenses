import streamlit as st
import pandas as pd
import joblib
import os

# === Page Config ===
st.set_page_config(page_title="Anomaly Expenses - Fraud Detector", page_icon="💸")

# === Title ===
st.title("💸 Anomaly Expenses")
st.caption("Detect fraudulent transactions using a trained Random Forest model.")

# === Load Model ===
@st.cache_resource
def load_model():
    model_path = "models/random_forest_fraud.pkl"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please train and save the model first.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# === Upload CSV ===
uploaded_file = st.file_uploader("📤 Upload a CSV file with transaction data", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
        st.stop()

    # === Drop 'is_fraud' if it exists to prevent data leakage ===
    if 'is_fraud' in df.columns:
        df = df.drop(columns=['is_fraud'])

    # === Validate Required Columns ===
    REQUIRED_FEATURES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    missing = [col for col in REQUIRED_FEATURES if col not in df.columns]

    if missing:
        st.error(f"❌ Uploaded file is missing required columns: {missing}")
        st.stop()

    # Ensure correct column order
    df = df[REQUIRED_FEATURES]

    # === Predict ===
    try:
        preds = model.predict(df)
        df['predicted_fraud'] = preds

        # === Show results ===
        fraud_df = df[df['predicted_fraud'] == 1]
        legit_df = df[df['predicted_fraud'] == 0]

        st.subheader("🔍 Prediction Summary")
        st.markdown(f"**Total Transactions:** {len(df)}")
        st.markdown(f"**Fraudulent Predictions:** {len(fraud_df)}")
        st.markdown(f"**Legitimate Predictions:** {len(legit_df)}")

        st.subheader("📌 Flagged Fraudulent Transactions")
        if fraud_df.empty:
            st.info("No frauds detected in this file.")
        else:
            st.dataframe(fraud_df.reset_index(drop=True))

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
