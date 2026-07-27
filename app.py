import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Page Configuration
st.set_page_config(page_title="Stroke Risk Predictor", layout="wide")

# Load Trained Model and Saved Feature Columns
model = joblib.load('stroke_model.pkl')
model_columns = joblib.load('model_columns.pkl')
# --- Header Section ---
st.title("Stroke Risk Predictor System")
st.write("Please enter patient details below to evaluate stroke risk.")
# --- Tab Layout Setup ---
tab1, tab2, tab3 =st.tabs(["🩺 Risk Predictor", "📊 Data Analytics", "⚙️ Model Info"])
# --- Tab 1: Input Form ---
with tab1:
    col1 , col2=st.columns(2)
    with col1:
        age=st.slider("Age", min_value=1, max_value=100, value=30)
        gender=st.selectbox("Gender",["Male","Female","Other"])
        hypertension=st.selectbox("Hypertension (High BP)",["Yes","No"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        ever_married = st.selectbox("Ever Married?", ["No", "Yes"])
    with col2:
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        avg_glucose =st.number_input("Average Glucose Level (mg/dL)", min_value=50.0, max_value=300.0, value=100.0)
        bmi = st.number_input("BMI Index", min_value=10.0, max_value=60.0, value=25.0)
        smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])
# Separator Line
    st.markdown("---")  
    # Predict Button
    if st.button("Predict Stroke Risk", type= "primary"):
    # 1. Store Raw User Inputs
        raw_data={
        'age':age,
        'hypertension':1 if hypertension=="Yes" else 0,
        'ever_married': 1.0 if ever_married == "Yes" else 0.0,
        'heart_disease': 1 if heart_disease == "Yes" else 0,
        'Residence_type': 1.0 if residence_type == "Urban" else 0.0,
        'avg_glucose_level': avg_glucose,
        'bmi': bmi,
        'gender': gender,
        'work_type': work_type,
        'smoking_status': smoking_status
        
     }
     # 2. Convert to DataFrame
        input_df=pd.DataFrame([raw_data])
    
        # 3. One-Hot Encoding for multi-class categorical features
        input_uncoded= pd.get_dummies(input_df,columns=['gender', 'work_type', 'smoking_status'], drop_first=True,dtype=int)
     
        # 4. Align with Model Columns (Missing columns set to 0)
        final_input=input_uncoded.reindex(columns=model_columns,fill_value=0)
    
        # 5. Predict Probability
        risk_probability=model.predict_proba(final_input)[0][1]
    
        # 6. Display Result using 0.30 Threshold
        st.subheader("Prediction Analysis Result")
        st.write(f"**Calculated Stroke Risk Probability:**{risk_probability*100:.2f}%")
        if risk_probability >= 0.30:
            st.error("⚠️ **High Risk Alert:** Patient shows a high risk of stroke! Immediate medical advice recommended.")
        else:
            st.success("✅ **Low Risk:** Patient stroke risk is currently within safe parameters.")
# --- Tab 2: Data Analytics ---
with tab2:
    st.header("📊 Stroke Dataset Analytics & Insights")
    st.write("Explore dataset summaries, distributions, and risk patterns.")
    # CSV File Load Karein (Dataset Check)
    try:
        df=pd.read_csv("healthcare-dataset-stroke-data.csv")
        # 1. Top Metrics Summary Cards
        col_m1, col_m2, col_m3, col_m4= st.columns(4)
        col_m1.metric("Total Patients",f"{len(df):,}")
        col_m2.metric("Average Age",f"{df['age'].mean():.1f} yrs")
        stroke_count=df['stroke'].sum()
        stroke_pct=(stroke_count/len(df))*100
        col_m3.metric("Stroke Positive Cases", f"{stroke_count}")
        col_m4.metric("Stroke Prevalence", f"{stroke_pct:.2f}%")
        st.markdown("---")
        # 2. Interactive Dataset Preview
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown("---")
         # 3. Visual Charts (Distribution & Risk Analysis)
        st.subheader("📈 Key Risk Factors Analysis") 
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("**Age Distribution of Patients**")
            st.bar_chart(df["age"].value_counts().sort_index())
        with chart_col2:
            st.markdown("**Stroke Risk by Work Type**")
            work_stroke = df.groupby('work_type')['stroke'].mean() * 100
            st.bar_chart(work_stroke)
        
    except FileNotFoundError:
        st.warning("⚠️ `healthcare-dataset-stroke-data.csv` file not found.")
        
# --- Tab 3: Model Information ---
with tab3:
    st.header("⚙️ Machine Learning Model Details")
    st.write("Overview of the trained binary classification model, features, and decision boundaries.")

    # 1. Model Overview Cards
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Model Architecture", "Random Forest / Logistic Regression")
    col_info2.metric("Decision Threshold", "0.30 (30%)")
    col_info3.metric("Total Input Features", f"{len(model_columns)}")

    st.markdown("---")

    # 2. Key Features Used by Model
    st.subheader("🛠️ Trained Feature Columns")
    st.write("The model evaluates risk based on the following preprocessed binary and encoded features:")
    
    # Feature columns ko clean table ya tags mein dikhayenge
    st.json(list(model_columns))

    st.markdown("---")

    # 3. Model Logic Note
    st.info(
        "💡 **Decision Threshold Note:** A standard classification threshold is usually 0.50. "
        "However, for medical stroke risk evaluation, a lower threshold of **0.30 (30%)** is applied "
        "to prioritize high sensitivity/recall and detect potential risk early."
    )