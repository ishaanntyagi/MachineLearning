import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Paths
# RF_MODEL_PATH = r"C:\Users\ishaan.narayan\Desktop\Ishaan's Workspace\Intern_ML-DL-Gen\Employee-Attrition-Prediction\Models\best_rf_model.pkl"
# XGB_MODEL_PATH = r"C:\Users\ishaan.narayan\Desktop\Ishaan's Workspace\Intern_ML-DL-Gen\Employee-Attrition-Prediction\Models\best_model_XGB.pkl"
# DATA_PATH = r"C:\Users\ishaan.narayan\Desktop\Ishaan's Workspace\Intern_ML-DL-Gen\Employee-Attrition-Prediction\Data\cleaned_employee_data.csv"

RF_MODEL_PATH = r"Intern_ML-DL-Gen\Employee-Attrition-Prediction\Models\best_rf_model.pkl"
XGB_MODEL_PATH = r"Intern_ML-DL-Gen\Employee-Attrition-Prediction\Models\best_model_XGB.pkl"
DATA_PATH = r"Intern_ML-DL-Gen\Employee-Attrition-Prediction\Data\cleaned_employee_data.csv"
 
@st.cache_resource
def load_all():
    df = pd.read_csv(DATA_PATH)
    rf = joblib.load(RF_MODEL_PATH)
    xgb = joblib.load(XGB_MODEL_PATH)
    means = df.mean(numeric_only=True)
    modes = df.mode().iloc[0]
    features = [col for col in df.columns if col not in ["Attrition"]]
    return df, rf, xgb, means, modes, features

df, model_rf, model_xgb, means, modes, all_features = load_all()

# 15 user-input features (from your data, exact names)
input_features = [
    ("Age", "slider"),
    ("DistanceFromHome", "slider"),
    ("Education", "selectbox"),
    ("EnvironmentSatisfaction", "selectbox"),
    ("JobInvolvement", "selectbox"),
    ("JobLevel", "selectbox"),
    ("JobSatisfaction", "selectbox"),
    ("MonthlyIncome", "number"),
    ("NumCompaniesWorked", "number"),
    ("StockOptionLevel", "selectbox"),
    ("TotalWorkingYears", "number"),
    ("TrainingTimesLastYear", "slider"),
    ("YearsAtCompany", "number"),
    ("JobRole_Research Scientist", "checkbox"),
    ("OverTime_Yes", "checkbox")
]

# Model selector
model_choice = st.sidebar.selectbox("Choose Model", ["Random Forest", "XGBoost"])
model = model_rf if model_choice == "Random Forest" else model_xgb

st.title("Employee Attrition Prediction(44 Features, 15 from User)")
st.write("15 main features below Rest are filled with Mode and Mean.")

with st.form("main_form"):
    cols = st.columns(3)
    user_input = {}
    for idx, (feature, widget) in enumerate(input_features):
        col = cols[idx % 3]
        if widget == "slider":
            minv = int(df[feature].min())
            maxv = int(df[feature].max())
            val = int(means[feature])
            user_input[feature] = col.slider(feature, minv, maxv, int(val))
        elif widget == "selectbox":
            options = sorted(df[feature].unique())
            user_input[feature] = col.selectbox(feature, options, index=options.index(modes[feature]) if modes[feature] in options else 0)
        elif widget == "number":
            minv = int(df[feature].min())
            maxv = int(df[feature].max())
            val = int(means[feature])
            user_input[feature] = col.number_input(feature, min_value=minv, max_value=maxv, value=val)
        elif widget == "checkbox":
            val = bool(modes[feature]) if feature in modes else False
            user_input[feature] = col.checkbox(feature.replace("_", " "), value=val)
    submitted = st.form_submit_button("Predict")

def build_input_vector(user_input, means, modes, all_features):
    row = {}
    for feat in all_features:
        if feat in user_input:
            val = user_input[feat]
            # For checkboxes, ensure int (for one-hot columns)
            if isinstance(val, bool):
                row[feat] = int(val)
            else:
                row[feat] = val
        else:
            # For numericvalues, using mean for other values , use mode.
            if feat in means:
                row[feat] = float(means[feat])
            else:
                
                row[feat] = int(modes[feat]) if feat in modes else 0
    return pd.DataFrame([row])

if submitted:
    input_df = build_input_vector(user_input, means, modes, all_features)
    pred = model.predict(input_df)[0]
    st.success(f"Prediction: {'Yes' if pred else 'No'}")
    with st.expander("Show all features used for prediction"):
        st.write(input_df)
    #for now removing the Probab Function as its not Correct.  
    st.button("Probab Functionality")  
st.caption("Auto-filling the remaining 29 features with mean/mode values. All 44 features are used for prediction as in the training data.")




