import streamlit as st 
import pandas as pd
import numpy as np
import pickle 
import json
import os
import sys

sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))

# Page Configuration
st.set_page_config(page_title = 'Churn Predictor',
                   page_icon="🔄",
                   layout = 'wide',
                   initial_sidebar_state="expanded")

# Load Models
@st.cache_resource

def load_models():
    """
    Loads all saved model files from models/ folder.

    Returns dictionary with all loaded objects.

    Returns None if any file is missing.
    """

    app_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(app_dir)
    models_dir = os.path.join(project_root, 'models')

    required_files = [
        'best_model.pkl',
        'scaler.pkl',
        'feature_columns.pkl',
        'model_info.json'
    ]

    for file in required_files:
        filepath = os.path.join(models_dir, file)
        if not os.path.exists(filepath):
            st.error(f'Missing File: {file}')
            return None
        
    # Load all files
    with open(os.path.join(models_dir, 'best_model.pkl'), 'rb') as f:
        model = pickle.load(f)

    with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    with open(os.path.join(models_dir, 'feature_columns.pkl'), 'rb') as f:
        feature_columns = pickle.load(f)

    with open(os.path.join(models_dir, 'model_info.json'), 'r') as f:
        model_info = json.load(f)

    return {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'model_info': model_info
    }

models = load_models()

# App Header
st.title("🔄 Customer Churn Predictor")
st.markdown("""
Predict whether a customer will leave the company
based on their profile and service usage.
""")
st.divider()

# Show Model Info in Sidebar
with st.sidebar:
    st.header("📊 Model Information")

    if models:
        info = models['model_info']

        st.metric('Model Type', info['model_name'])

        st.metric("AUC Score",
                  f"{info['auc_score']:.4f}")

        st.metric("Recall Score",
                  f"{info['recall']:.4f}")

        st.metric("Training Samples",
                  f"{info['train_size']:,}")

        st.divider()

        st.info("""
        **How to use:**
        1. Fill in customer details
        2. Click Predict button
        3. See churn probability!
        """)

# Main Content
if not models:
    st.error("❌ Could not load models!")
    st.stop()
    
col1, col2 = st.columns(2)

# LEFT COLUMN: Account Information
with col1:
    st.subheader("📋 Account Information")

    tenure = st.slider('Tenure (months)',
                       min_value = 0, 
                       max_value = 72,
                       value = 12,
                       help="How many months has this customer been with the company?")
    
    monthly_charges = st.number_input("Monthly Charges ($)",
                                      min_value = 0.0,
                                      max_value = 150.0,
                                      value = 65.0,
                                      format = '%.2f')
    
    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=float(tenure * monthly_charges),
        step=10.0,
        format="%.2f")

    contract = st.selectbox('Contract Type',
                            options = ["Month-to-month","One year","Two year"],
                            index = 0)
    
    payment_method = st.selectbox(
        "Payment Method",
        options=[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        options=["Yes", "No"]
    )

# RIGHT COLUMN: Service Information 
with col2:
    st.subheader("📡 Service Information")

    gender = st.selectbox("Gender", options=["Male", "Female"])

    senior_citizen = st.selectbox(
        "Senior Citizen",
        options=["No", "Yes"]
    )

    partner = st.selectbox(
        "Has Partner?",
        options=["Yes", "No"]
    )

    dependents = st.selectbox(
        "Has Dependents?",
        options=["Yes", "No"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        options=["DSL", "Fiber optic", "No"]
    )

    phone_service = st.selectbox(
        "Phone Service",
        options=["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        options=["Yes", "No", "No phone service"]
    )

# Full Width: Additional Services
st.subheader("🔒 Additional Services")

serv_col1, serv_col2, serv_col3, serv_col4 = st.columns(4)

with serv_col1:
    online_security = st.selectbox(
        "Online Security",
        options=["Yes", "No", "No internet service"]
    )

with serv_col2:
    online_backup = st.selectbox(
        "Online Backup",
        options=["Yes", "No", "No internet service"]
    )

with serv_col3:
    device_protection = st.selectbox(
        "Device Protection",
        options=["Yes", "No", "No internet service"]
    )

with serv_col4:
    tech_support = st.selectbox(
        "Tech Support",
        options=["Yes", "No", "No internet service"]
    )

stream_col1, stream_col2 = st.columns(2)

with stream_col1:
    streaming_tv = st.selectbox(
        "Streaming TV",
        options=["Yes", "No", "No internet service"]
    )

with stream_col2:
    streaming_movies = st.selectbox(
        "Streaming Movies",
        options=["Yes", "No", "No internet service"]
    )

st.divider()

# Predict Button
predict_button = st.button("🔮 Predict Churn",
                            type="primary",
                            use_container_width=True)

if predict_button:

    input_data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    input_df = pd.DataFrame([input_data])

    # Feature 1: tenure_group
    def get_tenure_group(t):
        if t <= 12:
            return 'New'
        elif t <= 36:
            return 'Mid'
        else:
            return 'Loyal'

    input_df['tenure_group'] = input_df['tenure'].apply(
        get_tenure_group
    )

    # Feature 2: total_services
    service_cols = [
        'PhoneService', 'MultipleLines',
        'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies'
    ]
    input_df['total_services'] = (
        input_df[service_cols] == 'Yes'
    ).sum(axis=1)

    # Feature 3: is_new_customer
    input_df['is_new_customer'] = (
        input_df['tenure'] <= 12
    ).astype(int)

    # Feature 4: avg_monthly_from_total
    input_df['avg_monthly_from_total'] = (
        input_df['TotalCharges'] /
        input_df['tenure'].replace(0, 1)
    )

    # Feature 5: high_value_customer
    TRAINING_MEDIAN_CHARGES = 70.35
    input_df['high_value_customer'] = (
        input_df['MonthlyCharges'] >
        TRAINING_MEDIAN_CHARGES
    ).astype(int)

    # One-Hot Encode
    categorical_cols = input_df.select_dtypes(include = ['object', 'category']).columns.tolist()

    input_encoded = pd.get_dummies(
        input_df,
        columns=categorical_cols,
        drop_first=True
    )

    feature_columns = models['feature_columns']
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[feature_columns]

    # Scale Features
    scaler = models['scaler']
    input_scaled = scaler.transform(input_encoded)
    
    # Predict
    model = models['model']
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    churn_probability = probability[1]
    no_churn_probability = probability[0]

    # Display results
    st.divider()
    st.subheader("🎯 Prediction Result")

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        st.metric(
            "Churn Probability",
            f"{churn_probability:.1%}",
            # :.1% → formats as percentage
            # 0.73 → "73.0%"
        )

    with res_col2:
        st.metric(
            "Retention Probability",
            f"{no_churn_probability:.1%}"
        )

    with res_col3:
        risk_level = (
            "🔴 HIGH" if churn_probability > 0.7
            else "🟡 MEDIUM" if churn_probability > 0.4
            else "🟢 LOW"
        )
        st.metric("Risk Level", risk_level)


    if prediction == 1:
        st.error(f"""
        ## 🚨 HIGH CHURN RISK DETECTED!

        This customer has a **{churn_probability:.1%}**
        probability of leaving.

        ### Recommended Actions:
        - 📞 Assign dedicated account manager
        - 💰 Offer loyalty discount or price lock
        - 📋 Suggest switching to annual contract
        - 🎁 Provide exclusive retention offer
        """)

    else:
        # Customer likely to stay!
        st.success(f"""
        ## ✅ LOW CHURN RISK

        This customer has a **{no_churn_probability:.1%}**
        probability of staying.

        ### Customer Profile:
        - Customer appears satisfied with service
        - Continue providing quality service
        - Consider upselling additional services
        """)

    st.subheader("📊 Probability Breakdown")

    import plotly.graph_objects as go

    fig = go.Figure(go.Bar(
        x=[no_churn_probability, churn_probability],
        y=['Will Stay', 'Will Churn'],
        orientation='h',
        marker_color=[
            '#2ecc71',    # green for stay
            '#e74c3c'     # red for churn
        ],
        text=[
            f'{no_churn_probability:.1%}',
            f'{churn_probability:.1%}'
        ],
        textposition='inside',
        textfont=dict(size=16, color='white')
    ))

    fig.update_layout(
        title='Churn vs Retention Probability',
        xaxis_title='Probability',
        xaxis=dict(range=[0, 1]),
        height=200,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("👤 Customer Summary")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.write(f"**Tenure:** {tenure} months")
        st.write(f"**Contract:** {contract}")
        st.write(f"**Monthly Charges:** "
                 f"${monthly_charges:.2f}")
        st.write(f"**Internet:** {internet_service}")

    with summary_col2:
        st.write(f"**Total Services:** "
                 f"{int(input_df['total_services'].iloc[0])}")
        st.write(f"**Customer Type:** "
                 f"{'New' if tenure <= 12 else 'Existing'}")
        st.write(f"**High Value:** "
                 f"{'Yes' if monthly_charges > 70.35 else 'No'}")
        st.write(f"**Senior Citizen:** {senior_citizen}")

        