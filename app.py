import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and feature columns
with open('customer_churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("📊 Churn Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["🏠 Overview", "🔮 Predict Churn", "💰 ROI Calculator"])


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================
if page == "🏠 Overview":
    st.title("Customer Churn Analysis Dashboard")
    st.markdown("A machine learning solution to identify and retain at-risk customers.")
    st.markdown("---")

    # Load data for overview
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])

    # KPI Metrics
    total_customers = len(df)
    churned = df[df['Churn'] == 'Yes'].shape[0]
    churn_rate = round((churned / total_customers) * 100, 1)
    avg_monthly = round(df['MonthlyCharges'].mean(), 2)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churned Customers", f"{churned:,}")
    col3.metric("Churn Rate", f"{churn_rate}%")
    col4.metric("Avg Monthly Charge", f"₹{avg_monthly}")

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots(figsize=(5, 3))
        df['Churn'].value_counts().plot(kind='bar', color=['#2ecc71', '#e74c3c'], ax=ax)
        ax.set_xticklabels(['Stayed', 'Churned'], rotation=0)
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with col2:
        st.subheader("Churn by Contract Type")
        fig, ax = plt.subplots(figsize=(5, 3))
        contract_churn = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()
        contract_churn.plot(kind='bar', color=['#2ecc71', '#e74c3c'], ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
        ax.set_ylabel("Proportion")
        ax.legend(['Stayed', 'Churned'])
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Key Insights")
    st.info("📌 Month-to-month customers churn at significantly higher rates than annual contract customers.")
    st.info("📌 Customers in their first 12 months are the most at risk — early tenure is the danger zone.")
    st.info("📌 Higher monthly charges correlate with higher churn — price sensitivity is a real driver.")

    st.markdown("---")
    st.subheader("🎯 High Risk Customer Segment Profile")
    st.markdown("Based on model analysis, here is who your high risk customers typically are:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.error("""
        **Contract & Tenure**
        - Month-to-month contract
        - Tenure under 12 months
        - New customers in first 3 months are the most vulnerable
        """)

    with col2:
        st.warning("""
        **Billing Profile**
        - Monthly charges above ₹65
        - Electronic check payment method
        - Paperless billing enabled
        - High monthly but low total charges
        """)

    with col3:
        st.info("""
        **Services Profile**
        - Fiber optic internet users
        - No online security add-on
        - No tech support add-on
        - Single line phone service
        """)

    st.markdown("---")
    st.subheader("📋 Strategic Recommendations")
    st.success("""
    **1. Early Intervention Programme**
    Target all month-to-month customers in their first 12 months with a proactive outreach campaign.
    Offer a discounted annual contract upgrade before loyalty erodes.
    """)
    st.success("""
    **2. Pricing Review for High-Charge Segments**
    Customers paying above ₹65/month show significantly higher churn.
    Review value perception for fiber optic plans and consider loyalty discounts at the 6-month mark.
    """)
    st.success("""
    **3. Service Bundling Strategy**
    Customers without security and support add-ons churn more.
    Bundle these services into base plans for new customers to increase switching costs and perceived value.
    """)


# ============================================================
# PAGE 2 — PREDICT CHURN
# ============================================================
elif page == "🔮 Predict Churn":
    st.title("🔮 Customer Churn Predictor")
    st.markdown("Enter customer details below to predict churn risk.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])

    with col2:
        st.subheader("Services")
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    with col3:
        st.subheader("Billing")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])
        monthly_charges = st.slider("Monthly Charges (₹)", 0, 120, 65)
        total_charges = st.slider("Total Charges (₹)", 0, 9000, 1000)

    st.markdown("---")

    if st.button("🔮 Predict Churn Risk", use_container_width=True):

        # Build input dictionary
        input_dict = {
            'gender': 1 if gender == 'Male' else 0,
            'SeniorCitizen': 1 if senior == 'Yes' else 0,
            'Partner': 1 if partner == 'Yes' else 0,
            'Dependents': 1 if dependents == 'Yes' else 0,
            'tenure': tenure,
            'PhoneService': 1 if phone == 'Yes' else 0,
            'MultipleLines': 1 if multiple_lines == 'Yes' else 0,
            'OnlineSecurity': 1 if online_security == 'Yes' else 0,
            'OnlineBackup': 1 if online_backup == 'Yes' else 0,
            'DeviceProtection': 1 if device_protection == 'Yes' else 0,
            'TechSupport': 1 if tech_support == 'Yes' else 0,
            'StreamingTV': 1 if streaming_tv == 'Yes' else 0,
            'StreamingMovies': 1 if streaming_movies == 'Yes' else 0,
            'PaperlessBilling': 1 if paperless == 'Yes' else 0,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'InternetService_Fiber optic': 1 if internet == 'Fiber optic' else 0,
            'InternetService_No': 1 if internet == 'No' else 0,
            'Contract_One year': 1 if contract == 'One year' else 0,
            'Contract_Two year': 1 if contract == 'Two year' else 0,
            'PaymentMethod_Credit card (automatic)': 1 if payment == 'Credit card (automatic)' else 0,
            'PaymentMethod_Electronic check': 1 if payment == 'Electronic check' else 0,
            'PaymentMethod_Mailed check': 1 if payment == 'Mailed check' else 0,
        }

        input_df = pd.DataFrame([input_dict])[feature_columns]
        probability = model.predict_proba(input_df)[0][1]
        probability_pct = round(probability * 100, 1)

        # Risk tier
        if probability_pct >= 70:
            risk = "🔴 High Risk"
            risk_color = "error"
        elif probability_pct >= 40:
            risk = "🟡 Medium Risk"
            risk_color = "warning"
        else:
            risk = "🟢 Low Risk"
            risk_color = "success"

        # Display result
        st.markdown("## Prediction Result")
        col1, col2 = st.columns(2)
        col1.metric("Churn Probability", f"{probability_pct}%")
        col2.metric("Risk Level", risk)

        # Retention strategy
        st.markdown("### 💡 Recommended Retention Strategy")
        if probability_pct >= 70:
            st.error("""
            **HIGH RISK — Immediate Action Required**
            - 📞 Priority callback within 48 hours
            - 🎁 Offer one month free if customer upgrades to annual contract
            - 🔒 Bundle Tech Support and Online Security at no extra cost for 3 months
            - 💬 Assign to senior retention specialist
            """)
        elif probability_pct >= 40:
            st.warning("""
            **MEDIUM RISK — Proactive Outreach**
            - 📧 Send personalised loyalty discount via email within 1 week
            - 📋 Offer a plan review — ensure customer is on the right tier
            - 🎯 Highlight unused features they are already paying for
            - 📊 Schedule a satisfaction check-in call
            """)
        else:
            st.success("""
            **LOW RISK — Maintain Engagement**
            - 🏆 Enrol in standard loyalty rewards programme
            - 📬 Include in quarterly satisfaction survey
            - ✅ No aggressive intervention needed — preserve retention budget
            """)

        # Churn drivers
        st.markdown("### 🔍 Key Risk Drivers for This Customer")
        drivers = []
        if contract == "Month-to-month":
            drivers.append("⚠️ Month-to-month contract — no switching cost, easiest to leave")
        if tenure < 12:
            drivers.append("⚠️ Early tenure (under 12 months) — loyalty not yet established")
        if monthly_charges > 65:
            drivers.append("⚠️ High monthly charges — above average, price sensitivity risk")
        if online_security == "No" and internet != "No":
            drivers.append("⚠️ No online security — less embedded in the service ecosystem")
        if tech_support == "No" and internet != "No":
            drivers.append("⚠️ No tech support — lower perceived value and switching cost")
        if not drivers:
            drivers.append("✅ No major risk drivers identified — customer profile is stable")
        for d in drivers:
            st.markdown(d)


# ============================================================
# PAGE 3 — ROI CALCULATOR
# ============================================================
elif page == "💰 ROI Calculator":
    st.title("💰 Retention ROI Calculator")
    st.markdown("Estimate the monthly revenue saved by proactively retaining at-risk customers.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Assumptions")
        total_customers = st.number_input("Total Customer Base", value=7000, step=100)
        churn_rate_input = st.slider("Current Churn Rate (%)", 1, 50, 27)
        avg_monthly_revenue = st.number_input("Average Monthly Revenue per Customer (₹)", value=65, step=5)
        retention_rate = st.slider("% of At-Risk Customers You Can Retain", 1, 100, 30)
        intervention_cost = st.number_input("Cost per Retention Intervention (₹)", value=500, step=100)

    with col2:
        st.subheader("Results")

        at_risk = int(total_customers * churn_rate_input / 100)
        customers_saved = int(at_risk * retention_rate / 100)
        revenue_saved = customers_saved * avg_monthly_revenue
        total_intervention_cost = customers_saved * intervention_cost
        net_roi = revenue_saved - total_intervention_cost

        st.metric("At-Risk Customers", f"{at_risk:,}")
        st.metric("Customers Saved", f"{customers_saved:,}")
        st.metric("Monthly Revenue Saved", f"₹{revenue_saved:,}")
        st.metric("Total Intervention Cost", f"₹{total_intervention_cost:,}")

        if net_roi > 0:
            st.success(f"✅ Net Monthly ROI: ₹{net_roi:,}")
        else:
            st.error(f"❌ Net Monthly ROI: ₹{net_roi:,} — Reduce intervention cost or improve retention rate")

    st.markdown("---")
    st.subheader("📊 Revenue Impact Breakdown")

    fig, ax = plt.subplots(figsize=(8, 4))
    categories = ['Revenue at Risk', 'Revenue Saved', 'Intervention Cost', 'Net ROI']
    values = [at_risk * avg_monthly_revenue, revenue_saved, total_intervention_cost, net_roi]
    colors = ['#e74c3c', '#2ecc71', '#f39c12', '#3498db']
    bars = ax.bar(categories, values, color=colors)
    ax.set_ylabel("Amount (₹)")
    ax.set_title("Monthly Retention ROI Breakdown")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                f'₹{val:,}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.info("""
    **How to use this calculator:**
    - Adjust the churn rate to match your current dataset findings (26.6%)
    - Set average monthly revenue based on your data (₹64.76 from EDA)
    - Model different retention scenarios by changing the retention rate slider
    - Net ROI tells you whether the retention programme pays for itself
    """)