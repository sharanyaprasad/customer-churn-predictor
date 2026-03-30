import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# ── Load model ────────────────────────────────────────────────────────────────
with open('customer_churn_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0A0F1E;
    color: #E8EAF0;
}

.stApp { background-color: #0A0F1E; }

[data-testid="stSidebar"] {
    background-color: #0D1425;
    border-right: 1px solid #1E2A45;
}

[data-testid="stSidebar"] .stRadio label {
    color: #8892A4 !important;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 10px 0;
    transition: color 0.2s;
}

#MainMenu, footer, header { visibility: hidden; }

h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.8rem !important;
    font-weight: 400 !important;
    color: #F0F2F8 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.15 !important;
}

h2 {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    color: #4A90D9 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
}

h3 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.4rem !important;
    font-weight: 400 !important;
    color: #D0D4E0 !important;
}

[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1E2A45;
    border-radius: 2px;
    padding: 20px 24px;
}

[data-testid="metric-container"] label {
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #5A6478 !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2rem !important;
    color: #F0F2F8 !important;
}

.stButton > button {
    background: #1A2540;
    color: #E8EAF0;
    border: 1px solid #2E4070;
    border-radius: 2px;
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 14px 32px;
    transition: all 0.2s;
}

.stButton > button:hover {
    background: #243560;
    border-color: #4A90D9;
    color: #FFFFFF;
}

hr { border: none; border-top: 1px solid #1E2A45; margin: 32px 0; }

.kpi-card {
    background: #111827;
    border: 1px solid #1E2A45;
    border-top: 2px solid #4A90D9;
    padding: 24px 28px;
    margin-bottom: 8px;
}

.kpi-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5A6478;
    margin-bottom: 8px;
}

.kpi-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #F0F2F8;
    line-height: 1;
}

.section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #4A90D9;
    margin-bottom: 4px;
}

.insight-card {
    background: #0D1425;
    border-left: 2px solid #4A90D9;
    padding: 16px 20px;
    margin-bottom: 8px;
}

.insight-text {
    font-size: 14px;
    color: #A0AABB;
    line-height: 1.6;
    font-weight: 300;
}

.result-card {
    background: #111827;
    border: 1px solid #1E2A45;
    padding: 32px;
    margin-top: 24px;
}

.prob-number { font-family: 'DM Serif Display', serif; font-size: 4rem; line-height: 1; margin-bottom: 4px; }
.risk-high { color: #E05555; }
.risk-medium { color: #D4913A; }
.risk-low { color: #3A9E6A; }

.strategy-card {
    background: #0D1425;
    border: 1px solid #1E2A45;
    padding: 20px 24px;
    margin-top: 16px;
}

.strategy-title {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.strategy-item {
    font-size: 13px;
    color: #8892A4;
    padding: 6px 0;
    border-bottom: 1px solid #1A2540;
    line-height: 1.5;
}

.driver-tag {
    display: inline-block;
    background: #1A2030;
    border: 1px solid #2E3A55;
    color: #8892A4;
    font-size: 11px;
    font-weight: 500;
    padding: 5px 12px;
    margin: 4px 4px 4px 0;
    letter-spacing: 0.04em;
}

.page-header {
    border-bottom: 1px solid #1E2A45;
    padding-bottom: 24px;
    margin-bottom: 40px;
}

.eyebrow {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #4A90D9;
    margin-bottom: 8px;
}

.footer {
    border-top: 1px solid #1A2540;
    margin-top: 80px;
    padding-top: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.footer-text { font-size: 11px; color: #3A4560; letter-spacing: 0.08em; }

.segment-card {
    background: #0D1425;
    border: 1px solid #1E2A45;
    padding: 24px;
}

.segment-title {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4A90D9;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid #1E2A45;
}

.segment-item { font-size: 13px; color: #7A8499; padding: 5px 0; line-height: 1.5; }

.roi-result {
    background: #111827;
    border: 1px solid #1E2A45;
    border-top: 2px solid #4A90D9;
    padding: 28px;
    text-align: center;
}

.roi-positive { border-top-color: #3A9E6A; }
.roi-negative { border-top-color: #E05555; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib theme ───────────────────────────────────────────────────────────
mpl.rcParams.update({
    'figure.facecolor': '#111827',
    'axes.facecolor': '#111827',
    'axes.edgecolor': '#1E2A45',
    'axes.labelcolor': '#5A6478',
    'axes.titlecolor': '#8892A4',
    'xtick.color': '#5A6478',
    'ytick.color': '#5A6478',
    'text.color': '#8892A4',
    'grid.color': '#1A2540',
    'grid.linewidth': 0.5,
    'font.family': 'sans-serif',
    'font.size': 11,
})

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 32px 0;'>
        <div style='font-family: DM Serif Display, serif; font-size: 1.3rem; color: #F0F2F8; line-height: 1.2;'>Churn<br>Intelligence</div>
        <div style='font-size: 10px; color: #3A4560; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 6px;'>Telco Analytics Platform</div>
    </div>
    <hr style='border-color: #1E2A45; margin-bottom: 24px;'>
    """, unsafe_allow_html=True)

    page = st.radio("", ["Overview", "Predict", "ROI Calculator"], label_visibility="collapsed")

    st.markdown("""
    <div style='margin-top: 48px;'>
        <div style='font-size: 10px; color: #2A3450; letter-spacing: 0.06em; line-height: 2;'>
            Dataset &nbsp; <span style='color: #3A4560;'>Telco Churn</span><br>
            Model &nbsp; <span style='color: #3A4560;'>Logistic Regression</span><br>
            ROC-AUC &nbsp; <span style='color: #3A4560;'>0.832</span><br>
            Records &nbsp; <span style='color: #3A4560;'>7,032</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":

    st.markdown("""
    <div class='page-header'>
        <div class='eyebrow'>Customer Analytics</div>
        <h1>Churn Overview</h1>
        <p style='color: #5A6478; font-size: 14px; margin-top: 8px; font-weight: 300;'>
            Exploratory analysis of 7,032 customer records. Identifying at-risk segments and retention opportunities.
        </p>
    </div>
    """, unsafe_allow_html=True)

    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])

    total = len(df)
    churned = df[df['Churn'] == 'Yes'].shape[0]
    churn_rate = round((churned / total) * 100, 1)
    avg_monthly = round(df['MonthlyCharges'].mean(), 2)
    avg_tenure = round(df['tenure'].mean(), 1)

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (c1, "Total Customers", f"{total:,}"),
        (c2, "Churned", f"{churned:,}"),
        (c3, "Churn Rate", f"{churn_rate}%"),
        (c4, "Avg Monthly", f"Rs.{avg_monthly}"),
        (c5, "Avg Tenure", f"{avg_tenure}mo"),
    ]
    for col, label, value in metrics:
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-label'>{label}</div>
                <div class='kpi-value'>{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<div class='section-label'>Distribution</div>", unsafe_allow_html=True)
        st.markdown("**Churn vs Retention**")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        counts = df['Churn'].value_counts()
        bars = ax.bar(['Retained', 'Churned'], [counts.get('No', 0), counts.get('Yes', 0)],
                      color=['#1E3A6E', '#4A90D9'], width=0.5)
        ax.set_ylabel("Customers", fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                    f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=10, color='#8892A4')
        fig.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("<div class='section-label'>By Contract</div>", unsafe_allow_html=True)
        st.markdown("**Churn Rate by Contract Type**")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        contract_churn = df.groupby('Contract')['Churn'].apply(
            lambda x: (x == 'Yes').sum() / len(x) * 100
        ).reset_index()
        contract_churn.columns = ['Contract', 'ChurnRate']
        bars = ax.barh(contract_churn['Contract'], contract_churn['ChurnRate'],
                       color=['#4A90D9', '#2E5A9E', '#1A3560'], height=0.5)
        ax.set_xlabel("Churn Rate (%)", fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3)
        for bar, val in zip(bars, contract_churn['ChurnRate']):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=10, color='#8892A4')
        fig.tight_layout()
        st.pyplot(fig)

    col3, col4 = st.columns(2, gap="large")

    with col3:
        st.markdown("<div class='section-label'>Tenure Analysis</div>", unsafe_allow_html=True)
        st.markdown("**Churn by Tenure**")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.hist(df[df['Churn'] == 'No']['tenure'], bins=30, alpha=0.6, color='#1E3A6E', label='Retained')
        ax.hist(df[df['Churn'] == 'Yes']['tenure'], bins=30, alpha=0.8, color='#4A90D9', label='Churned')
        ax.set_xlabel("Tenure (months)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=9, framealpha=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)

    with col4:
        st.markdown("<div class='section-label'>Billing Analysis</div>", unsafe_allow_html=True)
        st.markdown("**Monthly Charges by Churn Status**")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.boxplot(
            [df[df['Churn'] == 'No']['MonthlyCharges'], df[df['Churn'] == 'Yes']['MonthlyCharges']],
            labels=['Retained', 'Churned'],
            patch_artist=True,
            boxprops=dict(facecolor='#1E3A6E', color='#4A90D9'),
            medianprops=dict(color='#4A90D9', linewidth=2),
            whiskerprops=dict(color='#2E4070'),
            capprops=dict(color='#2E4070'),
            flierprops=dict(marker='o', color='#4A90D9', markersize=3, alpha=0.3)
        )
        ax.set_ylabel("Monthly Charges (Rs.)", fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Key Findings</div>", unsafe_allow_html=True)
    st.markdown("**What the data tells us**")
    st.markdown("<br>", unsafe_allow_html=True)

    for title, text in [
        ("Contract Type", "Month-to-month customers churn at 42% versus 3% for two-year contracts. The absence of contractual lock-in is the single most actionable lever for retention strategy."),
        ("Tenure Window", "Churn is heavily concentrated in the first 12 months. Customers who survive past year one develop loyalty that compounds over time. Early intervention is the highest-ROI play."),
        ("Price Sensitivity", "Churners pay significantly higher monthly charges than retained customers. High-value customers are paradoxically the most at risk — perceived value does not scale with price."),
        ("Service Bundling", "Customers without security and support add-ons churn more consistently. Bundling these services increases switching costs and perceived platform value at marginal incremental cost."),
    ]:
        st.markdown(f"""
        <div class='insight-card'>
            <div style='font-size: 11px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #4A90D9; margin-bottom: 6px;'>{title}</div>
            <div class='insight-text'>{text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Risk Profiling</div>", unsafe_allow_html=True)
    st.markdown("**High-Risk Customer Segment**")
    st.markdown("<br>", unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns(3, gap="medium")
    for col, title, items in [
        (sc1, "Contract & Tenure", ["Month-to-month contract", "Tenure under 12 months", "No switching cost or penalty", "First 3 months: highest risk window"]),
        (sc2, "Billing Profile", ["Monthly charges above average", "Electronic check payment", "Paperless billing enabled", "High monthly, low total charges"]),
        (sc3, "Services", ["Fiber optic internet", "No online security add-on", "No tech support subscription", "Single line phone service"]),
    ]:
        with col:
            items_html = "".join([f"<div class='segment-item'>{i}</div>" for i in items])
            st.markdown(f"""
            <div class='segment-card'>
                <div class='segment-title'>{title}</div>
                {items_html}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class='footer'>
        <div class='footer-text'>CHURN INTELLIGENCE PLATFORM</div>
        <div class='footer-text'>Project by Sharanya Prasad</div>
        <div class='footer-text'>Telco Customer Churn Dataset &nbsp;|&nbsp; Logistic Regression &nbsp;|&nbsp; ROC-AUC 0.832</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predict":

    st.markdown("""
    <div class='page-header'>
        <div class='eyebrow'>Prediction Engine</div>
        <h1>Churn Risk Assessment</h1>
        <p style='color: #5A6478; font-size: 14px; margin-top: 8px; font-weight: 300;'>
            Enter customer profile details to generate a churn probability score and retention recommendation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("<div class='section-label'>Demographics</div>", unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    with col2:
        st.markdown("<div class='section-label'>Services</div>", unsafe_allow_html=True)
        tenure = st.slider("Tenure (months)", 0, 72, 12)
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
        st.markdown("<div class='section-label'>Billing</div>", unsafe_allow_html=True)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.slider("Monthly Charges (Rs.)", 0, 120, 65)
        total_charges = st.slider("Total Charges (Rs.)", 0, 9000, 1000)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("RUN ASSESSMENT", use_container_width=True):

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
        prob_pct = round(probability * 100, 1)

        if prob_pct >= 70:
            risk_label, risk_class = "HIGH RISK", "risk-high"
            strategy_color = "#E05555"
            strategy_title = "IMMEDIATE INTERVENTION REQUIRED"
            strategies = [
                "Priority callback within 48 hours — escalate to senior retention specialist",
                "Offer one month complimentary service on annual contract upgrade",
                "Bundle Tech Support and Online Security at no cost for 90 days",
                "Flag for executive account review if monthly charges exceed Rs.80",
            ]
        elif prob_pct >= 40:
            risk_label, risk_class = "MEDIUM RISK", "risk-medium"
            strategy_color = "#D4913A"
            strategy_title = "PROACTIVE OUTREACH RECOMMENDED"
            strategies = [
                "Send personalised loyalty offer via email within 5 business days",
                "Conduct plan review — assess whether current tier matches usage pattern",
                "Surface unused features already included in the customer's current plan",
                "Schedule a satisfaction check-in within the month",
            ]
        else:
            risk_label, risk_class = "LOW RISK", "risk-low"
            strategy_color = "#3A9E6A"
            strategy_title = "STANDARD ENGAGEMENT"
            strategies = [
                "Enrol in loyalty rewards programme if not already active",
                "Include in quarterly Net Promoter Score survey",
                "No aggressive intervention required — preserve retention budget",
                "Monitor for change in usage patterns at next billing cycle",
            ]

        r1, r2 = st.columns([1, 2], gap="large")

        with r1:
            st.markdown(f"""
            <div class='result-card' style='text-align: center;'>
                <div class='kpi-label' style='margin-bottom: 16px;'>Churn Probability</div>
                <div class='prob-number {risk_class}'>{prob_pct}%</div>
                <div style='font-size: 10px; font-weight: 600; letter-spacing: 0.16em; color: {strategy_color}; margin-top: 12px; text-transform: uppercase;'>{risk_label}</div>
                <hr style='border-color: #1E2A45; margin: 20px 0;'>
                <div class='kpi-label'>Tenure</div>
                <div style='font-family: DM Serif Display, serif; font-size: 1.4rem; color: #D0D4E0;'>{tenure} months</div>
                <div class='kpi-label' style='margin-top: 16px;'>Contract</div>
                <div style='font-size: 13px; color: #8892A4;'>{contract}</div>
                <div class='kpi-label' style='margin-top: 16px;'>Monthly Charges</div>
                <div style='font-family: DM Serif Display, serif; font-size: 1.4rem; color: #D0D4E0;'>Rs.{monthly_charges}</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            strategies_html = "".join([f"<div class='strategy-item'>{s}</div>" for s in strategies])
            st.markdown(f"""
            <div class='strategy-card' style='margin-top: 24px;'>
                <div class='strategy-title' style='color: {strategy_color};'>{strategy_title}</div>
                {strategies_html}
            </div>
            """, unsafe_allow_html=True)

            drivers = []
            if contract == "Month-to-month":
                drivers.append("Month-to-month contract")
            if tenure < 12:
                drivers.append("Early tenure")
            if monthly_charges > 65:
                drivers.append("Above-average charges")
            if online_security == "No" and internet != "No":
                drivers.append("No online security")
            if tech_support == "No" and internet != "No":
                drivers.append("No tech support")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Risk Drivers</div>", unsafe_allow_html=True)
            if drivers:
                tags = "".join([f"<span class='driver-tag'>{d}</span>" for d in drivers])
            else:
                tags = "<span class='driver-tag'>No significant risk drivers identified</span>"
            st.markdown(f"<div style='margin-top: 8px;'>{tags}</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='footer'>
        <div class='footer-text'>CHURN INTELLIGENCE PLATFORM</div>
        <div class='footer-text'>Project by Sharanya Prasad</div>
        <div class='footer-text'>Logistic Regression &nbsp;|&nbsp; ROC-AUC 0.832 &nbsp;|&nbsp; Trained on 5,625 records</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ROI CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ROI Calculator":

    st.markdown("""
    <div class='page-header'>
        <div class='eyebrow'>Financial Modelling</div>
        <h1>Retention ROI Calculator</h1>
        <p style='color: #5A6478; font-size: 14px; margin-top: 8px; font-weight: 300;'>
            Model the monthly revenue impact of a proactive retention programme. Adjust assumptions to stress-test different scenarios.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("<div class='section-label'>Assumptions</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        total_customers = st.number_input("Total Customer Base", value=7000, step=100)
        churn_rate_input = st.slider("Current Churn Rate (%)", 1, 50, 27)
        avg_monthly_revenue = st.number_input("Average Monthly Revenue per Customer (Rs.)", value=65, step=5)
        retention_rate = st.slider("Estimated Retention Success Rate (%)", 1, 100, 30)
        intervention_cost = st.number_input("Cost per Retention Intervention (Rs.)", value=500, step=100)

    with col2:
        st.markdown("<div class='section-label'>Output</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        at_risk = int(total_customers * churn_rate_input / 100)
        customers_saved = int(at_risk * retention_rate / 100)
        revenue_saved = customers_saved * avg_monthly_revenue
        total_cost = customers_saved * intervention_cost
        net_roi = revenue_saved - total_cost
        roi_class = "roi-positive" if net_roi > 0 else "roi-negative"
        roi_color = "#3A9E6A" if net_roi > 0 else "#E05555"

        o1, o2 = st.columns(2)
        with o1:
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-label'>At-Risk Customers</div>
                <div class='kpi-value'>{at_risk:,}</div>
            </div>
            <div class='kpi-card' style='margin-top: 8px;'>
                <div class='kpi-label'>Customers Saved</div>
                <div class='kpi-value'>{customers_saved:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with o2:
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Revenue Saved</div>
                <div class='kpi-value' style='font-size: 1.6rem;'>Rs.{revenue_saved:,}</div>
            </div>
            <div class='kpi-card' style='margin-top: 8px;'>
                <div class='kpi-label'>Intervention Cost</div>
                <div class='kpi-value' style='font-size: 1.6rem;'>Rs.{total_cost:,}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='roi-result {roi_class}' style='margin-top: 8px;'>
            <div class='kpi-label'>Net Monthly ROI</div>
            <div style='font-family: DM Serif Display, serif; font-size: 2.4rem; color: {roi_color};'>Rs.{net_roi:,}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Visual Breakdown</div>", unsafe_allow_html=True)
    st.markdown("**Monthly Retention Economics**")
    st.markdown("<br>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    values = [at_risk * avg_monthly_revenue, revenue_saved, total_cost, net_roi]
    colors = ['#1E3A6E', '#1E5C3A', '#5C3A1E', '#3A9E6A' if net_roi >= 0 else '#9E3A3A']
    bars = ax.bar(['Revenue at Risk', 'Revenue Saved', 'Intervention Cost', 'Net ROI'],
                  values, color=colors, width=0.55)
    ax.set_ylabel("Amount (Rs.)", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                max(bar.get_height(), 0) + max(values) * 0.01,
                f'Rs.{val:,}', ha='center', va='bottom', fontsize=10, color='#8892A4')
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='insight-card'>
        <div style='font-size: 11px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #4A90D9; margin-bottom: 6px;'>How to use this calculator</div>
        <div class='insight-text'>
            Set churn rate to 26.6% to match the dataset baseline. Use Rs.64.76 as average monthly revenue based on EDA findings.
            Adjust the retention success rate to model conservative (20%), base (30%), and optimistic (50%) scenarios.
            Net ROI turns positive when revenue saved exceeds the cost of running the retention programme.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='footer'>
        <div class='footer-text'>CHURN INTELLIGENCE PLATFORM</div>
        <div class='footer-text'>Project by Sharanya Prasad</div>
        <div class='footer-text'>Telco Customer Churn Dataset &nbsp;|&nbsp; Logistic Regression &nbsp;|&nbsp; ROC-AUC 0.832</div>
    </div>
    """, unsafe_allow_html=True)
