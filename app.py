import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl

with open('customer_churn_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

st.set_page_config(page_title="Churn Predictor", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

* { font-family: 'Plus Jakarta Sans', sans-serif !important; }

html, body, [class*="css"], .stApp {
    background-color: #1a1a1a;
    color: #e8e4de;
}

[data-testid="stSidebar"] {
    background-color: #141414;
    border-right: 1px solid #2a2a2a;
}

[data-testid="stSidebar"] .stRadio > div {
    gap: 4px;
}

[data-testid="stSidebar"] .stRadio label {
    color: #6b6b6b !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 8px 12px !important;
    border-radius: 6px !important;
    transition: all 0.15s !important;
    width: 100%;
}

[data-testid="stSidebar"] .stRadio label:hover {
    color: #e8e4de !important;
    background: #222222 !important;
}

#MainMenu, footer, header { visibility: hidden; }

h1 {
    font-size: 1.75rem !important;
    font-weight: 600 !important;
    color: #f0ece6 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.3 !important;
    margin-bottom: 4px !important;
}

h2, h3 {
    font-weight: 600 !important;
    color: #c8c4be !important;
    letter-spacing: -0.01em !important;
}

p { color: #8a8680; font-size: 14px; line-height: 1.6; }

[data-testid="metric-container"] {
    background: #202020;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 18px 20px;
}

[data-testid="stMetricLabel"] {
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #5a5a5a !important;
    letter-spacing: 0.02em !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    color: #f0ece6 !important;
}

.stButton > button {
    background: #2a2a2a;
    color: #e8e4de;
    border: 1px solid #3a3a3a;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.01em;
    padding: 12px 28px;
    transition: all 0.15s;
}

.stButton > button:hover {
    background: #333333;
    border-color: #c8a96e;
    color: #f0ece6;
}

.stSelectbox > div > div {
    background: #202020 !important;
    border-color: #2a2a2a !important;
    border-radius: 8px !important;
    color: #e8e4de !important;
}

.stSlider > div > div > div {
    background: #c8a96e !important;
}

hr { border: none; border-top: 1px solid #242424; margin: 28px 0; }

.card {
    background: #202020;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 10px;
}

.tag {
    display: inline-block;
    background: #2a2a2a;
    border: 1px solid #333333;
    color: #8a8680;
    font-size: 12px;
    font-weight: 500;
    padding: 4px 10px;
    border-radius: 20px;
    margin: 3px;
}

.label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5a5a5a;
    margin-bottom: 6px;
}

.accent { color: #c8a96e; }

.insight {
    border-left: 2px solid #2a2a2a;
    padding: 12px 16px;
    margin-bottom: 8px;
    background: #1e1e1e;
    border-radius: 0 8px 8px 0;
}

.insight:hover { border-left-color: #c8a96e; }

.insight-title {
    font-size: 12px;
    font-weight: 600;
    color: #c8c4be;
    margin-bottom: 4px;
}

.insight-body {
    font-size: 13px;
    color: #6b6b6b;
    line-height: 1.6;
}

.prob-big {
    font-size: 3.5rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.03em;
}

.high { color: #e05555; }
.medium { color: #c8a96e; }
.low { color: #5a9e6a; }

.strat-item {
    font-size: 13px;
    color: #6b6b6b;
    padding: 8px 0;
    border-bottom: 1px solid #242424;
    line-height: 1.5;
}

.footer-bar {
    margin-top: 64px;
    padding-top: 20px;
    border-top: 1px solid #242424;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.footer-text { font-size: 12px; color: #3a3a3a; }
</style>
""", unsafe_allow_html=True)

mpl.rcParams.update({
    'figure.facecolor': '#202020', 'axes.facecolor': '#202020',
    'axes.edgecolor': '#2a2a2a', 'axes.labelcolor': '#6b6b6b',
    'xtick.color': '#5a5a5a', 'ytick.color': '#5a5a5a',
    'text.color': '#6b6b6b', 'grid.color': '#242424',
    'grid.linewidth': 0.5, 'font.size': 11,
})

with st.sidebar:
    st.markdown("""
    <div style='padding: 4px 0 28px 0;'>
        <div style='font-size: 15px; font-weight: 700; color: #f0ece6; letter-spacing: -0.01em;'>Churn Predictor</div>
        <div style='font-size: 12px; color: #3a3a3a; margin-top: 3px;'>Telco Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", ["Overview", "Predict", "ROI Calculator"], label_visibility="collapsed")

    st.markdown("""
    <div style='margin-top: 32px; padding: 14px; background: #1e1e1e; border-radius: 8px; border: 1px solid #242424;'>
        <div style='font-size: 11px; font-weight: 600; color: #3a3a3a; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 10px;'>Model Info</div>
        <div style='font-size: 12px; color: #5a5a5a; line-height: 2;'>
            Model &nbsp;<span style='color: #6b6b6b; float: right;'>Logistic Reg.</span><br>
            AUC &nbsp;<span style='color: #c8a96e; float: right;'>0.832</span><br>
            Records &nbsp;<span style='color: #6b6b6b; float: right;'>7,032</span><br>
            Churn Rate &nbsp;<span style='color: #e05555; float: right;'>26.6%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── PAGE 1: OVERVIEW ──────────────────────────────────────────────────────────
if page == "Overview":

    st.markdown("""
    <div style='margin-bottom: 28px;'>
        <div class='label accent'>Customer Analytics</div>
        <h1>Churn Overview</h1>
        <p>Exploratory analysis of 7,032 customer records — identifying who leaves, when, and why.</p>
    </div>
    """, unsafe_allow_html=True)

    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, val in [
        (c1, "Customers", f"{len(df):,}"),
        (c2, "Churned", f"{df[df['Churn']=='Yes'].shape[0]:,}"),
        (c3, "Churn Rate", "26.6%"),
        (c4, "Avg Monthly", f"Rs.{round(df['MonthlyCharges'].mean(),2)}"),
        (c5, "Avg Tenure", f"{round(df['tenure'].mean(),1)} mo"),
    ]:
        col.metric(label, val)

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<div class='label'>By Contract Type</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3))
        data = df.groupby('Contract')['Churn'].apply(lambda x: (x=='Yes').mean()*100).reset_index()
        data.columns = ['Contract', 'Rate']
        bars = ax.barh(data['Contract'], data['Rate'], color=['#c8a96e','#8a6a3e','#4a3a2e'], height=0.45)
        ax.set_xlabel("Churn Rate (%)")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar, val in zip(bars, data['Rate']):
            ax.text(val+0.5, bar.get_y()+bar.get_height()/2, f'{val:.1f}%', va='center', fontsize=10)
        fig.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("<div class='label'>Tenure Distribution</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(df[df['Churn']=='No']['tenure'], bins=28, alpha=0.5, color='#3a3a3a', label='Retained')
        ax.hist(df[df['Churn']=='Yes']['tenure'], bins=28, alpha=0.8, color='#c8a96e', label='Churned')
        ax.set_xlabel("Tenure (months)")
        ax.legend(framealpha=0, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='label'>Key Findings</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    for title, body in [
        ("Contract type is everything", "Month-to-month customers churn at 42% versus just 3% for two-year contracts. No lock-in means no reason to stay when something better comes along."),
        ("The first year is the danger zone", "Churn clusters heavily in the first 12 months. After that, customers tend to stay. Getting someone past year one is the real win."),
        ("Higher bills, higher risk", "Churners pay more per month than retained customers. The highest-paying customers are paradoxically the most at risk — they have the most to gain by switching."),
        ("Add-ons create stickiness", "Customers without security and support services churn more. Every add-on a customer uses is another reason not to leave."),
    ]:
        st.markdown(f"""
        <div class='insight'>
            <div class='insight-title'>{title}</div>
            <div class='insight-body'>{body}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='label'>Who churns</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns(3, gap="medium")
    for col, title, items in [
        (sc1, "Contract & Tenure", ["Month-to-month contract", "Under 12 months tenure", "First 3 months: highest risk", "No penalty for leaving"]),
        (sc2, "Billing", ["Above-average monthly charges", "Electronic check payment", "Paperless billing", "High monthly, low total"]),
        (sc3, "Services", ["Fiber optic internet", "No online security", "No tech support", "Single phone line"]),
    ]:
        with col:
            items_html = "".join([f"<div style='font-size:13px; color:#5a5a5a; padding: 5px 0; border-bottom: 1px solid #242424;'>{i}</div>" for i in items])
            st.markdown(f"""
            <div class='card'>
                <div class='label' style='margin-bottom: 12px;'>{title}</div>
                {items_html}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class='footer-bar'>
        <div class='footer-text'>Churn Predictor</div>
        <div class='footer-text'>Project by Sharanya Prasad</div>
        <div class='footer-text'>Logistic Regression &nbsp;·&nbsp; ROC-AUC 0.832</div>
    </div>
    """, unsafe_allow_html=True)


# ── PAGE 2: PREDICT ───────────────────────────────────────────────────────────
elif page == "Predict":

    st.markdown("""
    <div style='margin-bottom: 28px;'>
        <div class='label accent'>Prediction Engine</div>
        <h1>Churn Risk Assessment</h1>
        <p>Fill in a customer profile and run the model to get a churn probability and retention plan.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("<div class='label'>Demographics</div>", unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    with col2:
        st.markdown("<div class='label'>Services</div>", unsafe_allow_html=True)
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
        st.markdown("<div class='label'>Billing</div>", unsafe_allow_html=True)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.slider("Monthly Charges (Rs.)", 0, 120, 65)
        total_charges = st.slider("Total Charges (Rs.)", 0, 9000, 1000)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Run Assessment", use_container_width=True):
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
        prob = round(model.predict_proba(input_df)[0][1] * 100, 1)

        if prob >= 70:
            risk, risk_class = "High Risk", "high"
            strats = [
                "Call within 48 hours — flag as priority in CRM",
                "Offer one month free on upgrade to annual contract",
                "Bundle Tech Support and Security at no cost for 90 days",
                "Escalate to senior retention specialist",
            ]
        elif prob >= 40:
            risk, risk_class = "Medium Risk", "medium"
            strats = [
                "Send personalised loyalty offer within the week",
                "Run a plan review — are they on the right tier?",
                "Show them features they are paying for but not using",
                "Book a satisfaction check-in call",
            ]
        else:
            risk, risk_class = "Low Risk", "low"
            strats = [
                "Add to loyalty rewards programme",
                "Include in quarterly NPS survey",
                "No urgent action needed — save the budget",
                "Watch for usage pattern changes next cycle",
            ]

        st.markdown("<hr>", unsafe_allow_html=True)
        r1, r2 = st.columns([1, 2], gap="large")

        with r1:
            st.markdown(f"""
            <div class='card' style='text-align: center; padding: 28px;'>
                <div class='label' style='margin-bottom: 12px;'>Churn Probability</div>
                <div class='prob-big {risk_class}'>{prob}%</div>
                <div style='font-size: 13px; font-weight: 600; margin-top: 8px;' class='{risk_class}'>{risk}</div>
                <hr style='border-color: #2a2a2a; margin: 20px 0;'>
                <div class='label'>Tenure</div>
                <div style='font-size: 1.3rem; font-weight: 600; color: #c8c4be;'>{tenure} months</div>
                <div class='label' style='margin-top: 14px;'>Contract</div>
                <div style='font-size: 13px; color: #6b6b6b;'>{contract}</div>
                <div class='label' style='margin-top: 14px;'>Monthly</div>
                <div style='font-size: 1.3rem; font-weight: 600; color: #c8c4be;'>Rs.{monthly_charges}</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            strats_html = "".join([f"<div class='strat-item'>{s}</div>" for s in strats])
            st.markdown(f"""
            <div class='card' style='padding: 24px;'>
                <div class='label' style='margin-bottom: 14px;'>What to do</div>
                {strats_html}
            </div>
            """, unsafe_allow_html=True)

            drivers = []
            if contract == "Month-to-month": drivers.append("Month-to-month")
            if tenure < 12: drivers.append("Early tenure")
            if monthly_charges > 65: drivers.append("High charges")
            if online_security == "No" and internet != "No": drivers.append("No security")
            if tech_support == "No" and internet != "No": drivers.append("No support")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='label'>Risk factors</div>", unsafe_allow_html=True)
            if not drivers:
                drivers = ["None identified"]
            tags = "".join([f"<span class='tag'>{d}</span>" for d in drivers])
            st.markdown(f"<div style='margin-top: 8px;'>{tags}</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='footer-bar'>
        <div class='footer-text'>Churn Predictor</div>
        <div class='footer-text'>Project by Sharanya Prasad</div>
        <div class='footer-text'>Logistic Regression &nbsp;·&nbsp; ROC-AUC 0.832</div>
    </div>
    """, unsafe_allow_html=True)


# ── PAGE 3: ROI CALCULATOR ────────────────────────────────────────────────────
elif page == "ROI Calculator":

    st.markdown("""
    <div style='margin-bottom: 28px;'>
        <div class='label accent'>Financial Modelling</div>
        <h1>Retention ROI Calculator</h1>
        <p>How much revenue does a retention programme save? Adjust the inputs to model different scenarios.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<div class='label'>Inputs</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        total_cust = st.number_input("Total Customer Base", value=7000, step=100)
        churn_pct = st.slider("Current Churn Rate (%)", 1, 50, 27)
        avg_rev = st.number_input("Avg Monthly Revenue per Customer (Rs.)", value=65, step=5)
        ret_rate = st.slider("Retention Success Rate (%)", 1, 100, 30)
        cost_per = st.number_input("Cost per Intervention (Rs.)", value=500, step=100)

    with col2:
        st.markdown("<div class='label'>Results</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        at_risk = int(total_cust * churn_pct / 100)
        saved = int(at_risk * ret_rate / 100)
        rev_saved = saved * avg_rev
        total_cost = saved * cost_per
        net = rev_saved - total_cost

        o1, o2 = st.columns(2)
        o1.metric("At-Risk Customers", f"{at_risk:,}")
        o2.metric("Customers Saved", f"{saved:,}")
        o1.metric("Revenue Saved", f"Rs.{rev_saved:,}")
        o2.metric("Intervention Cost", f"Rs.{total_cost:,}")

        net_color = "#5a9e6a" if net > 0 else "#e05555"
        net_label = "Net positive" if net > 0 else "Net negative"
        st.markdown(f"""
        <div class='card' style='text-align: center; margin-top: 8px; border-top: 2px solid {net_color};'>
            <div class='label'>Net Monthly ROI</div>
            <div style='font-size: 2.2rem; font-weight: 700; color: {net_color}; letter-spacing: -0.02em;'>Rs.{net:,}</div>
            <div style='font-size: 12px; color: #5a5a5a; margin-top: 4px;'>{net_label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    vals = [at_risk * avg_rev, rev_saved, total_cost, net]
    colors = ['#2a2a2a', '#3a5a4a', '#5a3a2a', '#5a9e6a' if net >= 0 else '#9e3a3a']
    bars = ax.bar(['Revenue at Risk', 'Revenue Saved', 'Intervention Cost', 'Net ROI'],
                  vals, color=colors, width=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                max(bar.get_height(), 0) + max(vals)*0.01,
                f'Rs.{val:,}', ha='center', va='bottom', fontsize=10)
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    <div class='insight' style='margin-top: 20px;'>
        <div class='insight-title'>How to read this</div>
        <div class='insight-body'>Set churn rate to 26.6% and average revenue to Rs.64.76 to match the dataset baseline. Slide the retention rate between 20% (conservative) and 50% (optimistic) to stress-test the programme economics. Net ROI is positive when saved revenue exceeds intervention cost.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='footer-bar'>
        <div class='footer-text'>Churn Predictor</div>
        <div class='footer-text'>Project by Sharanya Prasad</div>
        <div class='footer-text'>Logistic Regression &nbsp;·&nbsp; ROC-AUC 0.832</div>
    </div>
    """, unsafe_allow_html=True)
