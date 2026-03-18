import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------
# Page configuration
# ------------------------------
st.set_page_config(
    page_title="E-Commerce Churn Predictor",
    page_icon="🛒",
    layout="centered"
)

# ------------------------------
# Load model
# ------------------------------
@st.cache_resource
def load_model():
    return joblib.load("Buwule_best_model.joblib")

model = load_model()

# ------------------------------
# Scaling parameters
# ------------------------------
scaling_params = {
    'recency_days'            : {'mean': 239.74, 'std': 152.62},
    'monetary_value'          : {'mean': 111.30, 'std':  80.68},
    'avg_delivery_delay'      : {'mean': -11.72, 'std':   9.76},
    'payment_type_encoded'    : {'mean':   0.29, 'std':   0.58},
    'category_encoded'        : {'mean':   9.62, 'std':   5.91},
    'customer_state_encoded'  : {'mean':  19.09, 'std':   6.95}
}

def manual_scale(value, feature):
    mean = scaling_params[feature]['mean']
    std  = scaling_params[feature]['std']
    return (value - mean) / std

# ------------------------------
# State encoding
# ------------------------------
state_encoding = {
    'AC': 0,  'AL': 1,  'AM': 2,  'AP': 3, 'BA': 4, 'CE': 5, 'DF': 6, 'ES': 7,
    'GO': 8,  'MA': 9,  'MG': 10, 'MS': 11, 'MT': 12, 'PA': 13, 'PB': 14,
    'PE': 15, 'PI': 16, 'PR': 17, 'RJ': 18, 'RN': 19, 'RO': 20, 'RR': 21,
    'RS': 22, 'SC': 23, 'SE': 24, 'SP': 25, 'TO': 26
}

state_names = {k: f"{k} — {v}" for k,v in {
    'AC':'Acre','AL':'Alagoas','AM':'Amazonas','AP':'Amapá','BA':'Bahia',
    'CE':'Ceará','DF':'Distrito Federal','ES':'Espírito Santo','GO':'Goiás',
    'MA':'Maranhão','MG':'Minas Gerais','MS':'Mato Grosso do Sul','MT':'Mato Grosso',
    'PA':'Pará','PB':'Paraíba','PE':'Pernambuco','PI':'Piauí','PR':'Paraná',
    'RJ':'Rio de Janeiro','RN':'Rio Grande do Norte','RO':'Rondônia','RR':'Roraima',
    'RS':'Rio Grande do Sul','SC':'Santa Catarina','SE':'Sergipe','SP':'São Paulo',
    'TO':'Tocantins'
}.items()}

# ------------------------------
# CSS for styling
# ------------------------------
st.markdown("""
<style>
    .main {
        background-color: #f0f4f8;
    }
    .rounded-container {
        background: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        color: white;
    }
    .header {
        text-align: center;
        color: #333;
    }
    .subheader {
        color: #555;
    }
    button {
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        cursor: pointer;
    }
    button:hover {
        background-color: #0056b3;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Header
# ------------------------------
st.markdown("<h1 class='header'>🛒 E-Commerce Customer Churn Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Predict if a customer will <strong>repeat purchase</strong> or <strong>churn</strong>.</p>", unsafe_allow_html=True)

# ------------------------------
# Input form in a rounded container
# ------------------------------
st.markdown("<div class='rounded-container'>", unsafe_allow_html=True)
st.subheader("Customer Profile")

col1, col2 = st.columns(2)
with col1:
    recency_days = st.number_input("Days since last purchase", 1, 730, 90)
    monetary_value = st.number_input("Total amount spent (BRL)", 0.0, 2000.0, 150.0, step=10.0)
    avg_delivery_delay = st.number_input("Average delivery delay (days)", -30, 60, -5)

with col2:
    payment_type = st.selectbox("Payment method", ["Credit card", "Boleto", "Voucher", "Debit card"])
    payment_map = {"Credit card":0, "Boleto":1, "Voucher":2, "Debit card":3}
    payment_type_encoded = payment_map[payment_type]

    category = st.selectbox("Primary product category", ["Electronics","Fashion","Home & Garden","Sports","Health & Beauty","Other"])
    category_map = {"Electronics":0,"Fashion":1,"Home & Garden":2,"Sports":3,"Health & Beauty":4,"Other":5}
    category_encoded = category_map[category]

    selected_state_label = st.selectbox("Customer state", options=list(state_names.values()), index=list(state_names.keys()).index('SP'))
    selected_state_code = selected_state_label.split(' — ')[0]
    customer_state_encoded  = state_encoding[selected_state_code]
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------
# Predict button
# ------------------------------
predict_clicked = st.button("Predict Churn Risk")
if predict_clicked:
    input_scaled = pd.DataFrame([{
        'recency_days'           : manual_scale(recency_days, 'recency_days'),
        'monetary_value'         : manual_scale(monetary_value, 'monetary_value'),
        'avg_delivery_delay'     : manual_scale(avg_delivery_delay, 'avg_delivery_delay'),
        'payment_type_encoded'   : manual_scale(payment_type_encoded, 'payment_type_encoded'),
        'category_encoded'       : manual_scale(category_encoded, 'category_encoded'),
        'customer_state_encoded' : manual_scale(customer_state_encoded, 'customer_state_encoded')
    }])
    feature_cols = ['recency_days','monetary_value','avg_delivery_delay','payment_type_encoded','category_encoded','customer_state_encoded']

    prediction = model.predict(input_scaled[feature_cols])[0]
    probability = model.predict_proba(input_scaled[feature_cols])[0][1]
    churn_prob  = 1 - probability

    # ------------------------------
    # Metrics cards with dynamic colors
    # ------------------------------
    st.subheader("Prediction Result")
    col1, col2, col3 = st.columns(3)

    def metric_card(label, value, color):
        st.markdown(f"""
        <div class='metric-card' style='background:{color}'>
        <h4 style='margin:5px'>{label}</h4>
        <h2 style='margin:5px'>{value}</h2>
        </div>
        """, unsafe_allow_html=True)

    repeat_color = "#28a745"  # green
    churn_color  = "#dc3545"  # red
    risk_color   = "#ffc107" if probability<0.6 and probability>=0.3 else "#28a745" if probability>=0.6 else "#dc3545"

    with col1:
        metric_card("Repeat Buy Probability", f"{probability*100:.1f}%", repeat_color)
    with col2:
        metric_card("Churn Probability", f"{churn_prob*100:.1f}%", churn_color)
    with col3:
        risk_text = "Low Risk" if probability>=0.6 else "Medium Risk" if probability>=0.3 else "High Risk"
        metric_card("Churn Risk Level", risk_text, risk_color)

    # ------------------------------
    # Verdict
    # ------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    if prediction == 1:
        st.success(f"✅ Customer in **{selected_state_code}** is likely to make a repeat purchase.")
    else:
        st.error(f"⚠️ Customer in **{selected_state_code}** is at risk of churning.")

    # ------------------------------
    # Recommendations
    # ------------------------------
    st.subheader("Recommended Action")
    if churn_prob >= 0.7:
        st.warning("🎯 High Priority — Send personalized retention offer immediately.")
    elif churn_prob >= 0.4:
        st.info("📧 Medium Priority — Include in next email campaign.")
    else:
        st.success("👍 Low Priority — Standard follow-up sufficient.")

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("B31373 | BUWULE RICHARD | J25M19/023")
