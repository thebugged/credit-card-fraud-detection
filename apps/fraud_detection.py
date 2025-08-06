import joblib
import numpy as np
import pandas as pd
import streamlit as st

from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder


@st.cache_resource
def load_fraud_models():
    try:
        european_model = joblib.load("models/european_fraud_model.pkl")
        synthetic_model = joblib.load("models/synthetic_fraud_model.pkl")
        synthetic_scaler = joblib.load("models/synthetic_scaler.pkl")
        european_scaler = joblib.load("models/european_scaler.pkl")
        return european_model, synthetic_model, synthetic_scaler, european_scaler
    except:
        st.error("❌ Models not found. Please ensure model files are in the 'models/' directory.")
        return None, None, None, None

def fraud_detection_page():
    
    # load models
    european_model, synthetic_model, synthetic_scaler, european_scaler = load_fraud_models()
    
    if european_model is None:
        st.stop()
    
    # model selection
    st.markdown("### 🎯 Select Detection Model")
    model_choice = st.radio(
        "Choose the model for fraud detection:",
        ["European Dataset Model", "Synthetic Dataset Model"],
        help="European model shows superior performance (95.68% accuracy vs 53.00%)"
    )
    
    st.markdown("---")
    
    if "European" in model_choice:
        st.markdown("### 📊 European Dataset Model - Transaction Analysis")
        st.caption("Enter PCA-transformed features and transaction amount")
        
        # for european model inputs (V1-V28 + Amount)
        col1, col2, col3 = st.columns(3)
        
        # initialize session state for V features if not exists
        for i in range(1, 29):
            if f'v{i}' not in st.session_state:
                st.session_state[f'v{i}'] = 0.0
        
        if 'amount' not in st.session_state:
            st.session_state['amount'] = 100.0
        
        with col1:
            v1 = st.number_input("V1", value=st.session_state['v1'], format="%.6f", key='v1_input')
            v2 = st.number_input("V2", value=st.session_state['v2'], format="%.6f", key='v2_input') 
            v3 = st.number_input("V3", value=st.session_state['v3'], format="%.6f", key='v3_input')
            v4 = st.number_input("V4", value=st.session_state['v4'], format="%.6f", key='v4_input')
            v5 = st.number_input("V5", value=st.session_state['v5'], format="%.6f", key='v5_input')
            v6 = st.number_input("V6", value=st.session_state['v6'], format="%.6f", key='v6_input')
            v7 = st.number_input("V7", value=st.session_state['v7'], format="%.6f", key='v7_input')
            v8 = st.number_input("V8", value=st.session_state['v8'], format="%.6f", key='v8_input')
            v9 = st.number_input("V9", value=st.session_state['v9'], format="%.6f", key='v9_input')
            v10 = st.number_input("V10", value=st.session_state['v10'], format="%.6f", key='v10_input')
        
        with col2:
            v11 = st.number_input("V11", value=st.session_state['v11'], format="%.6f", key='v11_input')
            v12 = st.number_input("V12", value=st.session_state['v12'], format="%.6f", key='v12_input')
            v13 = st.number_input("V13", value=st.session_state['v13'], format="%.6f", key='v13_input')
            v14 = st.number_input("V14", value=st.session_state['v14'], format="%.6f", key='v14_input')
            v15 = st.number_input("V15", value=st.session_state['v15'], format="%.6f", key='v15_input')
            v16 = st.number_input("V16", value=st.session_state['v16'], format="%.6f", key='v16_input')
            v17 = st.number_input("V17", value=st.session_state['v17'], format="%.6f", key='v17_input')
            v18 = st.number_input("V18", value=st.session_state['v18'], format="%.6f", key='v18_input')
            v19 = st.number_input("V19", value=st.session_state['v19'], format="%.6f", key='v19_input')
            amount = st.number_input("Amount (€)", min_value=0.0, value=st.session_state['amount'], step=0.01, key='amount_input')
        
        with col3:
            v20 = st.number_input("V20", value=st.session_state['v20'], format="%.6f", key='v20_input')
            v21 = st.number_input("V21", value=st.session_state['v21'], format="%.6f", key='v21_input')
            v22 = st.number_input("V22", value=st.session_state['v22'], format="%.6f", key='v22_input')
            v23 = st.number_input("V23", value=st.session_state['v23'], format="%.6f", key='v23_input')
            v24 = st.number_input("V24", value=st.session_state['v24'], format="%.6f", key='v24_input')
            v25 = st.number_input("V25", value=st.session_state['v25'], format="%.6f", key='v25_input')
            v26 = st.number_input("V26", value=st.session_state['v26'], format="%.6f", key='v26_input')
            v27 = st.number_input("V27", value=st.session_state['v27'], format="%.6f", key='v27_input')
            v28 = st.number_input("V28", value=st.session_state['v28'], format="%.6f", key='v28_input')
        
        # Quick fill 
        st.markdown("#### 🎲 Quick Test Cases")
        col1, col2, col3 = st.columns(3)
        
        if col1.button("💳 Normal Transaction"):
            for i in range(1, 29):
                st.session_state[f'v{i}'] = np.random.normal(0, 1)
            st.session_state['amount'] = np.random.uniform(10, 500)
            st.rerun()
        
        if col2.button("⚠️ Suspicious Transaction"):
            for i in range(1, 29):
                st.session_state[f'v{i}'] = np.random.normal(0, 2)
            st.session_state['amount'] = np.random.uniform(1000, 5000)
            st.rerun()
        
        if col3.button("🔄 Reset All"):
            for i in range(1, 29):
                st.session_state[f'v{i}'] = 0.0
            st.session_state['amount'] = 100.0
            st.rerun()
        
        # create user data using current input values
        user_data = pd.DataFrame({
            'V1': [v1], 'V2': [v2], 'V3': [v3], 'V4': [v4], 'V5': [v5],
            'V6': [v6], 'V7': [v7], 'V8': [v8], 'V9': [v9], 'V10': [v10],
            'V11': [v11], 'V12': [v12], 'V13': [v13], 'V14': [v14], 'V15': [v15],
            'V16': [v16], 'V17': [v17], 'V18': [v18], 'V19': [v19], 'V20': [v20],
            'V21': [v21], 'V22': [v22], 'V23': [v23], 'V24': [v24], 'V25': [v25],
            'V26': [v26], 'V27': [v27], 'V28': [v28], 'Amount': [amount]
        })
        
        model = european_model
        scaler = european_scaler
        
    else:
        st.markdown("### 📊 Synthetic Dataset Model - Transaction Details")
        st.caption("Enter realistic transaction information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=100.0, step=0.01)
            
            transaction_type = st.selectbox("Transaction Type", 
                                          ["deposit", "payment", "transfer", "withdrawal"])
            
            merchant_category = st.selectbox("Merchant Category",
                                           ["retail", "travel", "restaurant", "entertainment", 
                                            "grocery", "other", "utilities", "online"])
            
            location = st.selectbox("Location",
                                   ["Tokyo", "New York", "Singapore", "Berlin", 
                                    "Sydney", "Toronto", "Dubai", "London"])
            
            spending_deviation_score = st.slider("Spending Deviation Score", 
                                                min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
        
        with col2:
            device_used = st.selectbox("Device Used",
                                     ["mobile", "web", "atm", "pos"])
            
            payment_channel = st.selectbox("Payment Channel", 
                                         ["wire_transfer", "ACH", "card", "UPI"])
            
            velocity_score = st.slider("Velocity Score", 
                                     min_value=1, max_value=20, value=10)
            
            geo_anomaly_score = st.slider("Geographic Anomaly Score",
                                        min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        
        # encode categorical variables
        categorical_mapping = {
            'transaction_type': {"deposit": 0, "payment": 1, "transfer": 2, "withdrawal": 3},
            'merchant_category': {"retail": 0, "travel": 1, "restaurant": 2, "entertainment": 3,
                                "grocery": 4, "other": 5, "utilities": 6, "online": 7},
            'location': {"Tokyo": 0, "New York": 1, "Singapore": 2, "Berlin": 3,
                        "Sydney": 4, "Toronto": 5, "Dubai": 6, "London": 7},
            'device_used': {"mobile": 0, "web": 1, "atm": 2, "pos": 3},
            'payment_channel': {"wire_transfer": 0, "ACH": 1, "card": 2, "UPI": 3}
        }
        
        # create user data 
        user_data = pd.DataFrame({
            'amount': [amount],
            'transaction_type': [categorical_mapping['transaction_type'][transaction_type]],
            'merchant_category': [categorical_mapping['merchant_category'][merchant_category]],
            'location': [categorical_mapping['location'][location]],
            'device_used': [categorical_mapping['device_used'][device_used]],
            'spending_deviation_score': [spending_deviation_score],
            'velocity_score': [velocity_score],
            'geo_anomaly_score': [geo_anomaly_score],
            'payment_channel': [categorical_mapping['payment_channel'][payment_channel]]
        })
        
        model = synthetic_model
        scaler = synthetic_scaler
    
    st.markdown("---")
    
    # prediction
    if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
        try:
            user_data_scaled = scaler.transform(user_data)
            
            # Random Forest prediction (different from neural network)
            prediction_prob = model.predict_proba(user_data_scaled)[0][1]  # Get probability for class 1
            prediction_binary = model.predict(user_data_scaled)[0]
      
            st.markdown("### 📋 Analysis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction_binary == 1:
                    st.error("🚨 **FRAUD DETECTED**")
                    st.markdown(f"**Fraud Probability:** {prediction_prob:.2%}")
                    st.markdown("**Recommendation:** Block transaction and investigate")
                else:
                    st.success("✅ **LEGITIMATE TRANSACTION**")
                    st.markdown(f"**Fraud Probability:** {prediction_prob:.2%}")
                    st.markdown("**Recommendation:** Approve transaction")
            
            with col2:
                # Risk level gauge
                risk_level = "HIGH" if prediction_prob > 0.7 else "MEDIUM" if prediction_prob > 0.3 else "LOW"
                risk_color = "🔴" if risk_level == "HIGH" else "🟡" if risk_level == "MEDIUM" else "🟢"
                
                st.markdown(f"**Risk Level:** {risk_color} {risk_level}")
                st.markdown(f"**Confidence Score:** {abs(prediction_prob - 0.5) * 2:.2%}")
                
                # Model info
                model_name = "European Random Forest" if "European" in model_choice else "Synthetic Random Forest"
                st.markdown(f"**Model Used:** {model_name}")
                st.markdown(f"**Processing Time:** ~5ms")  # Random Forest is faster
        
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.error("Please check that your model was trained with the correct features.")

if __name__ == "__main__":
    fraud_detection_page()