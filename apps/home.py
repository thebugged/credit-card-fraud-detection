
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


def home_page():
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1>Credit Card Fraud Detection System</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style='text-align: center; font-family: "Arial", sans-serif; font-size: 16px; color: #666;'>
            Advanced machine learning system for real-time credit card fraud detection and prevention
        </div>
        """, 
        unsafe_allow_html=True
    )

    st.markdown("")
    st.markdown("")
    st.markdown("")
    
    # Key Statistics Section
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🎯 Model Accuracy",
            value="95.68%",
            delta="European Dataset"
        )

    with col2:
        st.metric(
            label="⚡ Detection Speed",
            value="< 50ms",
            delta="Real-time processing"
        )

    with col3:
        st.metric(
            label="🔍 Precision Rate",
            value="97.56%",
            delta="Low false alarms"
        )

    with col4:
        st.metric(
            label="📊 Recall Rate", 
            value="93.71%",
            delta="High fraud detection"
        )

        
    st.divider()

    # Main Content
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:

        # Global Fraud Statistics
        st.markdown("<h5 style='text-align: center;'> Global Fraud Statistics</h5>", unsafe_allow_html=True)

        fraud_stats = pd.DataFrame({
            'Year': [2021, 2022, 2023, 2024, 2025, 2026],
            'Fraud_Losses_Billions': [31.0, 35.5, 36.5, 39.5, 40.5, 45.0],
            'Fraud_Attempts_Millions': [140, 155, 162, 175, 185, 200] 
        })

        fig = px.line(fraud_stats, x='Year', y='Fraud_Losses_Billions', 
                    markers=True,
                    color_discrete_sequence=['#d32f2f'])  
                    
        fig.update_layout(
            height=300,
            xaxis_title="Year",
            yaxis_title="Fraud Losses (USD Billions)",
            font=dict(size=11)
        )

        # annotation
        fig.add_annotation(
            x=2025,
            y=42,
            text="Projected to reach<br>$43B by 2026",
            showarrow=True,
            arrowhead=2,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#d32f2f",
            font=dict(size=10)
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<h5 style='text-align: center;'> Model Metrics</h5>", unsafe_allow_html=True)
        
        # Model Metrics
        models_data = pd.DataFrame({
            'Dataset': ['European Model', 'Synthetic Model'],
            'Accuracy': [95.68, 53.00],
            'Precision': [97.56, 33.37],
            'Recall': [93.71, 41.11],
            'AUC-ROC': [99.08, 49.93]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='European Model', x=['Accuracy', 'Precision', 'Recall', 'AUC-ROC'],
                            y=[95.68, 97.56, 93.71, 99.08], marker_color="#3873E5"))
        fig.add_trace(go.Bar(name='Synthetic Model', x=['Accuracy', 'Precision', 'Recall', 'AUC-ROC'],
                            y=[53.00, 33.37, 41.11, 49.93], marker_color="#ff1e0e"))
        
        fig.update_layout(height=300, barmode='group')
        st.plotly_chart(fig, use_container_width=True)


    # Fraud Detection Process
    st.markdown("<h5 style='text-align: center;'> Fraud Detection Process</h5>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; padding: 20px; font-size: 16px;'>
        📝 Data Input → ⚙️ Preprocessing → 🧠 Model Analysis → 📊 Risk Assessment → ✅ Decision
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.divider()

    # System Features
    st.subheader('🎯 System Features')

    feat_col1, feat_col2, feat_col3 = st.columns(3)

    with feat_col1:
        st.markdown('''
        - **Real-time Detection**: Instant fraud assessment for transactions
        - **Multiple Models**: European and Synthetic dataset trained models  
        ''')

    with feat_col2:
        st.markdown('''
        - **High Accuracy**: 95%+ accuracy with low false positive rates
        - **Advanced Analytics**: Comprehensive fraud pattern analysis
        ''')

    with feat_col3:
        st.markdown('''
        - **User-Friendly**: Simple interface for transaction verification
        - **Cloud Integration**: Azure ML Studio compatible
        ''')

    
    st.markdown("")

    # Risk Factors 
    st.subheader('⚠️ Common Fraud Risk Factors')
    
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    
    with risk_col1:
        st.markdown("""
        **🌍 Geographic Patterns**
        - Unusual location activity
        - Cross-border transactions
        - High-risk regions
        """)
    
    with risk_col2:
        st.markdown("""
        **💰 Transaction Patterns**  
        - Unusually large amounts
        - Rapid successive transactions
        - Off-hours activity
        """)
    
    with risk_col3:
        st.markdown("""
        **🔧 Technical Indicators**
        - Device fingerprinting
        - Velocity scoring
        - Behavioral anomalies
        """)

    st.markdown("")
    st.divider()
    
    # Disclaimer
    st.markdown(
        """
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 20px 0;'>
            <h4 style='color: #1f77b4; margin-bottom: 10px;'>📋 Important Notice</h4>
            <p style='margin-bottom: 10px;'><strong>Academic Purpose:</strong> This system is developed for educational and research purposes as part of a machine learning project.</p>
            <p style='margin-bottom: 10px;'><strong>Not for Production:</strong> This demo should not be used for actual financial decision-making without proper validation and compliance measures.</p>
            <p style='margin: 0;'><strong>Data Privacy:</strong> No real financial data should be entered into this system.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    home_page()