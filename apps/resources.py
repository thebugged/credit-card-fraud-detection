import streamlit as st

def resources_page():
    
    # Project Information
    st.markdown("### 🎓 Project Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Project Title:** Credit Card Fraud Detection
        
        **Team:** Group 4
        - Cecil Oiku
        - Katelyn Siu  
        - Israel Maikyau
        - Meet Patel
        
        **Institution:** University of Calgary
        
        **Course:** DATA 695
        """)
    
    with col2:
        st.markdown("""
        **Technologies Used:**
        - Python & TensorFlow/Keras
        - Microsoft Azure ML Studio
        - Streamlit Web Framework
        - Scikit-learn & Pandas
        - Power BI Visualizations
        
        **Datasets:**
        - [European Credit Card Dataset (Real)](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)
        - [Financial Transactions Dataset (Synthetic)](https://www.kaggle.com/datasets/aryan208/financial-transactions-dataset-for-fraud-detection/data)
        """)
    
    st.markdown("---")
    
    # Technical Documentation
    st.markdown("### 🔧 Technical Documentation")
    
    tab1, tab2, tab3 = st.tabs(["Model Architecture", "Dataset Information", "Performance Metrics"])
    
    with tab1:
        st.markdown("""
        #### Random Forest Architecture
    
        **European Dataset Model:**
        - Input Layer: 29 features (V1-V28 + Amount)
        - Algorithm: Random Forest Classifier
        - Trees: 100 estimators
        - Max Depth: 10
        - Min Samples Split: 5
        - Feature Selection: All features used
        
        **Synthetic Dataset Model:**
        - Input Layer: 9 features (transaction details + engineered features)
        - Algorithm: Random Forest Classifier
        - Trees: 100 estimators
        - Max Depth: 10
        - Preprocessing: SMOTE balancing
        """)
    
    with tab2:
        st.markdown("""
        #### Dataset Characteristics
        
        **European Credit Card Dataset:**
        - Source: Real European cardholder transactions (2023)
        - Size: 568,630 transactions
        - Features: 29 (V1-V28 PCA features + Amount)
        - Class Distribution: 50-50 balanced (artificially)
        - Quality: High-quality, preprocessed data
        
        **Synthetic Financial Dataset:**
        - Source: Generated financial transaction data
        - Size: 538,659 transactions (after cleaning)
        - Features: 12 (transaction details + risk scores)
        - Class Distribution: 1:2 fraud to legitimate ratio
        - Challenges: Limited predictive signals
        """)
    
    with tab3:
        st.markdown("""
        #### Model Performance Comparison

        **Synthetic Dataset Results:**

        | Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
        |-------|----------|-----------|---------|----------|---------|
        | Neural Network | 52.51% | 33.04% | 41.38% | 36.74% | 49.81% |
        | Random Forest | 53.75% | 33.23% | 38.28% | 35.58% | 49.83% |
        | Logistic Regression | 49.96% | 33.38% | 50.33% | 40.14% | 49.91% |
        | SVM | 49.33% | 33.00% | 50.00% | 39.58% | 50.00% |
        | CatBoost | 49.98% | 34.37% | 54.88% | 42.27% | 51.61% |
        """)

        st.markdown("") 
        st.markdown("")  

        st.markdown("""
        **European Dataset Results:**

        | Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
        |-------|----------|-----------|---------|----------|---------|
        | Neural Network | 95.44% | 97.18% | 93.59% | 95.35% | 99.13% |
        | Random Forest | 98.59% | 99.87% | 97.32% | 98.58% | 99.95% |
        | Logistic Regression | 99.83% | 99.91% | 99.76% | 99.83% | 99.98% |
        | SVM | 99.80% | 99.85% | 98.50% | 99.17% | 99.90% |
        | CatBoost | 98.50% | 98.20% | 97.00% | 97.60% | 99.50% |
        """)

        st.markdown("")
        st.markdown("")
                                    
        st.markdown("""
                **Key Insights:**
                - European model shows excellent performance across all metrics
                - Synthetic model performs at random chance level
                - Real-world PCA features significantly outperform synthetic features
                """)
        

    st.markdown("---")


    # Research Papers & References
    st.markdown("### 📖 Research Papers & References")
    
    papers_data = [
        {
            "title": "Big data-driven distributed machine learning for scalable credit card fraud detection using PySpark, XGBoost, and CatBoost",
            "authors": "Theodorakopoulos, L., Theodoropoulou, A., Tsimakis, A., & Halkiopoulos, C.",
            "journal": "Electronics, 14(9), 1754",
            "year": "2025",
            "link": "https://doi.org/10.3390/electronics14091754"
        },
        {
            "title": "Credit card fraud detection: A deep learning approach",
            "authors": "Sharma, A., & Kumar, S.",
            "journal": "arXiv preprint arXiv:2409.13406v1",
            "year": "2024",
            "link": "https://arxiv.org/abs/2409.13406v1"
        },
        {
            "title": "Credit Card Fraud Detection Using CatBoost",
            "authors": "Raza, M. A., Ullah, S., Naqvi, H. S. H., & Ahsan, M.",
            "journal": "ResearchGate",
            "year": "2024",
            "link": "https://www.researchgate.net/publication/391708117_Credit_Card_Fraud_Detection_Using_CatBoost"
        },
        {
            "title": "A supervised machine learning algorithm for detecting and predicting fraud in credit card transactions",
            "authors": "Awoyemi, J. O., Adetunmbi, A. O., & Oluwadare, S. A.",
            "journal": "Machine Learning with Applications, 9, 100359",
            "year": "2023", 
            "link": "https://doi.org/10.1016/j.mlwa.2022.100359"
        },
        {
            "title": "Credit card fraud detection in the era of disruptive technologies: A systematic review",
            "authors": "Cherif, A., Badhib, A., Ammar, H., Alshehri, S., Kalkatawi, M., & Imine, A.",
            "journal": "Journal of King Saud University - Computer and Information Sciences, 35(1), 145-174",
            "year": "2023",
            "link": "https://doi.org/10.1016/j.jksuci.2022.11.008"
        },
        {
            "title": "Enhanced credit card fraud detection model using machine learning",
            "authors": "Al-Hashemi, R. R., & Al-Mosawi, A. I.",
            "journal": "Electronics, 11(4), 662",
            "year": "2022",
            "link": "https://doi.org/10.3390/electronics11040662"
        },
        {
            "title": "Enhanced credit card fraud detection based on attention mechanism and LSTM deep model", 
            "authors": "Benchaji, I., Douzi, S., & El Ouahidi, B.",
            "journal": "Journal of Big Data, 8(1), 151",
            "year": "2021",
            "link": "https://doi.org/10.1186/s40537-021-00541-8"
        },
        {
            "title": "Credit card fraud detection using machine learning: A survey",
            "authors": "Lucas, Y., & Jurgovsky, J.",
            "journal": "arXiv preprint arXiv:2010.06479",
            "year": "2020",
            "link": "https://arxiv.org/abs/2010.06479"
        },
        {
            "title": "Enhanced credit card fraud detection based on SVM-recursive feature elimination and hyper-parameters optimization",
            "authors": "Rtayli, N., & Enneya, N.",
            "journal": "Journal of Information Security and Applications, 55, 102596",
            "year": "2020",
            "link": "https://doi.org/10.1016/j.jisa.2020.102596"
        }
    ]
    
    for paper in papers_data:
        with st.expander(f"📄 {paper['title']} ({paper['year']})"):
            st.markdown(f"**Authors:** {paper['authors']}")
            st.markdown(f"**Published in:** {paper['journal']}")
            st.markdown(f"**Year:** {paper['year']}")
            st.markdown(f"**Link:** [{paper['link']}]({paper['link']})")
    
    
    
    
    
    st.markdown("---")
    
    # GitHub Repository
    st.markdown("### 💻 Code Repository")
    
    st.info("""
    **GitHub Repository: https://github.com/thebugged/data695-credit-card-fraud**
    
    The complete source code, datasets, trained models, and documentation are available in our GitHub repository.
    This includes:
    - Data preprocessing scripts
    - Model training notebooks  
    - Evaluation metrics and visualizations
    - Streamlit application code
    """)

if __name__ == "__main__":
    resources_page()