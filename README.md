<div align="center">
  <br />
    <a href="#" target="_blank">
      <img src="https://github.com/user-attachments/assets/fraud-guard-banner" alt="Fraud Guard Banner">
    </a>
  <br />

  <div>
    <img src="https://img.shields.io/badge/-Python-black?style=for-the-badge&logoColor=white&logo=python&color=3776AB" alt="python" />
    <img src="https://img.shields.io/badge/-TensorFlow-black?style=for-the-badge&logoColor=white&logo=tensorflow&color=FF6F00" alt="tensorflow" />
    <img src="https://img.shields.io/badge/-scikit_learn-black?style=for-the-badge&logoColor=white&logo=scikitlearn&color=F7931E" alt="scikit-learn" />
    <img src="https://img.shields.io/badge/-Streamlit-black?style=for-the-badge&logoColor=white&logo=streamlit&color=FF4B4B" alt="streamlit" />
    <img src="https://img.shields.io/badge/-Azure-black?style=for-the-badge&logoColor=white&logo=microsoftazure&color=0078D4" alt="azure" />
  </div>

  <h3 align="center">🛡️ Fraud Guard - Credit Card Fraud Detection</h3>

   <div align="center">
     Advanced machine learning system for real-time credit card fraud detection and prevention using neural networks and cloud-native architecture.
    </div>
</div>
<br/>

**Datasets** 🗃️
- [European Credit Card Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023) - Real anonymized European cardholder transactions
- [Financial Transactions Dataset](https://www.kaggle.com/datasets/aryan208/financial-transactions-dataset-for-fraud-detection) - Synthetic financial transaction data
- Processed datasets with 1:2 fraud-to-legitimate ratio for balanced training

<!-- **Key Features** ✨
- **Dual Model Architecture**: European (95.68% accuracy) and Synthetic dataset models
- **Real-time Detection**: Sub-50ms transaction processing
- **Interactive Dashboard**: Streamlit-powered web interface
- **Advanced Analytics**: Comprehensive fraud pattern analysis
- **Cloud Integration**: Microsoft Azure ML Studio compatible -->

## Setup & Installation

**Prerequisites**

Ensure the following are installed:
- [Git](https://git-scm.com/)
- [Python 3.8+](https://www.python.org/downloads/)
- [Jupyter Notebook](https://jupyter.org/install) (or install the Jupyter extension on [Visual Studio Code](https://code.visualstudio.com/))
  
To set up this project locally, follow these steps:

1. Clone the repository:
```shell
git clone https://github.com/thebugged/credit-card-fraud.git
```

2. Change into the project directory: 
```shell
cd credit-card-fraud
```

3. Create a virtual environment (recommended):
```shell
python -m venv fraud_env
source fraud_env/bin/activate  # On Windows: fraud_env\Scripts\activate
```

4. Install the required dependencies: 
```shell
pip install -r requirements.txt
```

5. Create the models directory and add your trained models:
```shell
mkdir models
# Place your trained model files:
# - european_fraud_model.h5
# - synthetic_fraud_model.h5
# - european_scaler.pkl
# - synthetic_scaler.pkl
```

<br/>

## Running the Application

1. Run the Streamlit application: 
```shell
streamlit run main.py
```

2. Alternatively, you can train the models first by running the notebooks:
   - `european_model_training.ipynb` - Train the European dataset model
   - `synthetic_model_training.ipynb` - Train the synthetic dataset model
   - Then run the command in step 1

The application will be available in your browser at http://localhost:8501.

## Model Performance 📊

| Dataset | Accuracy | Precision | Recall | AUC-ROC |
|---------|----------|-----------|---------|---------|
| European Model | 95.68% | 97.56% | 93.71% | 99.08% |
| Synthetic Model | 53.00% | 33.37% | 41.11% | 49.93% |

## Project Structure 📁

```
fraud-guard-detection/
├── main.py                 # Main Streamlit application
├── apps/                   # Application pages
│   ├── home.py            # Home dashboard
│   ├── fraud_detection.py # Fraud detection interface
│   └── resources.py       # Documentation and resources
├── models/                # Trained models and scalers
├── data/                  # Dataset files
├── notebooks/             # Jupyter notebooks for training
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

<!-- ## Academic Project 🎓

**Team:** Group 4
- Cecil Oiku
- Katelyn Siu  
- Israel Maikyau
- Meet Patel -->


## License & Disclaimer ⚠️

This project is developed for educational and research purposes. It should not be used for actual financial decision-making without proper validation, compliance measures, and regulatory approval.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](your-app-url-here)

