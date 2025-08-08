<div align="center">
  <br />
    <a href="#" target="_blank">
      <img src="https://github.com/user-attachments/assets/b30687ae-afb2-4f35-aa0d-9ea922809394" alt="Fraud Guard Banner">
    </a>
  <br />

  <div>
    <img src="https://img.shields.io/badge/-Python-black?style=for-the-badge&logoColor=white&logo=python&color=3776AB" alt="python" />
    <img src="https://img.shields.io/badge/-TensorFlow-black?style=for-the-badge&logoColor=white&logo=tensorflow&color=FF6F00" alt="tensorflow" />
    <img src="https://img.shields.io/badge/-scikit_learn-black?style=for-the-badge&logoColor=white&logo=scikitlearn&color=F7931E" alt="scikit-learn" />
    <img src="https://img.shields.io/badge/-Streamlit-black?style=for-the-badge&logoColor=white&logo=streamlit&color=FF4B4B" alt="streamlit" />
    </div>

  <h3 align="center">Fraud Guard - Credit Card Fraud Detection</h3>

   <div align="center">
     Advanced machine learning system for real-time credit card fraud detection and prevention using neural networks and cloud-native architecture.
    </div>
</div>

<br/>

**Datasets** 🗃️
- [European Credit Card Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023) - Real anonymized European cardholder transactions
- [Financial Transactions Dataset](https://www.kaggle.com/datasets/aryan208/financial-transactions-dataset-for-fraud-detection) - Synthetic financial transaction data
- Processed datasets with 1:2 fraud-to-legitimate ratio for balanced training

<br/>

## Setup & Installation

**Prerequisites**

Ensure the following are installed:
- [Git](https://git-scm.com/)
- [Python 3.8+](https://www.python.org/downloads/)
- [Jupyter Notebook](https://jupyter.org/install) (or install the Jupyter extension on [Visual Studio Code](https://code.visualstudio.com/))
  
To set up this project locally, follow these steps:

1. Clone the repository:
```shell
git clone https://github.com/thebugged/credit-card-fraud-detection
```

2. Change into the project directory: 
```shell
cd credit-card-fraud-detection
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
# - european_fraud_model.pkl
# - synthetic_fraud_model.pkl
# - european_scaler.pkl
# - synthetic_scaler.pkl
```

<br/>

## Running the Application

1. Run the Streamlit application: 
```shell
streamlit run main.py
```

2. Alternatively, you can train the models first by running this notebook `main_modelling.ipynb`. Then run the command in step 1.

The application will be available in your browser at http://localhost:8501.

<br/>


## App Structure

```
fraud-guard-detection/
├── main.py                # Main Streamlit application
├── apps/                  # Application pages
│   ├── home.py            # Home dashboard
│   ├── fraud_detection.py # Fraud detection interface
│   └── resources.py       # Documentation and resources
├── models/                # Trained models and scalers
├── data/                  # Dataset files
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

<br/>


## License & Disclaimer 

This project is developed for educational and research purposes. It should not be used for actual financial decision-making without proper validation, compliance measures, and regulatory approval.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ccfraud-guard.streamlit.app/)

