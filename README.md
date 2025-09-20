# Real-Time Transaction Anomaly Detection System 🛡️

This project is a solution for **Track 3: Real-Time Cybersecurity Anomaly Detection**. It implements a high-performance machine learning model to detect fraudulent or anomalous transactions in real-time, simulating a system that could identify cyber threats like unusual logins or data exfiltration.

## 📝 Problem Statement

> Cyberattacks don’t wait — suspicious logins, data exfiltration, or malware can spread in seconds. We need a system that can **spot anomalies in security data in real-time** and **explain them clearly**, so analysts know **why** something looks suspicious. This project serves as the core detection engine for such a system.

## ✨ Key Features

* **Advanced Feature Engineering**: Creates insightful features from raw transaction data, such as user's behavioral baselines (average location, transaction amount) and time-based patterns.
* **High-Performance XGBoost Model**: Utilizes a powerful Gradient Boosting classifier, optimized with GridSearchCV for maximum accuracy.
* **Handles Class Imbalance**: Implements SMOTE (Synthetic Minority Over-sampling Technique) to effectively train the model on highly imbalanced anomaly data.
* **Robust Evaluation**: The model is rigorously tested, showing a high ROC AUC score and excellent recall for detecting anomalies.
* **Ready for Deployment**: The trained model and data scaler are saved, allowing for quick and efficient predictions on new, incoming data streams.

## ⚙️ How It Works: The Machine Learning Pipeline

The project is divided into two main stages: training the detection model and using it for prediction on new data.

### 1. Model Training (`main.ipynb`)

1.  **Data Ingestion**: Loads the historical dataset (`transaction_dataset.csv`).
2.  **Feature Engineering**:
    * **Time-based Features**: Extracts the `hour_of_day` and `day_of_week` to capture temporal patterns.
    * **Behavioral Baselining**: For each user, it calculates their average transaction `amount`, `latitude`, and `longitude`. This creates a "normal behavior" profile.
    * **Spatial Anomaly Feature**: A `distance` feature is engineered to measure how far a new transaction is from the user's typical location. A large distance is a strong indicator of an anomaly (e.g., a login from a different country).
3.  **Handling Imbalance**: The dataset is highly imbalanced (few anomalies vs. many normal transactions). **SMOTE** is used to create synthetic examples of the minority class (anomalies), ensuring the model learns to identify them effectively.
4.  **Model Training**: An **XGBoost Classifier** is trained on the preprocessed data. **GridSearchCV** is used to systematically find the best hyperparameters, resulting in a highly optimized model.
5.  **Saving the Model**: The best-performing model (`xgb_model.pkl`) and the feature scaler (`scaler.pkl`) are saved to disk using `joblib` for later use.

### 2. Real-Time Prediction (`predict.py`)

This script simulates how the system would operate in a live environment.

1.  **Load Model**: The pre-trained model and scaler are loaded from the disk.
2.  **Process New Data**: It ingests a new data stream (simulated by `new_transactions.csv`).
3.  **Apply Transformations**: The exact same feature engineering and scaling steps from the training phase are applied to the new data.
4.  **Detect Anomalies**: The model predicts which of the new transactions are anomalous.
5.  **Evaluate Performance**: The predictions are evaluated to demonstrate the model's effectiveness on unseen data.

## 📊 Model Performance

The model achieves excellent results, particularly in its ability to correctly identify anomalies (high recall).

### Training & Validation Results

* **ROC AUC Score**: **0.9996**
* **Accuracy**: **99%**



### Performance on New, Unseen Data (`new_transactions.csv`)

* **ROC AUC Score**: **0.9683**
* **Accuracy**: **84%**
* **Classification Report**:
    | | precision | recall | f1-score | support |
    | :--- | :---: | :---: | :---: | :---: |
    | **False (Normal)** | 0.99 | 0.82 | 0.90 | 85062 |
    | **True (Anomaly)** | 0.48 | **0.97** | 0.65 | 14938 |

The key metric here is the **recall of 0.97 for anomalies**, meaning the model successfully caught 97% of all true anomalies in the new dataset.



## 💻 Technology Stack

* **Python 3.12**
* **Scikit-learn**: For data processing and model evaluation.
* **XGBoost**: For the core classification model.
* **Pandas**: For data manipulation.
* **imbalanced-learn**: For handling class imbalance with SMOTE.
* **Matplotlib & Seaborn**: For data visualization.
* **Joblib**: For saving and loading the model.

## 🚀 Setup and Usage

Follow these steps to run the project locally.

### 1. Prerequisites

* Python 3.8+
* pip package manager

### 2. Installation

First, clone the repository and create a virtual environment:

```bash
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

Install the required packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

*(You will need to create a `requirements.txt` file with the following content:)*
```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
joblib
ipykernel
```

### 3. Running the Project

**Step 1: Train the Model**

Run the Jupyter Notebook `main.ipynb`. This will process the `transaction_dataset.csv`, train the model, and save `xgb_model.pkl` and `scaler.pkl`.

**Step 2: Make Predictions on New Data**

Execute the prediction script. This will use the saved model to detect anomalies in `new_transactions.csv`.

```bash
python predict.py
```

You will see the evaluation metrics for the new dataset printed to the console.

## 📁 Project Structure

```
.
├── data
│   ├── transaction_dataset.csv   # Dataset for training
│   └── new_transactions.csv      # Dataset for prediction
├── main.ipynb                      # Notebook for model training and experimentation
├── predict.py                      # Script for making predictions on new data
├── xgb_model.pkl                   # Saved XGBoost model
├── scaler.pkl                      # Saved data scaler
├── requirements.txt                # Python dependencies
└── Readme.md                       # You are here!
```

## 🔮 Future Work

* **Real-Time Data Ingestion**: Replace the CSV reader with a real-time stream consumer like **Kafka** or a **REST API endpoint**.
* **LLM-Powered Explanations**: For each detected anomaly, send the feature data to a Large Language Model (LLM) to generate a human-readable alert, as suggested in the hackathon prompt (e.g., *“Unusual login detected: transaction amount is 50x higher than user's average and occurred 2000km from their normal location.”*).
* **Dashboarding & Alerting**: Create a live dashboard (using Streamlit or Dash) to visualize anomalies as they are detected and push alerts to platforms like Slack or Discord.
