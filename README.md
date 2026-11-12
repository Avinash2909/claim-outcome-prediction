# Insurance Claim Outcome Prediction

## Project Overview

This project focuses on predicting the outcome of insurance claims using Machine Learning. Based on details such as the insurance payer, healthcare provider, procedure code, diagnosis code, billed amount and patient age, the model predicts whether a claim will be Paid, Partially Paid or Denied.

The project covers the complete machine learning workflow, starting from generating a synthetic dataset and preparing the data to training a classification model and making predictions through a REST API.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Random Forest Classifier
- FastAPI
- Pydantic
- Joblib
- Matplotlib

## Dataset

Since a dataset was not provided, I created a synthetic dataset containing 400 insurance claim records.

The dataset includes the following columns:

- Payer
- Provider
- CPT Code
- Diagnosis Code
- Billed Amount
- Patient Age
- Claim Status

The target variable contains three categories:

| Claim Status | Records | Percentage |
|---|---:|---:|
| Paid | 280 | 70% |
| Partially_Paid | 80 | 20% |
| Denied | 40 | 10% |

The claim outcomes were generated using a risk-scoring approach based on selected claim characteristics. These are simulated assumptions and do not represent actual insurance company policies.

## Data Preparation and Feature Engineering

The dataset was divided into training and testing sets using an 80:20 stratified split to maintain the original class distribution.

I also created two additional features:

1. **Payer Denial Rate:** Calculates the historical proportion of denied claims for each payer using training data.
2. **Provider Average Billed Amount:** Calculates the average billed amount associated with each provider using training data.

The payer denial rate uses out-of-fold encoding for training records to reduce target leakage.

Categorical features were converted using OneHotEncoder, while numerical features were processed using missing-value imputation and StandardScaler.

## Model Training

A Random Forest Classifier was used to predict the claim outcome.

The model was trained with the following configuration:

- Number of estimators: 200
- Maximum depth: 12
- Minimum samples split: 4
- Minimum samples leaf: 2
- Class weighting: Balanced

The trained preprocessing and classification pipeline was saved using Joblib for later use.

## Model Evaluation

The model was evaluated using the test dataset of 80 records.

The following metrics were used:

- Precision
- Recall
- F1-score
- Accuracy
- Confusion Matrix

### Classification Results

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Denied | 0.67 | 0.50 | 0.57 |
| Paid | 0.92 | 0.88 | 0.90 |
| Partially_Paid | 0.48 | 0.62 | 0.54 |

**Overall Accuracy: 79%**

The model performs best when predicting Paid claims. Denied claims have the lowest recall at 50%, meaning that the model correctly identifies 4 out of the 8 actual denied claims in the test set.

One improvement I would explore is experimenting with different class weights, additional claim-related features and hyperparameter tuning to improve the prediction of minority classes.

## API Development

FastAPI was used to create a REST API that makes predictions using the saved machine learning model.

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Checks whether the API is running |
| POST | `/predict` | Predicts the outcome of an insurance claim |

### Sample Request

```json
{
  "payer": "Aetna",
  "provider": "Dr. Rao",
  "cpt_code": "99213",
  "diagnosis_code": "M54.5",
  "billed_amount": 220.0,
  "patient_age": 45
}
```

### Sample Response

```json
{
  "predicted_status": "Paid",
  "probabilities": {
    "Denied": 0.0345,
    "Paid": 0.8972,
    "Partially_Paid": 0.0684
  }
}
```

The API also validates incoming data to handle invalid input values.

## Project Structure

```text
claim-outcome-prediction/
│
├── data/
│   └── claims.csv
│
├── models/
│   ├── claim_model.pkl
│   └── feature_metadata.pkl
│
├── generate_data.py
├── train.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd claim-outcome-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the dataset (optional)

```bash
python generate_data.py
```

### 5. Train the model (optional)

```bash
python train.py
```

### 6. Start the API

```bash
uvicorn app:app --reload
```

### 7. Test the API

Open the following URL in your browser:

```text
http://127.0.0.1:8000/docs
```

Use the interactive Swagger documentation to test the `/predict` endpoint.

**Note:** The repository includes the trained model and feature metadata, so the API can be started directly without retraining, provided the saved files are present.

## Assumptions and Limitations

- The dataset is synthetically generated for this assessment.
- Claim outcomes are assigned using simulated risk scores and fixed class proportions.
- The model's performance on this dataset does not establish its effectiveness on real insurance claims.
- The dataset is relatively small, so the evaluation results should be interpreted with caution.
- Further improvements could include using real historical claims data, additional features and more extensive model evaluation.

## Conclusion

Through this project, I implemented an end-to-end machine learning workflow for insurance claim outcome prediction. It includes synthetic data generation, preprocessing, feature engineering, model training, evaluation and deployment using FastAPI.

The project helped me understand how to take a machine learning model beyond training and make it accessible through an API.
