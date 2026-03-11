# Credit Risk Modeling using Machine Learning

This project builds a credit risk prediction model to estimate the probability of borrower default using machine learning techniques.

The model is trained on a credit risk dataset containing borrower financial information and loan characteristics.

## Objectives

• Predict probability of default (PD)  
• Identify key drivers of credit risk  
• Compare machine learning models  
• Segment borrowers into risk categories  

## Dataset

The dataset includes borrower financial information such as:

- borrower income
- loan amount
- loan interest rate
- employment length
- credit history
- prior defaults

Target variable:

loan_status  
0 = non-default  
1 = default

## Methodology

1. Data preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature engineering
4. Model training

Models used:

- Logistic Regression
- Random Forest

## Model Evaluation

Models are evaluated using:

- ROC-AUC score
- Confusion matrix
- Feature importance

## Risk Segmentation

Borrowers are categorized into risk bands based on predicted probability of default:

Low Risk: <20%  
Medium Risk: 20–50%  
High Risk: >50%

## Visualization

Model outputs are visualized using a Tableau dashboard including:

- portfolio risk overview
- borrower risk segmentation
- feature importance analysis

## Technologies Used

Python  
Pandas  
Scikit-Learn  
Matplotlib  
Seaborn  
Tableau
