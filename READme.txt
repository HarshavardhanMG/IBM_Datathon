# AKI Prediction Project

## Overview
This project aims to predict patients in the Intensive Care Unit (ICU) who may require early renal replacement therapy (RRT) due to acute kidney injury (AKI). By leveraging machine learning techniques, we aim to enhance clinical decision-making and improve patient outcomes.

## Table of Contents
- [Data Cleaning and EDA](#data-cleaning-and-eda)
- [Preprocessing and Model Tuning](#preprocessing-and-model-tuning)
- [Model Implementations](#model-implementations)
  - [Logistic Regression](#logistic-regression)
  - [Random Forest](#random-forest)
  - [XGBoost](#xgboost)
  - [Ensemble Techniques](#ensemble-techniques)
- [Model Evaluation](#model-evaluation)
- [Usage](#usage)
- [Future Work](#future-work)

## Data Cleaning and EDA
The initial phase involved cleaning the dataset and performing exploratory data analysis (EDA) to understand the data distribution and correlations among features. Key steps included:
- Handling missing values
- Visualizing data distributions
- Identifying important features influencing AKI

## Preprocessing and Model Tuning
We implemented a preprocessing pipeline that included:
- Feature selection based on correlation analysis
- Applying SMOTE to address class imbalance
- Splitting the dataset into training, validation, and test sets

## Model Implementations

### Logistic Regression
- Implemented a Logistic Regression model with hyperparameter tuning using cross-validation.
- Evaluated model performance using metrics such as ROC AUC and classification reports.

### Random Forest
- Developed a Random Forest model to capture complex interactions between features.
- Analyzed feature importance to understand the most influential predictors of AKI.

### XGBoost
- Utilized XGBoost, a powerful gradient boosting technique, for classification tasks.
- Conducted grid search for hyperparameter optimization, significantly improving model performance.

### Ensemble Techniques
- Combined multiple models using a voting classifier to enhance prediction accuracy.
- Employed randomized search for hyperparameter tuning of the ensemble model.

## Model Evaluation
Each model was evaluated using various metrics, including:
- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC

The best-performing model was selected based on these metrics, ensuring robust predictions for early RRT recommendations.

## Usage
To use the models for predictions:
1. Load the cleaned dataset.
2. Select the relevant features.
3. Use the trained model to predict the likelihood of requiring early dialysis based on user input.

Example function for prediction:
def predict_early_dialysis(calcium_max, creatinine_min, aki_stage, aniongap_min, calcium_min, pt_max, model):
# Create a numpy array with the input data
input_data = np.array([[calcium_max, creatinine_min, aki_stage, aniongap_min, calcium_min, pt_max]])
# Use the trained model to predict the class (0 or 1)
prediction = model.predict(input_data)
return "Early dialysis is recommended." if prediction[0] == 1 else "Early dialysis is not required."

## Future Work
Future enhancements could include:
- Exploring additional machine learning algorithms.
- Implementing more advanced feature engineering techniques.
- Deploying the model in a clinical setting for real-time predictions.

## Conclusion
This project demonstrates the potential of machine learning in predicting AKI and improving patient care. We encourage further exploration and application of these techniques in healthcare settings.