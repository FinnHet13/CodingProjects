# Predicting Customer Churn with Logistic Regression and XGBoost

## Overview
As part of my Machine Learning course, I was tasked with training a machine learning model to maximize accuracy on a customer churn dataset of a telcom company.
To this end, I trained a logistic regression and XGBoost model. The XGBoost model performed best with an accuracy of 97% on the test split of the dataset.
The end-to-end pipeline of the best model is available as a Pickle file.

## Objectives
- Find the best prediction model for the dataset
- Build an end-to-end machine learning pipeline, including feature selection, train-test splitting, hyperparameter tuning and model evaluation

## Dataset
The dataset used is a slightly altered version of [Telecom Churn Dataset](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets), shown at [0_churn_dataset.csv](0_churn_dataset.csv). It consists of 2999 rows. Each row in the dataset represents one customer of the telcom company, with 19 features for each customer containing different attributes of that customer and a final column "Churn" as the label to predict.

The dataset is suitable for supervised machine learning as it is a binary classification task. As churned customers can be very costly, the dataset is additionally a good business use case for machine learning.

## Methodology
- Exploratory data analysis and end-to-end machine learning pipeline: [1_ML_pipeline.ipynb](1_ML_pipeline.ipynb)
- Pickle file containing preprocessing pipeline of best model: [2_preprocessor.pkl](2_preprocessor.pkl) 
- Pickle file containing best model (XGBoost): [3_best_model.pkl](3_best_model.pkl)
- Small validation dataset for testing the Pickle files: [4_validation_dataset.csv](4_validation_dataset.csv)

## Results
Comparison of accuracy of tested models:
| Model                           | Accuracy |
|---------------------------------|----------|
| Base Logstic Regression Model   | 85.50%   |
| Final Logistic Regression Model | 85.67%   |
| XGBoost Model                   | 97.00%   |

## Dependencies
See [requirements.txt](requirements.txt)

## References
- Mnassri, B. (2019). Telecom Churn Dataset. Kaggle. https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets
- With generous help from Claude 3.7 Sonnet :)

## Contact
Finn Hetzler

finn.he@protonmail.com