# Predicting Churn with Machine Learning

## Overview
Customer churn is a crucial issue for telcom companies since Telecom is largely a subscription-based business. This means each churned customer represents a direct loss of recurring monthly or annual revenue. Research has shown that **acquiring new clients** can cost **five to six times** more than **retaining existing ones** (Verbeke et al., 2012). Based on this insight, I developed and evaluated 2 machine learning models for churn prediction on a telcom churn dataset. A simple cost function representing the acquisition costs relative to retention costs was employed as the evaluation metric to place greater importance on correctly identifying the churners in the dataset. Both a logistic regression model and an XGBoost model were trained on the dataset, with the **XGBoost model achieving the lowest cost**, an over **75% cost reduction** relative to the **baseline scenario of retaining every customer**. The complete end-to-end pipeline for the best-performing model is available as a Pickle file.

## Objectives
- **Objective 1**: Build an end-to-end supervised machine learning pipeline, including feature selection, train-test splitting, hyperparameter tuning, model evaluation and model comparison.
- **Objective 2**: Find the best prediction model for the dataset
- **Objective 3**: Test the limits of the best current generative AI coding models and get better at using them.

## Dataset
The dataset used is a slightly altered version of [Telecom Churn Dataset](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets), shown at [0_churn_dataset.csv](0_churn_dataset.csv). It consists of 2999 rows. Each row in the dataset represents one customer of the telcom company, with 19 features for each customer containing different attributes of that customer and a final column "Churn" as the label to predict.

The dataset is suitable for supervised machine learning as it is a binary classification task. As churned customers can be very costly, the dataset is additionally a good business use case for machine learning.

## Methodology
- Exploratory data analysis and end-to-end machine learning pipeline: [1_ML_pipeline.ipynb](1_ML_pipeline.ipynb)
- Pickle file containing preprocessing pipeline of best model: [2_preprocessor.pkl](2_preprocessor.pkl) 
- Pickle file containing best model (XGBoost): [3_best_model.pkl](3_best_model.pkl)
- Small validation dataset for testing the Pickle files: [4_validation_dataset.csv](4_validation_dataset.csv)
- Script showing example application of the machine learning model on the validation dataset:
[5_apply_model.py](5_apply_model.py)

## Results
Regarding the 3 objectives for the project set above:
- **Objective 1**: I built one simpler (Logistic Regression) and one more complex (XGBoost) end-to-end machine learning pipeline and compared the performance of the two models.

- **Objective 2**: Comparison of costs of the different scenarios / models tested (note these are to be seen relative to each other and not as absolute values)

| Model                                 | Total Cost | Cost Reduction (vs Baseline Scenario 1) |
|---------------------------------------|------------|-----------------------------------------|
| Scenario 1: Retain every customer     | 600.0      | 0.0%                                    |
| Scenario 2: Retain no customer        | 478.5      | 20.3%                                   |
| Base Logistic Regression Model        | 323.0      | 46.2%                                   |
| Final Logistic Regression Model       | 302.0      | 49.7%                                   |
| **Final XGBoost Regression Model**    | **148.5**  | **75.3%**                               |

I was able to decrease the cost for the company by over a factor of 4 or by 75% from 600 units to 148.5 units using the best model.

- **Objective 3**: The project involved generous help from Claude 3.7 Sonnet and Gemini 2.5 Pro. I spent the largest part of this project specifying prompts to these models and checking their outputs, rather than coding myself. I actually felt my supervisory skills and ability to formulate precise questions were improved most by this project, more so than my coding skills. Most interesting was when Gemini 2.5 reformulated my functions to include explicit error handling, something I, as someone who largely self-taught myself to program, had not thought of implementing myself. In that moment, I actually felt like it was the model that was teaching **me** how to code better. 

## Dependencies
See [requirements.txt](requirements.txt)

## References
- Mnassri, B. (2019). Telecom Churn Dataset. Kaggle. https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets
- Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012). New insights into churn prediction in the telecommunication sector: A profit driven data mining approach. European Journal of Operational Research, 218(1), 211–229. https://doi.org/10.1016/j.ejor.2011.09.031
- Claude 3.7 and Gemini 2.5 Pro

## Contact
Finn Hetzler

finn.he@protonmail.com
