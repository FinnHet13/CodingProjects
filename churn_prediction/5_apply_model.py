# Script to load the XGBoost model, and make predictions on an example
# of a validation dataset.
# This script assumes that the preprocessing pipeline and the model have been saved as pickle files.
# Ex. 2_preprocessor.pkl and 3_best_model.pkl
# It also assumes the validation dataset is a csv file formatted correctly with the same features as the training dataset and without the "Churn" label.
# Ex. 4_validation_dataset.csv

# Import necessary libraries
import pickle
import pandas as pd

# First transformer to convert categorical to binary
def convert_to_binary(X_save):
    if 'International plan' in X_save.columns:
        X_save['International plan'] = X_save['International plan'].map({'Yes': 1, 'No': 0})
    if 'Voice mail plan' in X_save.columns:
        X_save['Voice mail plan'] = X_save['Voice mail plan'].map({'Yes': 1, 'No': 0})
    return X_save


# All the feature engineering steps for the XGBoost model
def feature_engineering(X_save):
    # Add total charge column
    X_save['Total charge'] = X_save['Total day charge'] + X_save['Total eve charge'] + X_save['Total night charge'] + X_save['Total intl charge']
    # Add total minutes column
    X_save['Total minutes'] = X_save['Total day minutes'] + X_save['Total eve minutes'] + X_save['Total night minutes'] + X_save['Total intl minutes']
    # Add total calls column
    X_save['Total calls'] = X_save['Total day calls'] + X_save['Total eve calls'] + X_save['Total night calls'] + X_save['Total intl calls']
    # Add average call duration
    X_save['Average call duration'] = X_save['Total minutes'] / X_save['Total calls']
    # Add Average Charge per Call
    X_save['Average charge per call'] = X_save['Total charge'] / X_save['Total calls']

    return X_save

# Load the preprocessing pipeline (including feature engineering)
with open("2_preprocessor.pkl", "rb") as file:
    loaded_preprocessor = pickle.load(file)

# Load the XGBoost model (pickled)
with open("3_best_model.pkl", "rb") as file:
    xgb_loaded_model = pickle.load(file)

print("Preprocessing pipeline (feature engineering + transformations) and XGBoost model loaded successfully!")

# New data to make a prediction
validation_df = pd.read_csv('4_validation_dataset.csv')
validation_df.head()

# Apply feature engineering and preprocessing together
new_data_transformed = loaded_preprocessor.transform(validation_df)

# Make predictions using the loaded XGBoost model
xgb_prediction = xgb_loaded_model.predict(new_data_transformed)

# Example of how to use the model to make a prediction
print(f"XGBoost Prediction: {xgb_prediction[0]}")
