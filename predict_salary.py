import pickle
import pandas as pd

def load_model_components(model_path='best_model.pkl',
                          scaler_path='scaler.pkl',
                          feature_columns_path='feature_columns.pkl'):
    """Loads the trained model, scaler, and feature columns."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(feature_columns_path, 'rb') as f:
        feature_columns = pickle.load(f)
    return model, scaler, feature_columns

def preprocess_new_data(new_data_df, scaler, loaded_feature_columns):
    """Preprocesses new input data for prediction."""
    # 1. One-hot encode categorical features
    categorical_cols = ['Gender', 'Education Level', 'Job Title']
    new_data_encoded = pd.get_dummies(new_data_df, columns=categorical_cols, drop_first=True)

    # 2. Align columns with the training data columns
    # Add missing columns with 0, and ensure order
    missing_cols = set(loaded_feature_columns) - set(new_data_encoded.columns)
    for c in missing_cols:
        new_data_encoded[c] = 0
    # Ensure the order of columns is the same as during training
    new_data_processed = new_data_encoded[loaded_feature_columns]

    # 3. Scale numerical features using the loaded scaler
    new_data_scaled = scaler.transform(new_data_processed)
    return new_data_scaled

def predict_salary(new_data_dict):
    """Predicts salary for new input data.

    Args:
        new_data_dict (dict): A dictionary containing new data points, e.g.,
                              {'Age': 30.0, 'Gender': 'Male', 'Education Level': "Bachelor's",
                               'Job Title': 'Software Engineer', 'Years of Experience': 4.0}

    Returns:
        float: The predicted salary.
    """
    model, scaler, feature_columns = load_model_components()
    new_data_df = pd.DataFrame([new_data_dict])
    processed_data = preprocess_new_data(new_data_df, scaler, feature_columns)
    predicted_salary = model.predict(processed_data)
    return predicted_salary[0]

if __name__ == '__main__':
    # Example Usage:
    print("--- Salary Prediction Script ---")

    # Simulate new input data
    example_data = {
        'Age': 30.0,
        'Gender': 'Male',
        'Education Level': "Bachelor's",
        'Job Title': 'Software Engineer',
        'Years of Experience': 4.0
    }

    # Make a prediction
    predicted_val = predict_salary(example_data)

    print(f"\nNew Input Data:\n{pd.DataFrame([example_data]).to_string(index=False)}")
    print(f"\nPredicted Salary: ${predicted_val:.2f}")

    print("\n--- Script Finished ---")
