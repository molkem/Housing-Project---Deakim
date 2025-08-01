import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Define the path to the saved model and scaler
model_path = './saved_models/decision_tree_regressor_model.pkl'
scaler_path = './saved_models/scaler.pkl'

# Load the trained model and scaler
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    st.success("Model and scaler loaded successfully!")
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'decision_tree_regressor_model.pkl' and 'scaler.pkl' are in the './saved_models/' directory.")
    st.stop() # Stop execution if files are not found
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()


st.title("Paris Housing Price Prediction")

st.write("Enter the property features to get a price prediction.")

# Get the list of features the model was trained on
# Assuming X_train is available and its columns represent the features the model expects
# If X_train is not available in the environment, you would need to load it or infer features from the scaler/model
try:
    model_features = X_train.columns.tolist()
except NameError:
    st.error("Error: X_train is not available. Cannot determine model features.")
    st.stop()


raw_input_data = {}
st.write("---")
st.write("Enter property features:")

# Collect raw input data for all features required by the model
for feature in model_features:
    if feature == 'propertyAge':
         # Calculate propertyAge from user input 'Year Made'
         # Assuming current_year is available
         try:
            made_year = st.number_input("Year Made", min_value=1900, max_value=current_year, value=2000, step=1) # Set a valid default value
            raw_input_data['propertyAge'] = current_year - made_year
         except NameError:
             st.error("Error: 'current_year' variable not found. Please ensure it's defined.")
             st.stop() # Stop execution if current_year is not available

    elif feature in ['hasYard', 'hasPool', 'isNewBuilt', 'hasStormProtector', 'hasStorageRoom']:
        # Binary features
        raw_input_data[feature] = st.selectbox(f"Has {feature.replace('has', '').replace('is', '').replace('Built', ' Built')}?", [0, 1])
    elif feature == 'cityPartRange':
        # City part range is likely categorical or ordinal, use a selectbox
        # Use the original data to get unique values for the selectbox
        # Assuming data_cleaned is available and contains the original (or scaled but with meaningful range) 'cityPartRange'
        # It's better to use the original data for user-friendly input values
        try:
             # Use original data for user input range if available, otherwise use a default range
             if 'data' in locals():
                 original_city_part_ranges = sorted(data['cityPartRange'].unique().tolist())
                 raw_input_data[feature] = st.selectbox(f"City Part Range", original_city_part_ranges, index=0) # Set default to the first item by index
             else:
                  raw_input_data[feature] = st.number_input(f"City Part Range", value=1) # Use a default value if original data is not accessible
        except NameError:
             # Fallback if original 'data' is not available
             raw_input_data[feature] = st.number_input(f"City Part Range", value=1) # Use a default value if original data is not accessible
    elif feature == 'numPrevOwners':
        # Number of previous owners is likely integer
        # Use original data's median for a more meaningful default
        try:
            if 'data' in locals():
                 original_median_owners = int(data['numPrevOwners'].median())
                 raw_input_data[feature] = st.number_input(f"Number of Previous Owners", min_value=0, step=1, value=original_median_owners)
            else:
                 raw_input_data[feature] = st.number_input(f"Number of Previous Owners", min_value=0, step=1, value=0) # Fallback default
        except NameError:
            raw_input_data[feature] = st.number_input(f"Number of Previous Owners", min_value=0, step=1, value=0) # Fallback default

    elif feature == 'hasGuestRoom':
        # Number of guest rooms
        # Use original data's median for a more meaningful default
        try:
            if 'data' in locals():
                 original_median_guestrooms = int(data['hasGuestRoom'].median())
                 raw_input_data[feature] = st.number_input(f"Number of Guest Rooms", min_value=0, step=1, value=original_median_guestrooms)
            else:
                raw_input_data[feature] = st.number_input(f"Number of Guest Rooms", min_value=0, step=1, value=0) # Fallback default
        except NameError:
             raw_input_data[feature] = st.number_input(f"Number of Guest Rooms", min_value=0, step=1, value=0) # Fallback default

    else:
        # Numerical features that need scaling
        # Use the mean of the original data for default values for user input
        try:
             if 'data' in locals():
                 original_mean = data[feature].mean() # Using the original data's mean
                 raw_input_data[feature] = st.number_input(feature, value=float(original_mean))
             else:
                 raw_input_data[feature] = st.number_input(feature, value=0.0) # Use 0 as a simple default
        except NameError:
             # Fallback if original 'data' is not available
             raw_input_data[feature] = st.number_input(feature, value=0.0) # Use 0 as a simple default


# Ensure all model features are in raw_input_data, adding None if not collected (shouldn't happen with the loop, but as a safeguard)
# This loop is more of a safety check and ensures order, but the primary collection loop above should cover all features
ordered_input_data = {feature: raw_input_data.get(feature, None) for feature in model_features}
input_df_raw = pd.DataFrame([ordered_input_data], columns=model_features)


# Apply scaling to the ordered input DataFrame
# The scaler expects all features that it was fitted on.
# Ensure the input DataFrame has the same columns and order as X_train
try:
    input_df_processed = scaler.transform(input_df_raw)
    # Convert the scaled numpy array back to a pandas DataFrame with correct column names
    input_df_processed = pd.DataFrame(input_df_processed, columns=model_features)
except NameError:
    st.error("Error: Scaler object 'scaler' not found.")
    st.stop()
except Exception as e:
    st.error(f"Error during scaling: {e}")
    st.stop()


if st.button("Predict Price"):
    try:
        # Predict on the processed input data (which is log-transformed price)
        predicted_price_log = model.predict(input_df_processed)[0]

        # Inverse transform the prediction to get the price in original scale
        predicted_price = np.expm1(predicted_price_log) # np.expm1 is the inverse of np.log1p

        st.success(f"Predicted Price: ${predicted_price:,.2f}")

    except NameError:
        st.error("Error: Model object 'model' not found.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
