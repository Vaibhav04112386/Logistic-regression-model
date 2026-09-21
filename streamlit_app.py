import streamlit as st
import joblib
import pandas as pd

st.title('Delivery Delay Prediction App')

# Load the trained model
model = joblib.load('logistic_regression_model.joblib')

# Define the input features based on the X.columns from the notebook
feature_columns = ['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
                   'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
                   'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
                   'Warehouse_Processing_Time']

# Create input fields for each feature
with st.sidebar:
    st.header('Input Features')
    inputs = {}
    inputs['Delivery_Distance'] = st.slider('Delivery Distance', 0.0, 100.0, 20.0)
    inputs['Traffic_Congestion'] = st.slider('Traffic Congestion (1-5)', 1, 5, 3)
    inputs['Weather_Condition'] = st.slider('Weather Condition (1-5)', 1, 5, 3)
    inputs['Delivery_Slot'] = st.slider('Delivery Slot (1-3)', 1, 3, 2)
    inputs['Driver_Experience'] = st.slider('Driver Experience (Years)', 0, 20, 10)
    inputs['Num_Stops'] = st.slider('Number of Stops', 1, 10, 5)
    inputs['Vehicle_Age'] = st.slider('Vehicle Age (Years)', 0, 15, 5)
    inputs['Road_Condition_Score'] = st.slider('Road Condition Score (1-5)', 1, 5, 3)
    inputs['Package_Weight'] = st.slider('Package Weight (kg)', 0.0, 50.0, 10.0)
    inputs['Fuel_Efficiency'] = st.slider('Fuel Efficiency (km/l)', 5.0, 25.0, 15.0)
    inputs['Warehouse_Processing_Time'] = st.slider('Warehouse Processing Time (min)', 0, 120, 60)

# Convert inputs to a DataFrame
input_df = pd.DataFrame([inputs])

st.subheader('Input Features:')
st.write(input_df)

# Make prediction when button is clicked
if st.button('Predict Delivery Delay'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[:, 1]

    st.subheader('Prediction:')
    if prediction[0] == 1:
        st.error('Predicted: Delivery Delay (1)')
    else:
        st.success('Predicted: No Delivery Delay (0)')

    st.subheader('Prediction Probability (Delay):')
    st.write(f'{prediction_proba[0]:.4f}')
