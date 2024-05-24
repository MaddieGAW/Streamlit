import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Preprocessing for numerical data
scaler = StandardScaler()

# Load your model
@st.cache
def load_model():
    # Load your trained model here
    model = RandomForestRegressor()  # Example model, replace with your actual model
    return model

model = load_model()

# Define function to preprocess input data
def preprocess_input(data):
    # Preprocess numerical features
    data[numerical_features] = scaler.transform(data[numerical_features])
    return data

# Function to make prediction
def predict(project_duration, eval_lag, sector_code, completion_year, project_size_USD_calculated):
    input_data = pd.DataFrame({
        'project_duration': [project_duration],
        'eval_lag': [eval_lag],
        'sector_code': [sector_code],
        'completion_year': [completion_year],
        'project_size_USD_calculated': [project_size_USD_calculated]
    })
    input_data = preprocess_input(input_data)
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit UI
st.title('Project Prediction App')

project_duration = st.slider('Project Duration', min_value=0, max_value=1000, step=1)
eval_lag = st.slider('Evaluation Lag', min_value=0, max_value=1000, step=1)
sector_code = st.selectbox('Sector Code', options=[1, 2, 3])  # Example options, replace with your actual options
completion_year = st.slider('Completion Year', min_value=2000, max_value=2030, step=1)
project_size_USD_calculated = st.number_input('Project Size (USD)', value=1000)

if st.button('Predict'):
    prediction = predict(project_duration, eval_lag, sector_code, completion_year, project_size_USD_calculated)
    st.write('Predicted Output:', prediction)
