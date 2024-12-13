import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

##load the model
with open("model.pkl","rb") as file:
    model = pickle.load(file)

## laod the scaler
with open("scaler.pkl","rb") as file:
    scaler = pickle.load(file)


##streamlit app
st.title("Diabetes Prediction app ✅✅")

##user input

pregnancies = st.number_input("Pregnancies",min_value=0,max_value=15)
glucose = st.number_input("Glucose")
bloodPressure = st.number_input("BloodPressure")
skinThickness = st.number_input("SkinThickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
diabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction")
age = st.slider("Age",min_value=10,max_value=90)




#example for input data..

input_data = pd.DataFrame({
    "Pregnancies":[pregnancies],
    "Glucose":[glucose],
    "BloodPressure":[bloodPressure],
    "SkinThickness":[skinThickness],
    "Insulin":[insulin],
    "BMI":[bmi],
    "DiabetesPedigreeFunction":[diabetesPedigreeFunction],
    "Age":[age]

})

scaled_df = scaler.transform(input_data)

prediction = model.predict(scaled_df)[0]

if prediction == 0:
    result_text = "The person is not diabetic.👍👍"
else:
    result_text = "The person is diabetic.😔😔"

        # Display result in bold, large font, and centered
# Display result in bold, large font, and centered
st.write(
    f"""
    <div style="text-align: center;">
        <span style="font-size: 30px; font-weight: bold;">{result_text}</span>
    </div>
    """,
    unsafe_allow_html=True
)

