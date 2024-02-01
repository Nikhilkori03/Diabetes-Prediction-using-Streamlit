# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:01:17 2020

"""



import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load



st.title('Model Deployment: Diabetes Prediction')

st.sidebar.header('User Input Parameters')

def user_input_features():
    Pregnancies = st.sidebar.number_input("Insert the Pregenancies")
    Glucose = st.sidebar.number_input("Insert the Glucose")
    BloodPressure = st.sidebar.number_input("Insert the BloodPresssure")
    SkinThickness = st.sidebar.number_input("Insert the SkinThickness")
    Insulin = st.sidebar.number_input("Insert the Insulin")
    BMI = st.sidebar.number_input("Insert the BMI")
    DiabetesPedigreeFunction = st.sidebar.number_input("Insert the DiabetesPedigreeFunction")
    AGE = st.sidebar.number_input("Insert the AGE")
    data = {'Pregnancies':Pregnancies,
            'Glucose':Glucose,
            'BloodPressure':BloodPressure,
            'SkinThickness':SkinThickness,
            'Insulin':Insulin,
            'BMI':BMI,
            'DiabetesPedigreeFunction':DiabetesPedigreeFunction,
            'AGE':AGE}
    features = pd.DataFrame(data,index = [0])
    return features 
df = user_input_features()
st.subheader('User Input parameters')
st.markdown(
    f'<div style="width: 870px; padding: 10px; border: 1px solid #ccc;">'
    f'{df.to_html(index=False, escape=False)}'
    '</div>',
    unsafe_allow_html=True)


# load the model from disk

loaded_model = load(open('Diabetes_model.sav', 'rb'))

prediction = loaded_model.predict(df)
prediction_proba = loaded_model.predict_proba(df)

st.subheader('Predicted Result')
st.write('Yes, the Patient is Diabetic' if prediction_proba[0][1] > 0.5 else 'No, the Patient is not Diabetic')

st.subheader('Prediction Probability')
st.write(prediction_proba)


