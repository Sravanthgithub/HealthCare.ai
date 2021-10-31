import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import streamlit as st 
import time

pickle_in = open("predictor.pkl","rb")
predictor=pickle.load(pickle_in)

def welcome():
    return "Welcome All"

def diabetes_prediction(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,Bmi,Age):
    
    """Let's Predict 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: Pregnancies
        in: query
        type: number
        required: true
      - name: Glucose
        in: query
        type: number
        required: true
      - name: BloodPressure
        in: query
        type: number
        required: true
      - name: SkinThickness
        in: query
        type: number
        required: true
      - name: Insulin
        in: query
        type: number
        required: true
      - name: Bmi
        in: query
        type: number
        required: true   
      - name: Age
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    prediction=predictor.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,Bmi,Age]])
    print(prediction)
    return prediction

def main():
    #st.title("Diabetes Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Diabetes Prediction App </h2>
    </div>
    """
    
    #Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	 Age
    st.markdown(html_temp,unsafe_allow_html=True)

    Pregnancies = st.text_input("Number of Pregnancies","Type Here")
    Glucose = st.text_input("Glucose","Type Here")
    BloodPressure = st.text_input("Blood Pressure","Type Here")
    SkinThickness = st.text_input("Skin Thickness","Type Here")
    Insulin = st.text_input("Insulin","Type Here")
    Bmi = st.text_input("Bmi","Type Here")
    Age = st.text_input("Age","Type Here")
    result=""
    if st.button("Predict"):
        result=diabetes_prediction(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,Bmi,Age)
        
        if (result==1):
            pos = "You are at high risk of diabetes. Please visit your nearest hospital for more information on managing the disease."
            st.success('{}'.format(pos))

        else:
            neg = "Your results look good. You are healthy and not at risk for diabetes :)"
            st.success('{}'.format(neg))

    with st.spinner('Almost done...'):
      time.sleep(2)
      st.success('Done!,Thank you for using this app!')


    st.info("Caution: This is just a prediction and not doctoral advice. Kindly see a doctor if you feel the symptoms persist.")

    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

st.sidebar.subheader("About App")

st.sidebar.info(
    "This web app uses some powerful Machine-Learning techniques and  helps you to find out whether you are at a risk of developing a heart disease.")
st.sidebar.info(
    "Enter the required fields and click on the 'Predict' button to check whether you have a healthy heart")





if __name__=='__main__':
    main()
