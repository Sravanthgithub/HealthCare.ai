
import streamlit as st
import time
import base64
import sklearn
import numpy as np
import pickle as pkl
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier


scal = MinMaxScaler()
model = pkl.load(open("final_model.p", "rb"))
df=pd.read_csv("heart-disease (2).csv")
scale=MinMaxScaler()
all=['age', 	'sex', 	'cp', 'trestbps', 'chol', 	'fbs', 	'restecg', 	'thalach' ,	'exang', 	'oldpeak' ,	'slope', 	'ca', 'thal']
df[all] = scale.fit_transform(df[all])
X=df.drop("target",axis=1).values
Y=df.target.values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0,test_size=0.2)
#st.title("HeartCare 🩺")
# st.page_icon("🤖")
st.set_page_config(page_title="HeartCare 🩺",
                   page_icon="🤖",
                   layout="centered",
                   initial_sidebar_state="expanded",
                   menu_items=({
                       'Get help': "https://github.com/Sravanthgithub/Hackoverflow",
                       'Report a bug': "https://github.com/Sravanthgithub/Hackoverflow/pulls",
                   }))


def Preprocess_info(age, sex, cp, trestbps, restecg, chol, fbs, thalach, exang, oldpeak, slope, ca, thal):

    # Pre-processing user input
    if sex == "male":
        sex = 1
    else:
        sex = 0

    if cp == "Typical angina":
        cp = 0
    elif cp == "Atypical angina":
        cp = 1
    elif cp == "Non-anginal pain":
        cp = 2
    elif cp == "Asymptomatic":
        cp = 3

    if exang == "Yes":
        exang = 1
    elif exang == "No":
        exang = 0

    if fbs == "Yes":
        fbs = 1
    elif fbs == "No":
        fbs = 0

    if slope == "Upsloping: better heart rate with excercise(uncommon)":
        slope = 0
    elif slope == "Flatsloping: minimal change(typical healthy heart)":
        slope = 1
    elif slope == "Downsloping: signs of unhealthy heart":
        slope = 2

    if thal == "fixed defect: used to be defect but ok now":
        thal = 6
    elif thal == "reversable defect: no proper blood movement when excercising":
        thal = 7
    elif thal == "normal":
        thal = 2.31
    
    if restecg == "Nothing to note":
        restecg = 0
    elif restecg == "ST-T Wave abnormality":
        restecg = 1
    elif restecg == "Possible or definite left ventricular hypertrophy":
        restecg = 2

    user_input = [age, sex, cp, trestbps, restecg, chol,
                  fbs, thalach, exang, oldpeak, slope, ca, thal]
    user_input = np.array(user_input)
    user_input = user_input.reshape(1, -1)
    user_input = scale.transform(user_input)
    model = KNeighborsClassifier()
    model.fit(X_train, Y_train)
    prediction = model.predict(user_input)

    return prediction


html_temp = """ 
    <div style ="background-color:orange;padding:13px;border-radius:8px"> 
    <h1 style ="color:black;text-align:center;">HeartCare 🩺</h1> 
    </div> 
    """

st.markdown(html_temp, unsafe_allow_html=True)
st.subheader("\n")
#st.subheader('by Sravanth ')
st.markdown(
    "It can basically classify whether a person has heart disease or not and yea lets gooo, **Enter the details**.")
st.markdown(
    "**Data Source**: https://www.kaggle.com/uciml/heart-disease-uci-mortality-dataset")

age = st.selectbox("Age", range(1, 121, 1))
sex = st.radio("Select Gender: ", ('male', 'female'))
cp = st.selectbox('Chest Pain Type', ("Typical angina",
                  "Atypical angina", "Non-anginal pain", "Asymptomatic"))
trestbps = st.slider('Resting Blood Sugar', 1, 500, 1)
restecg = st.selectbox('Resting Electrocardiographic Results', ("Nothing to note",
                       "ST-T Wave abnormality", "Possible or definite left ventricular hypertrophy"))
chol = st.slider('Serum Cholestoral in mg/dl', 1, 1000, 1)
fbs = st.radio("Fasting Blood Sugar higher than 120 mg/dl", ['Yes', 'No'])
thalach = st.slider('Maximum Heart Rate Achieved', 1, 300, 1)
exang = st.radio('Exercise Induced Angina', ["Yes", "No"])
oldpeak = st.number_input('Oldpeak')
slope = st.radio('Heart Rate Slope', ("Upsloping: better heart rate with excercise(uncommon)",
                                      "Flatsloping: minimal change(typical healthy heart)", "Downsloping: signs of unhealthy heart"))
ca = st.selectbox(
    'Number of Major Vessels Colored by Flourosopy', range(0, 5, 1))
thal = st.selectbox('Thalium Stress Result', range(1, 8, 1))


pred = Preprocess_info(age, sex, cp, trestbps, restecg, chol, fbs,
                       thalach, exang, oldpeak, slope, ca, thal)


if st.button("Predict"):
    if pred == 0:
        st.error('Warning! You have high risk of getting a heart attack!')
    else:
        st.success('You have lower risk of getting a heart disease!')


st.sidebar.subheader("About App")

st.sidebar.info(
    "This web app uses some powerful Machine-Learning techniques and  helps you to find out whether you are at a risk of developing a heart disease.")
st.sidebar.info(
    "Enter the required fields and click on the 'Predict' button to check whether you have a healthy heart")


with st.spinner('Almost done...'):
    time.sleep(2)
st.success('Done!,Thank you for using this app!')


st.info("Caution: This is just a prediction and not doctoral advice. Kindly see a doctor if you feel the symptoms persist.")
# %%
