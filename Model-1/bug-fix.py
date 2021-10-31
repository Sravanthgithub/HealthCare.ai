import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.write("""**1. Select Age :**""")
age = st.slider('', 0, 100, 25)
st.write("""**You selected this option **""",age)
    
st.write("""**2. Select Gender :**""")
sex = st.selectbox("(1=Male, 0=Female)",["1","0"])
st.write("""**You selected this option **""",sex)

df = user_input_features()
heart = pd.read_csv("heart-disease (2).csv")
X = heart.iloc[:,0:13].values
Y = heart.iloc[:,[13]].values
model = RandomForestClassifier()
model.fit(X, Y)
prediction = model.predict(df)
st.subheader('Prediction :')
df1=pd.DataFrame(prediction,columns=['0'])
df1.loc[df1['0'] == 0, 'Chances of Heart Disease'] = 'No'
df1.loc[df1['0'] == 1, 'Chances of Heart Disease'] = 'Yes'
st.write(df1)
prediction_proba = model.predict_proba(df)
st.subheader('Prediction Probability in % :')
st.write(prediction_proba * 100)

