# create a streamlit app for heart disease classification
import numpy as np
import pandas as pd
import streamlit as st
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

st.title('Heart disease classification')
st.markdown("It can basically classify whether a person has heart disease or not.")
st.markdown("**Data Source**: https://www.kaggle.com/uciml/heart-disease-uci-mortality-dataset")

model = pkl.load(open("final_model.p","rb"))

st.set_page_config(page_title="Heart Disease Prediction", 
page_icon="💗",
layout="centered",
initial_sidebar_state="expanded",
menu_items=({
    'Get help':"https://github.com/Sravanthgithub/Hackoverflow",
    'Report a bug': "https://github.com/Sravanthgithub/Hackoverflow/pulls"
})
)

