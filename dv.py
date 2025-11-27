import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

'''Load model'''

model=joblib.load(open("linear_regression_model.joblib",'rb'))
st.write("Model expects features:", model.n_features_in_)
st.title("Prediction")
MPG=st.number_input("MPG",min_value=0.0)
VOL=st.number_input("VOL",min_value=0.0)
SP=st.number_input("SP",min_value=0.0)


'''Mke pred'''
if st.button('predict sles'):
    input_data=np.array([[MPG,VOL,SP]])
    prediction=model.predict(input_data)[0]
    st.success(f'predict sles:{prediction:.2f}')


