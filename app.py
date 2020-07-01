import streamlit as st
import pandas as pd
import pickle
st.write("""# Salary Prediction App
This app predicts the **Salary** of the employees based on their position in the company !""")

st.header('User Input Features')
st.write("Select the Position of the Employee :")
st.selectbox('Business Analyst',('Business Analyst','Junior Consultant','Senior Consultant','Manager','Country Manager','Region Manager','Partner','Senior Partner','C-level','CEO'))
k = st.slider('Level in the company: ',1.0,10.0,4.0)
st.write("You selected : ",k)

st.header("The Predicted Salary of the Employee : ")
load_clf = pickle.load(open('salary_predictor.pkl','rb'))
ds = pd.read_csv('Position_Salaries.csv')
st.header(load_clf.predict([[k]]))