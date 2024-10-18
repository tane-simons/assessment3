'''
*******************************
Author:
u3253279 Group_Undergraduate_2 Assessment_3_Step_15, 10/03/2024
Programming:
*******************************
'''

import pandas as pd
import streamlit as st
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

og_data=pd.read_csv("laptop.csv") #og dataset
clean_data=og_data.drop_duplicates() #cleaned
num_only=clean_data.drop(columns=["Brand"]) #numonly dataset
z=num_only.apply(zscore).abs() #absolute value zscore of the dataset
no_outliers=num_only[(z<3).all(axis=1)] #no outlier dataset

x=no_outliers.drop('Price',axis=1) #this is the features dataset (without price)
y=no_outliers['Price'] #this is the target dataset (only price)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.5) #splits these new datasets in half again into testing and training
model=LinearRegression()
model.fit(xtrain,ytrain) #model trained using x and y training data

#streamlit interface
st.title("Laptop Price Prediction")
st.write("Fill in each field and click 'predict' to predict the price of the laptop")

#inputs
Processor_Speed=st.number_input("Processor Speed (GHz)",min_value=0.0,max_value=7.0)
Ram_Size=st.number_input("RAM Size (GB)",min_value=0,max_value=128)
Storage_Capacity=st.number_input("Storage (GB)",min_value=0,max_value=8000)
Screen_Size=st.number_input("Screen Size (In)",min_value=0,max_value=20)
Weight=st.number_input("Weight (kg)",min_value=0.0,max_value=5.0)

if st.button("Predict"):
    inputted=pd.DataFrame([[Processor_Speed,Ram_Size,Storage_Capacity,Screen_Size,Weight]],columns=['Processor_Speed','Ram_Size','Storage_Capacity','Screen_Size','Weight']) #dataframe with the inputs
    predicted=model.predict(inputted) #make prediction
    st.write(f"The predicted price for the laptop is: ${predicted[0]:.2f}")
    