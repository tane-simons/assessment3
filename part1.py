import pandas as pd
import streamlit as st

st.write("# Assessment 3 - Laptop Dataset")

st.write("## Step 1 - Reading a sample of clean data")
data=pd.read_csv("laptop.csv")
clean_dataset=data.drop_duplicates()
st.dataframe(clean_dataset.sample(10))

st.write(f"##Data Shape: {clean_dataset.shape}")
st.write(f"##Data Size: {clean_dataset.size}")

