import pandas as pd
import streamlit as st

st.write("# Assessment 3 - Laptop Dataset")

st.write("## Step 1 - Reading a sample of clean data")
data=pd.read_csv("laptop.csv")
clean_dataset=data.drop_duplicates()
st.dataframe(clean_dataset.sample(10))

#CHANGES
st.write(f"### Data Shape: {clean_dataset.shape} (rows/columns)")
st.write(f"### Data Size: {clean_dataset.size} cells")
st.write(f"### Data Attributes: {clean_dataset.attrs}")
st.write(f"### Correlations:")

corr1=clean_dataset['Processor_Speed'].corr(clean_dataset['Price'])
st.write(f"Processor: {corr1}")
corr2=clean_dataset['RAM_Size'].corr(clean_dataset['Price'])
st.write(f"RAM: {corr2}")
corr3=clean_dataset['Storage_Capacity'].corr(clean_dataset['Price'])
st.write(f"Storage: {corr3}")
corr4=clean_dataset['Screen_Size'].corr(clean_dataset['Price'])
st.write(f"Screennsize: {corr4}")
corr5=clean_dataset['Weight'].corr(clean_dataset['Price'])
st.write(f"Weight: {corr5}")

st.write("")