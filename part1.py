import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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
st.dataframe(clean_dataset.dtypes, use_container_width=True)

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

st.write(f"#### Using these correlations, the strongest correlation (storage) was selected. Therefore the independant variable is storage size, and the dependant variable is price.")

st.write("## Step 3 - Visualising the distribution of the target variable")
st.write("### Target/Dependant Variable: Price")
prices=clean_dataset['Price']
prices_hist=plt.hist(prices, bins=100)
plt.xlabel('Price')
plt.ylabel('Frequency')
st.pyplot(plt)

st.write("## Step 4 - Exploratory Data Analysis")
st.write("Due to the nature of the variables distribution and correlation, all the variables were chosen to aid in the prediction modelling as only 1 produced a significant enough correlation.")

#work in progress code to display .info in streamlit
info=io.StringIO()
clean_dataset.info(buf=info)
values=info.getvalue()
st.text(values)
st.dataframe(clean_dataset.info())
