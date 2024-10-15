import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import io
from scipy.stats import zscore

st.write("# Assessment 3 - Laptop Dataset")

st.write("## Step 1 - Reading a sample of clean data")
data=pd.read_csv("laptop.csv")
clean_dataset=data.drop_duplicates()
st.dataframe(clean_dataset.sample(10))

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

st.write(f"#### Using these correlations, the strongest correlation (storage) was most likely to be selected. Therefore the most important independant variable is storage size, and the dependant variable is price.")

st.write("## Step 3 - Visualising the distribution of the target variable")
st.write("### Target/Dependant Variable: Price")
prices=clean_dataset['Price']
prices_hist=plt.hist(prices, bins=100)
plt.xlabel('Price')
plt.ylabel('Frequency')
st.pyplot(plt)

st.write("## Step 4 - Exploratory Data Analysis")

st.write("### Head")
st.dataframe(clean_dataset.head())

st.write("### Info")
info=io.StringIO()
clean_dataset.info(buf=info)
values=info.getvalue()
st.text(values)

st.write("### Describe")
st.dataframe(clean_dataset.describe(include="all"))

st.write("### Nunique")
st.dataframe(clean_dataset.nunique())

st.write("Due to the nature of the variables distribution and correlation, all the variables are likely to be chosen to aid in the prediction modelling as only 1 produced a significant enough correlation. However, going forward brand will not be used as it is qualitative data")
num_only_dataset=clean_dataset.drop(columns=["Brand"])

st.write("## Step 5 - Visual Exploratory Data Analysis")
for column in num_only_dataset.columns:
    plt.hist(num_only_dataset[column], bins=50)
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    st.pyplot(plt)
    
st.write("## Step 6 - Outlier Value Analysis")
z=num_only_dataset.apply(zscore).abs()
no_outliers=num_only_dataset[(z<3).all(axis=1)]
st.dataframe(no_outliers.describe())

st.write("## Step 7 - Missing Value Analysis")
missing=no_outliers.isnull().sum()
st.write("Data frame for missing vals:")
st.dataframe(missing[missing>0])
st.write("There are no missing values. Each cell has a value")

st.write("## Step 8 - Feature Selection")
st.write("### Continuous")
for column in no_outliers.select_dtypes(include=["float64"]).columns:
    if column!="Price":
        plt.scatter(no_outliers[column],no_outliers["Price"])
        plt.title(f"Scatter plot of {column} vs Price")
        plt.xlabel(column)
        plt.xlim(0,no_outliers[column].max())
        plt.ylabel("Price")
        st.pyplot(plt)
        
        corr=no_outliers[column].corr(no_outliers['Price'])
        st.write(f"{column} correlation: {corr:.2f}")
        
st.write("### Integer")
for column in no_outliers.select_dtypes(include=["int64"]).columns:
    no_outliers.boxplot(column='Price',by=column)
    plt.xlabel(column)
    plt.ylabel('Price')
    st.pyplot(plt)
    corr=no_outliers[column].corr(no_outliers['Price'])
    st.write(f"{column} correlation: {corr:.2f}")

st.write("## Step 9 - ANOVA Test")
st.write("use brand column of clean_dataset for this")

st.write("## Step 10 - Final Selection")
st.write("All object, integer and float values will be used despite the lower correlations. This is due to the lack of high correlating data in the dataset. This will also be adequate to show the method of predicting itself, however the reliability of the outcome will be impacted")

st.write("## Step 11 - Conversion of nominal values")
nom_to_int=pd.get_dummies(clean_dataset['Brand'])
converted_df=pd.concat([clean_dataset,nom_to_int],axis=1)
converted_df=converted_df.drop('Brand',axis=1)
st.dataframe(converted_df.head())

st.write("## Step 12 - Splitting training and testing data")