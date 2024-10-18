'''
*******************************
Author:
u3253279 Group_Undergraduate_2 Assessment_3_Steps_1-14 18/10/2024
Programming:
*******************************
'''
import pandas as pd #used for the dfs
import streamlit as st #what we're using to display all the info
import matplotlib.pyplot as plt #for the graphs
import io #only used for the temp storage section of displaying .info() in streamlit
from scipy.stats import zscore, f_oneway #used for outliers and analysis of variance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

st.write("# Assessment 3 - Laptop Dataset")

st.write("## Step 1 - Reading a sample of clean data")
data=pd.read_csv("laptop.csv")
clean_dataset=data.drop_duplicates()
st.dataframe(clean_dataset.sample(10)) #random sample

st.write(f"### Data Shape: {clean_dataset.shape} (rows/columns)")
st.write(f"### Data Size: {clean_dataset.size} cells")
st.write(f"### Data Attributes: {clean_dataset.attrs}")
st.dataframe(clean_dataset.dtypes, use_container_width=True) #table of teh datatypes

st.write(f"### Correlations:") #brief look at the correlatons of each compared to price. will be explored further later on
corr1=clean_dataset['Processor_Speed'].corr(clean_dataset['Price'])
st.write(f"Processor: {corr1}")
corr2=clean_dataset['Ram_Size'].corr(clean_dataset['Price'])
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
prices_hist=plt.hist(prices, bins=100) #histogram of prices
plt.xlabel('Price')
plt.ylabel('Frequency')
st.pyplot(plt)

st.write("## Step 4 - Exploratory Data Analysis")

st.write("### Head")
st.dataframe(clean_dataset.head()) #top 10 or so rows

st.write("### Info")
info=io.StringIO() #info is acting as a memory storage here to store the output of .info() so it isnt directly printed
clean_dataset.info(buf=info) #'prints' summary to info
values=info.getvalue() #this gets the previous StringIO
st.text(values) #summary in streamlit

st.write("### Describe") #table of info about each column
st.dataframe(clean_dataset.describe(include="all"))

st.write("### Nunique") #no. unique vals in each
st.dataframe(clean_dataset.nunique())

st.write("Due to the nature of the variables distribution and correlation, all the variables are likely to be chosen to aid in the prediction modelling as only 1 produced a significant enough correlation. However, going forward brand will not be used as it is qualitative data")
num_only_dataset=clean_dataset.drop(columns=["Brand"]) #dataset without the category column if you want to work with just the numbers

st.write("## Step 5 - Visual Exploratory Data Analysis")
for column in num_only_dataset.columns: #loop that displays a histogram of each number only column
    plt.hist(num_only_dataset[column], bins=50) 
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    st.pyplot(plt)
    
st.write("## Step 6 - Outlier Value Analysis")
z=num_only_dataset.apply(zscore).abs() #absolute value zscore of the dataset
no_outliers=num_only_dataset[(z<3).all(axis=1)] #makes a new df without outliers (outliers being anything greater than 3 zscore away)
st.dataframe(no_outliers.describe()) #quick summary of the new dataset

st.write("## Step 7 - Missing Value Analysis")
missing=no_outliers.isnull().sum()
st.write("Data frame for missing vals:")
st.dataframe(missing[missing>0]) #dataframe to show the columns where theres more than no missing vals
st.write("There are no missing values. Each cell has a value")

st.write("## Step 8 - Feature Selection")
st.write("### Continuous")
for column in no_outliers.select_dtypes(include=["float64"]).columns: #loop for floats to display a scatter plot of them vs price
    if column!="Price":
        plt.scatter(no_outliers[column],no_outliers["Price"])
        plt.title(f"Scatter plot of {column} vs Price")
        plt.xlabel(column)
        plt.xlim(0,no_outliers[column].max()) #each will have its own xlim being the highest value of the column
        plt.ylabel("Price")
        st.pyplot(plt)
        
        corr=no_outliers[column].corr(no_outliers['Price']) #and also displays the r correlation value
        st.write(f"{column} correlation: {corr:.2f}")
        
st.write("### Integer")
for column in no_outliers.select_dtypes(include=["int64"]).columns: #same loop thing but for integers instead and therefore will show boxplot
    no_outliers.boxplot(column='Price',by=column)
    plt.xlabel(column)
    plt.ylabel('Price')
    st.pyplot(plt)
    corr=no_outliers[column].corr(no_outliers['Price']) #prints corrrelation
    st.write(f"{column} correlation: {corr:.2f}")

st.write("## Step 9 - ANOVA Test")
anova=[clean_dataset[clean_dataset['Brand']==group]['Price'] for group in clean_dataset['Brand'].unique()] #loop over all unique brand names and makes a subset of price for each. basically it's now a list of series of prices for a brand
stat, p=f_oneway(*anova) #f_oneway does an anova test (comparing avgs of groups), gets the anova list arguments from before, and stores the values in p and stat
#for reference sake, stat is the ratio between the averages of groups to the variance of groups. given null hypothesis (means ~ equal), p is probability of observing the stat
st.write(f"ANOVA test. Stats: {stat:.2f},p={p:.2f}")
if p<=0.05: #5% seems to be usual threshold for significance
    st.write(f"Reject null hypothesis as the p value is less than 0.05. There is a significant correlation.")
else:
    st.write(f"Null hypothesis. p value is more than 0.05. This suggests the Brand does not seemt to affect price.")

st.write("## Step 10 - Final Selection")
st.write("All integer and float values will be used despite the lower correlations. This is due to the lack of high correlating data in the dataset. This will be adequate to show the method of prediction, however the reliability of the outcome will be impacted. Also note, object types (Brand) will not be included due to the null hypothesis.")

st.write("## Step 11 - Conversion of nominal values")
nom_to_int=pd.get_dummies(clean_dataset['Brand']) #converts nominal data to binary (ys/no) new dataframe will have the 4 brands and a yes/no if it is the selected brand
converted_df=pd.concat([clean_dataset,nom_to_int],axis=1) #combines the dataframes
converted_df=converted_df.drop('Brand',axis=1) #gets rid of the categorical brand column
st.dataframe(converted_df.head())

st.write("## Step 12 - Splitting training and testing data")
train,test=train_test_split(num_only_dataset,test_size=0.5) #splits the dataset into train and test
st.write("Sample of test data")
st.write(test.sample(5))
st.write("Sample of training data")
st.write(train.sample(5)) #just samples of data to ensure they're there

x=num_only_dataset.drop('Price',axis=1) #this is the features dataset (without price)
y=num_only_dataset['Price'] #this is the target dataset (only price)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.5) #splits these new datasets in half again into testing and training

st.write("## Step 13 - The Models")
#this will be a dictionary of the regression models
models = {"Linear Regression":LinearRegression(),"Decision Tree Regressor":DecisionTreeRegressor(),"Random Forest Regressor":RandomForestRegressor(),"K-Nearest Neighbour Regressor":KNeighborsRegressor(),"SVM Regressor":SVR()}
for name,model in models.items(): #loop sover the dictionary, getting the name and the model
    model.fit(xtrain,ytrain) #model trained using x and y training data
    prediction=model.predict(xtest) #makes predictions for the xtest set
    r2=r2_score(ytest,prediction) #gets the r2 value of the ytest set and the prediction, testing the correlation
    st.write(f"### {name}")
    st.write(f"{name}'s r2 value: {r2:.5f}")
    st.write(f"{name}'s predictions per column (Speed, RAM, Storage, Screen, Weight): {model.predict(xtest)[:5]}") #outputs predictions (of the 5 cols)

st.write("## Step 14 - Selection of Model")
st.write("The linear regression model appeared to produce the best r2 value, indicating a very strong correlation between its prediction and the actual data")

st.write('## Step 15 - Final Model')
st.write("To interact with the model, navigate to the github repo: https://github.com/tane-simons/assessment3 and run model.py")
st.write("Dependancies include: pandas, streamlit, scikit-learn, scipy.stats")