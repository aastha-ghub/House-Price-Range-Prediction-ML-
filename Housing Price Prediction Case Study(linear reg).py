#!/usr/bin/env python
# coding: utf-8

# In[66]:


#supress warnings
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

#Data visualization

import matplotlib.pyplot as plt
import seaborn as sns


# In[67]:


housing=pd.DataFrame(pd.read_csv("/Users/Admin/Downloads/Housing.csv"))


# In[68]:


housing.head()


# In[69]:


#data inspection
housing.shape


# In[70]:


housing.info()


# In[71]:


housing.describe()


# In[72]:


#data cleaning
#checking null values

housing.isnull().sum()*100/housing.shape[0]

#no null values




# In[73]:


#outlier analysis
fig,axs=plt.subplots(2,3,figsize=(10,5))
plt1=sns.boxplot(housing['price'],ax = axs[0,0])
plt2=sns.boxplot(housing['area'],ax = axs[0,1])
plt3=sns.boxplot(housing['bedrooms'],ax = axs[0,2])
plt1=sns.boxplot(housing['bathrooms'],ax = axs[1,0])
plt2=sns.boxplot(housing['stories'],ax = axs[1,1])
plt3=sns.boxplot(housing['parking'],ax = axs[1,2])

plt.tight_layout()

#subplots() function in the matplotlib library, helps in creating multiple layouts of subplots.
#It provides control over all the individual plots that are created.
#Boxplots are a measure of how well distributed the data in a data set is. It divides the data set
#into three quartiles. This graph represents the minimum, maximum, median, first quartile and third
#quartile in the data set.
#tight_layout automatically adjusts subplot params so that the subplot(s) fits in to the figure area.
#This is an experimental feature and may not work for some cases.



# In[74]:


#outlier treatment for price


plt.boxplot(housing.price)
Q1=housing.price.quantile(0.25)
Q3=housing.price.quantile(0.75)
IQR=Q3-Q1
housing=housing[(housing.price >= Q1 - 1.5*IQR) & (housing.price <= Q3 + 1.5*IQR)]


# In[75]:


housing.shape


# In[76]:


#outlier treatment for area

plt.boxplot(housing.area)
Q1=housing.area.quantile(0.25)
Q3=housing.area.quantile(0.75)
IQR=Q3-Q1
housing=housing[(housing.area >= Q1 - 1.5*IQR) & (housing.area <= Q3 + 1.5*IQR)]


# In[77]:


housing.shape


# In[78]:


#outlier analysis
fig,axs=plt.subplots(2,3,figsize=(10,5))
plt1=sns.boxplot(housing['price'],ax = axs[0,0])
plt2=sns.boxplot(housing['area'],ax = axs[0,1])
plt3=sns.boxplot(housing['bedrooms'],ax = axs[0,2])
plt1=sns.boxplot(housing['bathrooms'],ax = axs[1,0])
plt2=sns.boxplot(housing['stories'],ax = axs[1,1])
plt3=sns.boxplot(housing['parking'],ax = axs[1,2])

plt.tight_layout()


# In[79]:


#Exploratory Data Analytics
#Let's now spend some time doing what is arguably the most important step - understanding the data.
#If there is some obvious multicollinearity going on, this is the first place to catch it
#Here's where you'll also identify if some predictors directly have a strong association with the
#outcome variable

#Visualising Numeric Variables
#Let's make a pairplot of all the numeric variables

sns.pairplot(housing)
plt.show()

#A pairplot plot a pairwise relationships in a dataset. The pairplot function creates a grid of Axes such that each variable
#in data will by shared in the y-axis across a single row and in the x-axis across a single column.
# A pairs plot allows us to see both distribution of single variables and relationships between two
#variables. The default pairs plot in seaborn only plots numerical columns
#The pairs plot builds on two basic figures, the histogram and the scatter plot. The histogram on the
#diagonal allows us to see the distribution of a single variable while the scatter plots on the upper
#and lower triangles show the relationship (or lack thereof) between two variables


# In[80]:


#visualizing categorical variables

plt.figure(figsize=(20,12))
plt.subplot(2,3,1)
sns.boxplot(x = 'mainroad', y = 'price', data=housing)
plt.subplot(2,3,2)
sns.boxplot(x = 'guestroom', y = 'price', data=housing)
plt.subplot(2,3,3)
sns.boxplot(x = 'basement', y = 'price', data=housing)
plt.subplot(2,3,4)
sns.boxplot(x = 'hotwaterheating', y = 'price', data=housing)
plt.subplot(2,3,5)
sns.boxplot(x = 'airconditioning', y= 'price', data=housing)
plt.subplot(2,3,6)
sns.boxplot(x = 'furnishingstatus', y = 'price', data=housing)

plt.show()


# In[81]:


#We can also visualise some of these categorical features parallely by using the hue argument.
#Below is the plot for furnishingstatus with airconditioning as the hue.

plt.figure(figsize = (10,5))
sns.boxplot(x ='furnishingstatus', y ='price', hue="airconditioning", data=housing)
plt.show()

#hue - When you use the hue parameter, you'll provide a categorical variable. When you pass a
#categorical variable to hue , sns. boxplot will create separate boxes for the different categories,
#and will color those boxes a different “hue"


# In[82]:


#Data Preparation
#You can see that the dataset has many columns with values as 'Yes' or 'No'.
#But in order to fit a regression line, we would need numerical values and not string.
#Hence, we need to convert them to 1s and 0s, where 1 is a 'Yes' and 0 is a 'No'.

#list of variables to map
varlist =['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

#defining Map function
def binary_map(x):
    return x.map({'yes' : 1, 'no' : 0 })

#applying the function to the housing list
housing[varlist]=housing[varlist].apply(binary_map)


# In[83]:


#check the housing dataframe now

housing.head()


# In[84]:


#Dummy Variables
#The variable furnishingstatus has three levels. We need to convert these levels into integer as well.
#For this, we will use something called dummy variables.

status=pd.get_dummies(housing['furnishingstatus'])
#check wht the dataset 'status' look like
status.head()

#get_dummies() is used for data manipulation.it converts categorical data into dumy.


# In[85]:


#Now, you don't need three columns. You can drop the furnished column, as the type of furnishing can
#be identified with just the last two columns where —
#00 will correspond to furnished
#01 will correspond to unfurnished
#10 will correspond to semi-furnished

#lets drop the first colmn from status df using 'drop_first=True'

status=pd.get_dummies(housing['furnishingstatus'],drop_first =True)
#add the rsults to original housing df

housing=pd.concat([housing, status], axis=1)

housing.head()
#axis=1 indicates colmn concatenation


# In[86]:


#drop 'furnishingstatus' as we have created dummies for it
housing.drop(['furnishingstatus'], axis =1, inplace = True)

housing.head()
#When inplace = True , the data is modified in place, which means it will return nothing and the
#dataframe is now updated. When inplace = False , which is the default, then the operation is
#performed and it returns a copy of the object. You then need to save it to something.


# In[87]:


#splitting the data into training and testing sets

from sklearn.model_selection import train_test_split

#we specify this so that the train and test data set always have the same rows, respectively
np.random.seed(0)
df_train,df_test =train_test_split(housing,train_size =0.7, test_size =0.3, random_state= 100)

#This is to check and validate the data when running the code multiple times. Setting random_state a
#fixed value will guarantee that same sequence of random numbers are generated each time you run the
#code


# In[88]:


#Rescaling the Features
#Here we can see that except for area, all the columns have small integer values. So, it is extremely
#important to rescale the variables so that they have a comparable scale. If we don't have comparable
#scales, then some of the coefficients as obtained by fitting the regression model might be very large
#or very small as compared to the other coefficients. This might become very annoying at the time of
#model evaluation. So, it is advised to use standardization or normalization so that the units of the
#coefficients obtained are all on the same scale. There are two common ways of rescaling:
#1.Min-Max scaling
#2.Standardization (mean-0, sigma-1)
#We will use MinMax scaling.


# In[89]:


from sklearn.preprocessing import MinMaxScaler

scaler= MinMaxScaler()
#MinMaxScaler. For each value in a feature, MinMaxScaler subtracts the minimum value in the feature
#and then divides by the range. The range is the difference between the original maximum and original
#minimum. MinMaxScaler preserves the shape of the original distribution.

#Apply scaler() to all the colmns except the 'yes-no' and 'dummy' variables
num_vars=['area', 'bedrooms', 'bathrooms', 'parking', 'price']

df_train[num_vars]= scaler.fit_transform(df_train[num_vars])

df_train.head()


# In[90]:


#lets check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (16,10))
sns.heatmap(df_train.corr(), annot = True, cmap= "YlGnBu")
plt.show()

#A heatmap is a two-dimensional graphical representation of data where the individual values that are
#contained in a matrix are represented as colors
#A heatmap contains values representing various shades of the same colour for each value to be plotted.
#Usually the darker shades of the chart represent higher values than the lighter shade
#annot: If True, write the data value in each cell.
#cmap: The mapping from data values to color space.
#As you might have noticed, area seems to the correlated to price the most.


# In[91]:


#dividing into X and Y sets for the model building

y_train=df_train.pop('price')
x_train=df_train


# In[92]:


#Model Building
#This time, we will be using the LinearRegression function from SciKit Learn for its compatibility
#with RFE (which is a utility from sklearn)
#RFE
#Recursive feature elimination
# Importing RFE and LinearRegression

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(x_train,y_train)

#Recursive Feature Elimination, or RFE for short, is a popular feature selection algorithm.
#RFE is popular because it is easy to configure and use and because it is effective at selecting those
#features (columns) in a training dataset that are more or most relevant in predicting the target variable.


# In[93]:


#Running RFE with the output number of the variable equal to 6

rfe = RFE(lm,6)               #running RFE
rfe = rfe.fit(x_train,y_train)


# In[94]:


list(zip(x_train.columns,rfe.support_,rfe.ranking_))

#support_ = an array tht indicates whether or not a feature was selected
#ranking_ = the ranking of the features


# In[95]:


col = x_train.columns[rfe.support_]
col


# In[96]:


x_train.columns[~rfe.support_]


# In[97]:


#building model using statsmodel,for the detailed statistics
#creating x_test dataframe with RFE selected variables

x_train_rfe = x_train[col]


# In[98]:


#adding a constant variable

import statsmodels.api as sm
x_train_rfe = sm.add_constant(x_train_rfe)

#add_constant adds a constant column to input data set
#By default, statsmodels fits a line passing through the origin, i.e. it doesn't fit an intercept.
#Hence, you need to use the command 'add_constant' so that it also fits an intercept.
#The intercept (often labeled the constant) is the expected mean value of Y when all X=0.
#The slope indicates the steepness of a line and the intercept indicates the location where it
#intersects an axis. The slope and the intercept define the linear relationship between two variables,
#and can be used to estimate an average rate of change.


# In[99]:


lm = sm.OLS(y_train, x_train_rfe).fit()

#The simplest linear regression algorithm assumes that the relationship between an
#independent variable (x) and dependent variable (y) is of the following form: y = mx + c, which is
#the equation of a line.
#In line with that, OLS is an estimator in which the values of m and c (from the above equation) are
#chosen in such a way as to minimize the sum of the squares of the differences between the observed
#dependent variable and predicted dependent variable. That’s why it’s named ordinary least squares.
#Also, it should be noted that when the sum of the squares of the differences is minimum, the loss is
#also minimum—hence the prediction is better.


# In[100]:


#lets see the summary of our linear model
print(lm.summary())


# In[101]:


#calculate the VIFs for the model
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[102]:


#The Variance Inflation Factor (VIF) is a measure of colinearity among predictor variables within a
#multiple regression.
#A multiple regression is used when a person wants to test the effect of multiple variables on a
#particular outcome. Using variance inflation factors helps to identify the severity of any
#multicollinearity issues so that the model can be adjusted.

vif = pd.DataFrame()
x = x_train_rfe
vif['Features'] = x.columns
vif['VIF'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

#a VIF above 10 indicates high correlation and is cause for concern


# In[103]:


#Residual Analysis of the train data
#So, now to check if the error terms are also normally distributed (which is infact, one of the major
#assumptions of linear regression), let us plot the histogram of the error terms and see what it looks
#like.

y_train_price = lm.predict(x_train_rfe)


# In[104]:


res = (y_train_price - y_train)


# In[105]:


#Importing the required libraries for python

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#%matplotlib inline sets the backend of matplotlib to the 'inline' backend: With this backend, the
#output of plotting commands is displayed inline within frontends like the Jupyter notebook, directly
#below the code cell that produced it.


# In[106]:


#Plot the histogram of the error terms

fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Errors Terms', fontsize = 20)        #plot heading
plt.xlabel('Errors', fontsize =18 )                #x label


# In[107]:


plt.scatter(y_train,res)
plt.show()

#there may be some relation in the error terms


# In[108]:


#Model Evaluation
#Applying the scaling on the test sets

num_vars = ['area', 'stories', 'bathrooms', 'airconditioning', 'prefarea', 'parking', 'price']


# In[109]:


df_test[num_vars] = scaler.fit_transform(df_test[num_vars])

#fit_transform() is used on the training data so that we can scale the training data and also learn
#the scaling parameters of that data. Here, the model built by us will learn the mean and variance of
#the features of the training set. These learned parameters are then used to scale our test data


# In[110]:


#dividing into x_test and y_test

y_test = df_test.pop('price')
x_test = df_test


# In[111]:


#adding constant variable to test dataframe

x_test = sm.add_constant(x_test)


# In[112]:


#now lets use our model to make predications
#creating x_test_new dataframe by dropping variables from x_test

x_test_rfe = x_test[x_train_rfe.columns]


# In[113]:


#making predictions

y_pred=lm.predict(x_test_rfe)


# In[114]:


from sklearn.metrics import r2_score

r2_score(y_test, y_pred)


# In[115]:


#plotting y_test and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y_test, y_pred)
fig.suptitle('y_test vs y_pred', fontsize = 20)
plt.xlabel('y_test',fontsize = 18 )
plt.ylabel('y_test',fontsize = 16)


# In[118]:


#We can see that the equation of our best fitted line is:

price=0.35×area+0.20×bathrooms+0.19×stories+0.10×airconditioning+0.10×parking+0.11×prefarea

