# STEP 1 is cleaning data
# Checking different variables of the dataset and cleaning the unwanted values from them
# Remove the null values 
# Remove the unwanted variables
import numpy as np 
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sb 
sb.set()
from sklearn.linear_model import LinearRegression
raw_data = pd.read_csv('1.04.+Real-life+example.csv')
data = raw_data.drop(['Model'] , axis = 1)

data_no_missingvalues = data.dropna(axis=0)

sb.displot((data_no_missingvalues['Price']))

q = data_no_missingvalues['Price'].quantile(0.99)
data1 = data_no_missingvalues[data_no_missingvalues['Price']<q]

q = data1['Mileage'].quantile(0.99)
data2 = data1[data1['Mileage']<q]

data3 = data2[data2['EngineV']<6.5]

q = data3['Year'].quantile(0.01)
data4 = data3[data3['Year']>q]

data_cleaned = data4.reset_index(drop=True)


#STEP 2 is checking the OLS Assumptions

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False, figsize =(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
ax3.set_title('Price and Mileage')


#STEP 3 is relaxing the dissatisfied OLS Assumptions
# ASSUMPTION 1 LINEARITY
log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price'] = log_price


f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['log_price'])
ax1.set_title('Log Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['log_price'])
ax2.set_title('Log Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['log_price'])
ax3.set_title('Log Price and Mileage')





data_cleaned = data_cleaned.drop(['Price'], axis=1)
print(data_cleaned)


# ASSUMPTION 2 NO ENDOGENEITY
# Will be discussed after the regression is created.

# ASSUMPTION 3 NORMALITY AND HOMOSCEDASTICITY AND ZERO MEAN
# Already Relaxed

# ASSUMPTION 4 NO AUTOCORRELATION 
# Dataset is not time series data. So assumption is already relaxed.

# ASSUMPTION 5 NO MULTICOLLINEARITY

from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['Mileage', 'Year' , 'EngineV']]
from statsmodels.tools.tools import add_constant
variables_con = add_constant(variables)
vif_con = pd.DataFrame()
vif_con["VIF"] = [variance_inflation_factor(variables_con.values, i) for i in range(variables_con.shape[1])]
vif_con["features"] = variables_con.columns
print(vif_con)

#STEP ADD DUMMMY VARAIBLES 

data_with_dummies = pd.get_dummies(data_cleaned, drop_first= True)
print(data_with_dummies.columns.values)
cols = ['log_price' , 'Mileage' , 'EngineV' , 'Year' , 'Brand_BMW' , 'Brand_Mercedes-Benz' ,
 'Brand_Mitsubishi' , 'Brand_Renault' , 'Brand_Toyota' , 'Brand_Volkswagen' ,
 'Body_hatch' , 'Body_other' , 'Body_sedan' , 'Body_vagon' , 'Body_van' ,
 'Engine Type_Gas' , 'Engine Type_Other' , 'Engine Type_Petrol' , 'Registration_yes']
data_pre_processed = data_with_dummies[cols]
print(data_pre_processed.head())


# EXTRA EXERCISE (Calculate the variance inflation factors for all variables contained in data_preprocessed. Anything strange?)
# def calculate_vif(dataset):
#     vif = pd.DataFrame()
#     vif['Features'] = dataset.columns
#     vif['VIF_Value'] = [variance_inflation_factor(dataset.values, i ) for i in range (dataset.shape[1])]
#     return(vif) 

# features = data_pre_processed.iloc[:,:]
# print(features.head())
# calculate_vif(features)



# variables = data_pre_processed
# vif = pd.DataFrame()
# vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
# vif["features"] = variables.columns
# vif["features"] = variables.columns
# print(vif)

# variables = data_pre_processed.drop(['log_price'],axis=1)
# vif = pd.DataFrame()
# vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
# vif["features"] = variables.columns
# print(vif)

# CREATING LINEAR REGRESSION
# STEP 1 - DECLARE THE INPUTS AND TARGETS
targets = data_pre_processed['log_price']
inputs = data_pre_processed.drop(['log_price'], axis=1)

# STEP 2 - SCALE THE DATA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)

# STEP 3 - TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(inputs_scaled, targets , test_size=0.2 , random_state=365)

# STEP 4 - CREATE THE REGRESSION
reg = LinearRegression()
reg.fit(x_train,y_train) # A simple way to check the final result is to plot the predicted values against the observed values.
