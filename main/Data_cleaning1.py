#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages
import os
import pandas as pd
import pandas_profiling
import numpy as np

import warnings
warnings.filterwarnings('ignore')
import xgboost
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import (GradientBoostingRegressor, GradientBoostingClassifier)
pd.set_option('max.columns',100)
pd.set_option('max.rows',500)


import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None

import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression

import missingno as msno


# In[2]:


#reading data 
os.listdir('/Users/sindhuveluguleti/Desktop/Semester -2 /Project/data files of project/data sets old ')
#data_path = '{}/data'.format(os.path.pardir)
#athlete_csv_file = '{}/{}'.format(data_path, 'Eduardo Oliveira (Intermediate).csv')
#print(athlete_csv_file)


# In[3]:


#reading eddy data and print its shape and data type
#eddy=pd.read_csv(athlete_csv_file)
eddy=pd.read_csv('/Users/sindhuveluguleti/Desktop/Semester -2 /Project/data files of project/data sets old /Eddy.csv')
print('eddy data shape: ', eddy.shape)#shape 
print(eddy.dtypes)#data type


# In[4]:


for col in eddy.columns: 
    print(col) 


# In[5]:


eddy.drop(['Favorite','Aerobic TE','Avg Run Cadence','Max Run Cadence','Avg Stride Length','Avg Vertical Ratio','Avg Vertical Oscillation','Avg Ground Contact Time'
,'Avg GCT Balance','L/R Balance','Grit','Flow','Total Reps','Total Sets','Bottom Time','Min Temp','Surface Interval','Decompression','Best Lap Time','Max Temp'], axis =1, inplace=True)


# In[6]:


print(eddy)


# In[7]:


eddy.columns= eddy.columns.str.replace(',', '')
print(eddy.columns)


# In[8]:


eddy.head(43)


# In[9]:


eddy = eddy.replace({ "--": np.nan, "...": np.nan })#missing values replaced by nan
eddy


# In[10]:


#capitalization
eddy['Activity Type'] = eddy['Activity Type'].str.lower()
eddy['Title'] = eddy['Title'].str.lower()
print(eddy)


# In[11]:


#formats
eddy['Elev Gain'] = eddy['Elev Gain'].str.replace(',', '')
eddy['Elev Gain'] = eddy['Elev Gain'].astype(float)


# In[12]:


eddy["Elev Gain"] = pd.to_numeric(eddy["Elev Gain"])
print(eddy.dtypes)


# In[13]:


eddy['Elev Loss'] = eddy['Elev Loss'].str.replace(',', '')
eddy['Elev Loss'] = eddy['Elev Loss'].astype(float)


# In[14]:


eddy['Elev Loss'] = pd.to_numeric(eddy['Elev Loss'])
print(eddy.dtypes)


# In[15]:


eddy['Distance'] = eddy['Distance'].str.replace(',', '')
eddy['Distance'] = eddy['Distance'].astype(float)


# In[16]:


eddy['Distance'] = pd.to_numeric(eddy['Distance'])
print(eddy.dtypes)


# In[17]:


eddy['Calories'] = eddy['Calories'].str.replace(',', '')
eddy['Calories'] = eddy['Calories'].astype(float)


# In[18]:


eddy['Calories'] = pd.to_numeric(eddy['Calories'])
print(eddy.dtypes)


# In[19]:


eddy['Max Power'] = eddy['Max Power'].str.replace(',', '')
eddy['Max Power'] = eddy['Max Power'].astype(float)
eddy['Max Power'] = pd.to_numeric(eddy['Max Power'])
print(eddy.dtypes)


# In[20]:


eddy['Avg Power'] = eddy['Avg Power'].astype(str)
eddy['Avg Power'] = eddy['Avg Power'].str.replace(',', '')
eddy['Avg Power'] = eddy['Avg Power'].astype(float)
eddy['Avg Power'] = pd.to_numeric(eddy['Avg Power'])
print(eddy.dtypes)


# In[21]:


eddy[['Max Avg Power (20 min)','Avg Power','Avg Stroke Rate','Avg HR', 'Max HR','Total Strokes','Avg. Swolf','Avg Bike Cadence','Max Bike Cadence','Normalized Power® (NP®)', 'Number of Laps']] = eddy[['Max Avg Power (20 min)','Avg Power','Avg Stroke Rate','Avg HR', 'Max HR','Total Strokes','Avg. Swolf','Avg Bike Cadence','Max Bike Cadence','Normalized Power® (NP®)' ,'Number of Laps']].apply(pd.to_numeric)
print(eddy.dtypes)


# In[22]:


eddy['Date'] = pd.to_datetime(eddy['Date'], errors='coerce')
eddy["Date"]= eddy["Date"].dt.strftime("%d/%m/%Y %H:%M:%S ")
#eddy['Date'] = pd.to_datetime(eddy['Date'], errors='coerce')
eddy['Date']=pd.to_datetime(eddy['Date'])


print(eddy["Date"])


# In[23]:


print(eddy.dtypes)


# In[24]:


#handling irregular data 
# select numeric columns
eddy_numeric = eddy.select_dtypes(include=[np.number])
numeric_cols = eddy_numeric.columns.values
print(numeric_cols)

# select non numeric columns
eddy_non_numeric = eddy.select_dtypes(exclude=[np.number])
non_numeric_cols = eddy_non_numeric.columns.values
print(non_numeric_cols)


# In[25]:


def find_missing_percent(data):
    """
    Returns dataframe containing the total missing values and percentage of total
    missing values of a column.
    """
    miss_eddy = pd.DataFrame({'ColumnName':[],'TotalMissingVals':[],'PercentMissing':[]})
    for col in data.columns:
        sum_miss_val = data[col].isnull().sum()
        percent_miss_val = round((sum_miss_val/data.shape[0])*100,2)
        miss_eddy = miss_eddy.append(dict(zip(miss_eddy.columns,[col,sum_miss_val,percent_miss_val])),ignore_index=True)
    return miss_eddy

miss_eddy = find_missing_percent(eddy)
#'''Columns with missing values'''
print(f"Number of columns with missing values: {str(miss_eddy[miss_eddy['PercentMissing']>0.0].shape[0])}")
display(miss_eddy[miss_eddy['PercentMissing']>0.0])
#'''Drop the columns with more than 90% of missing values'''
#drop_cols = miss_df[miss_df['PercentMissing'] >90.0].ColumnName.tolist()
#eddy = eddy.drop(drop_cols,axis=1)


# In[26]:


# In[51]:


#for ditecting the missing data visually by using  missingno library 
msno.bar(eddy)


# In[27]:


msno.matrix(eddy)#for visulaising the locations of the missing data 


# In[28]:


msno.heatmap(eddy)


# In[29]:


msno.dendrogram(eddy)# for grouping highly corelated   variable


# In[ ]:


pandas_profiling.ProfileReport(eddy)


# In[ ]:


#handling irregular data 
# select numeric columns
eddy_numeric = eddy.select_dtypes(include=[np.number])
numeric_cols = eddy_numeric.columns.values
print(numeric_cols)

# select non numeric columns
eddy_categoric = eddy.select_dtypes(exclude=[np.number])
categoric_cols = eddy_categoric.columns.values
print(categoric_cols)


# In[ ]:


#mean imputation
def mean_imputation(eddy_numeric):
    for col in eddy_numeric.columns:
        mean = eddy_numeric[col].mean()
        eddy_numeric[col] = eddy_numeric[col].fillna(mean)
    return eddy_numeric

eddy_mean_imp = mean_imputation(eddy_numeric)
eddy_mean_imp.head()


# In[ ]:


#regression imputaion
'''Select all the numeric columns for regression imputation'''
eddy_numeric_regr = eddy[numeric_cols]
'''Numeric columns with missing values which acts as target in training'''
target_cols = ['Distance','Calories','Avg HR','Max HR','Elev Gain','Elev Loss','Avg Bike Cadence']
'''Predictors for regression imputation'''
predictors = eddy_numeric_regr.drop(target_cols, axis =1)



# In[ ]:


def find_missing_index(eddy_numeric_regr, target_cols):
    """
    Returns the index of the missing values in the columns.
    """
    miss_index_dict = {}
    for tcol in target_cols:
        index = eddy_numeric_regr[tcol][eddy_numeric_regr[tcol].isnull()].index
        miss_index_dict[tcol] = index
    return miss_index_dict


# In[ ]:


def regression_imputation(eddy_numeric_regr, target_cols, miss_index_dict):
    """
    Fits XGBoost Regressor and replaces the missing values with
    the prediction.
    """
    for tcol in target_cols:
        y = eddy_numeric_regr[tcol]
        '''Initially impute the column with mean'''
        y = y.fillna(y.mean())
        xgb = xgboost.XGBRegressor(objective="reg:squarederror", random_state=42)
        '''Fit the model where y is the target column which is to be imputed'''
        xgb.fit(predictors, y)
        predictions = pd.Series(xgb.predict(predictors),index= y.index)    
        index = miss_index_dict[tcol]
        '''Replace the missing values with the predictions'''
        eddy_numeric_regr[tcol].loc[index] = predictions.loc[index]
    return eddy_numeric_regr

miss_index_dict = find_missing_index(eddy_numeric_regr, target_cols)
eddy_numeric_regr = regression_imputation(eddy_numeric_regr, target_cols, miss_index_dict)
eddy_numeric_regr.head()


# In[ ]:


def mode_imputation(eddy_categoric):
    """
    Mode Imputation
    """
    for col in eddy_categoric.columns:
        mode = eddy_categoric[col].mode().iloc[0]
        eddy_categoric[col] = eddy_categoric[col].fillna(mode)
    return eddy_categoric

eddy_mode_imp = mode_imputation(eddy_categoric)
'''Concatenate the mean and mode imputed columns'''
#eddy_imputed = pd.concat([eddy_mean_imp, eddy_mode_imp], axis = 1)
#eddy_imputed.head()
eddy_categoric.head()


# In[ ]:


def mice_imputation_numeric(eddy_numeric):
    """
    Impute numeric data using MICE imputation with Gradient Boosting Regressor.
    (we can use any other regressors to impute the data)
    """
    iter_imp_numeric = IterativeImputer(GradientBoostingRegressor())
    imputed_eddy = iter_imp_numeric.fit_transform(eddy_numeric)
    eddy_numeric_imp = pd.DataFrame(imputed_eddy, columns = eddy_numeric.columns, index= eddy_numeric.index)
    return eddy_numeric_imp


# In[ ]:


def mice_imputation_categoric(eddy_categoric):
    """
    Impute categoric data using MICE imputation with Gradient Boosting Classifier.
    Steps:
    1. Ordinal Encode the non-null values
    2. Use MICE imputation with Gradient Boosting Classifier to impute the ordinal encoded data
    (we can use any other classifier to impute the data)
    3. Inverse transform the ordinal encoded data.
    """
    ordinal_dict={}
    for col in eddy_categoric:
        '''Ordinal encode train data'''
        ordinal_dict[col] = OrdinalEncoder()
        nn_vals = np.array(eddy_categoric[col][eddy_categoric[col].notnull()]).reshape(-1,1)
        nn_vals_arr = np.array(ordinal_dict[col].fit_transform(nn_vals)).reshape(-1,)
        eddy_categoric[col].loc[eddy_categoric[col].notnull()] = nn_vals_arr

    '''Impute the data using MICE with Gradient Boosting Classifier'''
    iter_imp_categoric = IterativeImputer(GradientBoostingClassifier(), max_iter =5, initial_strategy='most_frequent')
    imputed_eddy = iter_imp_categoric.fit_transform(eddy_categoric)
    eddy_categoric_imp = pd.DataFrame(imputed_eddy, columns =eddy_categoric.columns,index = eddy_categoric.index).astype(int)
    
    '''Inverse Transform'''
    for col in eddy_categoric_imp.columns:
        oe = ordinal_dict[col]
        eddy_arr= np.array(eddy_categoric_imp[col]).reshape(-1,1)
        eddy_categoric_imp[col] = oe.inverse_transform(eddy_arr)
        
    return eddy_categoric_imp

eddy_numeric_imp  = mice_imputation_numeric(eddy_numeric)
eddy_categoric_imp = mice_imputation_categoric(eddy_categoric)

'''Concatenate Numeric and Categoric Training and Test set data '''
eddy_mice_imp = pd.concat([eddy_numeric_imp, eddy_categoric_imp], axis = 1)
eddy_mice_imp.head()


# In[ ]:


#eddy_mice_imp = pd.concat([eddy_numeric_imp, eddy_categoric_imp,], axis = 1)
#eddy_mice_imp.head()


# In[ ]:


eddy_numeric_imp.head()


# In[ ]:


def Linear_interpolation(eddy_categoric):
    for col in eddy_categoric.columns:
        eddy_categoric = eddy_categoric.interpolate(method='linear', limit_direction='forward', axis=0)
    return(eddy_categoric)


# In[ ]:


eddy_categoric


# In[ ]:




