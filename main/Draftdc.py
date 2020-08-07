#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import packages
import os
import pandas as pd
import numpy as np

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


# In[157]:


#reading data 
os.listdir('/Users/sindhuveluguleti/Desktop/Semester -2 /Project/data files of project/data sets old ')


# In[43]:


#reading eddy data and print its shape and data type
eddy=pd.read_csv('/Users/sindhuveluguleti/Desktop/Semester -2 /Project/data files of project/data sets old /Eddy.csv')
print('eddy data shape: ', eddy.shape)#shape 
print(eddy.dtypes)#data type


# In[158]:


eddy.head(43)#for displaying first few rows 


# In[164]:


eddy_clean = eddy.replace({ "--": np.nan, "...": np.nan })#missing values replaced by nan
eddy_clean


# In[202]:


def missing_values_table(eddy_clean):
        mis_val = eddy_clean.isnull().sum()#total missing values column wise 
        eddy_clean.isnull().sum().sum()#total number of missing values in the whole data set
        
        # Percentage of missing values
        mis_val_percent = 100 * eddy_clean.isnull().sum() / len(eddy_clean)
        
        # table for displaying the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # columns renaming 
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sorting
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Printing the summary
        print ("Your selected dataframe has " + str(eddy_clean.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        print("total missing values in the data frame:",eddy_clean.isnull().sum().sum())
        
        # retuns the missing information
        return mis_val_table_ren_columns


# In[61]:


eddy_clean_missing= missing_values_table(eddy_clean)
eddy_clean_missing


# In[51]:


#for ditecting the missing data visually by using  missingno library 
msno.bar(eddy_clean)


# In[52]:


msno.matrix(eddy_clean)#for visulaising the locations of the missing data 


# In[53]:


#msno.matrix(eddy_clean.sample(100)) for particular sample


# In[62]:


msno.heatmap(eddy_clean)


# In[64]:


msno.dendrogram(eddy_clean)# for grouping highly corelated   variables


# In[ ]:


#handling these missing values can be discussed later xg boost auto imputation


# In[160]:


#handling irregular data 
# select numeric columns
eddy_clean_numeric = eddy_clean.select_dtypes(include=[np.number])
numeric_cols = eddy_clean_numeric.columns.values
print(numeric_cols)

# select non numeric columns
eddy_clean_non_numeric = eddy_clean.select_dtypes(exclude=[np.number])
non_numeric_cols = eddy_clean_non_numeric.columns.values
print(non_numeric_cols)


# In[73]:


#outlier detection
eddy_clean['Avg Stride Length'].describe()


# In[161]:


#repetetiveness 
num_rows = len(eddy_clean.index)
low_information_cols = [] #

for col in eddy_clean.columns:
    cnts = eddy_clean[col].value_counts(dropna=False)
    top_pct = (cnts/num_rows).iloc[0]
    
    if top_pct > 0.95:
        low_information_cols.append(col)
        print('{0}: {1:.5f}%'.format(col, top_pct*100))
        print(cnts)
        print()


# In[128]:


# handiling inconsistencies- capitilisation and removal of white spaces 
#get all the unique values in the 'activitytype' column
activity = eddy_clean['Activity Type'].unique()



# sorting
activity.sort()
activity


# In[132]:


eddy_clean['Activity Type'] =eddy_clean['Activity Type'].str.lower()
eddy_clean['Activity Type'].value_counts(dropna=False)


# In[199]:


eddy_clean['Title'] =eddy_clean['Title'].str.lower()
eddy_clean['Title'].value_counts(dropna=False)


# In[201]:


eddy_clean['Activity Type'] = eddy_clean['Activity Type'].str.lstrip()#white spaces 
eddy_clean['Activity Type']


# In[200]:


eddy_clean['Title'] = eddy_clean['Title'].str.strip()#white spaces 
eddy_clean['Title']


# In[167]:


eddy_clean['Date']


# In[192]:


#eddy_clean['Date'] = pd.to_datetime(eddy_clean['Date'],format= "%d-%m-%Y %H:%M:%S")


# In[193]:


#eddy_clean['Date']


# In[194]:


eddy_clean["Date"]= eddy_clean["Date"].dt.strftime("%d/%m/%Y %H:%M:%S ")

print(eddy_clean["Date"])


# In[197]:


eddy_clean.head(20)


# In[ ]:




