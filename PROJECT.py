#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import sklearn as skl


# In[ ]:


os.chdir(r'D:\Data Science\Edwisor\Projects\Customer Segmentation')
os.getcwd()


# In[ ]:


credit_card = pd.read_csv('CC GENERAL.csv')
credit_card.head(10)


# # Basic Info

# In[ ]:


credit_card.info()


# In[ ]:


credit_card.describe().T


# In[ ]:


# There are some missing values in MINIMUM_PAYMENTS and CREDIT_LIMIT which we wil impute or drop later as per the requirement


# # EDA

# In[ ]:


# Missing Values Analysis
missing_values = pd.DataFrame(credit_card.isnull().sum())
missing_values = missing_values.reset_index()
missing_values = missing_values.rename(columns = {'index':'Variables', 0:'Percentage'})
missing_values['Percentage'] = (missing_values['Percentage']/len(credit_card))*100
missing_values = missing_values.sort_values('Percentage', ascending = False)
missing_values


# In[ ]:


plt.figure(figsize=(17,10))
sns.heatmap(credit_card.isna())


# In[ ]:


# Imputing the values using median
credit_card['MINIMUM_PAYMENTS'] = credit_card['MINIMUM_PAYMENTS'].fillna(credit_card['MINIMUM_PAYMENTS'].median())
credit_card['CREDIT_LIMIT'] = credit_card['CREDIT_LIMIT'].fillna(credit_card['CREDIT_LIMIT'].median())


# In[ ]:


# We don't need the customer id so let's drop it
credit_card.drop(['CUST_ID'],axis=1,inplace =True)
credit_card.head()


# In[ ]:


# Checking for outliers
plt.figure(figsize= (15,60))

for i in range(0,17):
    plt.subplot(17, 3, i+1)
    plt.xlabel(credit_card.columns[i])
    plt.boxplot(credit_card[credit_card.columns[i]])

#plt.show


# In[ ]:


credit_card.duplicated().sum()


# In[ ]:


# We have outliers in almost all the variables but we are not going to remove or impute them because these values are 
# important for the data as each customer has different spending which naturally makes the values to vary by a lot.
# Also a lot of variables have 0 value due to which log transformation can not be applied on the dataframe.


# In[ ]:


# let's check the correlation of variables
correlation_matrix = credit_card.corr()
correlation_matrix
plt.figure(figsize=(14,10))
sns.heatmap(correlation_matrix,vmin=1, vmax=-1, annot= True)


# In[ ]:


# Balance has positive correlation with CASH_ADVANCE, CASH_ADVANCE_FREQUENCY, CREDIT_LIMIT and limit_usage
# Purchase has high positive correlation with one_off_purchases


# # Visualizing Data

# In[ ]:


plt.figure(figsize=(15,70))
for i in range(0,17):
    plt.subplot(17,1,i+1)
    sns.distplot(credit_card[credit_card.columns[i]], kde_kws={'bw': 0.05,'lw':2}, color='red')
plt.tight_layout


# # Creating KPI'S

# In[ ]:


credit_card['limit_usage'] = credit_card['BALANCE']/credit_card['CREDIT_LIMIT']*100


# In[ ]:


# How much amount customer spends monthly
credit_card['monthly_amount_spent'] = credit_card['PURCHASES']/credit_card['TENURE']


# In[ ]:


credit_card['min_pay_ratio'] = credit_card['PAYMENTS']/credit_card['MINIMUM_PAYMENTS']


# In[ ]:


# creating different type of purchases
def purchase(credit_card):
    if credit_card['ONEOFF_PURCHASES'] > 0 and credit_card['INSTALLMENTS_PURCHASES'] > 0:
        return 'ot_oneoff_installment'
    elif credit_card['ONEOFF_PURCHASES'] > 0 and credit_card['INSTALLMENTS_PURCHASES'] == 0:
        return 'one_off'
    elif credit_card['ONEOFF_PURCHASES'] == 0 and credit_card['INSTALLMENTS_PURCHASES'] > 0:
        return 'installment_purchase'
    else:
        return 'none'


# In[ ]:


credit_card['purchase_type'] = credit_card.apply(purchase, axis = 1)


# In[ ]:


credit_card['monthly_cash_advance'] = credit_card['CASH_ADVANCE']/credit_card['TENURE']


# In[ ]:


corr1 = credit_card.corr()
plt.figure(figsize=(14,10))
corr1
sns.heatmap(corr1, vmin=-1, annot=True, )


# In[ ]:


credit_card.info()


# In[ ]:


credit_card_kpi = credit_card.copy()


# # Creating a model

# In[ ]:


credit_card.drop(['limit_usage','monthly_amount_spent','min_pay_ratio','purchase_type','monthly_cash_advance'], axis = 1, inplace = True)


# In[ ]:


# Let us make a copy of original dataset
df = credit_card.copy()


# In[ ]:


#from sklearn.metrics import mean_squared_error
import sklearn.cluster as cluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pca as PCA
from factor_analyzer import FactorAnalyzer
#from factor_analyzer import calculate_bartlett_sphericity
#from factor_analyzer import calculate_kmo


# In[ ]:


scaler = StandardScaler()
df_std = pd.DataFrame(scaler.fit_transform(df))


# In[ ]:


for i in range(0,17):
    df_std.rename(columns = {i : df.columns[i]},inplace = True)


# In[ ]:


df_std.head()


# In[ ]:


# Get eigenvalues to select number of factors for Factor Analysis
fa = FactorAnalyzer(n_factors=4, rotation=None).fit(df_std)
eigenvalues, vectors = fa.get_eigenvalues()
eigenvalues
# As we can see most 5 factors have eigenvalue more than 1, so the factors we can choose are from factors 3-5


# In[ ]:


# Perform Factor Analysis
fa = FactorAnalyzer(n_factors=3, rotation='varimax', method='ml')
credit_factored = fa.fit(df_std)
credit_factored


# In[ ]:


# get the loadings
credit_factored.loadings_ 


# In[ ]:


pd.DataFrame(credit_factored.loadings_, columns=['Factor1', 'Factor2', 'Factor3'], index=df_std.columns)


# In[ ]:


fa.get_communalities()


# In[ ]:


# Let us drop the variables which do not do well in any of the explaining factors
df_std.drop(['TENURE','PRC_FULL_PAYMENT','MINIMUM_PAYMENTS','BALANCE_FREQUENCY'], axis = 1, inplace = True)


# In[ ]:


credit_new = df_std
credit_new


#  # KMeans

# In[ ]:


# Checking the optimal number of clusters
Distortions = []
K = range(1,14)
for k in K:
    kmeanModel = KMeans(n_clusters = k).fit(credit_new)
    Distortions.append(kmeanModel.inertia_)
plt.plot(K, Distortions, 'x-')
plt.xlabel('k')
plt.ylabel('Distortions')
plt.show


# In[ ]:


# From the graph from Elbow Method it is clear that the optimal cluster would be from 4 to 6


# In[ ]:


kmeans = KMeans(init = 'random', n_clusters = 5, random_state = 1)
kmeans.fit(credit_new)
kmeans.cluster_centers_.shape
y_pred = kmeans.fit_predict(credit_new)
y_pred


# In[ ]:


kmeans.cluster_centers_.shape


# In[ ]:


credit_card_kpi['Cluster'] = y_pred


# In[ ]:


credit_card_kpi


# In[ ]:


df1 = pd.DataFrame(credit_card_kpi[credit_card_kpi['Cluster'] == 0])
df2 = pd.DataFrame(credit_card_kpi[credit_card_kpi['Cluster'] == 1])
df3 = pd.DataFrame(credit_card_kpi[credit_card_kpi['Cluster'] == 2])
df4 = pd.DataFrame(credit_card_kpi[credit_card_kpi['Cluster'] == 3])
df5 = pd.DataFrame(credit_card_kpi[credit_card_kpi['Cluster'] == 4])


# In[ ]:


df1.sample(20)


# In[ ]:


df2.sample(20)


# In[ ]:


df3.sample(20)


# In[ ]:


df4.sample(20)


# In[ ]:


df5.sample(20)


# In[ ]:


credit_card_kpi.drop('purchase_type', axis =1, inplace=True)


# In[ ]:


credit_card_kpi


# In[ ]:


for i in range(0,22):
    plt.figure(figsize= (10,8))
    plt.subplot(22,1, i+1)
    sns.jointplot(data=credit_card_kpi, x=credit_card_kpi.columns[i], y='Cluster')


# In[ ]:


sns.scatterplot(data=credit_card_kpi, x='BALANCE',y='PURCHASES', hue='Cluster', palette='deep')


# In[ ]:


sns.scatterplot(data=credit_card_kpi, x='BALANCE',y='CREDIT_LIMIT', hue='Cluster', palette='deep')


# In[ ]:


sns.scatterplot(data=credit_card_kpi, x='ONEOFF_PURCHASES',y='INSTALLMENTS_PURCHASES', hue='Cluster', palette='deep')


# In[ ]:


sns.scatterplot(data=credit_card_kpi, x='CREDIT_LIMIT',y='PURCHASES', hue='Cluster', palette='deep')


# In[ ]:


sns.scatterplot(data=credit_card_kpi, x='CASH_ADVANCE',y='CREDIT_LIMIT', hue='Cluster', palette='deep')


# In[ ]:


sns.scatterplot(data=credit_card_kpi, x='CREDIT_LIMIT',y='PAYMENTS', hue='Cluster', palette='deep')


# In[ ]:


sns.scatterplot(data=credit_card_kpi, x='PURCHASES',y='CASH_ADVANCE', hue='Cluster',palette='deep')


# In[ ]:


# Let us make a new dataset which contains only those variables which seems to define the activities of the customers that 
# can help in making marketing strategy for them
credit_final = credit_card_kpi[['BALANCE','PURCHASES','CASH_ADVANCE','CREDIT_LIMIT','PAYMENTS','MINIMUM_PAYMENTS','PRC_FULL_PAYMENT','Cluster']]
credit_final.head()


# In[ ]:


plt.figure(figsize=(14,10))
sns.pairplot(data=credit_final, hue='Cluster', palette='deep', diag_kws={'bw' : 0.5}, diag_kind='kde')

