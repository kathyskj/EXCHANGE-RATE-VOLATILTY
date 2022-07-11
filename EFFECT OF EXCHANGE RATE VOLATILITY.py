#!/usr/bin/env python
# coding: utf-8

# In[79]:


# necessary imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')


# In[80]:


import os

os.chdir('..')
#C:\Users\Khadijah Jasni\Desktop\SITC_EXPORT

df= pd.read_csv('C:/Users/Khadijah Jasni/Desktop/C_ANALYSIS.csv', encoding='latin-1')


# In[81]:


df.head()


# In[82]:


df.info()


# In[24]:


stat=df.describe()
stat.head()


# In[25]:



stat.to_csv(r'C:\Users\Khadijah Jasni\Desktop\sum_stat.csv')


# In[26]:


# looking for outliers using box plot

plt.figure(figsize = (20, 8))
sns.boxplot(data = df, width = 0.8)
plt.show()


# In[27]:


# creating features and label variable

X = df[['E_SITC4','E_SITC1','E_SITC5','E_SITC2','E_SITC0','E_SITC7','E_SITC6','E_SITC3','E_SITC8','E_SITC9','I_SITC4','I_SITC1','I_SITC5','I_SITC2','I_SITC0','I_SITC7','I_SITC6','I_SITC3','I_SITC8','I_SITC9','total_export','total_import','trade_balance','growthrate_export','growthrate_import']]
y= df[['CNY','KRW','THB','TWD','HKD','SGD','JPY','USD']]
  


# In[28]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[29]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[30]:


X_scaled


# In[31]:


# checking for multicollinearity using `VIF` and `correlation matrix`

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif['VIF'] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
vif['Features'] = X.columns

vif


# In[32]:


# Heatmap

fig, ax = plt.subplots(figsize = (16, 8))
sns.heatmap(df.corr(), annot = True, fmt = '1.2f', annot_kws = {'size' : 10}, linewidth = 1)
plt.show()


# In[33]:


#correlation analysis
corr = df.corr()
corr.head()


# In[34]:


corr.to_csv(r'C:\Users\Khadijah Jasni\Desktop\corr.csv')


# In[35]:


import statsmodels.formula.api as smf

lm = smf.ols(formula = 'CNY ~ growthrate_import', data = df).fit()
lm.summary()


# In[36]:


import statsmodels.formula.api as smf

lm = smf.ols(formula = 'CNY ~ total_export', data = df).fit()
lm.summary()


# In[37]:


# splitting data into training asnd test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.30, random_state = 0)


# In[38]:


# fitting training data to model

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)


# In[39]:


# prediction of model

lry_pred = lr.predict(X_test)


# In[40]:


# training accuracy of model

lr.score(X_train, y_train)


# In[41]:


# test accuracy of model

lr.score(X_test, y_test)


# In[42]:


# creating a function to create adhusted R-Squared

def adj_r2(X, y, model):
    r2 = model.score(X, y)
    n = X.shape[0]
    p = X.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    return adjusted_r2


# In[43]:


print(adj_r2(X_train, y_train, lr))


# In[44]:


print(adj_r2(X_test, y_test, lr))


# In[45]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Evaluate mean absolute error
print('Mean Absolute Error(MAE):', mean_absolute_error(y_test, lry_pred))
# Evaluate mean squared error
print("Mean Squared Error(MSE):", mean_squared_error(y_test, lry_pred))
# Evaluate root mean squared error
print("Root Mean Squared Error(RMSE):", np.sqrt(mean_squared_error(y_test, lry_pred)))
print("R-Square:",r2_score(y_test, lry_pred))


# # LASSO REGRESSION

# In[46]:


from sklearn.linear_model import Lasso, LassoCV, MultiTaskLassoCV

lasso_cv = MultiTaskLassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
lasso_cv.fit(X_train, y_train)


# In[47]:


# best alpha parameter

alpha = lasso_cv.alpha_
alpha


# In[48]:


lasso = Lasso(alpha = lasso_cv.alpha_)
lasso.fit(X_train, y_train)


# In[49]:


lasso.score(X_train, y_train)


# In[50]:


lasso.score(X_test, y_test)


# In[51]:


print(adj_r2(X_train, y_train, lasso))


# In[52]:


print(adj_r2(X_test, y_test, lasso))


# In[75]:


y_pred = lasso.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))


# # Ridge Regression

# In[53]:


from sklearn.linear_model import Ridge, RidgeCV

alphas = np.random.uniform(0, 10, 50)
ridge_cv = RidgeCV(alphas = alphas, cv = 10, normalize = True)
ridge_cv.fit(X_train, y_train)


# In[54]:


# best alpha parameter

alpha = ridge_cv.alpha_
alpha


# In[55]:


ridge = Ridge(alpha = ridge_cv.alpha_)
ridge.fit(X_train, y_train)


# In[56]:


ridge.score(X_train, y_train)


# In[57]:


ridge.score(X_test, y_test)


# In[58]:


print(adj_r2(X_train, y_train, ridge))


# In[59]:


print(adj_r2(X_test, y_test, ridge))


# In[74]:


y_pred = ridge.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))


# # Elastic

# In[61]:


from sklearn.linear_model import ElasticNet, ElasticNetCV, MultiTaskElasticNetCV

elastic_net_cv = MultiTaskElasticNetCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
elastic_net_cv.fit(X_train, y_train)


# In[62]:


# best alpha parameter

alpha = elastic_net_cv.alpha_
alpha


# In[63]:


# l1 ratio 

elastic_net_cv.l1_ratio


# In[64]:


elastic_net = ElasticNet(alpha = elastic_net_cv.alpha_, l1_ratio = elastic_net_cv.l1_ratio)
elastic_net.fit(X_train, y_train)


# In[65]:


elastic_net.score(X_train, y_train)


# In[66]:


elastic_net.score(X_test, y_test)


# In[67]:


print(adj_r2(X_train, y_train, elastic_net))


# In[68]:


print(adj_r2(X_test, y_test, elastic_net))


# In[73]:


y_pred = elastic_net.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))


# KNN

# In[ ]:


from sklearn import neighbors
knn_reg = neighbors.KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(X_train, y_train)
predictions = knn_reg.predict(X_test)

# Evaluate mean absolute error
print('Mean Absolute Error(MAE):', mean_absolute_error(y_test,predictions))
# Evaluate mean squared error
print("Mean Squared Error(MSE):", mean_squared_error(y_test, predictions))
# Evaluate root mean squared error
print("Root Mean Squared Error(RMSE):", np.sqrt(mean_squared_error(y_test,
predictions)))
print("R-Square:",r2_score(y_test, predictions))


# SVM

# In[ ]:


import numpy as np
from sklearn.svm import SVR

svr_reg = SVR(kernel="linear")
svr_reg.fit(X_train, y_train)
predictions = svr_reg.predict(X_test)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Evaluate mean absolute error
print('Mean Absolute Error(MAE):', mean_absolute_error(y_test,predictions))
# Evaluate mean squared error
print("Mean Squared Error(MSE):", mean_squared_error(y_test, predictions))
# Evaluate root mean squared error
print("Root Mean Squared Error(RMSE):", np.sqrt(mean_squared_error(y_test,
predictions)))
print("R-Square:",r2_score(y_test, predictions))

