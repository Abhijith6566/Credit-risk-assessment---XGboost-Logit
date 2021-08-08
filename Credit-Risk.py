#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt 
import sklearn as sk
from sklearn.model_selection import train_test_split
import xgboost 
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import preprocessing
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import pyplot
from xgboost import plot_importance


# In[2]:


data=pd.read_csv("C:/Users/asibh/OneDrive/Desktop/New folder/credit_risk_dataset.csv")


# In[3]:


data.isnull().sum()


# In[4]:


data.describe()


# In[5]:


data=data.dropna(axis=0)


# In[6]:


data.describe()


# In[7]:


boxplot=data.boxplot(column=['person_income'])


# In[ ]:





# In[8]:


data = data[data["person_age"]<=100]
data = data[data["person_emp_length"]<=100]
data = data[data["person_income"]<= 4000000]


# In[9]:


rat_non_default= data[data.loan_status==0]


# In[10]:


rat_non_default['loan_status'].count()/ data['loan_status'].count()


# In[11]:


import plotly.express as px
fig = px.parallel_categories(data, color_continuous_scale=px.colors.sequential.RdBu, color="loan_status",
dimensions=['person_home_ownership', 'loan_intent', "loan_grade", 'cb_person_default_on_file'], labels={col:col.replace('_', ' ') for col in data.columns})
fig.show()


# In[ ]:





# In[12]:


n_data=pd.get_dummies(data=data,columns=['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'])


# In[13]:


n_data.head()


# In[14]:


Y=n_data['loan_status']
X=n_data.drop('loan_status',axis=1)
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=0, test_size=0.20)


# In[15]:


lg=LogisticRegression(random_state=0)
lg.fit(x_train,y_train)
preds=lg.predict(x_test)
preds_prob=lg.predict_proba(x_test)
print('Logit','\n',classification_report(y_test,lg.predict(x_test)))


# In[16]:


XGBC=XGBClassifier(n_estimators=1000,learning_rate=0.01,use_label_encoder =False)
XGBC.fit(x_train,y_train)
predsX=XGBC.predict(x_test)
preds_probX=XGBC.predict_proba(x_test)
print('XG boost tree','\n',classification_report(y_test,XGBC.predict(x_test)))


# In[17]:


fig = plt.figure(figsize=(14,10))
plt.plot([0, 1], [0, 1],'r--')
probslg = preds_prob[:, 1]
fpr, tpr, thresh = metrics.roc_curve(y_test, probslg)
auclg = roc_auc_score(y_test, probslg)
plt.plot(fpr, tpr, label=f'Logistic Regression, AUC = {str(round(auclg,3))}')
probsxgb = preds_probX[:, 1]
fpr, tpr, thresh = metrics.roc_curve(y_test, probsxgb)
aucxgb = roc_auc_score(y_test, probsxgb)
plt.plot(fpr, tpr, label=f'XGBoost, AUC = {str(round(aucxgb,3))}')
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.title("ROC curve")
plt.rcParams['axes.titlesize'] = 18
plt.legend()
plt.show()


# In[19]:


plot_importance(XGBC, importance_type='gain',max_num_features=10)
ax1.set_title('Feature Importance by Information Gain', fontsize = 18)
ax1.set_xlabel('Gain')
plt.show()


# In[ ]:





# In[ ]:




