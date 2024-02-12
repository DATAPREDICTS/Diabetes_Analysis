#!/usr/bin/env python
# coding: utf-8

# # DIABETES PREDICTION ANALYSIS

# ### IMPORTING PACKAGES

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ### DATA COLLECTION 

# In[2]:


df=pd.read_csv('diabetes-dataset.csv', encoding='unicode_escape')


# In[3]:


df.head()


# In[4]:


df.shape


# ### DATA CLEANING

# In[5]:


df.info()


# In[6]:


pd.isnull(df).sum()


# In[7]:


df.describe()


# ### DATA VISUALIZATION

# In[12]:


sns.set(rc={'figure.figsize':(10,5)})
sns.barplot(x='Outcome',y='Pregnancies',data=df)


# In[46]:


diabetes_df=df.groupby(['Age'],as_index=False)['Pregnancies'].sum().sort_values(by='Pregnancies',ascending=False).head(10)

sns.barplot(x='Age',y='Pregnancies',data=diabetes_df)


# In[16]:


plt.figure(figsize=(10, 7))

sns.heatmap(df.corr(), annot=True, linewidths=0.2, fmt='.1f', cmap='coolwarm')
plt.show()


# ### FEATURE SELECTION

# In[27]:


feature_columns =df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']]

feature_columns.head()


# In[28]:


outcome_column =df['Outcome']
outcome_column.head()


# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


X_train, X_test, y_train, y_test = train_test_split( feature_columns, outcome_column, test_size=0.2, random_state=5)


# In[31]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ### LOADING MODEL FOR PREDICITION

# In[32]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# In[33]:


model = LogisticRegression()


# In[34]:


model = model.fit(X_train, y_train)


# In[35]:


score = model.predict(X_train)


# ### MODEL TESTING AND EVALUATION

# In[36]:


print("Training Score: ", model.score(X_train, y_train))
print("Testing Score:  ", model.score(X_test, y_test))


# In[37]:


pred = model.predict(X_test)
print("Model Accuracy is : ", pred)


# In[38]:


model.intercept_


# In[39]:


model.coef_


# In[40]:


accuracy_score(y_test, pred)


# In[41]:


df.columns


# ## THANK YOU!

# #### CONNECT WITH ME: 
# ##### LinkedIn: https://www.linkedin.com/in/harshita-sharma-b68154220/
# ##### GitHub: https://github.com/DATAPREDICTS
# ##### Instagram: https://www.instagram.com/datapredicts?utm_source=qr&igsh=czVzc2k5c3oxOWQ4
# ##### YouTube: https://youtube.com/@Datapredicts?si=eDKAqVciVxg23zab
