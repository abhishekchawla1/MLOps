#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import set_config
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder,OneHotEncoder


# In[5]:


get_ipython().system('pip install feature-engine')
from feature_engine.encoding import CountFrequencyEncoder
from feature_engine.outliers.winsorizer import Winsorizer


# In[7]:


import mlflow


# In[8]:


df=pd.read_csv(r"C:\Users\ASUS\Downloads\Titanic-Dataset.csv")


# In[9]:


df.head()


# In[10]:


df.isnull().sum()


# In[11]:


df.duplicated().any()


# In[13]:


df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)


# In[14]:


df.shape


# In[17]:


df.columns=[x.lower() for x in df.columns]


# In[18]:


df['family']=df['sibsp']+df['parch']


# In[19]:


df.drop(columns=['sibsp','parch'],inplace=True)


# In[20]:


df.head()


# In[22]:


ax=df['survived'].value_counts().plot(kind='bar')
for bars in ax.containers:
    ax.bar_label(bars)


# In[23]:


for col in ['sex','embarked','family']:
    ax=df[col].value_counts().plot(kind='bar',cmap='Pastel1')
    for bars in ax.containers:
        ax.bar_label(bars)
    plt.title(f'Distribution of {col}')
    plt.show()


# In[25]:


sns.boxplot(df)
plt.show()


# In[61]:


X=df.drop(columns=['survived'])
y=df['survived']


# In[62]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[28]:


age_pipeline=Pipeline(steps=[('imputation',SimpleImputer()),('outlier_removal',Winsorizer(capping_method='gaussian',fold=3)),('normalization',StandardScaler())])


# In[29]:


age_pipeline


# In[31]:


fare_pipeline=Pipeline(steps=[('outlier_removal',Winsorizer(capping_method='iqr',fold=2)),('normalization',StandardScaler())])


# In[32]:


fare_pipeline


# In[64]:


emb_pipeline=Pipeline(steps=[('imputation',SimpleImputer(strategy='most_frequent')),('encoding',CountFrequencyEncoder(encoding_method='count')),('normalization',StandardScaler())])


# In[65]:


emb_pipeline


# In[66]:


preprocessor=ColumnTransformer(transformers=[('age',age_pipeline,['age']),('fare',fare_pipeline,['fare']),('embarked',emb_pipeline,['embarked']),('sex',OneHotEncoder(sparse_output=False),['sex'])],remainder='passthrough',n_jobs=-1)


# In[67]:


preprocessor


# In[68]:


d=preprocessor.fit_transform(X_train)


# In[46]:


dt=preprocessor.get_params()


# In[50]:


params=RandomForestClassifier().get_params()


# In[54]:


model_params={'bootstrap': True,
 'ccp_alpha': 0.0,
 'class_weight': None,
 'criterion': 'gini',
 'max_depth': None,
 'max_features': 'sqrt',
 'max_leaf_nodes': None,
 'max_samples': None,
 'min_impurity_decrease': 0.0,
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'min_weight_fraction_leaf': 0.0,
 'n_estimators': 500,
 'n_jobs': None,
 'oob_score': False,
 'random_state': None,
 'verbose': 0,
 'warm_start': False}


# In[55]:


model=RandomForestClassifier(**model_params)


# In[76]:


model.fit(X_train,y_train)


# In[77]:


from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,ConfusionMatrixDisplay


# In[78]:


y_pred=model.predict(X_test)


# In[79]:


a=accuracy_score(y_test,y_pred)
p=precision_score(y_test,y_pred)
r=recall_score(y_test,y_pred)
c=confusion_matrix(y_test,y_pred)
ConfusionMatrixDisplay(c,display_labels=model.classes_).plot()


# In[80]:


mlflow.is_tracking_uri_set()


# In[81]:


mlflow.set_tracking_uri('http://127.0.0.0:8000')


# In[ ]:


mlflow.is_tracking_uri_set()

mlflow.set_experiment('titanic')

with mlflow.start_run():
    mlflow.log_params(preprocessor.get_params())
    mlflow.log_metrics({'accuracy':a,'precision':p,'recall':r})

