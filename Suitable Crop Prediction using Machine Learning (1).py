#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree


# ## The DataSet Taken

# In[2]:


df1 = pd.read_csv('cropred.csv')
df1.head()


# In[3]:


df1.shape


# In[4]:


df1.isnull().sum()


# In[5]:


df1['label'].unique()


# In[6]:


len(df1.label.unique())


# In[7]:


df1.describe()


# ## The Max Elements

# In[8]:


print(df1['N'].max())
print(df1['P'].max())
print(df1['K'].max())
print(df1['temperature'].max())
print(df1['humidity'].max())
print(df1['ph'].max())
print(df1['rainfall'].max())


# ## The Min Elements

# In[9]:


print(df1['N'].min())
print(df1['P'].min())
print(df1['K'].min())
print(df1['temperature'].min())
print(df1['humidity'].min())
print(df1['ph'].min())
print(df1['rainfall'].min())


# In[10]:


df=df1


# ## The Confusion Matrix

# In[11]:


df['label'].unique()
df.dtypes
df['label'].value_counts()
sns.heatmap(df.corr(),annot=True)
features = df[['N','P','K','temperature','humidity','ph','rainfall']]
target = df['label']
labels = df['label']


# # Training and Testing the data using Sciket Learning Library

# In[12]:


acc = []
model = []
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size=0.2,random_state=2)


# ## Descision Tree Algorithm

# In[13]:


from sklearn.tree import DecisionTreeClassifier
DecisionTree = DecisionTreeClassifier(criterion='entropy',random_state=2,max_depth=5)
DecisionTree.fit(Xtrain,Ytrain)

predicted_values = DecisionTree.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Decision Tree')
print("DecisionTrees's Accuracy is: ", x*100)
print(classification_report(Ytest,predicted_values))


# ## Naive bayes Algorithm

# In[14]:


from sklearn.naive_bayes import GaussianNB
NaiveBayes = GaussianNB()
NaiveBayes.fit(Xtrain,Ytrain)
predicted_values = NaiveBayes.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Naive Bayes')
print("Naive Bayes's Accuracy is: ", x*100)
print(classification_report(Ytest,predicted_values))


# ## Support Vector Machine Algorithm

# In[15]:


from sklearn.svm import SVC
SVM = SVC(gamma='auto')
SVM.fit(Xtrain,Ytrain)
predicted_values = SVM.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('SVM')
print("SVM's Accuracy is: ", x*100)
print(classification_report(Ytest,predicted_values))


# ## Random Forest Algorithm

# In[16]:


from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain,Ytrain)
predicted_values = RF.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('RF')
print("RF's Accuracy is: ", x*100)
print(classification_report(Ytest,predicted_values))


# ## Visualizing the results with plotting

# In[17]:


plt.figure(figsize=[10,5],dpi = 100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x = acc,y = model,palette='dark')


# # Python Input Taking Function

# In[18]:


def cpred(N,M,o,p,q,r,s):
    data = np.array([[N,M,o,p,q,r,s]])
    prediction = RF.predict(data)
    return prediction


# In[19]:


cpred(80,20,30,33,90,80,204)


# In[20]:


cpred(20,100,40,60,18,12,14)


# In[21]:


cpred(55,44,33,23,80,77,66)


# In[22]:


cpred(20,34,43,54,32,2,800)


# In[23]:


cpred(101,102,600,26,45,4,3)


# In[24]:


cpred(30,5,50,29,71,6,511)


# In[25]:


def cpredi(N,M,o,p,q,r,s):
    data = np.array([[N,M,o,p,q,r,s]])
    prediction = SVM.predict(data)
    return prediction


# In[26]:


cpredi(30,5,50,29,71,6,511)


# In[27]:


def cpredic(N,M,o,p,q,r,s):
    data = np.array([[N,M,o,p,q,r,s]])
    prediction = NaiveBayes.predict(data)
    return prediction


# In[28]:


cpredic(30,5,50,29,71,6,511)


# In[29]:


def cpredict(N,M,o,p,q,r,s):
    data = np.array([[N,M,o,p,q,r,s]])
    prediction = DecisionTree.predict(data)
    return prediction


# In[30]:


cpredict(30,5,50,29,71,6,511)


# In[31]:


cpred(40,65,80,80,40,8,200)


# In[32]:


cpred(60,30,70,75,25,8,500)


# In[33]:


cpred(30,40,35,72,30,8,700)


# In[34]:


cpred(800,600,180,70,30,12,1000)


# In[35]:


cpredi(800,600,180,70,30,12,1000)


# In[36]:


cpredic(800,600,180,70,30,12,1000)


# In[37]:


cpredict(800,600,180,70,30,12,1000)


# ### Cpred= Random Forest
# ### Cpredi= SVM
# ### Cpredic= Naive Bayes
# ### Cpredict= Decision Forest

# In[38]:


cpred(99,55,50,31,75,6,877)


# In[39]:


cpredic(99,55,50,31,75,6,877)


# In[40]:


cpredi(99,55,50,31,75,6,877)


# In[41]:


cpredict(99,55,50,31,75,6,877)


# In[42]:


cpred(20,25,50,40,80,4.5,500)


# In[43]:


cpred(20,25,50,40,80,4.5,500)


# In[44]:


cpred(20,25,50,40,80,14,500)


# In[45]:


cpred(20,45,50,30,72,6,712)


# In[46]:


cpred(30,50,25,38,80,7,712)


# In[47]:


cpredic(30,50,25,38,80,7,712)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




