
# coding: utf-8

# In[1]:

import pandas as pd


# ### Naming the header, to each columns (attribute) for csv file and read csv file and store in dataframe, df.head() peaks into first five values of the dataframe (df)

# In[2]:

columns = ['A1', 'A2','A3','A4','A5','A6', 'A7','A8','A9','A10','A11', 'A12','A13','A14','A15','A16']


# In[3]:

df = pd.read_csv('ml_assignment/crx.data',names=columns)


# In[4]:

df.head()


# In[5]:

'''Last column {+,-} denotes the labels, hence saving them for machine learning algorithm in future'''

labels = df.A16


# In[6]:

import numpy as np


# In[ ]:

'''Problem has two kinds of attributes, categorical and continuous. In each kind some values are missing ('?'), 
lets replace them with some values so that we can have nice mathematical and ML calculations. 
For continuous features, lets replace missing values with median of the column'''

continuous_values = ['A2','A3','A8','A14','A15']


# In[ ]:

'''df.A2[df.A2 != '?'].median()
df.A2[df.A2 == '?'] = 28.96'''


# In[8]:

df['A2'][df['A2'] == '?']


# In[6]:

median_value = df.A2[df.A2 != '?'].median()
#df.A2[df.A2 == '?'] = median_value


# In[9]:

median_value = df.A14[df.A14 != '?'].median()
df.A14[df.A14 == '?'] = median_value


# In[10]:

x = list() #get count of each unique value in the feature
a = list() #place each unique value in a list 

for values in df.A1.unique():
    if values == '?':
        continue
    a.append(values) 
    x.append(df.A1[df.A1 == values].count())

#print (x)
#x/np.sum(x)

#Generate random number with seed for consistency
png = np.random.RandomState(seed = 1)

#Replace the '?' missing values with randomnly generated values considering the probabibilty of occurance 
#of it in the feature.

df.A1[df.A1 == '?'] = png.choice(a,df.A1[df.A1 == '?'].count(),p = x/np.sum(x))


# In[11]:

x = list() #get count of each unique value in the feature
a = list() #place each unique value in a list 

for values in df.A4.unique():
    if values == '?':
        continue
    a.append(values) 
    x.append(df.A4[df.A4 == values].count())

#print (x)
#x/np.sum(x)

#Generate random number with seed for consistency
png = np.random.RandomState(seed = 1)

#Replace the '?' missing values with randomnly generated values considering the probabibilty of occurance 
#of it in the feature.

df.A4[df.A4 == '?'] = png.choice(a,df.A4[df.A4 == '?'].count(),p = x/np.sum(x))


# In[12]:

x = list() #get count of each unique value in the feature
a = list() #place each unique value in a list 

for values in df.A5.unique():
    if values == '?':
        continue
    a.append(values) 
    x.append(df.A5[df.A5 == values].count())

#print (x)
#x/np.sum(x)

#Generate random number with seed for consistency
png = np.random.RandomState(seed = 1)

#Replace the '?' missing values with randomnly generated values considering the probabibilty of occurance 
#of it in the feature.

df.A5[df.A5 == '?'] = png.choice(a,df.A5[df.A5 == '?'].count(),p = x/np.sum(x))


# In[13]:

x = list()
a = list()
for values in df.A6.unique():
    if values == '?':
        continue
    a.append(values)
    x.append(df.A6[df.A6 == values].count())

#print (x)
#x/np.sum(x)

png = np.random.RandomState(seed = 1)
df.A6[df.A6 == '?'] = png.choice(a,df.A6[df.A6 == '?'].count(),p = x/np.sum(x))


# In[14]:

x = list()
a = list()
for values in df.A7.unique():
    if values == '?':
        continue
    a.append(values)
    x.append(df.A7[df.A7 == values].count())

#print (x)
#x/np.sum(x)

png = np.random.RandomState(seed = 1)
df.A7[df.A7 == '?'] = png.choice(a,df.A7[df.A7 == '?'].count(),p = x/np.sum(x))


# df.A2.astype(np.float32).mean()

# In[15]:

#convert A2 attribute from object type float32 type, so calculation
df.A2 = df.A2.astype(np.float32)


# In[16]:

#Convert the labels from {+,-} to {1,2} for algorithm to understand each class labels
ls = list(map(lambda S: 1 if S == '+' else 2, labels))


# In[17]:

#take out the label column from data frame, because we need train_data (n_samples, n_features) 
#and label (n_samples) for scikit learn ML module
new_df = df.drop(labels='A16',axis=1)


# new_df.head()

# In[18]:

#just for understandable naming convention
labels = ls


# len(labels)

# In[19]:

'''Now, we have categorical features and continuous features. For categorical freatures each has many unique values, 
For module to understand the feature, we binarize each categorical features. for example df = {'A': ['x','y','z]},
Binarized output will be A_x = [1,0,0], A_y = [0,1,0], A_z = [0,0,1], following code does it for us'''

new_data = pd.get_dummies(new_df,columns=['A1','A4','A5','A6','A7','A9','A10','A12','A13'])


# In[20]:

'''A14 attribute has object type, lets convert it to int type '''
new_data.A14 = new_data.A14.astype(dtype=np.int16)


# In[21]:

'''Now we have added missing values, binarized the categorical data, but continuous data needs little preprocessing, 
as each continuous feature has different range, mean, std_dev. lets standardize them i.e. lets make mean = 0 
and std_dev = 1 for each continuous feature, sklearn has preprocessing method, which does this job for us'''
from sklearn import preprocessing


# new_data.head()

# In[22]:

scaling = preprocessing.StandardScaler().fit(new_data[['A2','A3','A8','A11','A14','A15']])

new_std_data = scaling.transform(new_data[['A2','A3','A8','A11','A14','A15']])

new_data[['A2','A3','A8','A11','A14','A15']] = new_std_data[:,:]


# new_data.head()

# In[23]:

'''Lets run Machine Learning code now, We have done preprocessing of data so that ML module will understand. 
Call sklearn and import SVM (support vector machine), Also split data into training and testing, 
so that we can later evaluate our trained model on unknown (unseen) data'''
from sklearn import svm
from sklearn.cross_validation import train_test_split


# In[24]:

get_ipython().magic('matplotlib inline')


# In[26]:

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# In[27]:

'''To visualize the data, use PCA to reduce the dimensionality (Dim = 2), plot it to see values of two classes'''

pca = KernelPCA(n_components=3)
pca.fit(new_data)
pca_values = pca.transform(new_data)
plt.scatter(pca_values[:,0],pca_values[:,1],c=labels)
plt.show()


# In[28]:

'''Split training and testing data'''
train_data,test_data,train_label,test_label = train_test_split(new_data,labels)


# In[52]:

'''Initialize the classifier which required hyperparameters, C = 10, which is penalty for wrong prediction
Fit classifier using the training data'''

classifier = svm.SVC(kernel='linear',gamma=0.001,C=10)
classifier.fit(X=train_data,y=train_label)


# svm.SVC?

# In[53]:

'''Accuracy On test data'''
print ('Accuracy = ', np.sum(np.equal(classifier.predict(test_data),test_label))/len(test_label))


# In[55]:

'''We can use in built sklearn method to calculate accuracy'''
from sklearn.metrics import accuracy_score
print ('Accuracy = ', accuracy_score(test_label,classifier.predict(test_data)))

#np.sum(np.equal(adaboost.predict(test_data),test_label))/len(test_label)


# from sklearn.feature_selection import SelectFromModel
# from sklearn.ensemble import ExtraTreesClassifier

# In[ ]:



