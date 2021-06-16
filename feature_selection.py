


# -*- coding: utf-8 -*-
"""
This code creates a heatmap showing the features that have higher correlation with the labels.

@author: ignas
"""


import pickle
from sklearn.model_selection import train_test_split
import numpy as np


fd_file = open(r'C:/Users/ignas/Downloads/dense.pkl', "rb")
df = pickle.load(fd_file)

features=[*df]
features.remove('pixel_number')
features.remove('image_name')
#features.remove('label')
features.remove('image')





x_train, x_test, y_train, y_test = train_test_split(df[features], df[['label', 'image_name']], random_state=None,
                                                    shuffle=False,
                                                    train_size=0.8)




#importing libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso #Loading the dataset
df['label']=np.array(df['label'],dtype='int')
features=[*df]
features.remove('pixel_number')
features.remove('image_name')
#features.remove('label')
features.remove('image')
X = df[features] #Feature Matrix
y =  np.ravel(df['label'])         #Target Variable
df.head()

import matplotlib.pyplot as plt
#Using Pearson Correlation

cor = df.corr()
plt.figure(figsize=(15,12))
sns.heatmap(cor, cmap='jet')
plt.savefig('ploting.svg')

#Correlation with output variable
cor_target = abs(cor['label'])#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
relevant_features