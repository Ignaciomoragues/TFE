# -*- coding: utf-8 -*-
"""
This code finds the weight values maximizing the accuracy.

@author: ignas
"""

from sklearn import svm
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

fd_file = open(r'/home/ignaciomoragues/PycharmProjects/mamo/suport/dense.pkl', "rb")
df = pickle.load(fd_file)

features = [*df]
features.remove('pixel_number')
features.remove('image_name')
features.remove('label')
features.remove('image')

x_train, x_test, y_train, y_test = train_test_split(df[features], df[['label', 'image_name']], random_state=None,
                                                    shuffle=False,
                                                    train_size=0.80)

x_test, x_test2, y_test, y_test2 = train_test_split(x_test, y_test,random_state=None,
                                                    shuffle=False,
                                                    train_size=0.50)


fd_file = open("/home/ignaciomoragues/PycharmProjects/mamo/suport/pred4.pk", "rb")
y_pred = pickle.load(fd_file)

from sklearn.metrics import accuracy_score

from collections import defaultdict
from tqdm import tqdm

match = []

images = np.ravel(y_test.take([1], axis=1))

matchings = defaultdict(list)
i = 0
for name in tqdm(images):
    matchings[name].append(y_pred[i])
    i += 1

import pandas as pd

from sklearn.metrics import confusion_matrix


def optm(params):
    a, b,d,e = params
    keys = [*matchings]
    classification = pd.DataFrame()
    from collections import Counter
    i = 0
    for key in keys:
        rows = pd.DataFrame()
        c = Counter(matchings[key])

        length = len(matchings[key])

        rows['image'] = [key]
        rows['dense'] = [length]
        rows['1'] = [(100 * c['1'] / length) * a]
        rows['2'] = [(100 * c['2'] / length) * b]
        rows['3'] = [(100 * c['3'] / length) * d]
        rows['4'] = [(100 * c['4'] / length) * e]
        rows['label'] = [key[-5]]

        if i == 0:
            classification = rows
            i = 1
        else:
            classification = classification.append(rows)

    classification['Max'] = classification[['1', '2', '3', '4']].idxmax(axis=1)

    return 1-accuracy_score(classification['label'], classification['Max'])
    #    return np.diagonal(1-confusion_matrix(classification['label'], classification['Max'],normalize='true'))[2]


from scipy.optimize import minimize


initial_guess = [ 0.94842885 , 2.94076792 , 2.72127382 ,11.66619276]
initial_guess = [ 0.5 , 1 , 2, 5]
result = minimize(optm, initial_guess,method='Powell', options={'disp': True})
if result.success:
    fitted_params = result.x
    print(fitted_params)
else:
    raise ValueError(result.message)
