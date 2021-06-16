# -*- coding: utf-8 -*-
"""
This codes classify the dense pixels into the 4 BI-RADS labels.

@author: ignas
"""

from sklearn import svm
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

fd_file = open(r'/home/ignaciomoragues/PycharmProjects/mamo/suport/dense.pkl', "rb")
df = pickle.load(fd_file)  

features=[*df]
features.remove('pixel_number')
features.remove('image_name')
features.remove('label')
features.remove('image')


x_train, x_test, y_train, y_test = train_test_split(df[['LBP2','LBP3','GLCM0','GLCM2','GLCM3','GLCM4','GLCM5','GLCM7','GLCM8','GLCM9','GLCM10','GLCM12','GLCM13','GLCM14','LAWS2','LAWS5','LAWS26']],df[['label','image_name']], random_state=None,shuffle=False,
train_size=0.80)


x_test, x_test2, y_test, y_test2 = train_test_split(x_test, y_test,random_state=None,
                                                    shuffle=False,
                                                    train_size=0.50)


neigh = KNeighborsClassifier(n_neighbors=3,n_jobs=61)
neigh.fit(x_train, np.ravel(y_train.take([0], axis=1)))

y_pred=neigh.predict(x_test)


a_file = open(rf"/home/ignaciomoragues/PycharmProjects/mamo/suport/pred4.pk", "wb")
pickle.dump(y_pred, a_file)
a_file.close()

fd_file = open(r'/home/ignaciomoragues/PycharmProjects/mamo/suport/pred3.pk', "rb")
y_pred = pickle.load(fd_file)



from sklearn.metrics import accuracy_score
accuracy_score(np.ravel(y_test.take([0], axis=1)), y_pred)


from collections import defaultdict
from tqdm import tqdm
match=[]

images = np.ravel(y_test.take([1], axis=1))

matchings=defaultdict(list)
i=0
for name in tqdm(images):
    matchings[name].append(y_pred[i])
    i+=1

import pandas as pd  
  

keys=[*matchings]
classification=pd.DataFrame()
from collections import Counter
i==0
for key in keys:
    rows=pd.DataFrame()
    c=Counter(matchings[key])
    
    length=len(matchings[key])
    
    rows['image']=[key]
    rows['dense']=[length]
    rows['1']=[100*c['1']/length*0.70111574]
    rows['2']=[100*c['2']/length*1.88195415]
    rows['3']=[100*c['3']/length*2.01130936]
    rows['4']=[100*c['4']/length*6.72028747]
    rows['label']=[key[-5]]
    
    if i == 0:
        classification = rows
        i=1
    else:
        classification = classification.append(rows)

#classification['Max']=classification[['1','2','3','4']].idxmax(axis=1)

from fcmeans import FCM
fcm = FCM(n_clusters=4)
fcm.fit(classification[['1','2','3','4']].to_numpy())
classification['Max'] = fcm.u.argmax(axis=1)+1

accuracy_score(classification['label'], classification['Max'])
accuracy_score(np.ravel(y_test.take([0], axis=1)), classification['Max'])

from sklearn.metrics import confusion_matrix



import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
array = confusion_matrix(classification['label'], classification['Max'])
df_cm = pd.DataFrame(array, index = ['BIRRADS1','BIRRADS2','BIRRADS3','BIRRADS4'],
                  columns = ['BIRRADS1','BIRRADS2','BIRRADS3','BIRRADS4'])

import numpy as np


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('demo.png')

plot_confusion_matrix(cm           = confusion_matrix(classification['label'], classification['Max']),
                      normalize    = True,
                      target_names =['BIRRADS1','BIRRADS2','BIRRADS3','BIRRADS4'],
                      title        = "Confusion Matrix")





