# -*- coding: utf-8 -*-
"""
Created on Sun May  2 11:19:24 2021

@author: ignas
"""

#%% Defining libraries

import pandas as pd
import os
import numpy as np
import pickle
from tqdm import tqdm
from fcmeans import FCM
import cv2
from PIL import Image

#%% The frist step is to group the pickle files that are from the same image

from collections import defaultdict
match=[]
path=r'/home/ignaciomoragues/PycharmProjects/mamo/suport/pickles'

plk_files = os.listdir(path)

matchings=defaultdict(list)
for plk in tqdm(plk_files):
    with open(f'{path}/{plk}','rb') as f:
        fd=pickle.load(f)
        matchings[fd["image_name"]].append(plk)

a_file = open(r'/home/ignaciomoragues/PycharmProjects/mamo/suport/matchings.plk', "wb")
pickle.dump(matchings, a_file)
a_file.close()




#%% The second step is to sort the data into a data frame, averaging in 2x2 blocks.
path=r'/home/ignaciomoragues/PycharmProjects/mamo/suport/pickles'

def rebin(a):
    r,c=a.shape
    r=int(r/2)
    c=int(c/2)
    shape=(r,c)
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

fd_file = open(r"/home/ignaciomoragues/PycharmProjects/mamo/suport/matchings.plk", "rb")
matchings = pickle.load(fd_file)


df_mother = pd.DataFrame() 

for image in tqdm(matchings):
   
    df = pd.DataFrame() 
    match = matchings[image]
    
    groups=defaultdict(list)
    for ftype in match:
        if 'GLCM' in ftype:
            groups['GLCM'].append(ftype)
        elif 'LAWS' in ftype:
            groups['LAWS'].append(ftype)
        elif 'LBP' in ftype:
            groups['LBP'].append(ftype)
            
    
    gGLCM=defaultdict(list)
    
    for window in groups['GLCM']:
        if 'w5' in window:
            gGLCM['w5'].append(window)
        elif 'w15' in window:
            gGLCM['w15'].append(window)
        elif 'w25' in window:
            gGLCM['w25'].append(window)
    
    df_GLCM1 = pd.DataFrame() 
    df_GLCM2 = pd.DataFrame() 
    df_GLCM3 = pd.DataFrame() 
    df_GLCM4 = pd.DataFrame() 
    
    glcm4=[df_GLCM1,df_GLCM2,df_GLCM3,df_GLCM4]
    
    for w in gGLCM:
        L=0
        for pkl in gGLCM[w]:
            fd_file = open(rf"{path}{pkl}", "rb")
            fd = pickle.load(fd_file)
            fd.pop('image_name', None)
            fd.pop('image', None)
            keys = [*fd]
            for key in keys:
                curr=glcm4[L]
                curr[key]=rebin(fd[key]).flatten()
            L+=1
            
    GLCM=pd.DataFrame()
    av_df=pd.DataFrame()
    names=[*df_GLCM1]
    
    for i in range(0,15):
        av_df["1"]=df_GLCM1.iloc[:, i]
        av_df["2"]=df_GLCM2.iloc[:, i]
        av_df["3"]=df_GLCM3.iloc[:, i]
        av_df["4"]=df_GLCM4.iloc[:, i]
        
        GLCM[f'{names[i][:-5]}']=av_df.mean(axis=1)
        
    
    df_LAWS1 = pd.DataFrame() 
    df_LAWS2 = pd.DataFrame() 
    df_LAWS3 = pd.DataFrame() 
    LAWS_3 = [df_LAWS1,df_LAWS2,df_LAWS3]
    
    L=0
    for pkl in groups['LAWS']:
            fd_file = open(f"{path}/{pkl}", "rb")
            fd = pickle.load(fd_file)
            fd.pop('image_name', None)
            fd.pop('image', None)
            keys = [*fd]
            for key in keys:
                curr=LAWS_3[L]
                curr[key]=rebin(fd[key]).flatten()
            L+=1
      
    LAWS=pd.DataFrame()
    av_df=pd.DataFrame()
    names=[*df_LAWS1]
    
    for i in range(0,len(names)):
        av_df["1"]=df_LAWS1.iloc[:, i]
        av_df["2"]=df_LAWS2.iloc[:, i]
        av_df["3"]=df_LAWS3.iloc[:, i]

        LAWS[f'{names[i]}']=av_df.mean(axis=1)


    df_LBP1 = pd.DataFrame() 
    df_LBP2 = pd.DataFrame() 
    df_LBP3 = pd.DataFrame() 
    LBP_3 = [df_LBP1,df_LBP2,df_LBP3]
    
    L=0
    for pkl in groups['LBP']:
            fd_file = open(f"{path}/{pkl}", "rb")
            fd = pickle.load(fd_file)
            fd.pop('image_name', None)
            fd.pop('image', None)
            keys = [*fd]
            for key in keys:
                curr=LBP_3[L]
                curr[key]=rebin(fd[key]).flatten()
            L+=1
      
    LBP=pd.DataFrame()
    av_df=pd.DataFrame()
    names=[*df_LBP1]
    
    for i in range(0,len(names)):
        av_df["1"]=df_LBP1.iloc[:, i]
        av_df["2"]=df_LBP2.iloc[:, i]
        av_df["3"]=df_LBP3.iloc[:, i]

        LBP[f'{names[i][:-6]}']=av_df.mean(axis=1)

    fd_file = open(f"{path}/{match[0]}", "rb")
    fd = pickle.load(fd_file)
    df1=pd.DataFrame()
    rebined=rebin(fd["image"])
    df1["image_name"]=np.full(len(rebined.flatten()), fd["image_name"]) 
    df1["label"]=np.full(len(rebined.flatten()), fd["image_name"][-5:][:-4])
    df1["pixel_number"]= list(range(0,len(rebined.flatten())))
    df1["image"]=rebined.flatten()
    
    df=pd.concat([df1,LBP,GLCM,LAWS], axis=1)

    df = df[df.image != 0]
           
    if len(df_mother) == 0:
        df_mother = df
    else:
        df_mother = df_mother.append(df)

a_file = open(r'/home/ignaciomoragues/PycharmProjects/mamo/suport/df_mother_rebin.pkl', "wb")
pickle.dump(df_mother, a_file)
a_file.close()

#%%SEGMENTATION
  
path=r'D:\tfg\input'
names=os.listdir(path)
a_file = open('image_names.pkl', "wb")
pickle.dump(names, a_file)
a_file.close()
  
fd_file = open(r"D:\df_mother_rebin.pkl", "rb")
df = pickle.load(fd_file)  

fd_file = open("image_names.pkl", "rb")
image_names = pickle.load(fd_file)  




df=df.drop(['GLCM_contrast_w5_d[','GLCM_energy_w5_d[','GLCM_homogeneity_w5_d[','GLCM_correlation_w5_d[','GLCM_dissimilarity_w5_d[', 'GLCM_contrast_w5_d[3]',
 'GLCM_energy_w5_d[3]',
 'GLCM_homogeneity_w5_d[3]',
 'GLCM_correlation_w5_d[3]',
 'GLCM_dissimilarity_w5_d[3]',], axis=1)
features = [*df]
features.remove('pixel_number')
features.remove('image_name')
features.remove('label')

#%% Creating the model
# features=['image','LBP_default',
#  'LBP_ror',
#  'LBP_uniform',
#  'LBP_var', 'GLCM_contrast_w15_d[',
#  'GLCM_energy_w15_d[',
#  'GLCM_homogeneity_w15_d[',
#  'GLCM_correlation_w15_d[',
#  'GLCM_dissimilarity_w15_d[',
#  'GLCM_contrast_w25_d[1',
#  'GLCM_energy_w25_d[1',
#  'GLCM_homogeneity_w25_d[1',
#  'GLCM_correlation_w25_d[1',
#  'GLCM_dissimilarity_w25_d[1', 'L5E5_mean',
#  'L5E5_var',
#  'L5E5_abs_mean',
#  'L5S5_mean',
#  'L5S5_var',
#  'L5S5_abs_mean',
#  'L5R5_mean',
#  'L5R5_var',
#  'L5R5_abs_mean',
#  'E5S5_mean',
#  'E5S5_var',
#  'E5S5_abs_mean',
#  'E5R5_mean',
#  'E5R5_var',
#  'E5R5_abs_mean',
#  'R5S5_mean',
#  'R5S5_var',
#  'R5S5_abs_mean',
#  'S5S5_mean',
#  'S5S5_var',
#  'S5S5_abs_mean',
#  'E5E5_mean',
#  'E5E5_var',
#  'E5E5_abs_mean',
#  'R5R5_mean',
#  'R5R5_var',
#  'R5R5_abs_mean']
fcm = FCM(n_clusters=3)
fcm.fit(df[features]) #Fitting the model

df['tissue'] = fcm.u.argmax(axis=1)

#%% Saving the model
names=os.listdir(path)
a_file = open(r"D:\fcm.pk", "wb")
pickle.dump(fcm, a_file)
a_file.close()

#%% Visualization of the mask

fd_file = open(r"D:\fcm.pk", "rb")
fcm = pickle.load(fd_file) 
df['tissue'] = fcm.u.argmax(axis=1)


def rebin(a):
    r,c=a.shape
    r=int(r/2)
    c=int(c/2)
    shape=(r,c)
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

j=0
for image in tqdm(image_names):
    
    image_df = df.loc[df['image_name'] == image] #image to reconstruct
    
    img=cv2.imread(f"{path}/{image}")
    gray=rebin(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    shap=gray.shape
    # MASK: MULTIPLY 0 container = gray.flatten()*0 
    container = gray.flatten()
    u=np.array(image_df['tissue'])
    pn=np.array(image_df['pixel_number'])
    
    for i in range(0,len(u)):
        if u[i]==2:
            container[pn[i]] = 0
            
    a,b = shap   
    rgbArray = np.zeros((a,b,3), 'uint8')
    rgbArray[..., 0] = gray
    rgbArray[..., 1] = container.reshape(shap)
    rgbArray[..., 2] = gray

    # im = Image.fromarray(container.reshape(shape).astype(np.uint8))
    im = Image.fromarray(rgbArray)
    # os.mkdir(rf"D:\mask\lbp_sin_image")
    im.save(rf'D:\mask\glcmok\dmask_{image}')
     
#%%

dense = df[df.tissue == 2]
dense = dense.drop(columns=["tissue"])
a_file = open('dense.pkl', "wb")
pickle.dump(dense, a_file)
a_file.close()

#%% CLASSIFICATION
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

fd_file = open(r"D:\dense.pkl", "rb")
df = pickle.load(fd_file)  

features=[*df]
features.remove('pixel_number')
features.remove('image_name')
features.remove('label')


x_train, x_test, y_train, y_test = train_test_split(df[features],df[['label','image_name']], random_state=1, 
train_size=0.8)


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, np.ravel(y_train.take([0], axis=1)))

y_pred=neigh.predict(x_test)

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
    rows['label']=[key[-5]]
    rows['dense']=[length]
    rows['1']=[100*c['1']/length]
    rows['2']=[100*c['2']/length]
    rows['3']=[100*c['3']/length]
    rows['4']=[100*c['4']/length]
    
    if i == 0:
        classification = rows
        i=1
    else:
        classification = classification.append(rows)

classification['Max']=classification[['1','2','3','4']].idxmax(axis=1)

from fcmeans import FCM
fcm = FCM(n_clusters=4)
fcm.fit(classification[['dense','1','2','3','4']])
classification['FCM'] = fcm.u.argmax(axis=1)+1
 

#CLASSIFICATION WHEIGHTENED

B=pd.DataFrame()
B['1']=np.ravel(classification['1'])*1.5
B['2']=np.ravel(classification['2'])*2.5
B['3']=np.ravel(classification['3'])*3.5
B['4']=np.ravel(classification['4'])*4.5
B['Max2']=B[['1','2','3','4']].idxmax(axis=1)
classification['Max2']=np.ravel(B['Max2'])

accuracy_score(np.ravel(classification['label']), np.ravel(classification['Max2']))
accuracy_score(np.ravel(classification['label']), np.ravel(classification['Max']))

a_file = open(r"D:\classification.pkl", "wb")
pickle.dump(matchings, a_file)
a_file.close()

