'''
This code takes the mammograms and texture images and flattens them. It also forms super-pixels. The output is used for the k-NN model
'''
# %% The second step is to sort the data into a data frame, averaging in 2x2 blocks.
path = r'/home/ignaciomoragues/PycharmProjects/mamo/suport/pickles'

import pandas as pd
import os
import numpy as np
import pickle
from tqdm import tqdm
from fcmeans import FCM
import cv2
from PIL import Image
from skimage.transform import rescale

def rebin(a):
    # r, c = a.shape
    # r = int(r / 2)
    # c = int(c / 2)
    # shape = (r, c)
    # sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    # return a.reshape(sh).mean(-1).mean(1)
    # rows,columns=a.shape
    # factor = 2
    # r=int(rows/factor+0.5)
    # c=int(columns/factor+0.5)

    # container=np.array([])

    # for i in range(0,rows,factor):
    #     for j in range(0,columns,factor):
    #         container=np.append(container,a[i:i+factor,j:j+factor].mean())
                                
    # return container.reshape(r,c)
    return rescale(a, 0.4, anti_aliasing=False)



fd_file = open(r"/home/ignaciomoragues/PycharmProjects/mamo/suport/matchings.plk", "rb")
matchings = pickle.load(fd_file)

df_mother = pd.DataFrame()


from collections import defaultdict

for image in tqdm(matchings):

    df = pd.DataFrame()
    match = matchings[image]

    groups = defaultdict(list)
    for ftype in match:
        if 'GLCM' in ftype:
            groups['GLCM'].append(ftype)
        elif 'LAWS' in ftype:
            groups['LAWS'].append(ftype)
        elif 'LBP' in ftype:
            groups['LBP'].append(ftype)

    gGLCM = defaultdict(list)

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

    glcm4 = [df_GLCM1, df_GLCM2, df_GLCM3, df_GLCM4]

    for w in gGLCM:
        L = 0
        for pkl in gGLCM[w]:
            fd_file = open(f"{path}/{pkl}", "rb")
            fd = pickle.load(fd_file)
            fd.pop('image_name', None)
            fd.pop('image', None)
            keys = [*fd]
            for key in keys:
                curr = glcm4[L]
                curr[key] = rebin(fd[key]).flatten()
            L += 1

    GLCM = pd.DataFrame()
    av_df = pd.DataFrame()
    names = [*df_GLCM1]

    for i in range(0, 15):
        av_df["1"] = df_GLCM1.iloc[:, i]
        av_df["2"] = df_GLCM2.iloc[:, i]
        av_df["3"] = df_GLCM3.iloc[:, i]
        av_df["4"] = df_GLCM4.iloc[:, i]

        GLCM[f'GLCM{i}'] = av_df.mean(axis=1)

    df_LAWS1 = pd.DataFrame()
    df_LAWS2 = pd.DataFrame()
    df_LAWS3 = pd.DataFrame()
    LAWS_3 = [df_LAWS1, df_LAWS2, df_LAWS3]

    L = 0
    for pkl in groups['LAWS']:
        fd_file = open(f"{path}/{pkl}", "rb")
        fd = pickle.load(fd_file)
        fd.pop('image_name', None)
        fd.pop('image', None)
        keys = [*fd]
        for key in keys:
            curr = LAWS_3[L]
            curr[key] = rebin(fd[key]).flatten()
        L += 1

    LAWS = pd.DataFrame()
    av_df = pd.DataFrame()
    names = [*df_LAWS1]

    for i in range(0, len(names)):
        LAWS[f'LAWS{i}'] = df_LAWS1.iloc[:, i]

    df_LBP1 = pd.DataFrame()
    df_LBP2 = pd.DataFrame()
    df_LBP3 = pd.DataFrame()
    LBP_3 = [df_LBP1, df_LBP2, df_LBP3]

    L = 0
    for pkl in groups['LBP']:
        fd_file = open(f"{path}/{pkl}", "rb")
        fd = pickle.load(fd_file)
        fd.pop('image_name', None)
        fd.pop('image', None)
        keys = [*fd]
        for key in keys:
            curr = LBP_3[L]
            curr[key] = rebin(fd[key]).flatten()
        L += 1

    LBP = pd.DataFrame()
    av_df = pd.DataFrame()
    names = [*df_LBP1]

    for i in range(0, len(names)):
        av_df["1"] = df_LBP1.iloc[:, i]
        av_df["2"] = df_LBP2.iloc[:, i]
        av_df["3"] = df_LBP3.iloc[:, i]

        LBP[f'LBP{i}'] = av_df.mean(axis=1)

    fd_file = open(f"{path}/{match[0]}", "rb")
    fd = pickle.load(fd_file)
    df1 = pd.DataFrame()
    rebined = rebin(fd["image"])
    df1["image_name"] = np.full(len(rebined.flatten()), fd["image_name"])
    df1["label"] = np.full(len(rebined.flatten()), fd["image_name"][-5:][:-4])
    df1["pixel_number"] = list(range(0, len(rebined.flatten())))
    df1["image"] = rebined.flatten()

    df = pd.concat([df1, LBP, GLCM, LAWS], axis=1)

    df = df[df.image != 0]

    if len(df_mother) == 0:
        df_mother = df
    else:
        df_mother = df_mother.append(df)

a_file = open(r'/home/ignaciomoragues/PycharmProjects/mamo/suport/df_mother_rebin.pkl', "wb")
pickle.dump(df_mother, a_file)
a_file.close()
