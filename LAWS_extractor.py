# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 15:15:35 2021

@author: ignas
"""

#%%function

#%% Defining libraries

import numpy as np
import cv2
from PIL import Image 
import os 
from scipy import ndimage
from tqdm import tqdm
import pickle
import sys
import time
from collections import defaultdict
import scipy.signal
import matplotlib.pyplot as plt

#%% Masks
#mask definition
L5 = [1, 4, 6, 4, 1]
E5 = [-1, -2, 0, 2, 1]
S5 = [-1,0,2,0,-1]
R5 = [1,-4,6,-4,1]
#W5 = [-1,2, 0,-2,1]

#5x5 masks have been computed using the np.outer(E5,L5)

L5L5 = [[ 1,  4,  6,  4,  1],
       [ 4, 16, 24, 16,  4],
       [ 6, 24, 36, 24,  6],
       [ 4, 16, 24, 16,  4],
       [ 1,  4,  6,  4,  1]]

L5E5 = [[ -1,  -2,   0,   2,   1],
       [ -4,  -8,   0,   8,   4],
       [ -6, -12,   0,  12,   6],
       [ -4,  -8,   0,   8,   4],
       [ -1,  -2,   0,   2,   1]]

L5S5 = [[-1,  0,  2,  0, -1],
       [-4,  0,  8,  0, -4],
       [-6,  0, 12,  0, -6],
       [-4,  0,  8,  0, -4],
       [-1,  0,  2,  0, -1]]

L5R5 = [[  1,  -4,   6,  -4,   1],
       [  4, -16,  24, -16,   4],
       [  6, -24,  36, -24,   6],
       [  4, -16,  24, -16,   4],
       [  1,  -4,   6,  -4,   1]]
#L5W5 = np.outer(L5,W5)


E5L5 = [[ -1,  -4,  -6,  -4,  -1],
       [ -2,  -8, -12,  -8,  -2],
       [  0,   0,   0,   0,   0],
       [  2,   8,  12,   8,   2],
       [  1,   4,   6,   4,   1]]

E5E5 = [[ 1,  2,  0, -2, -1],
       [ 2,  4,  0, -4, -2],
       [ 0,  0,  0,  0,  0],
       [-2, -4,  0,  4,  2],
       [-1, -2,  0,  2,  1]]

E5S5 = [[ 1,  0, -2,  0,  1],
       [ 2,  0, -4,  0,  2],
       [ 0,  0,  0,  0,  0],
       [-2,  0,  4,  0, -2],
       [-1,  0,  2,  0, -1]]

E5R5 = [[ -1,   4,  -6,   4,  -1],
       [ -2,   8, -12,   8,  -2],
       [  0,   0,   0,   0,   0],
       [  2,  -8,  12,  -8,   2],
       [  1,  -4,   6,  -4,   1]]
#E5W5 = np.outer(E5,W5)

S5L5 = [[-1, -4, -6, -4, -1],
       [ 0,  0,  0,  0,  0],
       [ 2,  8, 12,  8,  2],
       [ 0,  0,  0,  0,  0],
       [-1, -4, -6, -4, -1]]

S5E5 = [[ 1,  2,  0, -2, -1],
       [ 0,  0,  0,  0,  0],
       [-2, -4,  0,  4,  2],
       [ 0,  0,  0,  0,  0],
       [ 1,  2,  0, -2, -1]]

S5S5 = [[ 1,  0, -2,  0,  1],
       [ 0,  0,  0,  0,  0],
       [-2,  0,  4,  0, -2],
       [ 0,  0,  0,  0,  0],
       [ 1,  0, -2,  0,  1]]

S5R5 = [[-1,  4, -6,  4, -1],
       [ 0,  0,  0,  0,  0],
       [ 2, -8, 12, -8,  2],
       [ 0,  0,  0,  0,  0],
       [-1,  4, -6,  4, -1]]
#S5W5 = np.outer(S5,W5)

R5L5 = [[  1,   4,   6,   4,   1],
       [ -4, -16, -24, -16,  -4],
       [  6,  24,  36,  24,   6],
       [ -4, -16, -24, -16,  -4],
       [  1,   4,   6,   4,   1]]

R5E5 = [[ -1,  -2,   0,   2,   1],
       [  4,   8,   0,  -8,  -4],
       [ -6, -12,   0,  12,   6],
       [  4,   8,   0,  -8,  -4],
       [ -1,  -2,   0,   2,   1]]

R5S5 = [[-1,  0,  2,  0, -1],
       [ 4,  0, -8,  0,  4],
       [-6,  0, 12,  0, -6],
       [ 4,  0, -8,  0,  4],
       [-1,  0,  2,  0, -1]]

R5R5 = [[  1,  -4,   6,  -4,   1],
       [ -4,  16, -24,  16,  -4],
       [  6, -24,  36, -24,   6],
       [ -4,  16, -24,  16,  -4],
       [  1,  -4,   6,  -4,   1]]
#R5W5 = np.outer(R5,W5)

masks = [L5L5,L5E5,L5S5,L5R5,E5L5,E5E5,E5S5,E5R5,S5L5,S5E5,S5S5,S5R5,R5L5,R5E5,R5S5,R5R5];
names = ['L5E5','L5S5','L5R5','E5S5','E5R5','R5S5','S5S5','E5E5','R5R5'];
#%%function

def LAWSM_extractor(window_size: int = 3, input_folder: str = 'input', output_folder: str = 'output_LAWS'):
    """
    This function computes bla bla bla
    :param input_folder: path of the folder containing the segmented mammogram
    :param output_folder: path of the folder that will contain the image features in png
    :param window_size: n x n window used to extract local statistics.
    """
    
    if not os.path.exists(output_folder):
        print('Output directory for images does not exist, creating it...')
        os.mkdir(path=output_folder)
        
    if not os.path.exists(f"{output_folder}/pickles"):
        print('Output directory for pikcles does not exist, creating it...')
        os.mkdir(f"{output_folder}/pickles")
        
    #this loop checks wheter the output feature has been already computed, if someone is missing, line 146 executes.
    for imagePath in os.listdir(input_folder): 
        IS=0
        for c in range(0,9):
            if names[c]+'_var_'+imagePath in os.listdir(output_folder):
                IS=IS+1
            if names[c]+'_mean_'+imagePath in os.listdir(output_folder):
                IS=IS+1
            else:
               print(names[c]+' from '+imagePath+' missing, features extraction will begin') 
                

        if IS<18:
                
            inputPath = os.path.join(input_folder, imagePath) 
            # inputPath contains directory where mammograms are 
            img = cv2.imread(inputPath) 
            # image that needs to be generated 
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #img in gray color
            
            dictionary_data = {"image_name":imagePath,"image":gray_image}

            # zero size like image matrices are created. Will contain the convolution.
            L5L5_container = np.zeros_like(gray_image);
            L5E5_container = np.zeros_like(gray_image);
            L5S5_container = np.zeros_like(gray_image);
            L5R5_container = np.zeros_like(gray_image);
    
            E5L5_container = np.zeros_like(gray_image);
            E5E5_container = np.zeros_like(gray_image);
            E5S5_container = np.zeros_like(gray_image);
            E5R5_container = np.zeros_like(gray_image);
            
            S5L5_container = np.zeros_like(gray_image);
            S5E5_container = np.zeros_like(gray_image);
            S5S5_container = np.zeros_like(gray_image);
            S5R5_container = np.zeros_like(gray_image);
            
            R5L5_container = np.zeros_like(gray_image);
            R5E5_container = np.zeros_like(gray_image);
            R5S5_container = np.zeros_like(gray_image);
            R5R5_container = np.zeros_like(gray_image);
            
            containers=[L5L5_container, L5E5_container,L5S5_container,L5R5_container,E5L5_container, E5E5_container, E5S5_container, E5R5_container,
                        S5L5_container,S5E5_container, S5S5_container,S5R5_container, R5L5_container, R5E5_container,R5S5_container, R5R5_container];
    
            for i in range(0,16):
                containers[i]=scipy.signal.convolve2d(gray_image,masks[i],mode='same'); #used ndimage since numpy does not allow 2D convolutions
            
            #averange is performed of paired masks as seen in literature:
            av_L5E5 = (containers[1]+containers[4])/2
            av_L5S5 = (containers[2]+containers[8])/2
            av_L5R5 = (containers[3]+containers[12])/2
            av_E5S5 = (containers[6]+containers[9])/2
            av_E5R5 = (containers[7]+containers[13])/2
            av_S5R5 = (containers[11]+containers[14])/2
            
            #L5L5 not used since mean is not 0 (is 10.24)
            
            new_containers=[av_L5E5,av_L5S5,av_L5R5,av_E5S5,av_E5R5,av_S5R5,containers[10],containers[5],containers[15]];
                
            for c in range(0,9): 
                # zero size like image matrices are created. Will be filed by the feature.
                mean=np.zeros_like(new_containers[c]);
                var=np.zeros_like(new_containers[c]);
                abs_mean=np.zeros_like(new_containers[c]);
                
                rows= new_containers[c].shape[0]
                columns = new_containers[c].shape[1]
                
                # imgPadding = cv2.copyMakeBorder(new_containers[c] , int(window_size/2), int(window_size/2), int(window_size/2), int(window_size/2), cv2.BORDER_CONSTANT)
                
                padding_width = int(window_size / 2 + .5)
                padded_image = cv2.copyMakeBorder(src=new_containers[c],
                                          top=padding_width,
                                          bottom=padding_width,
                                          left=padding_width,
                                          right=padding_width,
                                          borderType=cv2.BORDER_CONSTANT,
                                          value=0)
                
                #the mean and var are computed in a w size window.
                

                
                
                    
                for i in tqdm(range(0,rows-1)):
                    for j in range(0, columns-1):
                        mean[i,j]=np.mean(padded_image[i:i+window_size,j:j+window_size])

                        var[i,j]=np.var(padded_image[i:i+window_size,j:j+window_size])

                        abs_mean[i,j]=np.mean(abs(padded_image[i:i+window_size,j:j+window_size]))
    
                        
                # mean = (mean / mean.max())*255;  #normalizes data in range 0 - 255   
                # mean8 = mean.astype(np.uint8); #converts float64 to uint8
                mean8=cv2.normalize(mean, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
                # var = (var / var.max())*255;  #normalizes data in range 0 - 255            
                # var8 = var.astype(np.uint8); #converts float64 to uint8
                var8=cv2.normalize(var, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

                
                # var = (abs_mean / abs_mean.max())*255; 
                # abs_mean8 = abs_mean.astype(np.uint8);
                abs_mean8=cv2.normalize(abs_mean, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

             
                OutPathLAWSmean = os.path.join(output_folder, f"{names[c]}_mean_{imagePath}")
                im = Image.fromarray(mean8)
                im.save(OutPathLAWSmean)
                # print('\nLAWS mean feature saved in '+OutPathLAWSmean)
                
                OutPathLAWSvar = os.path.join(output_folder, f"{names[c]}_var_{imagePath}")
                im = Image.fromarray(var8)
                im.save(OutPathLAWSvar)
                # print('LAWS var feature saved in '+OutPathLAWSvar)
                
                OutPathLAWSabs = os.path.join(output_folder, f"{names[c]}_abs_{imagePath}")
                im = Image.fromarray(abs_mean8)
                im.save(OutPathLAWSabs)
                # print('LAWS abs feature saved in '+OutPathLAWSabs)
                
                # fig = plt.figure()
                # ax = fig.add_subplot(1,1,1)
                # plot = plt.imshow(mean8,cmap='jet')
                # fig.colorbar(plot);
                # fig.savefig(r"D:\plot.png")

                
                dictionary_data[f'{names[c]}_mean'] = mean
                dictionary_data[f'{names[c]}_var'] = var
                dictionary_data[f'{names[c]}_abs_mean'] = abs_mean
                    
            
                a_file = open(f"{output_folder}/pickles/{imagePath[:-4]}_LAWS.pkl", "wb")
                pickle.dump(dictionary_data, a_file)
                a_file.close()
  

        else:
            print('All features extracted') 
            
if __name__ == '__main__':
    LAWSM_extractor(window_size=5,
                  input_folder='input', 
                  output_folder='output_LAWS')