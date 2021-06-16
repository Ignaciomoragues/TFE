# -*- coding: utf-8 -*-
"""
Main code to execute all the extractors with the parameteres chosen.

@author: ignas
"""
import numpy as np
from LBP_extractor import lbp_extractor
from GLCM_extractor import glcm_extractor
from LAWS_extractor import LAWSM_extractor


path='input'


LAWSM_extractor(window_size=5,
                  input_folder=path, 
                  output_folder='output_LAWS1')

LAWSM_extractor(window_size=15,
                  input_folder=path, 
                  output_folder='output_LAWS2')

LAWSM_extractor(window_size=25,
                  input_folder=path, 
                  output_folder='output_LAWS3')

lbp_extractor(radius=2,
                  n_points=4,
                  input_folder=path,
                  output_folder='output_LBP1')

lbp_extractor(radius=5,
                  n_points=8,
                  input_folder=path,
                  output_folder='output_LBP2')

lbp_extractor(radius=10,
                  n_points=16,
                  input_folder=path,
                  output_folder='output_LBP3')

glcm_extractor(window_size=5,distances=[3],angles=[0],step=1,levels=32,properties=['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity'],input_dir=path)
glcm_extractor(window_size=5,distances=[3],angles=[np.pi/4,],step=1,levels=32,properties=['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity'],input_dir=path)
glcm_extractor(window_size=5,distances=[3],angles=[np.pi/2],step=1,levels=32,properties=['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity'],input_dir=path)
glcm_extractor(window_size=5,distances=[3],angles=[3*np.pi/4],step=1,levels=32,properties=['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity'],input_dir=path)

glcm_extractor(window_size=15,distances=[5],angles=[0],step=1,levels=32,properties=['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity'],input_dir=path)
glcm_extractor(window_size=15,distances=[5],angles=[np.pi/4,],step=1,levels=32,properties=['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity'],input_dir=path)
glcm_extractor(window_size=15,distances=[5],angles=[np.pi/2],step=1,levels=32,properties=['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity'],input_dir=path)
glcm_extractor(window_size=15,distances=[5],angles=[3*np.pi/4],step=1,levels=32,properties=['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity'],input_dir=path)

glcm_extractor(window_size=25,distances=[10],angles=[0],step=1,levels=32,properties=['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity'],input_dir=path)
glcm_extractor(window_size=25,distances=[10],angles=[np.pi/4,],step=1,levels=32,properties=['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity'],input_dir=path)
glcm_extractor(window_size=25,distances=[10],angles=[np.pi/2],step=1,levels=32,properties=['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity'],input_dir=path)
glcm_extractor(window_size=25,distances=[10],angles=[3*np.pi/4],step=1,levels=32,properties=['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity'],input_dir=path)