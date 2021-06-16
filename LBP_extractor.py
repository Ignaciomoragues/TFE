import os
import cv2
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
import pickle


def lbp_extractor(radius: int = 3, n_points: int = 24, input_folder: str = 'input', output_folder: str = 'output_LBP'):
    """
    This function extracts the LBP from the images in the input folder
    :param input_folder: path of the folder containing the segmented mammogram
    :param output_folder: path of the folder that will contain the image features in png and the pickles
    :param radius: Radius of circle (spatial resolution of the operator)
    :param n_points: Number of circularly symmetric neighbour set points (quantization of the angular space).
    """
    if not os.path.exists(output_folder):
        print('Output directory for images does not exist, creating it...')
        os.mkdir(path=output_folder)
        
    if not os.path.exists(f"{output_folder}/pickles"):
        print('Output directory for pikcles does not exist, creating it...')
        os.mkdir(f"{output_folder}/pickles")
        
    # this loop checks whether the output feature has been already computed
    files_in_output_folder = set(os.listdir(output_folder))
    for image_name in os.listdir(input_folder):
        if ('LBP_default_' + image_name in files_in_output_folder and
                'LBP_ror_' + image_name in files_in_output_folder and
                'LBP_uniform_' + image_name in files_in_output_folder and
                'LBP_var_' + image_name in files_in_output_folder):
            print(f'LBP features for {image_name} already in output_folder [{output_folder}]')
        else:
            bgr_image = cv2.imread(os.path.join(input_folder, image_name))  # image that needs to be processed
            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)  # image in gray color

            methods = ['default', 'ror', 'uniform','var'] #nri_uniform not used since is not rotationally invariant
            
        
            
            
            dictionary_data = {"image_name":image_name,"image":gray_image}

            for method in methods:
                lbp_float64 = local_binary_pattern(gray_image, n_points, radius, method)
                
                if method == "var":
                    lbp_float64 = np.nan_to_num(lbp_float64, copy=True, nan=0.0, posinf=None, neginf=None) #in the var function the divided 0/0 are NaN, replacing by zero.
                
                normalized = (lbp_float64 / lbp_float64.max()) * 255  # normalizes data in range 0 - 255
                lbp_uint8 = normalized.astype(np.uint8)  # converts float64 to uint8
                
                lbp_uint8 = lbp_float64.astype(np.uint8)  # converts float64 to uint8
                output_filepath = os.path.join(output_folder, f'LBP_{method}_{image_name}')
                
               
                print(f'LBP feature {method} saved in {output_filepath}')
                im = Image.fromarray(lbp_uint8)
                im.save(output_filepath)
                
                dictionary_data[f'LBP_{method}_r{radius}_n{n_points}'] = lbp_float64
                
            a_file = open(f"{output_folder}/pickles/{image_name[:-4]}_LBP_{method}_r{radius}_n{n_points}.pkl", "wb")
            pickle.dump(dictionary_data, a_file)
            a_file.close()



if __name__ == '__main__':
    lbp_extractor(radius=3,
                  n_points=3 * 8,
                  input_folder='input',
                  output_folder='output_LBP')
