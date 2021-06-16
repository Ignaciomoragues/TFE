import os
import sys
import cv2
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
from math import ceil
from collections import defaultdict
from skimage.feature import greycomatrix, greycoprops
import pickle

"""
to open a pickle:
a_file = open("pickles/tdb005mlol_b3_GLCM_w15_d[3]_a[0].pkl", "rb")
output = pickle.load(a_file)
"""


def glcm_extractor(window_size: int = 8, distances: list = None, angles: list = None, step: int = 1, levels: int = 256,
                   properties: list = None, input_dir: str = 'input', output_dir: str = 'output_GLCM'):
    """
    :param input_folder: path of the folder containing the segmented mammogram
    :param output_folder: path of the folder that will contain the image features in png and the pickles
    :param window_size: size of the slice for the GLCM computation
    :param distances: d in GLCM computatiom
    :param angles: angle in GLCM computatiom
    :param step: number of pixels per step in the iteration of the window slicing
    :param levels: size of the GLCM (binning)
    :param properties: type of the statistic computations performed from the GLCM

    """
    if not os.path.exists(output_dir):
        print('Output directory for images does not exist, creating it...')
        os.mkdir(path=output_dir)
        
    if not os.path.exists(f"{output_dir}/pickles"):
        print('Output directory for pikcles does not exist, creating it...')
        os.mkdir(f"{output_dir}/pickles")
        
    distances = distances or [1]
    angles = angles or [0]
    properties = properties or ['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity']

    # files_in_output_dir = set(os.listdir(output_dir))
    for image_name in os.listdir(input_dir):
        # if ('img_contrast_' + image_name in files_in_output_dir and
        #         'img_energy_' + image_name in files_in_output_dir and
        #         'img_homogeneity_' + image_name in files_in_output_dir and
        #         'img_correlation_' + image_name in files_in_output_dir and
        #         'img_dissimilarity_' + image_name in files_in_output_dir):
        #     print(f'GLCM features for {image_name} already saved')
        #     continue

        empty_patch = np.zeros((window_size, window_size), dtype=np.uint8)
        empty_glmc = greycomatrix(empty_patch,
                                  distances=distances,
                                  angles=angles,
                                  levels=levels,
                                  symmetric=True,
                                  normed=True)
        empty_prop_values = dict()
        for prop in properties:
            empty_prop_values[prop] = greycoprops(empty_glmc, prop=prop)[0][0]

        bgr_image = cv2.imread(os.path.join(input_dir, image_name))  # image that needs to be processed
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)  # img in gray color

        rows, columns = gray_image.shape
        out_rows, out_cols = int(ceil(rows / step)), int(ceil(columns / step))

        prop_values = dict()
        for prop in properties:
            prop_values[prop] = np.zeros((out_rows, out_cols))

        padding_width = int(window_size / 2 + .5)
        centering_shift = padding_width // 2
        padded_image = cv2.copyMakeBorder(src=gray_image,
                                          top=padding_width,
                                          bottom=padding_width,
                                          left=padding_width,
                                          right=padding_width,
                                          borderType=cv2.BORDER_CONSTANT,
                                          value=0)

        if levels < 256:
            padded_image = np.rint(padded_image / 255 * (levels - 1)).astype(np.uint8)

        timings = defaultdict(int)
        try:
            n_patches, n_empty = 0, 0
            with tqdm(total=rows * columns // step ** 2) as progress_bar:
                x = 0
                for i in range(centering_shift, rows + centering_shift, step):
                    y = 0
                    for j in range(centering_shift, columns + centering_shift, step):
                        n_patches += 1
                        patch = padded_image[i:i + window_size, j:j + window_size]
                        if patch.mean() <= 0:
                            n_empty += 1
                            for prop, value in prop_values.items():
                                value[x, y] = empty_prop_values[prop]
                        else:
                            init_time = time.time()
                            glcm = greycomatrix(patch,
                                                distances=distances,
                                                angles=angles,
                                                levels=levels,
                                                symmetric=True,
                                                normed=True)
                            timings['glcm'] += time.time() - init_time
                            for prop, value in prop_values.items():
                                init_time = time.time()
                                value[x, y] = greycoprops(glcm, prop=prop)[0][0]
                                timings[prop] += time.time() - init_time
                        y += 1
                        progress_bar.update(1)
                    x += 1
            print(f'{n_empty} out of {n_patches} patches ({n_empty / n_patches * 100:.2f}%) where empty')
            
            dictionary_data = {"image_name":image_name,"image":gray_image}
            for prop, value in prop_values.items():
                output_path = os.path.join(output_dir, f'img_{prop}_w{window_size}_d{distances}_a{int(angles[0]*360/(2*np.pi))}_{image_name}')
                # Normalize values in range 0-255, needed to display as image, remove when storing as feature vectors.
                normalized_value = (value - value.min()) / (value.max() - value.min()) * 255
                normalized_value = normalized_value.astype(np.uint8)
                im = Image.fromarray(normalized_value)
                im.save(output_path)
                print(f'{prop} feature saved in {output_path}')
                dictionary_data[f'GLCM_{prop}_w{window_size}_d{distances}_a{int(angles[0]*360/(2*np.pi))}'] = value
                
            a_file = open(f"{output_dir}/pickles/{image_name[:-4]}_GLCM_w{window_size}_d{distances}_a{int(angles[0]*360/(2*np.pi))}.pkl", "wb")
            pickle.dump(dictionary_data, a_file)
            a_file.close()

        except KeyboardInterrupt:
            for k, v in timings.items():
                print(f'{k} took {v:.2f}s')
            sys.exit()


if __name__ == '__main__':
    glcm_extractor(window_size=15,
                   distances=[3],
                   angles=[0],
                   step=1,
                   levels=10,
                   properties=['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity'])
