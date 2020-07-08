"""
Common functions for visualization in different ipython notebooks
"""
import os
import random
from matplotlib.pyplot import figure, imshow, axis, show
from matplotlib.image import imread

N_CLASSES = 200
N_ATTRIBUTES = 312

def get_class_attribute_names(img_dir = 'CUB_200_2011/images/', feature_file='CUB_200_2011/attributes/attributes.txt'):
    """
    Returns:
    class_to_folder: map class id (0 to 199) to the path to the corresponding image folder (containing actual class names)
    attr_id_to_name: map attribute id (0 to 311) to actual attribute name read from feature_file argument
    """
    class_to_folder = dict()
    for folder in os.listdir(img_dir):
        class_id = int(folder.split('.')[0])
        class_to_folder[class_id - 1] = os.path.join(img_dir, folder)

    attr_id_to_name = dict()
    with open(feature_file, 'r') as f:
        for line in f:
            idx, name = line.strip().split(' ')
            attr_id_to_name[int(idx) - 1] = name
    return class_to_folder, attr_id_to_name

def sample_files(class_label, class_to_folder, number_of_files=10):
    """
    Given a class id, extract the path to the corresponding image folder and sample number_of_files randomly from that folder
    """
    folder = class_to_folder[class_label]
    class_files = random.sample(os.listdir(folder), number_of_files)
    class_files = [os.path.join(folder, f) for f in class_files]
    return class_files

def show_img_horizontally(list_of_files):
    """
    Given a list of files, display them horizontally in the notebook output
    """
    fig = figure(figsize=(40,40))
    number_of_files = len(list_of_files)
    for i in range(number_of_files):
        a=fig.add_subplot(1,number_of_files,i+1)
        image = imread(list_of_files[i])
        imshow(image)
        axis('off')
    show(block=True)
