import glob
import pickle
from os import listdir, path, makedirs
from os.path import isdir

import cv2
import matplotlib.pyplot as plt

from code.const import PATH_TO_TRAIN_DATA_VEHICLES, PATH_TO_TRAIN_DATA_NON_VEHICLES, PATH_TO_PICKLED_OBJECTS, \
    PATH_OUTPUT_IMAGES


def all_files_matching(directory, supported_extensions=("jpg",)):
    """ Returns a generator yielding tuples of file_name and full path to file """
    for file_name in listdir(directory):
        for extentiosn in supported_extensions:
            if file_name.endswith(extentiosn):
                yield file_name, path.join(directory, file_name)


def show_image(image, window_title=None):
    """ Provide an image and creates a new figure and plots color or greyscale images """
    plt.figure()
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)

    if window_title is not None:
        fig = plt.gcf()
        fig.canvas.set_window_title(window_title)
    plt.show()


def bgr_to_rgb(image):
    """ Convert OpenCV read image to RGB """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def get_vehicles():
    return glob.iglob(path.join(PATH_TO_TRAIN_DATA_VEHICLES, '*', '*.png'))


def get_not_vehicles():
    return glob.iglob(path.join(PATH_TO_TRAIN_DATA_NON_VEHICLES, '*', '*.png'))


def store_object(obj, name):
    """ Saves a given python obj pickled to a file"""
    if not isdir(PATH_TO_PICKLED_OBJECTS):
        makedirs(PATH_TO_PICKLED_OBJECTS)

    save_path = path.join(PATH_TO_PICKLED_OBJECTS, name)
    with open(save_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_object(name):
    """ Loads a python object form a pickled file"""
    load_path = path.join(PATH_TO_PICKLED_OBJECTS, name)
    with open(load_path, 'rb') as handle:
        return pickle.load(handle)


def store_image(image, name, extension='jpg'):
    """ Store to file with extension """
    name = name.replace('.png', '').replace('.jpg', '')
    output_path = path.join(PATH_OUTPUT_IMAGES, '%s.%s' % (name, extension))
    print('Saving to %s' % output_path)
    cv2.imwrite(output_path, image)


def make_output_path(file_name):
    return path.join(PATH_OUTPUT_IMAGES, file_name)
