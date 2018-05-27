import matplotlib.image as mpimg
from moviepy.video.io.VideoFileClip import VideoFileClip

from code.car_detection import CarDetector
from code.car_finder import find_cars
from code.const import TRAINED_MODEL_NAME, TRAINED_MODEL_SVC, TRAINED_MODEL_PARAMS, TRAINED_MODEL_ACCURACY, \
    TRAINED_MODEL_X_SCALER, PATH_TEST_IMAGES, PATH_TO_TEST_VIDEO, PATH_TO_TEST_VIDEO_OUT, PATH_TO_PROJECT_VIDEO, \
    PATH_TO_PROJECT_VIDEO_OUT, SEARCH_AREA_AND_SCALE
from code.trainer import train
from code.utils import load_object, all_files_matching, store_image, bgr_to_rgb


def train_model():
    """ Train the model with selected params """

    # Training params
    color_space = 'YCrCb'
    spatial_size = (32, 32)
    hist_bins = 32

    # Hog parameters
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL'

    train(color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins,
          orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)


def trained_svc_detect_on_images():
    """ Apply trained model to test images """

    # Load trained model
    model = load_object(TRAINED_MODEL_NAME)

    svc = model[TRAINED_MODEL_SVC]
    extract_features_params = model[TRAINED_MODEL_PARAMS]
    accuracy_score = model[TRAINED_MODEL_ACCURACY]
    X_scaler = model[TRAINED_MODEL_X_SCALER]

    # remap stored params
    color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel = extract_features_params

    print('Loaded trained model with accuracy: %s' % accuracy_score)

    y_start, y_stop, scale = SEARCH_AREA_AND_SCALE[1]

    for file_name, path in all_files_matching(PATH_TEST_IMAGES):
        print('Loading %s' % path)
        image = mpimg.imread(path)

        result = find_cars(image, y_start, y_stop, scale, svc, X_scaler,
                           color_space, spatial_size, hist_bins,
                           orient, pix_per_cell, cell_per_block, hog_channel)

        store_image(bgr_to_rgb(result), "trained_svc_%s" % file_name)


def car_detection_on_images():
    """ Performs car detection on each test image"""

    for file_name, path in all_files_matching(PATH_TEST_IMAGES):
        print('Loading %s' % path)
        image = mpimg.imread(path)

        # being optimized for video car detection, when processing images a new object is needed for each image
        car_detector = CarDetector()
        detected_cars = car_detector.detect_cars(image)

        store_image(bgr_to_rgb(detected_cars), "car_detections_%s" % file_name)


def car_detection_on_video(in_path, out_path):
    """ Performs car detection on each frame of the video """

    car_detector = CarDetector()
    clip = VideoFileClip(in_path)

    video = clip.fl_image(car_detector.detect_cars)
    video.write_videofile(out_path, audio=False)


if __name__ == '__main__':
    # train_model()
    # trained_svc_detect_on_images()
    # car_detection_on_images()
    car_detection_on_video(PATH_TO_TEST_VIDEO, PATH_TO_TEST_VIDEO_OUT)
    car_detection_on_video(PATH_TO_PROJECT_VIDEO, PATH_TO_PROJECT_VIDEO_OUT)
    pass
