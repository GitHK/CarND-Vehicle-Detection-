from random import randint

import cv2
import matplotlib.pyplot as plt
import numpy as np

from code.car_detection import CarDetector
from code.const import TRAINED_MODEL_NAME, TRAINED_MODEL_PARAMS, PATH_TEST_IMAGES, SEARCH_AREA_AND_SCALE
from code.features import get_hog_features
from code.utils import get_vehicles, get_not_vehicles, make_output_path, load_object, all_files_matching, store_image, \
    bgr_to_rgb


def car_not_car_image(car_path, not_car_path):
    plt.subplot(1, 2, 1)
    car = plt.imread(car_path)
    plt.imshow(car)
    plt.title('Car')

    plt.subplot(1, 2, 2)
    not_car = plt.imread(not_car_path)
    plt.imshow(not_car)
    plt.title('Not Car')

    plt.savefig(make_output_path('report_car_not_car.png'))


def image_hog(path, label, save_name, orient, pix_per_cell, cell_per_block):
    image = plt.imread(path)

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(label)

    plt.subplot(1, 2, 2)
    features, hog = get_hog_features(image[:, :, 0], orient=orient, pix_per_cell=pix_per_cell,
                                     cell_per_block=cell_per_block, vis=True,
                                     feature_vec=True)
    plt.imshow(hog, cmap='gray')
    plt.title('HOG Image')
    plt.savefig(make_output_path('%s.png' % save_name))


def draw_multi_scale_windows(img, y_start, y_stop, scale, pix_per_cell):
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_to_search = img[y_start:y_stop, :, :]
    shape = img_to_search.shape
    img_to_search = cv2.resize(img_to_search, (np.int(shape[1] / scale), np.int(shape[0] / scale)))

    nx_blocks = (img_to_search.shape[1] // pix_per_cell) - 1
    ny_blocks = (img_to_search.shape[0] // pix_per_cell) - 1

    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nx_blocks - nblocks_per_window) // cells_per_step
    nysteps = (ny_blocks - nblocks_per_window) // cells_per_step

    draw_first_rect = True

    first_rect_start = None
    first_rect_end = None

    for xb in range(nxsteps + 1):
        for yb in range(nysteps + 1):
            y_pos = yb * cells_per_step
            x_pos = xb * cells_per_step

            x_left = x_pos * pix_per_cell
            y_top = y_pos * pix_per_cell

            x_box_left = np.int(x_left * scale)
            y_top_draw = np.int(y_top * scale)
            win_draw = np.int(window * scale)
            rect_start = (x_box_left, y_top_draw + y_start)
            rect_end = (x_box_left + win_draw, y_top_draw + win_draw + y_start)
            cv2.rectangle(draw_img, rect_start, rect_end, (0, 0, 255), 4)

            # keep the first rect for later usage
            if draw_first_rect:
                draw_first_rect = False
                first_rect_start = rect_start
                first_rect_end = rect_end

    cv2.rectangle(draw_img, first_rect_start, first_rect_end, (255, 0, 0), 4)

    return draw_img


def car_detector_result(image, output_name):
    car_detector = CarDetector()
    result = car_detector.detect_cars(image)

    plt.figure(figsize=(15, 3))

    plt.subplot(141)
    plt.imshow(image)
    plt.title('Original')

    plt.subplot(142)
    plt.imshow(car_detector.get_heatmap(image)) # bad way of using the function
    plt.title('Heatmap')

    plt.subplot(143)
    plt.imshow(result)
    plt.title('Label box')

    plt.savefig(make_output_path('ticd_%s' % output_name))


if __name__ == '__main__':
    model = load_object(TRAINED_MODEL_NAME)
    extract_features_params = model[TRAINED_MODEL_PARAMS]
    # remap stored params
    color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel = extract_features_params

    cars = [x for x in get_vehicles()]
    not_cars = [x for x in get_not_vehicles()]

    car_path = cars[randint(0, len(cars))]
    not_car_path = not_cars[randint(0, len(not_cars))]

    car_not_car_image(car_path, not_car_path)
    image_hog(car_path, "Car", "report_car_hog", orient, pix_per_cell, cell_per_block)
    image_hog(not_car_path, "Not car", "report_not_car_hog", orient, pix_per_cell, cell_per_block)

    test_image_name, test_image_path = all_files_matching(PATH_TEST_IMAGES).__next__()
    test_image = plt.imread(test_image_path.replace('test6', 'test3'))

    for y_start, y_stop, scale in SEARCH_AREA_AND_SCALE:
        grid = draw_multi_scale_windows(test_image, y_start, y_stop, scale, pix_per_cell)
        store_image(bgr_to_rgb(grid), 'test_image_grid_scale[%s]' % scale)

    for image_name, image_path in all_files_matching(PATH_TEST_IMAGES):
        image = plt.imread(image_path)
        car_detector_result(image, image_name)

