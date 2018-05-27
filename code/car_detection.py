from collections import deque

import cv2
import numpy as np
from scipy import ndimage

from code.const import TRAINED_MODEL_NAME, TRAINED_MODEL_SVC, TRAINED_MODEL_PARAMS, TRAINED_MODEL_X_SCALER, \
    SEARCH_AREA_AND_SCALE
from code.features import convert_color, get_hog_features, bin_spatial, color_hist
from code.utils import load_object


class CarDetector:
    def __init__(self):
        # Load trained model
        model = load_object(TRAINED_MODEL_NAME)

        self.svc = model[TRAINED_MODEL_SVC]
        self.X_scaler = model[TRAINED_MODEL_X_SCALER]

        # remap stored params
        train_params = model[TRAINED_MODEL_PARAMS]
        color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel = train_params

        self.color_space = color_space
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel

        self.heatmap_history = deque(maxlen=6)
        self.heat_threshold = 2

        self.frame_counter = 0
        self.frames_limit = 5

        # local search params
        self.x_start = 600
        self.dilatation_kernel = np.ones((50, 50))

    def get_search_areas_and_scale(self):
        """ We need to search in different areas and at different scales """
        # y_start, y_stop, scale
        return SEARCH_AREA_AND_SCALE

    def detect_cars(self, img):
        """ Detect a car in the image? """

        draw_img = np.copy(img)

        # Find card, bur returns boxes for detection
        self.boxes_list = self.get_boxes(img)

        # Find final boxes from heatmap using label function
        self.draw_cars_on_image(draw_img)

        return draw_img

    def get_boxes(self, img):
        box_list = []

        img = img.astype(np.float32) / 255

        if self.frame_counter % self.frames_limit == 0:
            mask = np.ones_like(img[:, :, 0])
        else:
            mask = np.sum(np.array(self.heatmap_history), axis=0)
            mask[(mask > 0)] = 1
            mask = cv2.dilate(mask, self.dilatation_kernel, iterations=1)

        self.frame_counter += 1

        for y_start, y_stop, scale in self.get_search_areas_and_scale():
            self.search_for_boxes_at_scale(box_list, img, mask, y_start, y_stop, scale)

        return box_list

    def search_for_boxes_at_scale(self, box_list, img, mask, y_start, y_stop, scale):
        """ Search at scale """

        nonzero = mask.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        if len(nonzero_y) > 0:
            y_start = max(np.min(nonzero_y), y_start)
            y_stop = min(np.max(nonzero_y), y_stop)

        if len(nonzero_x) > 0:
            x_start = max(np.min(nonzero_x), self.x_start)
            x_stop = np.max(nonzero_x)
        else:
            return

        if x_stop <= x_start or y_stop <= y_start:
            return

        img_to_search = img[y_start:y_stop, x_start:x_stop, :]
        ctrans_to_search = convert_color(img_to_search, color_space=self.color_space)
        if scale != 1:
            imshape = ctrans_to_search.shape
            ys = np.int(imshape[1] / scale)
            xs = np.int(imshape[0] / scale)
            if (ys < 1 or xs < 1):
                return
            ctrans_to_search = cv2.resize(ctrans_to_search, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        if ctrans_to_search.shape[0] < 64 or ctrans_to_search.shape[1] < 64:
            return

        ch1 = ctrans_to_search[:, :, 0]
        ch2 = ctrans_to_search[:, :, 1]
        ch3 = ctrans_to_search[:, :, 2]

        nx_blocks = (ch1.shape[1] // self.pix_per_cell) - 1
        ny_blocks = (ch1.shape[0] // self.pix_per_cell) - 1

        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nx_blocks - nblocks_per_window) // cells_per_step
        nysteps = (ny_blocks - nblocks_per_window) // cells_per_step

        if self.hog_channel == 'ALL':
            hog1 = get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
            hog2 = get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
            hog3 = get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        else:
            hog = get_hog_features(self.hog_channel, self.orient, self.pix_per_cell, self.cell_per_block,
                                   feature_vec=False)

        for xb in range(nxsteps + 1):
            for yb in range(nysteps + 1):
                y_pos = yb * cells_per_step
                x_pos = xb * cells_per_step

                # Extract HOG for this patch
                if self.hog_channel == 'ALL':
                    hog_feat1 = hog1[y_pos:y_pos + nblocks_per_window, x_pos:x_pos + nblocks_per_window].ravel()
                    hog_feat2 = hog2[y_pos:y_pos + nblocks_per_window, x_pos:x_pos + nblocks_per_window].ravel()
                    hog_feat3 = hog3[y_pos:y_pos + nblocks_per_window, x_pos:x_pos + nblocks_per_window].ravel()
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                else:
                    hog_features = hog[y_pos:y_pos + nblocks_per_window, x_pos:x_pos + nblocks_per_window].ravel()

                x_left = x_pos * self.pix_per_cell
                y_top = y_pos * self.pix_per_cell

                # Extract the image patch
                sub_img = ctrans_to_search[y_top:y_top + window, x_left:x_left + window]

                # Get color features
                spatial_features = bin_spatial(sub_img, size=self.spatial_size)
                hist_features = color_hist(sub_img, nbins=self.hist_bins)

                # Scale features and make a prediction
                test_features = self.X_scaler.transform(
                    np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                test_prediction = self.svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = x_start + np.int(x_left * scale)
                    ytop_draw = np.int(y_top * scale)
                    win_draw = np.int(window * scale)
                    box_list.append(
                        ((xbox_left, ytop_draw + y_start), (xbox_left + win_draw, ytop_draw + win_draw + y_start))
                    )

    def draw_cars_on_image(self, img):
        """ Draws detected cars on output image """

        labels = ndimage.label(self.get_heatmap(img))
        for car_number in range(1, labels[1] + 1):
            nonzero = (labels[0] == car_number).nonzero()
            nonzero_y = np.array(nonzero[0])
            nonzero_x = np.array(nonzero[1])
            bbox = ((np.min(nonzero_x), np.min(nonzero_y)), (np.max(nonzero_x), np.max(nonzero_y)))
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 4)

    def get_heatmap(self, draw_img):
        """ Averages the heatmap across multiple detections """

        current_heatmap = np.zeros_like(draw_img[:, :, 0]).astype(np.float)
        for box in self.boxes_list:
            current_heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        self.heatmap_history.append(current_heatmap)

        heatmap = np.sum(np.array(self.heatmap_history), axis=0)
        heatmap[heatmap <= self.heat_threshold] = 0
        return heatmap
