import cv2
import numpy as np

from code.features import convert_color, get_hog_features, bin_spatial, color_hist


def find_cars(img, y_start, y_stop, scale, svc, X_scaler,
              color_space, spatial_size, hist_bins,
              orient, pix_per_cell, cell_per_block, hog_channel):
    """ Uses the trained model to detect cars and returns and image with the detections drawn on top """

    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[y_start:y_stop, :, :]
    ctrans_to_search = convert_color(img_tosearch, color_space=color_space)
    if scale != 1:
        shape = ctrans_to_search.shape
        ctrans_to_search = cv2.resize(ctrans_to_search, (np.int(shape[1] / scale), np.int(shape[0] / scale)))

    ch1 = ctrans_to_search[:, :, 0]
    ch2 = ctrans_to_search[:, :, 1]
    ch3 = ctrans_to_search[:, :, 2]

    # Define blocks and steps as above
    nx_blocks = (ch1.shape[1] // pix_per_cell) - 1
    ny_blocks = (ch1.shape[0] // pix_per_cell) - 1
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nx_steps = (nx_blocks - nblocks_per_window) // cells_per_step
    ny_steps = (ny_blocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    if hog_channel == 'ALL':
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    else:
        hog = get_hog_features(hog_channel, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nx_steps):
        for yb in range(ny_steps):
            y_pos = yb * cells_per_step
            x_pos = xb * cells_per_step
            # Extract HOG for this patch

            if hog_channel == 'ALL':
                hog_feat1 = hog1[y_pos:y_pos + nblocks_per_window, x_pos:x_pos + nblocks_per_window].ravel()
                hog_feat2 = hog2[y_pos:y_pos + nblocks_per_window, x_pos:x_pos + nblocks_per_window].ravel()
                hog_feat3 = hog3[y_pos:y_pos + nblocks_per_window, x_pos:x_pos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog[y_pos:y_pos + nblocks_per_window, x_pos:x_pos + nblocks_per_window].ravel()

            x_left = x_pos * pix_per_cell
            y_top = y_pos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_to_search[y_top:y_top + window, x_left:x_left + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(x_left * scale)
                ytop_draw = np.int(y_top * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + y_start),
                              (xbox_left + win_draw, ytop_draw + win_draw + y_start), (0, 0, 255), 4)

    return draw_img
