import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from code.const import TRAINED_MODEL_NAME, TRAINED_MODEL_SVC, TRAINED_MODEL_PARAMS, TRAINED_MODEL_ACCURACY, \
    TRAINED_MODEL_X_SCALER
from code.features import extract_features
from code.utils import get_vehicles, get_not_vehicles, store_object


def train(color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel):
    # for better training accuracy use more data for car and not car
    cars = get_vehicles()
    not_cars = get_not_vehicles()

    # Feature extraction
    t = time.time()

    car_features = extract_features(cars, color_space=color_space, spatial_size=spatial_size,
                                    hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block, hog_channel=hog_channel,
                                    spatial_feat=True, hist_feat=True, hog_feat=True)

    notcar_features = extract_features(not_cars, color_space=color_space, spatial_size=spatial_size,
                                       hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block, hog_channel=hog_channel,
                                       spatial_feat=True, hist_feat=True, hog_feat=True)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to extract features...')

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)

    # Fit a per-column scaler only on the training data
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X_train and X_test
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)

    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train  and save SVC...')

    # Check the score of the SVC
    accuracy_score = round(svc.score(X_test, y_test), 4)
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

    # Store the model for later usage
    params = color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel
    store_object({
        TRAINED_MODEL_SVC: svc,
        TRAINED_MODEL_PARAMS: params,
        TRAINED_MODEL_ACCURACY: accuracy_score,
        TRAINED_MODEL_X_SCALER: X_scaler
    }, TRAINED_MODEL_NAME)
