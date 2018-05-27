import os

HERE = os.path.dirname(os.path.abspath(__file__))

PATH_TEST_IMAGES = os.path.join(HERE, '..', 'test_images')
PATH_OUTPUT_IMAGES = os.path.join(HERE, '..', 'output_images')
PATH_TO_TRAIN_DATA_VEHICLES = os.path.join(HERE, '..', 'train_data', 'vehicles')
PATH_TO_TRAIN_DATA_NON_VEHICLES = os.path.join(HERE, '..', 'train_data', 'non-vehicles')
PATH_TO_PICKLED_OBJECTS = os.path.join(HERE, '..', 'output_images', 'objects')

PATH_TO_TEST_VIDEO = os.path.join(HERE, '..', 'test_video.mp4')
PATH_TO_TEST_VIDEO_OUT = os.path.join(HERE, '..', 'out_test_video.mp4')
PATH_TO_PROJECT_VIDEO = os.path.join(HERE, '..', 'project_video.mp4')
PATH_TO_PROJECT_VIDEO_OUT = os.path.join(HERE, '..', 'out_project_video.mp4')

TRAINED_MODEL_NAME = "model.pickle"
TRAINED_MODEL_SVC = 'svc'
TRAINED_MODEL_PARAMS = 'extract_features_params'
TRAINED_MODEL_ACCURACY = 'accuracy_score'
TRAINED_MODEL_X_SCALER = 'X_scaler'

SEARCH_AREA_AND_SCALE = [
    # y_start, y_stop, scale
    (380, 480, 1.0),
    (400, 600, 1.5),
    (480, 680, 2.5),
]
