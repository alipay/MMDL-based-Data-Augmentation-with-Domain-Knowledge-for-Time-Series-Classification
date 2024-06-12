from utils import load_dataset
from MDL_shape import MDL_shape
import numpy as np
from tsml_eval.publications.y2023.tsc_bakeoff import _set_bakeoff_classifier
from tsml_eval.utils.validation import is_sklearn_classifier
from tsml_eval.estimators import SklearnToTsmlClassifier
from sklearn.metrics import accuracy_score
import os


if __name__ == '__main__':

    # current_file_path = os.path.abspath(__file__)
    # current_folder_path = os.path.dirname(current_file_path)
    # dataset_dir_path = current_folder_path + '/dataset'

    dataset_dir_path = './dataset'

    dataset = 'RiseFall'
    domain_type = 1

    print(f'dataset: {dataset}')

    X_train, y_train, X_test, y_test = load_dataset(dataset, dataset_dir_path)

    # print(X_train.shape)

    MDL_shape = MDL_shape(series_length=X_train.shape[1])
    X_train_shape = MDL_shape.generate_shape(X_train)
    X_train_residual = X_train - X_train_shape

    # print(X_train_shape.shape)

    if domain_type == 1:
        X_train_augmentation = np.vstack((X_train, X_train_shape))
        y_train_augmentation = np.hstack((y_train, y_train))
    else:
        X_train_augmentation = np.vstack((X_train, X_train_residual))
        y_train_augmentation = np.hstack((y_train, y_train))

    classifier_origin = _set_bakeoff_classifier('Hydra-MultiROCKET', random_state=1)
    if is_sklearn_classifier(classifier_origin):
        classifier_origin = SklearnToTsmlClassifier(
            classifier=classifier_origin, concatenate_channels=True, random_state=1
        )

    classifier_origin.fit(X_train, y_train)
    y_pred_origin = classifier_origin.predict(X_test)
    accuracy_origin = accuracy_score(y_test, y_pred_origin)

    classifier_augmentation = _set_bakeoff_classifier('Hydra-MultiROCKET', random_state=1)
    if is_sklearn_classifier(classifier_augmentation):
        classifier_shape = SklearnToTsmlClassifier(
            classifier=classifier_augmentation, concatenate_channels=True, random_state=1
        )

    classifier_augmentation.fit(X_train_augmentation, y_train_augmentation)
    y_augmentation_pred = classifier_augmentation.predict(X_test)
    accuracy_augmentation = accuracy_score(y_test, y_augmentation_pred)

    print(f'The original classification accuracy is: {accuracy_origin}')
    print(f'The MMDL augmentation classification accuracy is: {accuracy_augmentation}')
