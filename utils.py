import numpy as np
import pandas as pd


def read_tsv(filename):
    data_df = pd.read_csv(filename, sep='\t', header=None)
    data = np.array(data_df)

    Y = data[:, 0]
    X = data[:, 1:]

    return X, Y


def load_dataset(dataset, dataset_dir_path):
    data_folder = dataset_dir_path + "/" + dataset + "/"

    train_file = data_folder + dataset + "_TRAIN.tsv"
    X_train, y_train = read_tsv(train_file)

    test_file = data_folder + dataset + "_TEST.tsv"
    X_test, y_test = read_tsv(test_file)

    return X_train, y_train, X_test, y_test
