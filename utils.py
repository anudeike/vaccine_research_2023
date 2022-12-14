import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


# CONSTANTS


# BASIC HELPER FUNCTIONS
def convert_type_to_number(tp, orgsAreBots=True):
    """
    Used to convert the raw type to a discrete number for training.
    0 - bot
    1 - human
    2 - organization
    :param orgsAreBots: whether Organization will be counted as bots or not
    :param tp: incoming type string
    :return: 0, 1 or 2
    """
    if tp == "bot":
        return 0
    if tp == "human":
        return 1
    if orgsAreBots:
        return 0
    return 2  # ORGANIZATION


def prepare_data_for_training(df, test_size=0.3, columnsToExclude=None, outputColumnName="type"):
    """
    Prepares the data for training.
    :param df: The training set data
    :param test_size: what proportion of the data will be used to test the model
    :param columnsToExclude: Columns to exclude for testing
    :param outputColumnName: the "Y" variable or output variable
    :return: The train and test datasets, properly formatted and ready to be used
    """
    if columnsToExclude is None:
        columnsToExclude = [outputColumnName, 'index', 'id']

    labels = np.array(df[outputColumnName].values)

    df.drop(columnsToExclude, axis=1, inplace=True)

    train, test, train_labels, test_labels = train_test_split(df, labels,
                                                              stratify=labels,
                                                              test_size=test_size)

    return train, test, train_labels, test_labels, labels
