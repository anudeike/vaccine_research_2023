"""
This file trains a random forest classifer using the tools provided by sklearn.
This does NOT use XGBoost nor contain the logic for independent model validation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score
import utils

path_to_raw_data = r"data/master_training_set_raw_discrete_orgsAreBots.csv"


# for the specific file
def evaluate_model(model, test_data, test_labels, full=True):
    if full:
        # base score
        test_labels = test_labels.reshape(-1, 1)
        baseScore = model.score(test_data, test_labels)

        # balanced accuracy score
        modelPredicitons = model.predict(test_data)
        balancedScore = balanced_accuracy_score(modelPredicitons, test_labels)

        # precision score
        precisionScore = precision_score(modelPredicitons, test_labels)

        # recall score
        recallScore = recall_score(modelPredicitons, test_labels)
        return baseScore, balancedScore, precisionScore, recallScore



    return


def main():
    modelTrainingDataRows = []
    NUMBER_OF_ESTIMATORS = 10
    MAX_FEATURES = "sqrt"
    NUMBER_OF_RUNS = 50
    ESTIMATORS_INCREMENT = 5
    INCREASE_ESTIMATORS_OVER_TIME = True

    df = pd.read_csv(path_to_raw_data)

    train, test, train_labels, test_labels, true_labels = utils.prepare_data_for_training(df,
                                                                                          columnsToExclude=['type',
                                                                                                            'index',
                                                                                                            'id'])
    for i in range(0, NUMBER_OF_RUNS):
        print(f"Run {i}/{NUMBER_OF_RUNS}")

        if INCREASE_ESTIMATORS_OVER_TIME:
            if i % 10 == 0:
                NUMBER_OF_ESTIMATORS += ESTIMATORS_INCREMENT
                print(NUMBER_OF_ESTIMATORS)

        modelTrainingData = {}

        model = RandomForestClassifier(n_estimators=NUMBER_OF_ESTIMATORS, bootstrap=True, max_features=MAX_FEATURES)
        model.fit(train, train_labels)

        n_nodes = []
        max_depths = []

        for ind_tree in model.estimators_:
            n_nodes.append(ind_tree.tree_.node_count)
            max_depths.append(ind_tree.tree_.max_depth)

        modelTrainingData["Avg Node Count"] = int(np.mean(n_nodes))
        modelTrainingData["Avg Max Depth"] = int(np.mean(max_depths))
        modelTrainingData["Number of Estimators"] = NUMBER_OF_ESTIMATORS

        baseScore, balancedScore, precisionScore, recallScore = evaluate_model(model, test_labels=test_labels, test_data=test)
        modelTrainingData["Basic Accuracy Score"] = f"{np.round(baseScore * 100, 2)}%"
        modelTrainingData["Balanced Accuracy Score"] = f"{np.round(balancedScore * 100, 2)}%"
        modelTrainingData["Precision Score"] = f"{np.round(precisionScore * 100, 2)}%"
        modelTrainingData["Recall Score"] = f"{np.round(recallScore * 100, 2)}%"

        print(modelTrainingData)
        modelTrainingDataRows.append(modelTrainingData)

    metadata_df = pd.DataFrame(modelTrainingDataRows)

    metadata_df.to_csv(f"Runs-{NUMBER_OF_RUNS}_Estimators-{NUMBER_OF_ESTIMATORS}_Increment-{INCREASE_ESTIMATORS_OVER_TIME}.csv")


main()
