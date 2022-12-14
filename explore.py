import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve

path_to_raw_data = r"data/master_training_set_raw_discrete_orgsAreBots.csv"


def prepare_data_for_training(df, test_size=0.3, columnsToExclude=None, outputColumnName="type"):
    if columnsToExclude is None:
        columnsToExclude = [outputColumnName, 'index', 'id']

    labels = np.array(df[outputColumnName].values)

    df.drop(columnsToExclude, axis=1, inplace=True)

    train, test, train_labels, test_labels = train_test_split(df, labels,
                                                              stratify=labels,
                                                              test_size=test_size)

    return train, test, train_labels, test_labels, labels


# for the specific file
def evaluate_model(model, test_data, test_labels, full=True):
    if full:
        test_labels = test_labels.reshape(-1, 1)

        baseScore = model.score(test_data, test_labels)
        return baseScore

    return


def main():
    modelTrainingData = {}
    df = pd.read_csv(path_to_raw_data)

    train, test, train_labels, test_labels, true_labels = prepare_data_for_training(df,
                                                                                    columnsToExclude=['type', 'index',
                                                                                                      'id'])

    model = RandomForestClassifier(n_estimators=10, bootstrap=True, max_features='sqrt')

    model.fit(train, train_labels)

    # get the amount of nodes are in tree on average and the max depth. about 100 trees total
    n_nodes = []
    max_depths = []

    for ind_tree in model.estimators_:
        n_nodes.append(ind_tree.tree_.node_count)
        max_depths.append(ind_tree.tree_.max_depth)

    modelTrainingData["Avg Node Count"] = int(np.mean(n_nodes))
    modelTrainingData["Avg Max Depth"] = int(np.mean(max_depths))

    train_rf_predictions = model.predict(train)
    train_rf_probs = model.predict_proba(train)[:, 1]

    rf_predictions = model.predict(test)
    rf_probs = model.predict_proba(test)[:, 1]

    baseScore = evaluate_model(model, test_labels=test_labels, test_data=test)

    modelTrainingData["Basic Accuracy Score"] = f"{np.round(baseScore * 100, 2)}%"
    print(modelTrainingData)
    pass


main()
