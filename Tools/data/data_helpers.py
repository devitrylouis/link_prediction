import pandas as pd
import csv

def kaggle_submission(predictions, name):
    """
    Transform a np.array of predictions into a
    Kaggle expected *.csv file with title name.
    """
    predictions = zip(range(len(predictions)), predictions)
    predictions = pd.DataFrame(predictions)
    predictions.columns = ['id', 'category']
    predictions.to_csv(name + '.csv', index = False)

def retrieve_data():
    """
    Retrieve data sets:
    - training_set: Directed Graphs at time $t_{0}$
    - testing_set: Directed Graphs at time $t_{1}$
    - node_info:
    - IDs:
    """
    with open("data/testing_set.txt", "r") as f:
        reader = csv.reader(f)
        testing_set  = list(reader)

    testing_set = [element[0].split(" ") for element in testing_set]

    with open("data/training_set.txt", "r") as f:
        reader = csv.reader(f)
        training_set  = list(reader)

    training_set = [element[0].split(" ") for element in training_set]

    with open("data/node_information.csv", "r") as f:
        reader = csv.reader(f)
        node_info  = list(reader)

    IDs = [element[0] for element in node_info]

    return(training_set, testing_set, node_info, IDs)
