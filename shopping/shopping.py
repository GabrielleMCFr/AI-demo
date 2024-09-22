import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    if len(sys.argv) != 2:
        sys.exit("Wrong arguments count")

    # load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    datalists = []
    labellist = []

    with open("shopping.csv") as f:
        reader = csv.reader(f)
        next(reader)

        
        for row in reader:
            evidence = []
            label = 0
            for x in range(18):
                if x == 0 or x == 2 or x == 4 or x == 11 or x == 12 or x == 13 or x == 14:
                    evidence.append(int(row[x]))
                elif x == 1 or x == 3 or x == 5 or x == 6 or x == 7 or x == 8 or x == 9:
                    evidence.append(float(row[x]))
                elif x == 10:
                    if row[10] == 'Jan':
                        evidence.append(0)
                    elif row[10] == 'Feb':
                        evidence.append(1)
                    elif row[10] == 'Mar':
                        evidence.append(2)
                    elif row[10] == 'Apr':
                        evidence.append(3)
                    elif row[10] == 'May':
                        evidence.append(4)
                    elif row[10] == 'June':
                        evidence.append(5)
                    elif row[10] == 'Jul':
                        evidence.append(6)
                    elif row[10] == 'Aug':
                        evidence.append(7)
                    elif row[10] == 'Sep':
                        evidence.append(8)
                    elif row[10] == 'Oct':
                        evidence.append(9)
                    elif row[10] == 'Nov':
                        evidence.append(10)
                    else:
                        evidence.append(11)
                elif x == 15:
                    if row[15] == 'Returning_Visitor':
                        evidence.append(1)
                    else :
                        evidence.append(0)
                elif x == 16:
                    if row[16] == 'FALSE':
                        evidence.append(0)
                    else:
                        evidence.append(1)
                # or just else
                elif x == 17:
                    datalists.append(evidence)
                    if row[17] == 'TRUE':
                        labellist.append(1)
                    else:
                        labellist.append(0)

        
    return (datalists, labellist)



def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    returns a tuple (sensitivity, specificity).

    Each label is either a 1 (positive) or 0 (negative).

    `sensitivity` is a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` is a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    totalneg = 0
    totalpos = 0
    rightneg = 0
    rightpos = 0

    for actual, predicted in zip(labels, predictions):
        # if the actual label correspond to a sell (pos)
        if actual == 1:
            totalpos += 1
            if actual == predicted:
                rightpos += 1
        # if the actual label correspond to no sell (neg)
        else:
            totalneg += 1
            if actual == predicted:
                rightneg += 1
    
    sensitivity = float(rightpos/totalpos)
    specificity = float(rightneg/totalneg)

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
