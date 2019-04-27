from sklearn import metrics as metric

# calculating the f1 score
def f1_score(true_y, predicted_y):
    print(metric.f1_score(true_y, predicted_y, average=None))

#using a confusion matrix method to calculate all the performance measures

def measure_model(true_y, predicted_y, debug=False):
    feed_data = zip(true_y, predicted_y)
    true_positive = 0.0
    false_positive = 0.0
    true_negative = 0.0
    false_negative = 0.0


    for d in feed_data:
        if d == (-1, -1):
            true_negative += 1
        elif d == (1, 1):
            true_positive += 1
        elif d == (1, -1):
            false_negative += 1
        else:
            false_positive += 1


    #calculating recall, precision, and f1 values

    recall = true_positive / (true_positive + false_negative)
    precision = true_positive / (false_positive + true_positive)
    f1 = 2 * (precision * recall) / (precision + recall)
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    MCC = ((true_positive * true_negative - false_positive * false_negative) / (true_positive + false_positive) * (false_negative + true_negative) * (false_positive + true_negative) * (true_positive + false_negative))**(1/2)
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (false_positive + true_negative)
    if debug:
        print("Performance Measures: \n")

        print("Value of True Positive:", true_positive)
        print("Value of True Negative:", true_negative)

        print("Value of False Positive:", false_positive)
        print("Value of False Negative:", false_negative)

        print()

        print("Accuracy of the Model:", accuracy)
        print()

        print("F1 Value:", f1)
        print()

        print("Recall Value:", recall)
        print()

        print("Precison Value:", precision)
        print()

        print("Matthews Correlation Coefficient Value:", MCC)
        print()

        print("Sensitivity:", sensitivity)
        print()

        print("Specificity:", specificity)
        print()


    return accuracy, f1, recall, precision, MCC, sensitivity, specificity

#main
def main():
    true_y = [-1,1,1,-1,1,-1] ###dummy values
    predicted_y = [-1,1,1,-1,-1,1] ###dummy values

    f1_score(true_y, predicted_y)
    measure_model(true_y, predicted_y, True)

if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))





