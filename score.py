from sklearn.metrics import f1_score, classification_report, confusion_matrix

LABELS = [0, 1]  # 0 = not disaster, 1 = disaster

def score_submission(y_true, y_pred):
    """
    Compute the binary F1 score and return confusion matrix and classification report.
    """
    f1 = f1_score(y_true, y_pred, average='binary')  # target class = 1
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    report = classification_report(y_true, y_pred, labels=LABELS, target_names=["not_disaster", "disaster"])
    return f1, cm, report

def print_confusion_matrix(cm):
    """
    Nicely print the confusion matrix.
    """
    labels = ["not_disaster", "disaster"]
    header = "|{:^15}|{:^15}|{:^15}|".format('', *labels)
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        print("|{:^15}|{:^15}|{:^15}|".format(labels[i], *row))
        print("-" * len(header))

def report_score(y_true, y_pred):
    f1, cm, report = score_submission(y_true, y_pred)
    print_confusion_matrix(cm)
    print("\nClassification Report:\n", report)
    print(f"\n Binary F1 Score (target: 1): {f1:.4f}")
    return f1

if __name__ == "__main__":
    # Example usage
    actual = [0, 1, 0, 1, 1, 0, 1, 0]
    predicted = [0, 1, 0, 0, 1, 1, 1, 0]
    report_score(actual, predicted)
