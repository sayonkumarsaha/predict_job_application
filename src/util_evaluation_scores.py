from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def get_scores(y_pred, Y_test):
    """
    :param y_pred: Actual predictions from the model.
    :param Y_test: Labelled data for comparing predictions.
    :return: scores: eEvaluation scores for the model based on prediction on test data.
    """
    accuracy = round(accuracy_score(Y_test, y_pred) * 100, 3)
    precision = round(precision_score(Y_test, y_pred) * 100, 3)
    recall = round(recall_score(Y_test, y_pred) * 100, 3)
    f1 = round(f1_score(Y_test, y_pred) * 100, 3)
    auroc = round(roc_auc_score(Y_test, y_pred) * 100, 3)
    scores = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc
    }
    return scores