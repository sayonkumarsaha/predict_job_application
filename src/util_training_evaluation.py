from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.util_evaluation_scores import get_scores

import sys
sys.setrecursionlimit(100000)


class TrainingEvaluation:
    """
    Class used for splitting the data into train/ test set, training, and evaluating a given model on the test set.
    """

    def __init__(self, train_data, model):
        """
        :param train_data: Cleaned data with features ready to be trained upon.
        :param model: Model initialized with parameters ready for fitting and training.
        """
        self.train_data = train_data
        self.model = model

    def handler(self):
        """
        Handle method to perform all the functionality.
        :return: model: Trained model fitted to the parameters using training data.
                 eval_scores: Evaluation scores for the model based on prediction on test data.
        """

        print("Splitting Train/Test Data..")
        X_train, X_test, Y_train, Y_test = self._get_split_data(test_ratio=0.25)

        print("Scaling Data..")
        X_train_scaled, X_test_scaled = self._get_scaled_data(X_train, X_test)

        print("Training Model..")
        self.model.fit(X_train_scaled, Y_train)

        print("Evaluating Model..")
        y_pred = self.model.predict(X_test_scaled)
        eval_scores = get_scores(y_pred, Y_test)

        return self.model, eval_scores

    def _get_split_data(self, test_ratio):
        """
        :param test_ratio: Ratio in which Train/ Test Data is to be split
        :return: Training and Testing Data with separate prediction columns after splitting.
        """
        pred_col = self.train_data.has_applied
        train_features = self.train_data.loc[:, self.train_data.columns != 'has_applied']
        return train_test_split(train_features, pred_col, test_size=test_ratio, stratify=pred_col)

    def _get_scaled_data(self, X_train, X_test):
        """
        :param X_train: Training Data to be scaled
        :param X_test: Testing Data to be scaled
        :return: Standardized features by removing the mean and scaling to unit variance
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
