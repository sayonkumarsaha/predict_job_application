import sys
import re
import sys
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MultiLabelBinarizer


class DataPreprocessing:
    """
    Class used for Labelled Data Pre-processing: Data Cleaning and Feature Extraction
    """

    def __init__(self, data_job_desc_path, data_user_path):
        """
        :param data: Labelled training data for user-applied job description
                     Labelled training data for user features
        """
        self.data_job_desc = pd.read_csv(data_job_desc_path)
        self.data_user = pd.read_csv(data_user_path)

    def handler(self):
        """
        Handle method to perform all the functionality.
        :return: train_data: cleaned and featured engineered dataset ready to be used for training
        """

        raw_data = pd.merge(self.data_job_desc, self.data_user, on=['user_id']).drop(['user_id'], axis=1)

        data_company_encoded = self._encode_company_features(raw_data)
        data_jobtitle_encoded = self._encode_job_title_features(data_company_encoded)
        data_imputed = self._fill_missing_data(data_jobtitle_encoded)

        trained_data = data_imputed

        return trained_data

    def _encode_company_features(self, data):

        data_company_encoded = pd.get_dummies(data, prefix=['company'], columns =['company'])
        return data_company_encoded

    def _get_jobtitle_keywords(self, jobtitle):

        manual_analysis_word_removal_list = ['(m/f/d)', 'mfd', 'team', '-']
        keywords = re.split(',| | &| ;', jobtitle)
        keywords = [re.sub(r'|\.|\(|\)|\]|\[|\/|-|!|\?', '', word) for word in keywords if len(word) > 1]
        keywords = [word.lower() for word in keywords if
                    word.lower() not in manual_analysis_word_removal_list]
        keywords = list(set(' '.join(keywords).split()))
        return keywords

    def _encode_job_title_features(self, data):

        data['jobtitle_keywords'] = ''
        for index, row in data.iterrows():
            data['jobtitle_keywords'].values[index]= self._get_jobtitle_keywords(row['job_title_full'])

        mlb = MultiLabelBinarizer()
        jobtitle_encoded = pd.DataFrame(mlb.fit_transform(data['jobtitle_keywords']),
                                        columns='jobtitle_keyword_' + mlb.classes_)

        data_jobtitle_encoded = pd.concat([data, jobtitle_encoded], axis=1, join='inner')
        data_jobtitle_encoded = data_jobtitle_encoded.drop(['job_title_full', 'jobtitle_keywords'], axis=1)

        return data_jobtitle_encoded

    def _fill_missing_data(self, data):

        sys.setrecursionlimit(100000)

        non_imput_cols = ['has_applied']
        data_to_imput = data.drop(non_imput_cols, axis=1)
        imput_cols = list(data_to_imput)
        non_imputed_data = data.drop(imput_cols, axis=1)

        imp_mean_knn = KNNImputer(n_neighbors=30)
        imputed_data = imp_mean_knn.fit_transform(data_to_imput)
        imputed_data = pd.DataFrame(imputed_data, columns=imput_cols)
        resultant_imputed_data = pd.concat([non_imputed_data, imputed_data], axis=1, join='inner')
        return resultant_imputed_data

