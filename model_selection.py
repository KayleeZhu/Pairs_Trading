import pandas as pd
from pathlib import Path
import datetime
import numpy as np

from timeseriescv import cross_validation as kfoldcv
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

import spread_feature_engineering as spd


def get_train_val_indices(data):

    cv = kfoldcv.CombPurgedKFoldCV(n_splits=5, n_test_splits=1, embargo_td=pd.Timedelta(days=10))

    data = data.sort_values('prediction_date')
    fold_indices = list(cv.split(data, pred_times=data['prediction_date'], eval_times=data['evaluation_date']))

    return fold_indices


def read_features_label_data():

    y_pkl_path = Path('data/pairs_label.pkl')
    X_pkl_path = Path('data/pairs_features.pkl')

    labels = pd.read_pickle(y_pkl_path)
    features = pd.read_pickle(X_pkl_path)

    return features, labels


class Modelling:

    def __init__(self, X_data, y_data, train_year):
        self.train_year = train_year
        self.X = self.get_historical_data_for_given_year(X_data)
        self.y = self.get_historical_data_for_given_year(y_data)
        self.param_dist = self.get_parameter_distribution()

    def get_historical_data_for_given_year(self, data):
        # Select historical dates data
        year = datetime.datetime.strptime(self.train_year, '%Y')
        data = data[data['evaluation_date'] < year]
        return data

    @staticmethod
    def get_parameter_distribution():
        # Define a dictionary of parameters for each model

        param_dist = {
            'logistic': {'classifier__penalty': ['l2', 'none'],
                         'classifier__dual': [True, False],
                         'classifier__C': np.arange(0.5, 5, 0.5),
                         'classifier__multi_class': ['auto', 'ovr', 'multinomial']
                         },
            'decision_tree': {'classifier__criterion': ['gini', 'entropy'],
                              'classifier__splitter': ['best', 'random'],
                              'classifier__max_depth': np.arange(3, 20, 1),
                              'classifier__min_samples_split': np.arange(2, 10, 1),
                              'classifier__min_samples_leaf': np.arange(2, 20, 1)
                              },
            'random_forest': {'classifier__n_estimators': np.arange(20, 250, 5),
                              'classifier__criterion': ['gini', 'entropy'],
                              'classifier__max_depth': np.arange(5, 20, 1),
                              'classifier__min_samples_split': np.arange(2, 10, 1),
                              'classifier__min_samples_leaf': np.arange(2, 20, 1)
                              }
        }
        return param_dist

    def hyperparameter_tunning(self, model_type, score_method):
        """ This function will get the best hyperparameter of the given model type based on training set
            Supported model type includes Logistic, Decision Tree, Random Forest & SVC
            score_method: {str} -- specify what scoring method to use
        """

        # Get fold indicies:
        fold_indices = get_train_val_indices(self.X)

        # Generate pipeline for the given model type
        if model_type == 'logistic':
            model = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', LogisticRegression())])
        elif model_type == 'decision_tree':
            model = Pipeline(steps=[('classifier', DecisionTreeClassifier())])
        elif model_type == 'random_forest':
            model = Pipeline(steps=[('classifier', RandomForestClassifier())])

        # Search for best params
        reg = RandomizedSearchCV(model, self.param_dist[model_type], random_state=4,
                                 scoring=score_method, n_jobs=-1, cv=fold_indices)

        # Drop non-features / non-labels columns
        data_X = self.X.copy().drop(columns=['GVKEY_asset1', 'GVKEY_asset2', 'prediction_date', 'evaluation_date'])
        data_y = self.y.copy()['y']

        # Fit
        search = reg.fit(data_X, data_y)
        return search

    def choose_the_best_model(self, score_method, model_type_list):
        # Record run time
        start_time = datetime.datetime.now()

        score_list = []
        param_list = []
        estimators_list = []
        for model in model_type_list:
            search = self.hyperparameter_tunning(model, score_method)
            score_list.append(search.best_score_)
            param_list.append(search.best_params_)
            estimators_list.append(search.best_estimator_)
            # print(f"The {score_method} score of model {model} is {search.best_score_}")

        # Get the best model based on score
        best_model_index = score_list.index(max(score_list))
        best_model = model_type_list[best_model_index]
        best_score = score_list[best_model_index]
        best_param = param_list[best_model_index]
        best_estimator = estimators_list[best_model_index]
        print(f"The chosen model for year {self.train_year} is {best_model}, with model score of {best_score}")

        # Run time record
        end_time = datetime.datetime.now()
        run_time = end_time - start_time
        print(f'{run_time.seconds} seconds')

        return best_score, best_param, best_estimator


if __name__ == '__main__':

    # # Get all_data & pairs data
    # data_path = Path('data/cleaned_data.pkl')
    # data = spd.get_crsp_data(data_path)
    #
    # pairs_path = Path('data/pairs_for_all_days.pkl')
    # pairs_data = spd.get_pairs_data(pairs_path)
    #
    # # Get spread features & label
    # spread_features = spd.SpreadFeature(all_data=data, pairs=pairs_data)
    # pairs_features = spread_features.generate_label_y(upper_threshold_factor=1, lower_threshold_factor=1)

    # Get features & labels data
    X, y = read_features_label_data()
    # results = hyperparameter_tunning(model_type='logistic', X_data=X, y_data=y)
    # print(results.best_score)

    model = Modelling(X, y, train_year='2012')
    model_score, model_params, model_estimators = model.choose_the_best_model(score_method='f1_macro',
                                                                              model_type_list=['logistic',
                                                                                               'decision_tree',
                                                                                               'random_forest'])
    print(f"Model score is {model_score}")
    print(f"Model parameters are {model_params}")
    print(f"Model estimators are {model_estimators}")

