import pandas as pd
from pathlib import Path
import datetime
import numpy as np

from timeseriescv import cross_validation as kfoldcv
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

import spread_feature_engineering as spd


def read_features_label_data():

    y_pkl_path = Path('data/pairs_label.pkl')
    X_pkl_path = Path('data/pairs_features.pkl')

    labels = pd.read_pickle(y_pkl_path)
    features = pd.read_pickle(X_pkl_path)

    return features, labels


def get_train_val_indices(data):

    cv = kfoldcv.CombPurgedKFoldCV(n_splits=5, n_test_splits=1, embargo_td=pd.Timedelta(days=10))

    data = data.sort_values('prediction_date')
    fold_indices = list(cv.split(data, pred_times=data['prediction_date'], eval_times=data['evaluation_date']))

    return fold_indices


def get_parameter_distribution(dict_num):
    # Define a dictionary of parameters for each model

    param_dist = {
        1: {'logistic': {'classifier__penalty': ['l2', 'none'],
                         'classifier__dual': [True, False],
                         'classifier__C': np.arange(0.5, 5, 0.5),
                         'classifier__multi_class': ['auto', 'ovr', 'multinomial']
                         },
            'decision_tree': {'classifier__criterion': ['gini', 'entropy'],
                              'classifier__splitter': ['best', 'random'],
                              'classifier__max_depth': np.arange(5, 10, 2),
                              'classifier__max_leaf_nodes': np.arange(20, 40, 5),
                              'classifier__min_samples_split': np.arange(2, 10, 3),
                              'classifier__min_samples_leaf': np.arange(10, 80, 2)
                              },
            'random_forest': {'classifier__n_estimators': np.arange(50, 250, 5),
                              'classifier__criterion': ['gini', 'entropy'],
                              'classifier__max_depth': np.arange(3, 6, 2),
                              'classifier__min_samples_split': np.arange(3, 10, 2),
                              'classifier__min_samples_leaf': np.arange(20, 80, 5)
                              }
            },

        2: {'logistic': {'classifier__penalty': ['l1', 'l2', 'none'],
                         'classifier__dual': [True, False],
                         'classifier__C': np.arange(0.2, 5, 0.2),  # smaller means stronger regularization
                         'classifier__multi_class': ['auto', 'ovr', 'multinomial']
                         },
            'decision_tree': {'classifier__criterion': ['gini', 'entropy'],
                              'classifier__splitter': ['best', 'random'],
                              'classifier__max_depth': np.arange(5, 10, 2),
                              'classifier__max_leaf_nodes': np.arange(20, 40, 5),
                              'classifier__min_samples_split': np.arange(2, 10, 3),
                              'classifier__min_samples_leaf': np.arange(10, 80, 2)
                              },
            'random_forest': {'classifier__n_estimators': np.arange(50, 250, 5),
                              'classifier__criterion': ['gini', 'entropy'],
                              'classifier__max_depth': np.arange(3, 6, 2),
                              'classifier__min_samples_split': np.arange(3, 10, 2),
                              'classifier__min_samples_leaf': np.arange(20, 80, 5)
                              }
            },

        3: {'logistic': {'classifier__penalty': ['l2', 'none'],
                         'classifier__dual': [True, False],
                         'classifier__C': np.arange(0.5, 5, 0.5),
                         'classifier__multi_class': ['auto', 'ovr', 'multinomial']
                         },
            'decision_tree': {'classifier__criterion': ['gini', 'entropy'],
                              'classifier__splitter': ['best', 'random'],
                              'classifier__max_depth': np.arange(5, 10, 2),
                              'classifier__max_leaf_nodes': np.arange(20, 40, 5),
                              'classifier__min_samples_split': np.arange(2, 10, 3),
                              'classifier__min_samples_leaf': np.arange(10, 80, 2)
                              },
            'random_forest': {'classifier__n_estimators': np.arange(50, 250, 5),
                              'classifier__criterion': ['gini', 'entropy'],
                              'classifier__max_depth': np.arange(3, 6, 2),
                              'classifier__min_samples_split': np.arange(3, 10, 2),
                              'classifier__min_samples_leaf': np.arange(20, 80, 5)
                              }
            }
    }
    return param_dist[dict_num]


class ModelPipeline:

    def __init__(self, model_type, score_method, param_dist_num, random_state):
        self.random_state = random_state
        self.X_info_columns = ['GVKEY_asset1', 'GVKEY_asset2', 'prediction_date', 'evaluation_date']
        self.y_info_columns = ['prediction_date', 'evaluation_date', 'GVKEY_asset1', 'GVKEY_asset2',
                               'spread_return_60d_std',
                               'spread_t0', 'spread_t1', 'spread_t2', 'spread_t3', 'spread_t4', 'spread_t5',
                               'spread_return_1d', 'spread_return_2d', 'spread_return_3d', 'spread_return_4d',
                               'spread_return_5d']
        self.model_type = model_type
        self.score_method = score_method

        self.param_dist = get_parameter_distribution(param_dist_num)
        self.pipeline = self.set_up_pipeline()

    def set_up_pipeline(self):
        # Generate pipeline for the given model type
        if self.model_type == 'logistic':
            pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', LogisticRegression())])
        elif self.model_type == 'decision_tree':
            pipeline = Pipeline(steps=[('classifier', DecisionTreeClassifier())])
        elif self.model_type == 'random_forest':
            pipeline = Pipeline(steps=[('classifier', RandomForestClassifier())])

        return pipeline

    def get_features(self, X_data):
        X = X_data.copy().drop(columns=self.X_info_columns)
        return X

    def get_labels(self, y_data):
        return y_data.copy()['y']

    def hyperparameter_tunning(self, X_data, y_data):
        # Get fold indicies:
        fold_indices = get_train_val_indices(X_data)

        # Drop non-features / non-labels columns
        X = self.get_features(X_data)
        y = self.get_labels(y_data)

        # Search for best params
        cv = RandomizedSearchCV(self.pipeline, self.param_dist[self.model_type], random_state=self.random_state,
                                scoring=self.score_method, n_jobs=-1, cv=fold_indices, n_iter=20)
        cv.fit(X, y)
        self.pipeline = cv.best_estimator_

    def model_training(self, X_data, y_data):
        # Drop non-features / non-labels columns
        X = self.get_features(X_data)
        y = self.get_labels(y_data)

        # Train the model
        self.pipeline.fit(X, y)

    def get_prediction(self, X_data, y_data):
        # Drop non-features columns
        X = self.get_features(X_data)

        # Get predicted labels and prediction probabilities
        y_pred = self.pipeline.predict(X)
        prob = self.pipeline.predict_proba(X)

        # Attach the prediction and prob to data
        y_data['y_pred'] = y_pred
        y_data[['prob_short', 'prob_hold', 'prob_long']] = prob
        return y_data


if __name__ == '__main__':

    # Get all_data & pairs data
    data_path = Path('data/cleaned_data.pkl')
    data = spd.get_crsp_data(data_path)

    pairs_path = Path('data/pairs_for_all_days.pkl')
    pairs_data = spd.get_pairs_data(pairs_path)

    # Get spread features & label
    spread_features = spd.SpreadFeature(all_data=data, pairs=pairs_data)
    pairs_features = spread_features.generate_label_y(upper_threshold_factor=0.8, lower_threshold_factor=0.8)

    # Get features & labels data
    X, y = read_features_label_data()
    ppl = ModelPipeline(model_type='random_forest', score_method='f1_macro')
