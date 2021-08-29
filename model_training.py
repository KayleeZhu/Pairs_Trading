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
import CONFIG


def get_train_val_indices(data, lag=CONFIG.max_holding_period_days):

    cv = kfoldcv.CombPurgedKFoldCV(n_splits=5, n_test_splits=1, embargo_td=pd.Timedelta(days=lag+10))
    data = data.sort_values('prediction_date')
    fold_indices = list(cv.split(data, pred_times=data['prediction_date'], eval_times=data['evaluation_date']))

    return fold_indices


def time_series_sample_weighting_vix(y_data, vix_data, higher_weight_factor=1.2, weighted=True):
    if weighted:
        y_data['year'] = y_data['prediction_date'].dt.year
        y_data['month'] = y_data['prediction_date'].dt.month
        y_data = y_data.merge(vix_data, on=['year', 'month'], how='left')

        # Get VIX value of last month --> current_vix
        last_month_list = y_data.groupby(['year', 'month'])['prediction_date'].unique().values[-1]
        last_month_mask = y_data.prediction_date.isin(last_month_list)
        current_vix = y_data[last_month_mask].vix.mean()
        y_data['vix_diff'] = abs(y_data['vix'] - current_vix)

        # Reassign sample weights
        avg_weight = 1 / len(y_data)
        mask = y_data['vix_diff'] <= y_data['vix_diff'].median()

        # Calculate the weight factor applied to avg weight, such that sum of sample weight = 1
        similar_vix_len = len(y_data[mask])
        dissimilar_vix_len_vix_len = len(y_data[~mask])
        lower_weight_factor = (1 - higher_weight_factor) * similar_vix_len / dissimilar_vix_len_vix_len + 1
        y_data['sample_weight'] = np.where(mask, avg_weight * higher_weight_factor, avg_weight * lower_weight_factor)
        sample_weights = y_data['sample_weight'].sort_index()
    else:
        sample_weights = 1 / len(y_data)

    return sample_weights


def get_parameter_distribution(dict_num):
    # Define a dictionary of parameters for each model

    param_dist = {
        1: {'logistic': {'classifier__penalty': ['l2'],
                         'classifier__dual': [False],
                         'classifier__C': np.arange(0.5, 5, 0.5),
                         'classifier__multi_class': ['auto', 'ovr', 'multinomial'],
                         # 'classifier__random_state': [0],
                         # 'classifier__solver': ['saga'],
                         'classifier__max_iter': [3000]
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
        self.model_type = model_type
        self.score_method = score_method
        self.random_state = random_state
        self.X_info_columns = CONFIG.X_info_columns
        self.y_info_columns = CONFIG.y_info_columns

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
        return y_data['y'].copy()

    def hyperparameter_tunning(self, X_data, y_data):
        # Get fold indicies:
        fold_indices = get_train_val_indices(X_data)

        # Drop non-features / non-labels columns
        X = self.get_features(X_data)
        y = self.get_labels(y_data)

        # Search for best params
        cv = RandomizedSearchCV(self.pipeline, self.param_dist[self.model_type], random_state=self.random_state,
                                scoring=self.score_method, n_jobs=-1, cv=fold_indices, n_iter=5)
        cv.fit(X, y)
        self.pipeline = cv.best_estimator_

    def model_training(self, X_data, y_data, vix_data, higher_weight_factor, weighted):
        # Get sample weights for model fitting
        sample_weights = time_series_sample_weighting_vix(y_data, vix_data, higher_weight_factor, weighted)

        # Drop non-features / non-labels columns
        X = self.get_features(X_data)
        y = self.get_labels(y_data)

        # Train the model
        kwargs = {self.pipeline.steps[-1][0] + '__sample_weight': sample_weights}
        self.pipeline.fit(X, y, **kwargs)

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
    cleaned_data = pd.read_pickle(CONFIG.cleaned_data_path)
    pairs_data = spd.get_pairs_data(CONFIG.pairs_data_path)

    # Get features & labels data
    y = pd.read_pickle(CONFIG.pairs_label_data_path)
    X = pd.read_pickle(CONFIG.pairs_features_data_path)

    ppl = ModelPipeline(model_type=CONFIG.model_type,
                        score_method=CONFIG.score_method,
                        param_dist_num=CONFIG.param_dist,
                        random_state=CONFIG.random_state_num
                        )

