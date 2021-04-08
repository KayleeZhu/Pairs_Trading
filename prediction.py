import pandas as pd
from pathlib import Path
import datetime
import numpy as np

from timeseriescv import cross_validation as kfoldcv
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

import spread_feature_engineering as spd

# Define a dictionary of parameters for each model
param_dist = {
    'logistic': {'classifier__penalty': ['l1', 'l2', 'elasticnet'],
                 'classifier__dual': [True, False],
                 'classifier__C': np.arange(0.5, 5, 0.5),
                 'classifier__multi_class': ['auto', 'ovr', 'multinomial']
                 },
    'decision_tree': {'classifier__criterion': ['gini', 'entropy'],
                      'classifier__splitter': ['best', 'random'],
                      'classifier__max_depth': np.arange(3, 20, 1),
                      'classifier__min_samples_split': np.arange(2, 10, 1),
                      'classifier__min_samples_leaf': np.arange(2, 10, 1)
                      },
    'random_forest': {'classifier__n_estimators': np.arange(20, 200, 5),
                      'classifier__criterion': ['gini', 'entropy'],
                      'classifier__max_depth': np.arange(3, 11, 1),
                      'classifier__min_samples_split': np.arange(2, 10, 1),
                      'classifier__min_samples_leaf': np.arange(2, 10, 1)
                      },
    'svc': {'classifier__C': np.arange(0.5, 5, 0.5),
            'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'classifier__degree': np.arange(2, 8, 1),
            'classifier__gamma': ['scale', 'auto']
            }
}


def get_train_val_indices(data):

    cv = kfoldcv.CombPurgedKFoldCV(n_splits=5, n_test_splits=1, embargo_td=pd.Timedelta(days=10))

    data = data.sort_values('prediction_date')
    fold_indices = list(cv.split(data, pred_times=data['prediction_date'], eval_times=data['evaluation_date']))

    return fold_indices


def hyperparameter_tunning(model_type, X_data, y_data):
    """ This function will get the best hyperparameter of the given model type based on training set
        Supported model type includes Ridge, Decision Tree, Random Forest
    """

    # Get fold indicies:
    fold_indices = get_train_val_indices(X_data)

    # Generate pipeline for the given model type
    if model_type == 'logistic':
        model = Pipeline(steps=['classifier', LogisticRegression()])
    elif model_type == 'decision_tree':
        model = Pipeline(steps=['classifier', DecisionTreeClassifier()])
    elif model_type == 'random_forest':
        model = Pipeline(steps=['classifier', RandomForestClassifier()])
    elif model_type == 'svc':
        model = Pipeline(steps=['classifier', SVC()])

    # Search for best params
    reg = RandomizedSearchCV(model, param_dist[model_type], random_state=4,
                             scoring='f1_macro', n_jobs=-1, cv=fold_indices)

    # Drop non-features / non-labels columns
    X_data = X_data.drop(columns=['GVKEY_asset1', 'GVKEY_asset2', 'prediction_date', 'evaluation_date'])
    y_data = y_data['y']

    # Fit
    search = reg.fit(X_data, y_data)
    return search


def read_features_label_data():

    y_pkl_path = Path('data/pairs_label.pkl')
    X_pkl_path = Path('data/pairs_features.pkl')

    labels = pd.read_pickle(y_pkl_path)
    features = pd.read_pickle(X_pkl_path)

    return features, labels


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
    results = hyperparameter_tunning(model_type='logistic', X_data=X, y_data=y)
    print(results.best_score, )