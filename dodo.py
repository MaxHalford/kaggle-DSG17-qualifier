def task_merge_kaggle_data():
    return {
        'actions': ['python scripts/merge_kaggle_data.py'],
        'file_dep': [
            'data/kaggle/extra_infos.json',
            'data/kaggle/test.csv',
            'data/kaggle/train.csv'
        ],
        'targets': ['data/kaggle/merged.csv']
    }


def task_extract_features():
    return {
        'actions': ['python scripts/extract_features.py'],
        'file_dep': ['data/kaggle/merged.csv'],
        'targets': ['data/features.csv']
    }


TRAINING_SETS = [
    'data/X_train.csv',
    'data/y_train.csv',
    'data/X_train_sample.csv',
    'data/y_train_sample.csv'
]

TEST_SETS = [
    'data/X_test.csv',
    'data/y_test.csv'
]


def task_split_train_test():
    return {
        'actions': ['python scripts/split_train_test.py'],
        'file_dep': ['data/features.csv'],
        'targets': TRAINING_SETS + TEST_SETS
    }


def task_fit_keras():
    return {
        'actions': ['python models/keras/fit.py'],
        'file_dep': TRAINING_SETS,
        'targets': ['models/keras/pipeline.pkl', 'models/keras/model.h5'],
        'verbosity': 2 # To display training progress
    }


def task_predict_keras():
    return {
        'actions': ['python models/keras/predict.py'],
        'file_dep': TEST_SETS + ['models/keras/pipeline.pkl', 'models/keras/model.h5'],
        'targets': ['models/keras/submission_keras.csv']
    }


def task_fit_xgboost():
    return {
        'actions': ['python models/xgboost/fit.py'],
        'file_dep': TRAINING_SETS,
        'targets': ['models/xgboost/pipeline.pkl'],
        'verbosity': 2 # To display training progress
    }


def task_predict_xgboost():
    return {
        'actions': ['python models/xgboost/predict.py'],
        'file_dep': TEST_SETS + ['models/xgboost/pipeline.pkl'],
        'targets': ['models/xgboost/submission_xgboost.csv']
    }

def task_plot_xgboost():
    return {
        'actions': [
            'python models/xgboost/plot_learning_curve.py',
            'python models/xgboost/plot_feature_importance.py',
        ],
        'file_dep': ['models/xgboost/pipeline.pkl'],
        'targets': [
            'models/xgboost/roc_auc_learning_curve.png',
            'models/xgboost/feature_importance.png'
        ]
    }

