import pandas as pd
from sklearn import model_selection
from sklearn import pipeline
import xam
import xgboost as xgb


# Load data
X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv')['is_listened']

# Create a validation set with 20% of the training set
X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.2)

pipe = pipeline.Pipeline([
    ('gbm', xgb.XGBClassifier(
        n_estimators=50,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.9,
        scale_pos_weight=sum(y_train == 0) / sum(y_train == 1)
    ))
])

def split(row):
    return row['user_listen_count'] < 11

split_pipe = xam.splitting.SplittingEstimator(pipe, split)


scores = model_selection.cross_val_score(
    estimator=pipe,
    X=X_train,
    y=y_train,
    cv=model_selection.KFold(n_splits=5, random_state=42),
    scoring='roc_auc',
    fit_params={
        'gbm__eval_set': [(X_train, y_train), (X_val, y_val)],
        'gbm__eval_metric': ['auc'],
        'gbm__early_stopping_rounds': 10,
        'gbm__verbose': False
    }
)

print('ROC AUC: %0.5f (± %0.5f)' % (scores.mean(), 1.96 * scores.std()))

print('...................')

scores = model_selection.cross_val_score(
    estimator=split_pipe,
    X=X_train,
    y=y_train,
    cv=model_selection.KFold(n_splits=5, random_state=42),
    scoring='roc_auc',
    fit_params={
        'gbm__eval_set': [(X_train, y_train), (X_val, y_val)],
        'gbm__eval_metric': ['auc'],
        'gbm__early_stopping_rounds': 10,
    }
)

print('ROC AUC: %0.5f (± %0.5f)' % (scores.mean(), 1.96 * scores.std()))

