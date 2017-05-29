import os

import pandas as pd
from sklearn.externals import joblib


X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')

directory = os.path.dirname(os.path.realpath(__file__))
pipe = joblib.load(os.path.join(directory, 'pipeline.pkl'))

pred = pipe.predict_proba(X_test)[:, 1]

submission = pd.DataFrame(data={
    'sample_id': y_test['sample_id'].astype(int),
    'is_listened': pred
}).sort_values('sample_id')

submission.to_csv(os.path.join(directory, 'submission_lightgbm.csv'), index=False)
