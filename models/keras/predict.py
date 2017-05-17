import os

from keras import models
import pandas as pd
from sklearn.externals import joblib


X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')

directory = os.path.dirname(os.path.realpath(__file__))
pipe = joblib.load(os.path.join(directory, 'pipeline.pkl'))
model = models.load_model(os.path.join(directory, 'model.h5'))
pipe.steps.append(('nn', model))

pred = pipe.predict_proba(X_test)[:, 0]

submission = pd.DataFrame(data={
    'sample_id': y_test['sample_id'].astype(int),
    'is_listened': pred
}).sort_values('sample_id')

submission.to_csv(os.path.join(directory, 'submission_keras.csv'), index=False)
