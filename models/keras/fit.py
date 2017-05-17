import os

from keras import backend as K
from keras import callbacks
from keras import layers
from keras import models
from keras.wrappers.scikit_learn import KerasClassifier
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.externals import joblib


# Load data
X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv')['is_listened']

# Use Tenserflow backend
sess = tf.Session()
K.set_session(sess)


def model():
    model = models.Sequential([
        layers.Dense(64, input_dim=X_train.shape[1], activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')

pipe = pipeline.Pipeline([
    ('rescale', preprocessing.StandardScaler()),
    ('nn', KerasClassifier(build_fn=model, nb_epoch=10, batch_size=128,
                           validation_split=0.2, callbacks=[early_stopping]))
])


pipe.fit(X_train.values, y_train.values)

directory = os.path.dirname(os.path.realpath(__file__))
model_step = pipe.steps.pop(-1)[1]
joblib.dump(pipe, os.path.join(directory, 'pipeline.pkl'))
models.save_model(model_step.model, os.path.join(directory, 'model.h5'))
