import os

import matplotlib.pyplot as plt
from sklearn.externals import joblib
from xgboost import plot_importance


directory = os.path.dirname(os.path.realpath(__file__))
pipe = joblib.load(os.path.join(directory, 'pipeline_best.pkl'))
gbm = pipe.steps[-1][1]

plot_importance(gbm)
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(directory, 'feature_importance.png'), dpi=300)
