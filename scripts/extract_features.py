import datetime as dt

import pandas as pd


merged = pd.read_csv('data/kaggle/merged.csv').sort_values('ts_listen')

# User media listening frequency along time
merged['user_media_listen_freq'] = pd.concat([
    g['is_listened'].shift().rolling(min_periods=1, window=len(g)).mean()
    for _, g in merged.groupby(['user_id', 'media_id'])
])
merged['user_media_listen_freq'].fillna(0.5, inplace=True)

# User listening frequency along time
merged['user_listen_freq'] = pd.concat([
    g['is_listened'].shift().rolling(min_periods=1, window=len(g)).mean()
    for _, g in merged.groupby('user_id')
])
merged['user_listen_freq'].fillna(
    merged.groupby('user_id').nth(1)['user_listen_freq'].mean(),
    inplace=True
)

# User number of listens along time
merged['user_listen_count'] = pd.concat([
    g['is_listened'].shift().rolling(min_periods=1, window=len(g)).count()
    for _, g in merged.groupby('user_id')
])

# Only keep Flow observations
test_mask = merged['is_listened'].isnull()
train_mask = (merged['is_listened'].notnull()) & (merged['listen_type'] == 1)
flow = merged[(test_mask) | (train_mask)].copy()

# Track is first played in at least 10 minutes
flow['first_of_session'] = pd.concat([
    g['ts_listen'].diff().fillna(601) > 600
    for _, g in flow.groupby('user_id')
]).astype(int)

# User Flow listening frequency along time
flow['user_listen_freq_flow'] = pd.concat([
    g['is_listened'].shift().rolling(min_periods=1, window=len(g)).mean()
    for _, g in flow.groupby('user_id')
])
flow['user_listen_freq_flow'].fillna(
    flow.groupby('user_id').nth(1)['user_listen_freq_flow'].mean(),
    inplace=True
)

# User number of Flow listens along time
flow['user_listen_count_flow'] = pd.concat([
    g['is_listened'].shift().rolling(min_periods=1, window=len(g)).count()
    for _, g in flow.groupby('user_id')
])

# User Flow ratio
flow['user_flow_ratio'] = (flow['user_listen_count_flow'] / flow['user_listen_count']).fillna(0.5)

# Platform
flow['platform'] = flow['platform_family'].apply(str) + flow['platform_name'].apply(str)

# Listening time
flow['dt_listen'] = flow['ts_listen'].apply(dt.datetime.fromtimestamp)
flow['listen_at_hour'] = flow['dt_listen'].dt.hour
flow['listen_at_weekday'] = flow['dt_listen'].dt.weekday

# Release year
def get_year(date):
    return dt.datetime.strptime(str(date), '%Y%m%d').year

flow['release_year'] = flow['release_date'].apply(get_year)

# Select features
features = flow[[
    'context_type',
    'first_of_session',
    'listen_at_hour',
    'listen_at_weekday',
    'platform',
    'release_year',
    'user_age',
    'user_flow_ratio',
    'user_gender',
    'user_listen_count',
    'user_listen_count_flow',
    'user_listen_freq',
    'user_listen_freq_flow',
    'user_media_listen_freq',

    'is_listened',
    'sample_id'
]]

# One-hot encoding
features = pd.get_dummies(features, columns=['context_type', 'platform'])

# Drop context types that are not present in the test set
test_features = features[features['is_listened'].isnull()]
for feature in features.columns:
    if feature.startswith('context_type') and test_features[feature].sum() == 0:
        features.drop(feature, axis='columns', inplace=True)

features.to_csv('data/features.csv', index=False)
