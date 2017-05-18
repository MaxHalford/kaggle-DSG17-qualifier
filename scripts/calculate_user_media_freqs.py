import pandas as pd


merged = pd.read_csv('data/kaggle/merged.csv').sort_values('ts_listen')

user_media_freqs = pd.concat([
    g['is_listened'].shift().rolling(min_periods=1, window=len(g)).mean()
    for _, g in merged.groupby(['user_id', 'media_id'])
])

user_media_freqs.to_frame('user_media_freq').to_csv('data/user_media_freqs.csv')
