import pandas as pd


train = pd.read_csv('data/kaggle/train.csv')
test = pd.read_csv('data/kaggle/test.csv')
extra = pd.read_json('data/kaggle/extra_infos.json', lines=True)

train = pd.merge(train, extra, on='media_id')
test = pd.merge(test, extra, on='media_id')

data = pd.concat((train, test), axis='rows')
data.to_csv('data/kaggle/merged.csv')
