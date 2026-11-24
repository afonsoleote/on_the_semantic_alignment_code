import pandas as pd

from src.utils.common_paths import CUB_CSV_PATH, DATASETS_PATH

df = pd.read_csv(CUB_CSV_PATH)


ids = pd.read_csv(DATASETS_PATH / "CUB_200_2011" / "images.txt", sep=r'\s+', header=None, names=['image_id', 'image_name'])

print(df.columns[0], '->', df.iloc[0, 0])
print('images.txt   ->', ids['image_name'].iloc[0])

df = df.merge(ids, left_on='filepath', right_on='image_name', how='left')
df = df.drop(columns='image_name')
print('unmatched:', df['image_id'].isna().sum())

print(len(set(df['filepath']) & set(ids['image_name'])), '/', len(df))
col = df.pop('image_id')
df.insert(0, 'image_id', col)

df.to_csv("CUB_attributes_id.csv",index=False)