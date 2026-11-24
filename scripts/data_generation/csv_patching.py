import pandas as pd

from src.generation.xtrains.concept_definitions import has_passenger_long_wagon
from src.generation.xtrains.intervention_registry import InterventionManagerXTRAINS
from src.utils.common_paths import XTRAINS_NEW_CSV_PATH, XTRAINS_NEW_CONFIG_PATH
from src.utils.pickle import read_configs

im = InterventionManagerXTRAINS()

needed_concepts = list(im._registry.keys())
print(len(needed_concepts))
print(needed_concepts)

df = pd.read_csv(XTRAINS_NEW_CSV_PATH)

existing = list(df.columns)
print(df.columns)
missing = []
for c in needed_concepts:
    if c not in existing:
        missing.append(c)
print(missing)

df['3Wagon'] = df['NumberOfWagons'].apply(lambda x: 1 if x>=3 else 0)
df['2LongWagon'] = df['NumberOfLongWagons'].apply(lambda x: 1 if x>=2 else 0)
df['2PassengerCar'] = df['NumberOfPassengerCars'].apply(lambda x: 1 if x>=2 else 0)
df['notEmptyTrain'] = 1 - df['EmptyTrain']
df['notLongWagon'] = 1 - df['LongWagon']
res = []
cfgs = read_configs(XTRAINS_NEW_CONFIG_PATH, as_dict=True)
for c in cfgs:
    res.append(has_passenger_long_wagon(cfgs[c]))
# df['PassengerLongWagon'] = res
df['PassengerLongWagon'] = df['image_id'].map(
    lambda i: has_passenger_long_wagon(cfgs[i])
)
df.to_csv("labels.csv", index=False)
