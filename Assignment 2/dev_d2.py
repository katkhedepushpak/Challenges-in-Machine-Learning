### AUTHOR : PUSHPAK VIJAY KATKHEDE
### AI 539 - ML Challenges
### Assignment 2: Give Your Models a Grade
### DATE : 02/04/2023

from imblearn.over_sampling import SMOTE
import pandas as pd
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from pathlib import Path

filepath = Path('dev2.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)


data = pd.read_csv('activity-dev.csv')

X = data[['G_front', 'G_vert', 'G_lat', 'ant_id', 'RSSI', 'phase', 'freq', "person"]]
y = data['activity']

counter = Counter(y)
print(counter)

dict_o = {3: 2368, 1: 1400, 2: 1400, 4: 1400}

dict_u = { 3: 1400, 1: 1400, 2: 1400, 4: 1400}

oversample = SMOTE(sampling_strategy=dict_o, random_state=9)
undersample = RandomUnderSampler(sampling_strategy=dict_u, random_state=9)

X, y = oversample.fit_resample(X, y)
counter = Counter(y)
print(counter)
X, y = undersample.fit_resample(X, y)
counter = Counter(y)
print(counter)

result = pd.concat([X, y], axis=1).reindex(y.index)

result.to_csv(filepath, index=False)

