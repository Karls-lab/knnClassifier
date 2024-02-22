import pandas as pd
import numpy as np
import torch
import os

print(os.getcwd())
curFolder = os.path.dirname(os.path.realpath(__file__))
df = pd.read_csv(os.path.join(curFolder, 'diabetes.csv'))

# Standardize the data
for feature in df.columns[:-1]:
    mean = df[feature].mean()
    std = df[feature].std()
    df[feature] = (df[feature] - mean) / std
print(df.head())

# Save the preprocessed data
df.to_csv(os.path.join(curFolder, 'diabetes_preprocessed.csv'), index=False)

# Run the data on the knn model
knnModelFolder = os.path.join(curFolder, '..', 'heartDisease')



