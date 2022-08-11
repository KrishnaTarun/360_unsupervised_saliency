import pandas as pd
import numpy as np
import os


data = pd.read_csv('./example_SalmapComparisons.csv')
data.head()
result = np.array([0, 0, 0, 0, 0])
for i in range(0, 500, 5):
    result = result + (data.iloc[i:i + 5, -1]).to_numpy()
result = result / 100
my_result = {'AUC_Judd': result[0], 'NSS': result[1], 'CC': result[2], 'SIM': result[3], 'KLD': result[4]}
print(my_result)