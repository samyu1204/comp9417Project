# Using dask here, to work with such a big dataset:
# pip isntall dask!
from tokenize import String
import pyarrow.parquet as pq
import datatable as dt
import numpy as np

# Documentations:https://docs.dask.org/en/stable/dataframe.html

import pandas as pd
# pip install pyarrow
# List of columns that has more than 50% missing values:
remove_list = ['S_2', 'D_42', 'D_49', 'D_50', 'D_53', 'D_56', 'S_9', 'B_17', 'D_73', 'D_76', 'R_9', 'D_82', 'B_29', 'D_87', 'D_88', 'D_105', 'D_106', 'R_26', 'D_108', 'D_110', 'D_111', 'B_39', 'B_42', 'D_132', 'D_134', 'D_135', 'D_136', 'D_137', 'D_138', 'D_142']
# * ========================================================================================
col_avg = [0.64833, 0.1517, 0.12663, 0.62104, 0.08019, 0.23053, 0.0586, 0.1279, 0.15394, 0.11844, 0.16987, 0.23778, 0.0801, 0.04781, 0.47652, 0.39687, 0.3722, 0.1601, 0.18896, 0.48097, 
          0.13929, 0.19194, 0.1308, 0.17673, 0.59666, 0.233, 0.08582, 0.11292, 0.27279, 0.98497, 0.03136, 0.22669, 0.08998, 0.32588, 0.28028, 0.09838, 0.03516, 0.21323, 0.09808, 0.37178, 
          0.37806, 0.41668, 0.05039, 0.37355, 0.18567, 0.94061, 1.53744, 0.04338, 0.33364, 0.59843, 0.14693, -0.78343, 0.22572, 4.57168, 0.2418, 0.05967, 0.25429, 0.06159, 0.15561, 0.10395, 
          0.11046, 0.06751, 0.03749, 0.37708, 0.1734, 0.18833, 0.15318, 0.17019, 0.03934, 0.08784, 0.21532, 0.10203, 0.08877, 0.06323, 0.07125, 0.03871, 0.05455, 0.11016, 0.06559, 0.05286, 
          0.00501, 0.05299, 0.03972, 0.98006, 0.15109, 0.0055, 0.04361, 0.23662, 0.02089, 0.04487, 0.06148, 0.15474, 0.03052, 0.03565, 0.00532, 0.00501, 0.99682, 0.00503, 0.022, 0.02621, 0.0186, 
          0.03662, 0.02305, 0.61571, 0.00645, 0.00844, 0.00528, 0.05993, 0.08659, 0.01472, 0.02126, 0.01891, 0.00826, 0.03435, 0.77966, 0.17907, 0.73526, 0.92474, 0.06384, 0.18698, 0.46985, 0.45306,
          0.20805, 0.00627, 0.12589, 0.89, 2.6769, 0.00658, 0.84634, 0.20188, 0.28365, 0.16246, 0.51284, 0.27009, -0.06233, 2.09566, 0.27565, 0.27162, 0.06238, 0.52214, 0.398, 0.05325, 0.30387, 
          0.08873, 0.67078, 0.10278, 0.57121, 0.43069, 0.03189, 0.20487, 0.10649, 0.04616, 0.00608, 0.18294, 0.02757, 0.16804, 0.18281, 0.05168, 0.06415]

# Data getter functions:
def get_sample_sub():
  return pd.read_csv("../data/sample_submission.csv")

def get_test_data():
  return pd.read_parquet("../data/test.parquet")

# Train labels show if the person defaults or not:
def get_train_label():
  return pd.read_csv("../data/train_labels.csv")

def get_train_data():
  data = pd.read_feather("../data/train.feather")
  for i in remove_list:
    del data[i]
  return data

def get_merge_train_data():
  data = get_train_data()
  col_names = list(data.columns)
  del col_names[0]
  
  i = 0
  for col in col_names:
    data[col].fillna(col_avg[i])
    i += 1

  return data.groupby(['customer_ID']).mean()

  
def get_column_avg():
  avg = ['hello']
  data = get_train_data()
  data = data.groupby(['customer_ID']).mean()
  print(data)
  for col in data.columns:
    print(col)
    total = 0
    count = 0
    for e in data[col]:
      if (col == "customer_ID"):
        break
      if (np.isnan(e) == False):
        total += e
        count += 1
    avg.append(round(total/ count, 5))
  
  print(avg)
  return
# * ========================================================================================

# * ========================================================================================
# .head(x) can display x number of rows in the dataframe
# get_train_data().head(100)
# * ========================================================================================
