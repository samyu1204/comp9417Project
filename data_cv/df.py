# Using dask here, to work with such a big dataset:
# pip isntall dask!
import dask.dataframe as dd
import pyarrow.parquet as pq
import datatable as dt
# Documentations:https://docs.dask.org/en/stable/dataframe.html

import pandas as pd
# pip install pyarrow
# List of columns that has more than 50% missing values:
remove_list = ['S_2', 'D_42', 'D_49', 'D_50', 'D_53', 'D_56', 'S_9', 'B_17', 'D_73', 'D_76', 'R_9', 'D_82', 'B_29', 'D_87', 'D_88', 'D_105', 'D_106', 'R_26', 'D_108', 'D_110', 'D_111', 'B_39', 'B_42', 'D_132', 'D_134', 'D_135', 'D_136', 'D_137', 'D_138', 'D_142']
# * ========================================================================================
# Data getter functions:
def get_sample_sub():
  return dd.read_csv("../data/sample_submission.csv")

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

def get_merge_train():
  data = get_train_data()
  for col in data.columns:
    data = data.fillna(value=data[col].mean(), inplace=True)
    print(col)
  return data
  
# * ========================================================================================

# * ========================================================================================
# .head(x) can display x number of rows in the dataframe
# get_train_data().head(100)
# * ========================================================================================
