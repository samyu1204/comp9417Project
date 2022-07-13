# Using dask here, to work with such a big dataset:
# pip isntall dask!
import dask.dataframe as dd
import pyarrow.parquet as pq
import datatable as dt
# Documentations:https://docs.dask.org/en/stable/dataframe.html

import pandas as pd
# pip install pyarrow

# * ========================================================================================
# Data getter functions:
def get_sample_sub():
  return dd.read_csv("../data/sample_submission.csv")

def get_test_data():
  return pd.read_parquet("../data/test.parquet")

def get_train_data():
  return pd.read_csv("../data/train_data.csv")

# Train labels show if the person defaults or not:
def get_train_label():
  return pd.read_csv("../data/train_labels.csv")

def get_par_training_data():
  # return dt.fread("../data/train.parquet")
  return pd.read_feather("../data/train.feather")
  # return pd.read_parquet("../data/train.parquet")


# * ========================================================================================

# * ========================================================================================
# .head(x) can display x number of rows in the dataframe
# get_train_data().head(100)
# * ========================================================================================

print(get_par_training_data())