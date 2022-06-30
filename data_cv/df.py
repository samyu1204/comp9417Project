# Using dask here, to work with such a big dataset:
# pip isntall dask!
import dask.dataframe as dd
# Documentations:https://docs.dask.org/en/stable/dataframe.html


# * ========================================================================================
# Data getter functions:
def get_sample_sub():
  return dd.read_csv("../data/sample_submission.csv")

def get_test_data():
  return dd.read_csv("../data/test_data.csv")

def get_train_data():
  return dd.read_csv("../data/train_data.csv")

def get_train_label():
  return dd.read_csv("../data/train_labels.csv")
# * ========================================================================================

# * ========================================================================================
# .head(x) can display x number of rows in the dataframe
# get_train_data().head(100)
# * ========================================================================================