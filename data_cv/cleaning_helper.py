import pandas as pd
import numpy as np

def select_by_letter(letter, df):
  """Filter data frame by the first letter of the column.

  Args:
      letter (character): letter you want to filter by

  Returns:
      dataframe: filtered dataframe where the returned columns correspond to the letter filtered
  """
  get_col = ['customer_ID']
  for name in df:
    if name[0] == letter.upper():
      get_col.append(name)

  return df[get_col]

def check_na_by_column(df):
  """Checks each column of dataframe for NA and prints a summary:

  Args:
      df (object): data frame (pandas)
  """
  for i in df.columns:
    print(f"Column: {i} and null: {df[i].isnull().any()}")
    
  return

  # Checking for NA in the data:
def data_check_NA(data):
  """Checks for NA in each column and if a column exceeds over 60% NA
      it will be returned in the list.

  Args:
      data (dataframe): pandas data frame

  Returns:
      list: list of all high NA percentage columns
  """
  to_remove = []
  no_rows = 5531451
  for column in data:
    count = 0
    for value in data[column]:
      if pd.isnull(value):
        count += 1
    
    if (count / no_rows) > 0.6:
      to_remove.append(column)
    
    print(to_remove)
  return to_remove

