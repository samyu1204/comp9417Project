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