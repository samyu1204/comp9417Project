import pandas as pd
import df
import numpy as np
import customised_df as c_df
import matplotlib.pyplot as plt
import cleaning_helper

df = df.get_par_training_data()

# Returns 5531451 customers here in the training set:
# print(len(df['customer_ID']))

# Checking for NA in the data:
'''
to_remove = []
no_rows = 5531451
for column in df:
  count = 0
  for value in df[column]:
    if pd.isnull(value):
      count += 1
  
  if (count / no_rows) > 0.6:
    to_remove.append(column)
  
  print(to_remove)
'''

# print(c_df.df_sig_columns())

# c_df.df_sig_columns().groupby(['customer_ID'])

print(cleaning_helper.select_by_letter('S', df))

# for x in df['D_66']:
#   if x == 2.0:
#     print(x)


