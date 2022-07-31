import pandas as pd
import df
import numpy as np
import matplotlib.pyplot as plt
import cleaning_helper
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Returns 5531451 customers here in the training set:
# * ========================================================================================
# Look at all correlation between every variable and return the highest ones
def get_corr_df():
  highly_correlated = []
  data_frame = df.get_train_data()
  data_frame = data_frame.groupby(['customer_ID']).mean()
  # Insert the default column:
  data_frame = data_frame.assign(default = df.get_train_label()['target'].tolist())
  corr_df = data_frame.corr()
  return

# Gives all pair of variables that are strongly correlated:
def corr_analysis():
  corr_df = pd.read_csv("../generated_df/correlation_map.csv")
  # corr_df = corr_df.rename(columns=corr_df.iloc[0])
  corr_df.set_index('Unnamed: 0', inplace=True)
  names = corr_df.axes[0].tolist()
  sig_list = []

  for i in names:
    index = 0
    for j in corr_df[i]:
      if j > 0.4 or j < -0.4 :
        sig_list.append([i, names[index]])
      index += 1
  
  for i in sig_list:
    if i[0] == i[1]:
      sig_list.remove(i)
  return sig_list

def get_default_var():
  corr_list = corr_analysis()
  var = []
  for e in corr_list:
    if e[0] == 'default' and e[1] != 'default':
      var.append(e[1])
    elif e[1] == 'default' and e[0] != 'default':
      var.append(e[0])
  
  # Remove duplicates
  var = list(set(var)) 
  return var

