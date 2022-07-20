import pandas as pd
import df
import numpy as np
import customised_df as c_df
import matplotlib.pyplot as plt
import cleaning_helper
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Returns 5531451 customers here in the training set:
# print(len(df['customer_ID']))

# print(c_df.df_sig_columns())

# c_df.df_sig_columns().groupby(['customer_ID'])

# * ========================================================================================

def info():
  # Column names:
  # ['customer_ID', 'S_2', 'S_3', 'S_5', 'S_6', 'S_7', 'S_8', 'S_9', 'S_11',
  #   'S_12', 'S_13', 'S_15', 'S_16', 'S_17', 'S_18', 'S_19', 'S_20', 'S_22',
  #   'S_23', 'S_24', 'S_25', 'S_26', 'S_27']
  # customer_ID     object
  # S_2             object - date
  # S_3            float32 
  # S_5            float32
  # S_6               int8
  # S_7            float32
  # S_8              int16
  # S_9            float32
  # S_11              int8
  # S_12           float32
  # S_13             int16
  # S_15              int8
  # S_16           float32
  # S_17           float32
  # S_18              int8
  # S_19           float32
  # S_20              int8
  # S_22           float32
  # S_23           float32
  # S_24           float32
  # S_25           float32
  # S_26           float32
  # S_27           float32
  return 

# Returns the scaled data:
def data_preprocessing():
  data_frame = df.get_par_training_data()
  # Data analysis for column "s":
  data_S = cleaning_helper.select_by_letter('S', data_frame)
  # Fill all NA in the dataframe with 0's
  data_S = data_S.fillna(0)

  # Checking rows with NA:
  # cleaning_helper.check_na_by_column(data_S)

  # Group by customer_id and take the average of the columns:
  data_S_avg = data_S.groupby(['customer_ID']).mean()

  # Standardise each column
  scalar = StandardScaler().fit(data_S_avg)
  data_S_avg = scalar.transform(data_S_avg)

  scaled_df = pd.DataFrame(data_S_avg, columns= ['S_3', 'S_5', 'S_6', 'S_7', 'S_8', 'S_9', 'S_11',
    'S_12', 'S_13', 'S_15', 'S_16', 'S_17', 'S_18', 'S_19', 'S_20', 'S_22', 'S_23', 'S_24', 'S_25', 'S_26', 'S_27'])

  return scaled_df.assign(default = df.get_train_label()['target'].tolist())

# Look at all correlation between every variable and return the highest ones
def get_corr_df():
  highly_correlated = []
  data_frame = df.get_par_training_data()
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

print(get_default_var())


# Examplar model:
# This model will focus on the columns: S_ 3, 7, 25
def trial_model():
  result = np.zeros(924621)
  customer = []
  test = df.get_test_data().groupby(['customer_ID']).mean()
  test = test.assign(prediction = result)

  # df_avg = test.groupby(['customer_ID']).mean()

  # for row in df_avg.iterrows():
  #   print(row)


  solution = test[['prediction']].copy().reset_index()
  print(solution)

  solution.to_csv(r'solution.csv', index = False)
  return






