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

# Checking for NA in the data:
def data_check_NA(data):
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

  # print(data_S.columns)
  # print(data_S.dtypes)
  # print(str(data_frame.iloc[1]))
  # print(str(data_frame.iloc[1][0]) == "0000099d6bd597052cdcda90ffabf56573fe9d7c79be5fbac11a8ed792feb62a")

  # print(data_S['S_27'].isnull().any())
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

df = data_preprocessing()
print(df)
# Correlation study:
# print(scaled_df.corr())

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

# print(trial_model())


# print(df.get_test_data().head())







