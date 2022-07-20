import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import cleaning_helper
import df
import pandas as pd
from sklearn.preprocessing import StandardScaler

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

def test_data_process():
  data_frame = df.get_test_data()
  data_S = cleaning_helper.select_by_letter('S', data_frame)
  data_S = data_S.fillna(0)
  data_S_avg = data_S.groupby(['customer_ID']).mean()

  # Standardise each column
  scalar = StandardScaler().fit(data_S_avg)
  data_S_avg = scalar.transform(data_S_avg)

  scaled_df = pd.DataFrame(data_S_avg, columns= ['S_3', 'S_5', 'S_6', 'S_7', 'S_8', 'S_9', 'S_11',
    'S_12', 'S_13', 'S_15', 'S_16', 'S_17', 'S_18', 'S_19', 'S_20', 'S_22', 'S_23', 'S_24', 'S_25', 'S_26', 'S_27'])

  return scaled_df[['S_3', 'S_7', 'S_25']].copy()

fit_data = data_preprocessing()

X = fit_data[['S_3', 'S_7', 'S_25']].copy()
Y = fit_data['default'].copy().to_numpy()
# Using the training model to fit:
model = LogisticRegression(solver='liblinear', random_state=0).fit(X, Y)

print(model.predict_proba(test_data_process()))
pd.DataFrame(model.predict_proba(test_data_process())).to_csv(r'solution.csv', index = False)


