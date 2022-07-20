import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import cleaning_helper
import df
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ['D_58', 'B_37', 'B_9', 'D_74', 'B_3', 'D_55', 'B_22', 'B_33', 'R_1', 'B_38', 'D_75', 'D_44', 'D_48', 'B_2', 'B_7', 'B_18', 'B_30', 'B_1', 'D_61', 'B_4', 'P_2', 'R_10', 'B_17', 'B_23']
def data_preprocessing():
  data_frame = df.get_par_training_data()
  # Data analysis for column "s":
  data = data_frame[['customer_ID', 'D_58', 'B_37', 'B_9', 'D_74', 'B_3', 'D_55', 'B_22', 'B_33', 'R_1', 'B_38', 'D_75', 'D_44', 'D_48', 'B_2', 'B_7', 'B_18', 'B_30', 'B_1', 'D_61', 'B_4', 'P_2', 'R_10', 'B_17', 'B_23']]

  # Fill all NA in the dataframe with 0's
  data = data.fillna(0)

  # Checking rows with NA:
  # cleaning_helper.check_na_by_column(data_S)

  # Group by customer_id and take the average of the columns:
  return data.groupby(['customer_ID']).mean()


def test_data_process():
  data_frame = df.get_test_data()
  data = data_frame[['customer_ID', 'D_58', 'B_37', 'B_9', 'D_74', 'B_3', 'D_55', 'B_22', 'B_33', 'R_1', 'B_38', 'D_75', 'D_44', 'D_48', 'B_2', 'B_7', 'B_18', 'B_30', 'B_1', 'D_61', 'B_4', 'P_2', 'R_10', 'B_17', 'B_23']]
  data = data.fillna(0)
  data_avg = data.groupby(['customer_ID']).mean()

  return data_avg

X = data_preprocessing()

Y = df.get_train_label()['target'].to_numpy()
# Using the training model to fit:
model = LogisticRegression(solver='liblinear', random_state=1).fit(X, Y)

print(model.predict_proba(test_data_process()))


# pd.DataFrame(model.predict_proba(test_data_process())).to_csv(r'solution.csv', index = False)


