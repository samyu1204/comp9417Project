import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import cleaning_helper
import df
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# cov_list = ['customer_ID', 'D_58', 'B_37', 'B_9', 'D_74', 'B_3', 'D_55', 'B_22', 'B_33', 'R_1', 'B_38', 'D_75', 'D_44', 'D_48', 'B_2', 'B_7', 'B_18', 'B_30', 'B_1', 'D_61', 'B_4', 'P_2', 'R_10', 'B_17', 'B_23']
cov_list = ['B_30', 'B_1', 'B_23', 'B_7', 'D_55', 'B_37', 'B_3', 'D_58', 'D_61', 'B_4', 'B_22', 'D_44', 'D_75', 'R_1', 'B_9', 'B_38', 'D_48', 'R_10', 'D_74']

def data_preprocessing():
  data_frame = df.get_merge_train_data()
  data = data_frame[cov_list]
  return data

def test_data_process():
  data_frame = df.get_test_data().groupby(['customer_ID']).mean()
  data = data_frame[cov_list]
  return data

X = data_preprocessing()

Y = df.get_train_label()['target'].to_numpy()

# # Using the training model to fit:
# model = LogisticRegression(solver='liblinear', random_state=1).fit(X, Y)
# print(model.predict_proba(test_data_process()))

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=23)

def XGBoost_model():
  dtrain = xgb.DMatrix(x_train, label=y_train)
  dtest = xgb.DMatrix(x_test, label=y_test)

  # Paramter spec
  param = {'max_depth': 6, 'eta': 1, 'objective': 'binary:logistic'}
  param['nthread'] = 4
  param['eval_metric'] = 'auc'

  evallist = [(dtest, 'eval'), (dtrain, 'train')]
  num_round = 10
  bst = xgb.train(param, dtrain, num_round, evallist)

  ypred = bst.predict(xgb.DMatrix(test_data_process()), iteration_range=(0, bst.best_iteration + 1))
  print(ypred)
  pd.DataFrame(pd.DataFrame(ypred).to_csv(r'solution.csv', index = False))

XGBoost_model()


# LOGISTIC REGRESSION MODEL
def logistic_model():
  lr_model = LogisticRegression(solver='liblinear', max_iter=200).fit(x_train, y_train)
  y_pred = lr_model.predict(x_test)

  print(confusion_matrix(y_test, y_pred))

  # pd.DataFrame(model.predict_proba(test_data_process())).to_csv(r'solution.csv', index = False)


