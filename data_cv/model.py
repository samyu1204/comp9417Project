import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import cleaning_helper
import df
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb 
from xgboost import XGBClassifier
import feature_selection
from sklearn import metrics
import matplotlib.pyplot as plt

# cov_list = ['customer_ID', 'D_58', 'B_37', 'B_9', 'D_74', 'B_3', 'D_55', 'B_22', 'B_33', 'R_1', 'B_38', 'D_75', 'D_44', 'D_48', 'B_2', 'B_7', 'B_18', 'B_30', 'B_1', 'D_61', 'B_4', 'P_2', 'R_10', 'B_17', 'B_23']

# Based on correlation
# cov_list = ['B_30', 'B_1', 'B_23', 'B_7', 'D_55', 'B_37', 'B_3', 'D_58', 'D_61', 'B_4', 'B_22', 'D_44', 'D_75', 'R_1', 'B_9', 'B_38', 'D_48', 'R_10', 'D_74'] 
corr_list = ['B_30', 'B_1', 'B_23', 'B_7', 'D_55', 'B_37', 'B_3', 'D_58', 'D_61', 'B_4', 'B_22', 'D_44', 'D_75', 'R_1', 'B_9', 'B_38', 'D_48', 'R_10', 'D_74'] 

# Based on p-value logit selection
# cov_list = feature_selection.logit_selection()
logit_list = feature_selection.logit_selection()

# Based on lasso feature selection
# cov_list = feature_selection.get_sig_list()
lasso_list = feature_selection.get_sig_list()


def data_preprocessing():
  data_frame = df.get_sample_train_data()
  data = data_frame[corr_list]
  return data

def data_preprocessing_list(l):
  data_frame = df.get_sample_train_data()
  data = data_frame[l]
  return data

def test_data_process():
  data_frame = df.get_test_data().groupby(['customer_ID']).mean()
  data = data_frame[cov_list]
  return data

X = data_preprocessing()

Y = df.get_sample_train_data()['target'].to_numpy()

# # Using the training model to fit:
# model = LogisticRegression(solver='liblinear', random_state=1).fit(X, Y)
# print(model.predict_proba(test_data_process()))

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=1)

# Handling the categorial
def XGBoost_model():
  dtrain = xgb.DMatrix(x_train, label=y_train)
  dtest = xgb.DMatrix(x_test, label=y_test)

  # Paramter spec
  param = {'max_depth': 3, 'eta': 1, 'objective': 'binary:logistic'}
  param['nthread'] = 4
  param['eval_metric'] = 'auc'

  evallist = [(dtest, 'eval'), (dtrain, 'train')]
  num_round = 10
  bst = xgb.train(param, dtrain, num_round, evallist)

  ypred = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
  return ypred
  # pd.DataFrame(pd.DataFrame(ypred).to_csv(r'solution.csv', index = False))
  # print(confusion_matrix(y_test, ypred))
  # print("Accuracy: %.3f" % accuracy_score(y_test, ypred))

def XGBoost_model_classifier():
  model = XGBClassifier()
  model.fit(x_train, y_train)
  y_pred = model.predict(x_test)

  print(confusion_matrix(y_test, y_pred))
  # roc 
  y_pred_proba = XGBoost_model()
  fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
  print("Accuracy: %.3f" % accuracy_score(y_test, y_pred))
  # #create ROC curve
  plt.plot(fpr,tpr)
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.show()

# XGBoost_model()
# XGBoost_model_classifier()

# LOGISTIC REGRESSION MODEL
def logistic_model():
  lr_model = LogisticRegression(solver='liblinear', max_iter=200).fit(x_train, y_train)
  y_pred = lr_model.predict(x_test)

  print(confusion_matrix(y_test, y_pred))
  print(y_pred)
  print("Accuracy: %.3f" % accuracy_score(y_test, y_pred))

  # roc 
  # y_pred_proba = lr_model.predict_proba(x_test)[::,1]
  # fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
  # #create ROC curve
  # plt.plot(fpr,tpr)
  # plt.ylabel('True Positive Rate')
  # plt.xlabel('False Positive Rate')
  # plt.show()

def ROC_curve_logistic_model():
  # Data preprocessing ofr different lists:
  X_corr = data_preprocessing_list(corr_list)
  X_logit = data_preprocessing_list(logit_list)
  X_lasso = data_preprocessing_list(lasso_list)

  Y = df.get_sample_train_data()['target'].to_numpy()

  x_corr_train, x_corr_test, y_train, y_test = train_test_split(X_corr, Y, random_state=1)
  X_logit_train, X_logit_test, _, _ = train_test_split(X_logit, Y, random_state=1)
  X_lasso_train, X_lasso_test, _, _ = train_test_split(X_lasso, Y, random_state=1)

  lr_model_corr = LogisticRegression(solver='liblinear', max_iter=200).fit(x_corr_train, y_train)
  lr_model_logit = LogisticRegression(solver='liblinear', max_iter=200).fit(X_logit_train, y_train)
  lr_model_lasso = LogisticRegression(solver='liblinear', max_iter=200).fit(X_lasso_train, y_train)

  # ROC Curve:
  y_pred_proba_corr = lr_model_corr.predict_proba(x_corr_test)[::,1]
  y_pred_proba_logit = lr_model_logit.predict_proba(X_logit_test)[::,1]
  y_pred_proba_lasso = lr_model_lasso.predict_proba(X_lasso_test)[::,1]
  fpr_corr, tpr_corr, _ = metrics.roc_curve(y_test,  y_pred_proba_corr)
  fpr_logit, tpr_logit, _ = metrics.roc_curve(y_test,  y_pred_proba_logit)
  fpr_lasso, tpr_lasso, _ = metrics.roc_curve(y_test,  y_pred_proba_lasso)

  #create ROC curve
  plt.plot(fpr_corr,tpr_corr, label='Correlation Features')
  plt.plot(fpr_logit,tpr_logit, label='Logit P-value Features')
  plt.plot(fpr_lasso,tpr_lasso, label='Lasso Features')
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  leg = plt.legend()
  plt.show()
  return

def ROC_curve_xgb_model():
  # Data preprocessing ofr different lists:
  X_corr = data_preprocessing_list(corr_list)
  X_logit = data_preprocessing_list(logit_list)
  X_lasso = data_preprocessing_list(lasso_list)

  Y = df.get_sample_train_data()['target'].to_numpy()

  x_corr_train, x_corr_test, y_train, y_test = train_test_split(X_corr, Y, random_state=1)
  X_logit_train, X_logit_test, _, _ = train_test_split(X_logit, Y, random_state=1)
  X_lasso_train, X_lasso_test, _, _ = train_test_split(X_lasso, Y, random_state=1)

  xgb_model_corr = XGBClassifier().fit(x_corr_train, y_train)
  xgb_model_logit = XGBClassifier().fit(X_logit_train, y_train)
  xgb_model_lasso = XGBClassifier().fit(X_lasso_train, y_train)

  # ROC Curve:
  y_pred_proba_corr = xgb_model_corr.predict_proba(x_corr_test)[::,1]
  y_pred_proba_logit = xgb_model_logit.predict_proba(X_logit_test)[::,1]
  y_pred_proba_lasso = xgb_model_lasso.predict_proba(X_lasso_test)[::,1]
  fpr_corr, tpr_corr, _ = metrics.roc_curve(y_test,  y_pred_proba_corr)
  fpr_logit, tpr_logit, _ = metrics.roc_curve(y_test,  y_pred_proba_logit)
  fpr_lasso, tpr_lasso, _ = metrics.roc_curve(y_test,  y_pred_proba_lasso)

  # Create ROC curve
  plt.plot(fpr_corr,tpr_corr, label='Correlation Features')
  plt.plot(fpr_logit,tpr_logit, label='Logit P-value Features')
  plt.plot(fpr_lasso,tpr_lasso, label='Lasso Features')
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  leg = plt.legend()
  plt.show()
  return

def ROC_model_comparison():
  # Data preprocessing ofr different lists:
  X_logit = data_preprocessing_list(logit_list)

  Y = df.get_sample_train_data()['target'].to_numpy()

  X_logit_train, X_logit_test, _, _ = train_test_split(X_logit, Y, random_state=1)

  xgb_model_logit = XGBClassifier().fit(X_logit_train, y_train)
  lr_model_logit = LogisticRegression(solver='liblinear', max_iter=200).fit(X_logit_train, y_train)

  # Prediction values:
  y_pred_proba_logit = xgb_model_logit.predict_proba(X_logit_test)[::,1]
  fpr_logit, tpr_logit, _ = metrics.roc_curve(y_test,  y_pred_proba_logit)

  y_pred_proba_logit = lr_model_logit.predict_proba(X_logit_test)[::,1]
  l_fpr_logit, l_tpr_logit, _ = metrics.roc_curve(y_test,  y_pred_proba_logit)

  # Create ROC curve
  plt.plot(fpr_logit,tpr_logit, label='XGBoost Logit Feature Selection')
  plt.plot(l_fpr_logit,l_tpr_logit, label='Logistic Logit Feature Selection')
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  leg = plt.legend()
  plt.show()
  return

ROC_model_comparison()


# pd.DataFrame(model.predict_proba(test_data_process())).to_csv(r'solution.csv', index = False)


