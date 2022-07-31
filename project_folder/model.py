import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
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


# * ========================================================================================
# Lists for feature selections:
# Based on correlation
print('Initialising all feature lists. Please wait...')
corr_list = ['B_30', 'B_1', 'B_23', 'B_7', 'D_55', 'B_37', 'B_3', 'D_58', 'D_61', 'B_4', 'B_22', 'D_44', 'D_75', 'R_1', 'B_9', 'B_38', 'D_48', 'R_10', 'D_74'] 

# Based on p-value logit selection
# cov_list = feature_selection.logit_selection()
logit_list = feature_selection.logit_selection()

# Based on lasso feature selection
# cov_list = feature_selection.get_sig_list()
lasso_list = feature_selection.get_sig_list()
# This function is for selection of list:
def welcome():
  val = input('Please select a feature list by typing in a letter of the following: a) correlation b) logit c) lasso: ')
  if val == 'a':
    return corr_list
  elif val == 'b':
    return logit_list
  else:
    return lasso_list
cov_list = welcome()

# Training data preprocessing:
def data_preprocessing():
  print('Processing data...')
  data_frame = df.get_sample_train_data()
  data = data_frame[cov_list]
  return data

def data_preprocessing_list(l):
  data_frame = df.get_sample_train_data()
  data = data_frame[l]
  return data

# Test data preprocessing:
def test_data_process():
  print('Processing test data...')
  data_frame = df.get_test_data().groupby(['customer_ID']).mean()
  data = data_frame[cov_list]
  return data

# Training and Test data sets:
X = data_preprocessing()
Y = df.get_sample_train_data()['target'].to_numpy()
# Splitting:
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=1)

# Handling the categorial
def XGBoost_model():
  print("Running the XGBoost Model...")
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
  print(f'Predicted values: {ypred}')
  return ypred


def XGBoost_model_classifier():
  print("Running the XGBoost classifier Model...")
  model = XGBClassifier()
  model.fit(x_train, y_train)
  y_pred = model.predict(x_test)

  print(f'The confusion matrix is: {confusion_matrix(y_test, y_pred)}')
  # roc 
  y_pred_proba = XGBoost_model()
  fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
  print("Accuracy: %.3f" % accuracy_score(y_test, y_pred))


# * ========================================================================================
# LOGISTIC REGRESSION MODEL
def logistic_model():
  print("Running the logistic model...")
  lr_model = LogisticRegression(solver='liblinear', max_iter=200).fit(x_train, y_train)
  y_pred = lr_model.predict(x_test)

  print(f'The confusion matrix is: {confusion_matrix(y_test, y_pred)}')
  print("Accuracy: %.3f" % accuracy_score(y_test, y_pred))


# * ========================================================================================
# ROC plots:
def ROC_curve_logistic_model():
  print("Drawing the ROC curve...")
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
  print("Drawing the ROC curve...")
  # Data preprocessing for different lists:
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
  print("Drawing the ROC curve...")
  # Data preprocessing for different lists:
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

# * ========================================================================================
# To run the ROC models: uncomment one of these:

# For writing results:
# pd.DataFrame(model.predict_proba(test_data_process())).to_csv(r'solution.csv', index = False)


# * ========================================================================================
# Model running:
# Comment models you want to run:
# XGBoost_model()
# XGBoost_model_classifier()
# logistic_model()

# Running ROC Curves:
# ROC_curve_logistic_model()
# ROC_curve_xgb_model()
# ROC_model_comparison()
