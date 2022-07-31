from operator import index
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, RepeatedKFold, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import df
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector
import statsmodels.api as sm
from sklearn.pipeline import Pipeline

# cov_list = ['B_30', 'B_1', 'B_23', 'B_7', 'D_55', 'B_37', 'B_3', 'D_58', 'D_61', 'B_4', 'B_22', 'D_44', 'D_75', 'R_1', 'B_9', 'B_38', 'D_48', 'R_10', 'D_74']
# Using Lasso Regression to perform feature selectio
def select_features():
  data = df.get_sample_train_data()
  X, Y = data.drop('target',axis=1), data['target']

  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

  param = {
    'alpha':[.00001, 0.0001,0.001, 0.01],
    'fit_intercept':[True,False],
    'normalize':[True,False],
    'positive':[True,False],
    'selection':['cyclic','random'],
    }

  model = Lasso()
  cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

  search = GridSearchCV(model, param, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv)
  result = search.fit(X, Y)

  print('Best Score: %s' % result.best_score_)
  print('Best Hyperparameters: %s' % result.best_params_)

  #Fitting the model
  # model.fit(X_train,y_train)
  # y_pred = model.predict(X_test)
  # print(y_pred)
  # #Model Predictions
  # print('='*30)
  # print(r2_score(y_test,y_pred))
  # print(model.coef_)

def lasso(alphas):
  data = df.get_sample_train_data()
  X, y = data.drop('target',axis=1), data['target']
  # Create an empty data frame
  frame = pd.DataFrame()
  
  # Create a column of feature names
  col_names = list(data.columns)
  del col_names[-1]

  frame = frame.assign(feature_name = col_names)

  # For each alpha value in the list of alpha values,
  for alpha in alphas:
  # Create a lasso regression with that alpha value,
      lasso = Lasso(alpha=alpha)
      
      # Fit the lasso regression
      lasso.fit(X, y)
      
      # Create a column name for that alpha value
      column_name = 'Alpha = %f' % alpha

      # Create a column of coefficient values
      frame[column_name] = lasso.coef_
      
  # Return the dataframe    
  return frame

def select_lasso():
  pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Lasso())
  ])
  data = df.get_sample_train_data()
  X, y = data.drop('target',axis=1), data['target']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

  search = GridSearchCV(pipeline,
    {'model__alpha':np.arange(0.0001, 1, 0.01)},
    cv = 5,
    scoring = 'neg_mean_squared_error',
    verbose = 3
  )
  print(search.fit(X_train, y_train))
  print(search.best_params_)

# select_lasso()

# print(lasso([0.5, 0.1, 0.001, 0.0001]))

# lasso([0.5, 0.1, 0.01, 0.001, 0.0001]).to_csv(r'solution.csv', index = False)
# print(lasso([0.01])['Alpha = 0.010000'])

# Retrieves the important features
def get_sig_list():
  data = df.get_sample_train_data()
  col_names = list(data.columns)
  del col_names[-1]

  j = 0
  index_list = []
  for i in lasso([0.01])['Alpha = 0.010000']:
    if i > 0 or i < 0:
      index_list.append(j)
    j += 1
  

  cov_list = []
  for i in index_list:
    cov_list.append(col_names[i])
  
  return cov_list

# select_features()
def logit_selection():
  data = df.get_sample_train_data()
  X, Y = data.drop('target',axis=1), data['target']
  logit_model=sm.Logit(Y, X)
  result=logit_model.fit()
  cov_list = []
  table = result.summary2().tables[1]
  row_names = table.index.values
  
  counter = 0
  for i in result.summary2().tables[1]['P>|z|']:
    if i < 0.05:
      cov_list.append(row_names[counter])
    counter += 1
  return cov_list
