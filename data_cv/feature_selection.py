import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
import df

# Using Lasso Regression to perform feature selectio
def select_features():
  print(df.get_merge_train_data())
  
  return

select_features()