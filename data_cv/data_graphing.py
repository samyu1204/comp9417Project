import pandas as pd
#from data_cv.cleaning_helper import select_by_letter
import df
import numpy as np
import customised_df as c_df
import matplotlib.pyplot as plt
import cleaning_helper
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Look at all correlation between every variable and return the highest ones
def get_corr_df():
  highly_correlated = []
  data_frame = df.get_train_data()
  data_frame = data_frame.groupby(['customer_ID']).mean()
  # Insert the default column:
  data_frame = data_frame.assign(default = df.get_train_label()['target'].tolist())
  for col in ['B_30', 'B_1', 'B_23', 'B_7', 'D_55', 'B_37', 'B_3', 'D_58', 'D_61', 'B_4', 'B_22', 'D_44', 'D_75', 'R_1', 'B_9', 'B_38', 'D_48', 'R_10', 'D_74']:
    sns.boxplot(y = data_frame[col], x = data_frame['default'], showfliers = False).set_title('Boxplot for ' + str(col) + ' grouped by default value')
    plt.show()
  plt.clf()
  for letter in ['D', 'S', 'P', 'B', 'R']:
    data_frame[letter] = data_frame[[col for col in data_frame.columns if col[0] == letter]].mean(axis=1)
  data_frame = data_frame[['D', 'S', 'P', 'B', 'R', 'default']]
  for letter in ['D', 'S', 'P', 'B', 'R']:
    sns.boxplot(y = data_frame[letter], x = data_frame['default'], showfliers = False).set_title('Boxplot for category ' + str(letter) + ' grouped by default value')
    plt.show()
  return

# Gets correlation of these variables against each other
def corr_variables():
  corr_df = pd.read_csv("generated_df\correlation_map.csv")
  corr_df = corr_df[corr_df['Unnamed: 0'].isin(['B_30', 'B_1', 'B_23', 'B_7', 'D_55', 'B_37', 'B_3', 'D_58', 'D_61', 'B_4', 'B_22', 'D_44', 'D_75', 'R_1', 'B_9', 'B_38', 'D_48', 'R_10', 'D_74'])]
  corr_df.set_index('Unnamed: 0', inplace=True)
  corr_df = corr_df[['B_30', 'B_1', 'B_23', 'B_7', 'D_55', 'B_37', 'B_3', 'D_58', 'D_61', 'B_4', 'B_22', 'D_44', 'D_75', 'R_1', 'B_9', 'B_38', 'D_48', 'R_10', 'D_74']]
  corr_df = corr_df.reindex(sorted(corr_df.columns), axis = 1)
  corr_df = corr_df.sort_index()

  sns.heatmap(corr_df)
  plt.show()

# Gets correlation of ['B_30', 'B_1', 'B_23', 'B_7', 'D_55', 'B_37', 'B_3', 'D_58', 'D_61', 'B_4', 'B_22', 'D_44', 'D_75', 'R_1', 'B_9', 'B_38', 'D_48', 'R_10', 'D_74'] these variables
# against the default value
def correlation_of_selected_variables():
  corr_df = pd.read_csv("generated_df/correlation_map.csv")
  corr_df.set_index('Unnamed: 0', inplace=True)
  corr_df = corr_df.sort_values(by='default', key = abs)
  corr_df = corr_df['default']
  corr_df = corr_df.loc[['B_30', 'B_1', 'B_23', 'B_7', 'D_55', 'B_37', 'B_3', 'D_58', 'D_61', 'B_4', 'B_22', 'D_44', 'D_75', 'R_1', 'B_9', 'B_38', 'D_48', 'R_10', 'D_74']]
  print(corr_df)

# Correlation of each category against default
def correlation_by_category():
    corr_df = pd.read_csv("generated_df/correlation_map.csv")
    corr_df['category'] = corr_df['Unnamed: 0'].str[0]
    corr_df = corr_df.groupby(by='category').mean()
    corr_df = corr_df['default']
    print(corr_df)
