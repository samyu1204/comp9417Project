import pandas as pd
import df

def df_sig_columns():
  return df.get_train_data().drop(['D_42', 'D_53', 'D_73', 'D_76', 'B_29', 'D_88', 'D_110', 'B_39', 'B_42', 'D_132', 'D_134', 'D_142'], axis=1)

