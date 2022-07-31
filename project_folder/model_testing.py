from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict, train_test_split
import df
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

# Based on correlation
features = ['B_30', 'B_1', 'B_23', 'B_7', 'D_55', 'B_37', 'B_3', 'D_58', 'D_61', 'B_4', 'B_22', 'D_44', 'D_75', 'R_1', 'B_9', 'B_38', 'D_48', 'R_10', 'D_74'] 

# Based on p-value logit selection
#features = feature_selection.logit_selection()

# Based on lasso feature selection
#features = feature_selection.get_sig_list()


def data_preprocessing():
    data_frame = df.get_sample_train_data()
    data = data_frame[features]
    return data

X = data_preprocessing()

y = df.get_sample_train_data()['target'].to_numpy()

kf = KFold(n_splits=5, shuffle = True, random_state=5)

# Using 5 five fold cross validation for logistic regression
# prints the confusion matrix
def test_logistic_regression():
    print("logistic regression")
    total = 0
    for train_index, test_index in kf.split(y):
        X_train , X_test = X.iloc[train_index,:] , X.iloc[test_index,:]
        y_train , y_test = y[train_index] , y[test_index]       

        lr_model = LogisticRegression(solver='liblinear', max_iter=300).fit(X_train, y_train)
        
        y_pred = lr_model.predict(X_test)
        total += confusion_matrix(y_test, y_pred)
        print(confusion_matrix(y_test, y_pred))
    print(total/5)

# Using 5 five fold cross validation for logistic regression
# prints the confusion matrix 
def test_XGBoost_model():
    total = 0
    print("xgboost")
    for train_index, test_index in kf.split(y):
        X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
        y_train , y_test = y[train_index] , y[test_index]       
        #print(y_train.mean())
        model = XGBClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(confusion_matrix(y_test, y_pred))
        total += confusion_matrix(y_test, y_pred)
        print(confusion_matrix(y_test, y_pred))
    print(total)
    

test_logistic_regression()
test_XGBoost_model()