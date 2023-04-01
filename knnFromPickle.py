## Author: Patrick Tibbals

import numpy as np
import joblib
from sklearn import metrics
from sklearn.feature_extraction.text import HashingVectorizer


def buckets(number):

    if(number==1):
        return "xx-24"
    elif(number==2):
        return "25-34"
    elif(number==3):
        return "34-49"
    else:
        return "50-xx"

dataset2 = joblib.load('knnpickle_TrainData.pkl')
users = joblib.load('knnpickle_Users.pkl')



n=8000
train_Ids = users[0:n]
data_train = dataset2.loc[users]


test_Ids = users[n:]
data_test = dataset2.loc[test_Ids]

hash_vect = HashingVectorizer()

X_train = hash_vect.fit_transform(data_train['likes'])
X_test = hash_vect.transform(data_test['likes'])


y_train_gender = data_train['gender']
knn_Gender = joblib.load('knnpickle_Gender.pkl')



y_train_age = data_train['age'].astype('int')
print(y_train_age)

knn_age = joblib.load('knnpickle_Age.pkl')

print("Fitted")


#######################################################
#######################################################
print("=================================")

y_train_ope = data_train['ope'].astype(float)

svr_ope = joblib.load('svr_Ope.pkl')

print("Ope Fitted")
X_test_ope = hash_vect.transform(data_test['likes'])
y_test_ope = data_test['ope'].astype(float)
y_test_pred = svr_ope.predict(X_test_ope)

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test_ope, y_test_pred)))
print("MSE:", metrics.mean_squared_error(y_test_ope, y_test_pred))
print("MAE:", metrics.mean_absolute_error(y_test_ope, y_test_pred))
print("=================================")

#######################################################
#######################################################

y_train_con = data_train['con'].astype(float)

svr_con = joblib.load('svr_Con.pkl')

print("con Fitted")
X_test_con = hash_vect.transform(data_test['likes'])
y_test_con = data_test['con'].astype(float)
y_test_pred = svr_con.predict(X_test_con)

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test_con, y_test_pred)))
print("MSE:", metrics.mean_squared_error(y_test_con, y_test_pred))
print("MAE:", metrics.mean_absolute_error(y_test_con, y_test_pred))
print("=================================")

#######################################################
#######################################################

y_train_ext = data_train['ext'].astype(float)

svr_ext = joblib.load('svr_Ext.pkl')

print("ext Fitted")
X_test_ext = hash_vect.transform(data_test['likes'])
y_test_ext = data_test['ext'].astype(float)
y_test_pred = svr_ext.predict(X_test_ext)

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test_ext, y_test_pred)))
print("MSE:", metrics.mean_squared_error(y_test_ext, y_test_pred))
print("MAE:", metrics.mean_absolute_error(y_test_ext, y_test_pred))
print("=================================")

#######################################################
#######################################################

y_train_agr = data_train['agr'].astype(float)

svr_agr = joblib.load('svr_Agr.pkl')

print("agr Fitted")
X_test_agr = hash_vect.transform(data_test['likes'])
y_test_agr = data_test['agr'].astype(float)
y_test_pred = svr_agr.predict(X_test_agr)

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test_agr, y_test_pred)))
print("MSE:", metrics.mean_squared_error(y_test_agr, y_test_pred))
print("MAE:", metrics.mean_absolute_error(y_test_agr, y_test_pred))
print("=================================")

#######################################################
#######################################################

y_train_neu = data_train['neu'].astype(float)

svr_neu = joblib.load('svr_Neu.pkl')


print("neu Fitted")
X_test_neu = hash_vect.transform(data_test['likes'])
y_test_neu = data_test['neu'].astype(float)
y_test_pred = svr_neu.predict(X_test_neu)

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test_neu, y_test_pred)))
print("MSE:", metrics.mean_squared_error(y_test_neu, y_test_pred))
print("MAE:", metrics.mean_absolute_error(y_test_neu, y_test_pred))
print("=================================")

#######################################################
#######################################################

test_likes = hash_vect.fit_transform(data_test['likes'])



  
true_count_age =0
true_count_gender =0

ope_predictions = {}
for i in range(len(data_test)):


    gender = (knn_Gender.predict(test_likes[i]))
    
    if(gender == [1.] and data_test.at[8000+i,'gender'] == True):
        true_count_gender +=1
    elif(gender == [0.] and data_test.at[8000+i,'gender'] == False):
        true_count_gender +=1
         
    temp = buckets((knn_age.predict(test_likes[i])))
    real = buckets(int(data_test.at[8000+i,'age']))

    if(temp == real):
        true_count_age +=1

        

print("Gender true count: ", true_count_gender)
print("Age true count: ", true_count_age)

