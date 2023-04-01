## Author: Patrick Tibbals


import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR

def buckets(number):
    if(number<25):
        return "1"
    elif(number<35):
        return "2"
    elif(number<50):
        return "3"
    else:
        return "4"


df = pd.read_csv("training/profile/profile.csv")

userIDs = df.loc[:,['userid']]
userInfo = df.loc[:,['userid','gender','age','ope','con','ext','agr','neu']]

all_like_data = {}

df = pd.read_csv("training/relation/relation.csv")
posterID = df.loc[:,['userid']]
likeID = df.loc[:,['like_id']]



j = 0
currID = posterID.loc[0,'userid']
print(len(likeID))
#print(currID)
likeList = ""
for i in range(len(likeID)):
    if(currID == (posterID.loc[i,'userid'])):
        likeList += str(likeID.loc[i,'like_id'])+" "
    else:
        a = userInfo['age'].where(userInfo['userid'] == currID).dropna().tolist()
        a = buckets(int(a[0]))

        g = userInfo['gender'].where(userInfo['userid'] == currID).dropna().tolist()
        o = userInfo['ope'].where(userInfo['userid'] == currID).dropna().tolist()
        c = userInfo['con'].where(userInfo['userid'] == currID).dropna().tolist()
        e = userInfo['ext'].where(userInfo['userid'] == currID).dropna().tolist()
        ag = userInfo['agr'].where(userInfo['userid'] == currID).dropna().tolist()
        n = userInfo['neu'].where(userInfo['userid'] == currID).dropna().tolist()

        
        all_like_data[j] = [ currID , likeList, g[0], a, o[0], c[0], e[0], ag[0], n[0]]
        currID = posterID.loc[i,'userid']
        likeList = str(likeID.loc[i,'like_id'])+" "
        j+=1
        
all_like_data[j] = [ currID , likeList, g[0], a, o[0], c[0], e[0], ag[0], n[0]]    



dataset = pd.DataFrame.from_dict(all_like_data)
dataset = dataset.T
dataset2 = dataset.set_axis(['userid', 'likes', 'gender', 'age','ope','con','ext','agr','neu'], axis=1, copy= False)
dataset2['gender'] = dataset2['gender'].astype('bool')
joblib.dump(dataset2, 'knnpickle_TrainData.pkl')


users = np.arange(len(dataset2))
joblib.dump(users, 'knnpickle_Users.pkl')




train_Ids = users
data_train = dataset2.loc[users]


hash_vect = HashingVectorizer()

X_train = hash_vect.fit_transform(data_train['likes'])


y_train_gender = data_train['gender']
knn_Gender = KNeighborsClassifier(n_neighbors = 25,algorithm='auto', metric='minkowski') #setting up the KNN model to use 5NN
knn_Gender.fit(X_train, y_train_gender) #fitting the KNN

joblib.dump(knn_Gender, 'knnpickle_Gender.pkl')





y_train_age = data_train['age'].astype('int')

knn_age = KNeighborsClassifier(n_neighbors = 25,algorithm='auto', metric='minkowski') #setting up the KNN model to use 5NN
knn_age.fit(X_train, y_train_age) #fitting the KNN
joblib.dump(knn_age, 'knnpickle_Age.pkl')
print("Fitted")

#######################################################
#######################################################
print("=================================")

y_train_ope = data_train['ope'].astype(float)

svr_ope = SVR()
svr_ope.fit(X_train, y_train_ope)
joblib.dump(svr_ope, 'svr_Ope.pkl')

print("Ope Fitted")


#######################################################
#######################################################

y_train_con = data_train['con'].astype(float)

svr_con = SVR()
svr_con.fit(X_train, y_train_con)
joblib.dump(svr_con, 'svr_Con.pkl')

print("con Fitted")

#######################################################
#######################################################

y_train_ext = data_train['ext'].astype(float)

svr_ext = SVR()
svr_ext.fit(X_train, y_train_ext)
joblib.dump(svr_ext, 'svr_Ext.pkl')

print("ext Fitted")


#######################################################
#######################################################

y_train_agr = data_train['agr'].astype(float)

svr_agr = SVR()
svr_agr.fit(X_train, y_train_agr)
joblib.dump(svr_agr, 'svr_Agr.pkl')

print("agr Fitted")


#######################################################
#######################################################

y_train_neu = data_train['neu'].astype(float)

svr_neu = SVR()
svr_neu.fit(X_train, y_train_neu)
joblib.dump(svr_neu, 'svr_Neu.pkl')

print("neu Fitted")


#######################################################
#######################################################
