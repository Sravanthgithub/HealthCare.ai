import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("heart-disease (2).csv")
from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
all=['age', 	'sex', 	'cp', 'trestbps', 'chol', 	'fbs', 	'restecg', 	'thalach' ,	'exang', 	'oldpeak' ,	'slope', 	'ca', 'thal']
df[all] = scale.fit_transform(df[all])
X=df.drop("target",axis=1).values
Y=df.target.values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0,test_size=0.2)

from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score,roc_auc_score,confusion_matrix

def evaluation(Y_test,Y_pred):
  acc=accuracy_score(Y_test,Y_pred)
  rcl=recall_score(Y_test,Y_pred)
  f1=f1_score(Y_test,Y_pred)
 

  metric_dict={'accuracy': round(acc,3),
               'recall': round(rcl,3),
               'F1 score': round(f1,3),
              }
  return print(metric_dict)

np.random.seed(42)
from sklearn.neighbors import KNeighborsClassifier
Knn_clf=  KNeighborsClassifier()
Knn_clf.fit(X_train,Y_train)
Knn_Y_pred=Knn_clf.predict(X_test)
Knn_score=Knn_clf.score(X_test,Y_test)
#print(Knn_score)
evaluation(Y_test,Knn_Y_pred)

neighbors = range(1, 21) 
knn = KNeighborsClassifier()
for i in neighbors:
    knn.set_params(n_neighbors = i) # set neighbors value
    print(f"Accuracy with {i} no. of neighbors: {knn.fit(X_train, Y_train).score(X_test,Y_test)}%")
knn_grid={'n_neighbors': np.arange(1,30,1),
          'leaf_size': np.arange(1,50,1)}
from sklearn.model_selection import GridSearchCV

gs_knn=GridSearchCV(KNeighborsClassifier(),param_grid=knn_grid,cv=5,verbose=True)
gs_knn.fit(X_train, Y_train)
gs_knn.best_params_
print(f"Accuracy score:{gs_knn.score(X_test,Y_test)*100}%")
#print(f"Final accuracy (highest) - {Knn_score*100}")


from sklearn.metrics import confusion_matrix

fig,ax=plt.subplots()
ax=sns.heatmap(confusion_matrix(Y_test,Knn_Y_pred),annot=True,cbar=True);


user_input=input("Enter the values one by one - ")
user_input=user_input.split(",")


for i in range(len(user_input)):
    #convert each item to int type
    user_input[i] = float(user_input[i])

user_input=np.array(user_input)
user_input=user_input.reshape(1,-1)
user_input=scale.transform(user_input)
Knn_clf.fit(X_train,Y_train)
Knn_Y_pred=Knn_clf.predict(user_input)
if(Knn_Y_pred[0]==0):
  print("Warning! You have chances of getting a heart disease!")
else:
  print("You are healthy and are less likely to get a heart disease!")

import pickle as pkl
pkl.dump(Knn_clf,open("final_model.p","wb"))