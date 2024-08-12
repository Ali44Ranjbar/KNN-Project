#K-N-N:(K-Nearest-Neighbors)

import pandas as pd

df=pd.read_csv("iris.csv",names=["sepal.length","sepal.width","petal.length","petal.width","variety"])
print(df)

X=df.iloc[: , : -1].values
Y=df.iloc[: , 4].values
print(X)
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
#Find best k and best accuracy
bestaccuracy=0
bestk=0
for k in range(1,51,2):
    
    model=KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train,Y_train)
    
    p=model.predict(X_test)
    from sklearn.metrics import accuracy_score
    accuracy=accuracy_score(p,Y_test)
    print(f"accuracy for k={k} is",accuracy_score(p,Y_test))
    if accuracy>bestaccuracy:
        bestaccuracy=accuracy
        best=k
        
print("---------------------------")
print(f"Best k is {bestk} with Accuracy {bestaccuracy}")
        