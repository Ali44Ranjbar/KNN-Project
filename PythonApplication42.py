#K-N-N:
import pandas as pd 

df=pd.read_csv("breast-cancer.csv")

x=df.iloc[: , : -1]
y=df.iloc[: , 9]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)

from sklearn.neighbors import KNeighborsClassifier

bestaccuracy=0
bestk=0

for k in range(1,57,2):
    
    model=KNeighborsClassifier(n_neighbors=k)
    model.fit(xtrain,ytrain)
    
    p=model.predict(xtest)
    from sklearn.metrics import accuracy_score
    accuracy=accuracy_score(p,ytest)
    print(f"Accuracy for K={k} is",accuracy)
    
    if accuracy>bestaccuracy:
        bestaccuracy=accuracy
        bestk=k
        
print("------------------------")
print(f"Best K is {bestk} with Accuracy {bestaccuracy}")

