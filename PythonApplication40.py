import numpy as np
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

df = pd.read_csv('iris.csv') #, names= ["sepal.length","sepal.width","petal.length","petal.width","variety"])

x = df.iloc[: , : -1].values
y = df.iloc[: , 4].values

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size= 0.2 ) #, random_state= 10)

from sklearn.neighbors import KNeighborsClassifier

bestaccuracy = 0
bestk = 0

for k in range(1,51, 2):

    model = KNeighborsClassifier(n_neighbors= k)
    model.fit(xtrain, ytrain)

    p = model.predict(xtest)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(p, ytest)
    print(f'Accuracy for K={k} is ', accuracy)
    
    if accuracy > bestaccuracy:
        bestaccuracy = accuracy
        bestk = k

print('#############################')
print(f'Best K is {bestk} with Accuracy {bestaccuracy}')



# p = model.predict([[1,3,2,4]])
# print(p)
