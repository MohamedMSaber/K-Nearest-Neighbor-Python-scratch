import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter


def E_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance


class KNN:
    def __init__(self, K=3):
        self.K = K

    def fit(self, X, y):  # store training examples
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        pre = [self.getCommon(i) for i in X]
        return np.array(pre)

    def getCommon(self, i):
        # calculate Distances
        distance = [E_distance(i, x_train) for x_train in self.X_train]
        #get the K nearest examples
        k_index = np.argsort(distance)[:self.K]    #Sort and []slice the first K vlaues
        k_nearest = [self.y_train[i] for i in k_index]
        
        #get most common class
        common_class = Counter(k_nearest).most_common()      # get the common class (class , no of times )
        return common_class[0][0]    #return only the class

    def accuracy(self,Y_pred,Y_test):
        return np.sum(Y_pred==Y_test)/len(Y_test)

    


if __name__ == "__main__":
    # Importing dataset

    df = pd.read_csv("BankNote_Authentication.csv")

    X = df.drop('class' , axis= 1)
    y = df['class']

    # Splitting dataset into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30)

    #Normalize X-train
    X_train["variance"] = (X_train["variance"]-np.mean(X_train["variance"]))/(np.std(X_train["variance"]))
    X_train["skewness"] = (X_train["skewness"]-np.mean(X_train["skewness"]))/(np.std(X_train["skewness"]))
    X_train["curtosis"] = (X_train["curtosis"]-np.mean(X_train["curtosis"]))/(np.std(X_train["curtosis"]))
    X_train["entropy"] = (X_train["entropy"]-np.mean(X_train["entropy"]))/(np.std(X_train["entropy"]))

    #Normalize X-test
    X_test["variance"] = (X_test["variance"]-np.mean(X_train["variance"]))/(np.std(X_train["variance"]))
    X_test["skewness"] = (X_test["skewness"]-np.mean(X_train["skewness"]))/(np.std(X_train["skewness"]))
    X_test["curtosis"] = (X_test["curtosis"]-np.mean(X_train["curtosis"]))/(np.std(X_train["curtosis"]))
    X_test["entropy"] = (X_test["entropy"]-np.mean(X_train["entropy"]))/(np.std(X_train["entropy"]))


    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)

   

    
   


    for i in range(2,6):
        k = i
        model = KNN(K=k)
        model.fit(X_train , Y_train)
        Y_pred = model.predict(X_test)
        accuracy = model.accuracy(Y_pred , Y_test)*100
        print ("k value :" , k)
        print("Number of correctly classified instances :",np.sum(Y_pred == Y_test) ,"Total number of instances : ",len(Y_test))
        print("Accuracy :",accuracy)
     
    


