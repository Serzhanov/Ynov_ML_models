import numpy as np
import pandas as pd
import models.decision_tree
import models.linear_reg
import models.logistic_reg
import models.bagging
import seaborn as sns
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

def test_decision_tree():

    csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

    col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']

    data=pd.read_csv(csv_url, skiprows=1, header=None, names=col_names)

    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values.reshape(-1,1)
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)


    classifier = models.decision_tree.DecisionTreeClassifier(min_samples_split=3, max_depth=3)
    classifier.fit(X_train,Y_train)
    classifier.print_tree()


    Y_pred = classifier.predict(X_test) 
    from sklearn.metrics import accuracy_score
    accuracy_score(Y_test, Y_pred)

def test_linear_reg():
    #Imports
    
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def r2_score(y_true, y_pred):
        corr_matrix = np.corrcoef(y_true, y_pred)
        corr = corr_matrix[0, 1]
        return corr ** 2

    X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    regressor = models.linear_reg.LinearRegression(learning_rate=0.01, n_iters=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    print("MSE:", mse)

    accu = r2_score(y_test, predictions)
    print("Accuracy:", accu)

    y_pred_line = regressor.predict(X)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
    plt.show()

def test_logistic_reg():
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    

    bc = datasets.load_iris()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=500
    )

    regressor = models.logistic_reg.LogisticRegression(learning_rate=0.0001, n_iters=1000)
    regressor.fit(X_train, y_train)

    predictions = regressor.predict(X_test)
    probs = regressor.get_probs(X_test)

    print("LR classification accuracy:", accuracy(y_test, predictions))

def test_bagging():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

    col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']

    data=pd.read_csv(csv_url, skiprows=1, header=None, names=col_names)

    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values.reshape(-1,1)

    #Encoding
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            if Y[i][j]=='Iris-setosa':
                Y[i][j]=1
            if Y[i][j]=='Iris-versicolor':
                Y[i][j]=2
            if Y[i][j]=='Iris-virginica':
                Y[i][j]=3

    bagger = models.bagging.Bagger()


    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=500
    )

    bagger.fit(X_train, y_train, B = 30, max_depth = 20, min_size = 5, seed = 123)
    y_test_hat = bagger.predict(X_test)

    print("Bagging accuracy:", accuracy(y_test, y_test_hat))
