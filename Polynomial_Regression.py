from tkinter import Y
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

def readFile():

    dataset = pd.read_csv('C:\\Users\\Bilal Khan\\PycharmProjects\\Tutorials\\baseballplayer.csv')

    X = dataset.iloc[0, 0].values
    Y = dataset.iloc[0,1].values
    return X,Y

def split_data(X,Y):

    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.4, random_state=0) 
    return train_X, test_X, train_Y, test_Y


def model_train (X,Y):

    X1 = np.reshape(X, (-1,1))
    Y1 = np.reshape(Y, (-1,1))

    poly_reg = PolynomialFeatures(degree=2)

    tran_X = poly_reg.fit_transform(X1)

    regressor = LinearRegression()
    regressor = fit(tran_X, Y1)

    y_pred = regressor.predict(tran_X)

    r_square = r2_score(Y1,y_pred)


def visualize_result(train X, train Y, test_X, test_Y, regressor,X1,Y1):
    
    plt.scatter(X1, Y1, color = 'red')
    plt.plot(X1, regressor.predict())

    plt.title("Distance covered by the ball")

    plt.xlabel("Angle")
    plt.ylabel("Distance")

    plt.show()


def main():
    X, Y = readFile()
    X_train, X_test, y_train, y_test = split_data(X,Y)
    classifier = model_train(X_train, y_train, X_test, X_test)
    visualize_result(X_train,y_train,X_test,y_test,regressor)

if __name__ == "__main__":
    main()

