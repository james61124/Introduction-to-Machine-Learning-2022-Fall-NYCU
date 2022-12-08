import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


def Mean_square_error(n,y_pred,y):
    return (1/n)*sum((y - y_pred)*(y - y_pred))

def Cross_entropy_error(y_pred,y):
    return sum((-y) * np.log(y_pred) - (1 - y)*np.log(1-y_pred))

def main():
    x_train, x_test, y_train, y_test = np.load('regression_data.npy', allow_pickle=True)
    X_train = np.array(x_train).flatten()
    Y_train = np.array(y_train)
    X_test = np.array(x_test).flatten()
    Y_test = np.array(y_test)

    beta_0 = np.random.normal(0,1)
    beta_1 = np.random.normal(0,1)

    learning_rate = 0.2
    epochs = 50

    n_train = float(len(X_train))
    n_test = float(len(X_test)) 
    training_curve = []

    for i in range(epochs): 
        Y_pred = beta_0 * X_train + beta_1
        D_beta_0 = (-2/n_train) * sum(X_train * (Y_train - Y_pred))
        D_beta_1 = (-2/n_train) * sum(Y_train - Y_pred) 
        beta_0 = beta_0 - learning_rate * D_beta_0 
        beta_1 = beta_1 - learning_rate * D_beta_1
        Y_pred_test = beta_0 * X_test + beta_1
        training_curve.append(Mean_square_error(n_train,Y_pred,Y_train))

    MSE = Mean_square_error(n_test,Y_pred_test,Y_test)
    Y_pred = beta_0*X_train + beta_1

    print("Mean_square_error:",MSE)
    print("weights:",beta_0)
    print("intercepts:",beta_1)

    plt.plot(training_curve)
    plt.show()



    learning_rate = 0.005 
    epochs = 60

    x_train, x_test, y_train, y_test = np.load('classification_data.npy', allow_pickle=True)
    X_train = np.array(x_train).flatten()
    Y_train = np.array(y_train)
    X_test = np.array(x_test).flatten()
    Y_test = np.array(y_test)

    n_train = float(len(X_train))
    n_test = float(len(X_test))

    w = np.random.normal(0,1)
    b = np.random.normal(0,1)

    training_curve = []
    for i in range(epochs): 
        z = w*X_train + b
        a = 1/(1+np.exp(z))
        D_w = sum((a - Y_train) * X_train)
        D_b = sum(a - Y_train)
        w = w + learning_rate * D_w
        b = b + learning_rate * D_b
        training_curve.append(Cross_entropy_error(a,Y_train))
        z_test = w*X_test + b
        a_test = 1/(1+np.exp(z_test))

    cross_entropy = Cross_entropy_error(a_test,Y_test)
    print("Cross Entropy Error:",cross_entropy)
    print("weights:",w)
    print("intercepts:",b)

    plt.plot(training_curve)
    plt.show()

if __name__ == '__main__':
    main()


