import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def neighbor(X_test_projected, target):
    distance = []
    for i in range(len(X_test_projected)):
        dtype = [('index', int), ('distance', float)]
        unit = (i,np.absolute(X_test_projected[i]-target))
        distance.append(unit)
    distance = np.sort(np.array(distance, dtype=dtype), order='distance')
    return distance  

def knn(X_test_projected, Y_test, n):
    Y_pred = []
    for i in range(len(X_test_projected)):
        distance = neighbor(X_test_projected,X_test_projected[i])
        c1 = 0
        c2 = 0
        for j in range(1,n+1):
            if Y_test[distance[j][0]] == 0:
                c1 = c1 + 1
            if Y_test[distance[j][0]] == 1:
                c2 = c2 + 1
        if c1 >= c2:
            Y_pred.append(0)
        else:
            Y_pred.append(1)
    return np.array(Y_pred)


def main():
    x_train, x_test, y_train, y_test = np.load('classification_data.npy', allow_pickle=True)   
    X_train = np.array(x_train)
    Y_train = np.array(y_train)
    X_test = np.array(x_test)
    Y_test = np.array(y_test)
    # print(X_train)
    # print(Y_train)
    n1=0
    n2=0
    m1=0
    m2=0
    for i in range(len(X_train)):
        if Y_train[i] == 0:
            n1 = n1 + 1
            m1 = m1 + X_train[i]
        if Y_train[i] == 1:
            n2 = n2 + 1
            m2 = m2 + X_train[i]
    m1 = m1 / n1
    m2 = m2 / n2
    #np.reshape(m1, (2,1)) 
    print(f"mean vector of class 1: {m1}", f"mean vector of class 2: {m2}")

    Sw = np.array([[0,0],[0,0]])
    for i in range(len(X_train)):
        if Y_train[i] == 0:
            tmp = X_train[i] - m1
            Sw = Sw + np.dot(tmp[:,None],tmp[None,:])
            #print((X_train[i] - m1).transpose())
        if Y_train[i] == 1:
            tmp = X_train[i] - m2
            Sw = Sw + np.dot(tmp[:,None],tmp[None,:])
    print(f"Within-class scatter matrix SW: {Sw}")

    Sb = np.array([[0,0],[0,0]])
    tmp = m2 - m1
    Sb = Sb + np.dot(tmp[:,None],tmp[None,:])
    print(f"Between-class scatter matrix SB: {Sb}")

    w = np.dot(np.linalg.inv(Sw),m2-m1)
    w = w / np.linalg.norm(w)
    print(f" Fisher’s linear discriminant: {w}")

    X_train_projected = np.dot(X_train,w)
    X_test_projected = np.dot(X_test,w)

    for i in range(5):
        acc = accuracy_score(y_test, knn(X_test_projected, Y_test, i+1))
        print(f"For K={i+1}, Accuracy of test-set {acc}")

    X_train_x1 = []
    X_train_x2 = []
    for i in range(len(X_train)):
        X_train_x1.append(X_train[i][0])
        X_train_x2.append(X_train[i][1])

    w = w.reshape((2, ))
    X_train_projected = X_train_projected.reshape((3750,1))
    X_train_projected_plot = X_train_projected * w

    slope = (X_train_projected_plot[1][1] - X_train_projected_plot[0][1]) / (X_train_projected_plot[1][0] - X_train_projected_plot[0][0])
    intercept = X_train_projected_plot[0][1] - slope * X_train_projected_plot[0][0]

    plt.title(f"Projection Line: w={slope}, b={intercept}")
    plt.plot(X_train_projected_plot.T[0], X_train_projected_plot.T[1])

    for i in range(len(X_train)):
        x = [X_train[i][0], X_train_projected_plot[i][0]]
        y = [X_train[i][1], X_train_projected_plot[i][1]]
        plt.plot(x, y, linewidth=0.1, c='k', zorder=0)

    color = np.where(Y_train == 1, 'r', 'b')
    plt.scatter(X_train_projected_plot.T[0], X_train_projected_plot.T[1], c=color, s=1, zorder=10)
    plt.scatter(X_train_x1, X_train_x2, c=color, s=1)
    plt.show()


if __name__ == '__main__':
    main()


