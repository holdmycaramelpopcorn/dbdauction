import numpy as np
import sys

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)

    for i in range(1, num_iters):
        hypothesis = np.dot(X, theta)
        errors = hypothesis - y

        newDecrement = (alpha * (1/m) * np.dot(np.transpose(errors), X))
        theta = theta - np.transpose(newDecrement)

    return theta

def featureNormalize(X, n):
    X_norm = X
    mu = np.zeros(n, dtype=np.int)
    sigma = [0] * n

    for i in range(0, n):
        meanOfCurrentFeatureInX = np.mean(X[:, i])
        mu[i] = meanOfCurrentFeatureInX

        X_norm[:, i] = [x - mu[i] for x in X_norm[:, i]]

        standardDeviationOfCurrentFeatureInX = np.std(X[:, i])
        sigma[i] = standardDeviationOfCurrentFeatureInX

        X_norm[:, i] = [x / sigma[i] for x in X_norm[:, i]]

    return X_norm, mu, sigma

def run():
    # load data
    print("Enter crop for which you need prediction: ")
    crop=input()
    if crop == "rose":
        data = np.loadtxt('lr-multi-data-rose.txt', dtype=np.int, delimiter=",")
    elif crop == "potato":
        data = np.loadtxt('lr-multi-data-potato.txt', dtype=np.int, delimiter=",")
    elif crop == "tomato":
        data = np.loadtxt('lr-multi-data-tomato.txt', dtype=np.int, delimiter=",")
    else:
        print("Wrong Crop, but showing rose anyway")
        data = np.loadtxt('lr-multi-data-rose.txt', dtype=np.int, delimiter=",")
    # no of features
    n = 2
    # load features
    X = data[:, 0:n]
    # load prices
    y = data[:, n:n+1]
    m = len(y)

    X, mu, sigma = featureNormalize(X, n)

    # Adding intercept term to X
    ones = np.ones(m, dtype=np.int)
    X = np.column_stack((ones, X))

    alpha = 0.01
    num_iters = 400
    theta = np.zeros((n+1, 1), dtype=np.int)

    theta = gradientDescent(X, y, theta, alpha, num_iters)
    print("Enter month in numeral: 1-Jan, 2-Feb, etc.")
    month=int(input())
    print("Enter MSP of your "+ crop +" crop.")
    msp=int(input())
    predict_norm = [1, ((int(month) - mu[0]) / sigma[0]), ((int(msp) - mu[1]) / sigma[1])]
    price = np.dot(predict_norm, theta)
    switcher = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December"
    }
    print('Predicted Sales for '+ crop +' in (kg) in Month {} with MSP Rs.{} is {}'.format(switcher.get(int(month),"Invalid"), msp, price[0]))

if __name__ == '__main__':
    run()