import math

data = pd.read_csv("data.csv")
bias = 0

w = [0] * len(data.columns)

def calc_z(x):
    z = bias
    for i in range(len(w)):
        z += w[i] * x[i]
    return z

def sigmoid(x):
    return 1 / (1 + math.e **(-calc_z(x)))

def Likelihood():
    result = 0

    for i in range(len(data)):
        x = data.iloc[i][:-1]
        y = data.iloc[i][-1]

        z = calc_z(x)

        if y == 1:
            result += math.log(sigmoid(x))
        else:
            result += math.log(1 - sigmoid(x))
    
    return result

def Gradient(j):
    grad = 0
    for i in range(len(data)):
        x = data.iloc[i][:-1]
        y = data.iloc[i][-1]
        grad += y  - sigmoid(x) * x[j]
        
    return grad
