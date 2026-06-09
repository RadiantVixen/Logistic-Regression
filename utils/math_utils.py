import math


def sigmoid(z):

    return 1 / (1 + math.exp(-z))


def dot(a, b):

    total = 0

    for i in range(len(a)):
        total += a[i] * b[i]

    return total


def vector_add(a, b):

    result = []

    for i in range(len(a)):
        result.append(a[i] + b[i])

    return result


def vector_sub(a, b):

    result = []

    for i in range(len(a)):
        result.append(a[i] - b[i])

    return result


def scalar_multiply(vector, scalar):

    result = []

    for value in vector:
        result.append(value * scalar)

    return result