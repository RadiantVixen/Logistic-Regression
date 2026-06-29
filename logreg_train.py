import math
import sys

from utils.normalization import fit_normalization, apply_normalization
from utils.csv_reader import read_csv
from utils.stats import get_numeric_columns


EPOCHS = 10
LEARNING_RATE = 0.01


def calc_z(row, weights, bias, features):
    z = bias
    for feature in features:
        z += weights[feature] * row[feature]
    return z


def sigmoid(row, weights, bias, features):
    z = calc_z(row, weights, bias, features)
    return 1 / (1 + math.exp(-z))


def likelihood(data, house, weights, bias, features):
    result = 0

    for row in data:
        y = 1 if row["Hogwarts House"] == house else 0
        p = sigmoid(row, weights, bias, features)

        # avoid log(0)
        p = max(min(p, 1 - 1e-15), 1e-15)

        if y == 1:
            result += math.log(p)
        else:
            result += math.log(1 - p)

    return result


def gradient_weight(data, feature, house, weights, bias, features):
    grad = 0

    for row in data:
        y = 1 if row["Hogwarts House"] == house else 0
        p = sigmoid(row, weights, bias, features)
        grad += (y - p) * row[feature]

    return grad


def gradient_bias(data, house, weights, bias, features):
    grad = 0

    for row in data:
        y = 1 if row["Hogwarts House"] == house else 0
        p = sigmoid(row, weights, bias, features)
        grad += (y - p)

    return grad


def train_one_vs_all(data, features):
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

    all_weights = {}
    all_biases = {}

    print("Training...")

    for house in houses:
        weights = {feature: 0.0 for feature in features}
        bias = 0.0

        for _ in range(EPOCHS):
            for feature in features:
                grad = gradient_weight(data, feature, house, weights, bias, features)
                weights[feature] += LEARNING_RATE * grad

            bias += LEARNING_RATE * gradient_bias(data, house, weights, bias, features)

        all_weights[house] = weights
        all_biases[house] = bias

        print(f"{house} done")

    return all_weights, all_biases


def save_model(path, weights, biases, stats):
    with open(path, "w") as file:
        file.write("WEIGHTS\n")
        for house, house_weights in weights.items():
            file.write(f"{house}:{house_weights}\n")

        file.write("\nBIASES\n")
        for house, bias in biases.items():
            file.write(f"{house}:{bias}\n")

        file.write("\nNORMALIZATION\n")
        for feature, (mu, sigma) in stats.items():
            file.write(f"{feature}:{mu},{sigma}\n")
        file.write("\n")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py dataset.csv")
        sys.exit(1)

    path = sys.argv[1]
    data = read_csv(path)

    features = get_numeric_columns(data)

    stats = fit_normalization(data, features)
    data = apply_normalization(data, features, stats)

    weights, biases = train_one_vs_all(data, features)

    save_model("weights.txt", weights, biases, stats)