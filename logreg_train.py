import math
import sys
import argparse
import random

from utils.normalization import fit_normalization, apply_normalization
from utils.csv_reader import read_csv
from utils.stats import get_numeric_columns


def sigmoid(z):
    # Numerically stable sigmoid function
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    else:
        return math.exp(z) / (1 + math.exp(z))


def train_one_vs_all(data, features, method, epochs, lr, batch_size):
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

    all_weights = {}
    all_biases = {}

    print(f"Training using {method} gradient descent (epochs={epochs}, lr={lr}, batch_size={batch_size})...")

    for house in houses:
        weights = {feature: 0.0 for feature in features}
        bias = 0.0

        for epoch in range(epochs):
            if method == "batch":
                # Batch Gradient Descent: gradient calculated over the entire dataset
                grad_w = {f: 0.0 for f in features}
                grad_b = 0.0
                for row in data:
                    y = 1 if row["Hogwarts House"] == house else 0
                    z = bias + sum(weights[f] * row[f] for f in features)
                    p = sigmoid(z)
                    diff = y - p
                    for f in features:
                        grad_w[f] += diff * row[f]
                    grad_b += diff

                # Update step (scaled by dataset size)
                for f in features:
                    weights[f] += lr * (grad_w[f] / len(data))
                bias += lr * (grad_b / len(data))

            elif method == "sgd":
                # Stochastic Gradient Descent: updates weights example by example
                random.seed(epoch)
                shuffled_data = list(data)
                random.shuffle(shuffled_data)
                for row in shuffled_data:
                    y = 1 if row["Hogwarts House"] == house else 0
                    z = bias + sum(weights[f] * row[f] for f in features)
                    p = sigmoid(z)
                    diff = y - p
                    for f in features:
                        weights[f] += lr * diff * row[f]
                    bias += lr * diff

            elif method == "mini-batch":
                # Mini-batch Gradient Descent: updates weights batch by batch
                random.seed(epoch)
                shuffled_data = list(data)
                random.shuffle(shuffled_data)
                for i in range(0, len(data), batch_size):
                    batch = shuffled_data[i:i+batch_size]
                    grad_w = {f: 0.0 for f in features}
                    grad_b = 0.0
                    for row in batch:
                        y = 1 if row["Hogwarts House"] == house else 0
                        z = bias + sum(weights[f] * row[f] for f in features)
                        p = sigmoid(z)
                        diff = y - p
                        for f in features:
                            grad_w[f] += diff * row[f]
                        grad_b += diff

                    # Update step (scaled by batch size)
                    for f in features:
                        weights[f] += lr * (grad_w[f] / len(batch))
                    bias += lr * (grad_b / len(batch))

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


def main():
    parser = argparse.ArgumentParser(description="Train a multi-classifier logistic regression model.")
    parser.add_argument("dataset", type=str, help="Path to dataset_train.csv")
    parser.add_argument("-m", "--method", type=str, choices=["batch", "sgd", "mini-batch"], default="batch",
                        help="Optimization method (default: batch)")
    parser.add_argument("-l", "--lr", type=float, default=None,
                        help="Learning rate (default: 0.5 for batch, 0.01 for sgd, 0.1 for mini-batch)")
    parser.add_argument("-e", "--epochs", type=int, default=None,
                        help="Number of epochs (default: 150 for batch, 15 for sgd, 30 for mini-batch)")
    parser.add_argument("-b", "--batch-size", type=int, default=32,
                        help="Batch size for mini-batch gradient descent (default: 32)")

    args = parser.parse_args()

    # Apply default parameters based on selected optimization method
    if args.method == "batch":
        lr = args.lr if args.lr is not None else 0.5
        epochs = args.epochs if args.epochs is not None else 150
    elif args.method == "sgd":
        lr = args.lr if args.lr is not None else 0.01
        epochs = args.epochs if args.epochs is not None else 15
    elif args.method == "mini-batch":
        lr = args.lr if args.lr is not None else 0.1
        epochs = args.epochs if args.epochs is not None else 30

    data = read_csv(args.dataset)
    features = get_numeric_columns(data)

    stats = fit_normalization(data, features)
    data = apply_normalization(data, features, stats)

    weights, biases = train_one_vs_all(data, features, args.method, epochs, lr, args.batch_size)

    save_model("weights.txt", weights, biases, stats)


if __name__ == "__main__":
    main()