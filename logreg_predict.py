import ast
import sys
from utils.csv_reader import read_csv
from utils.normalization import apply_normalization
from utils.stats import get_numeric_columns
import math

houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
data = {}
features = []
biases = {}
weights = {}
stats = {}


def read_weights():
    weights = {}
    biases = {}
    stats = {}
    section = None

    with open("weights.txt", "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            if line in ("WEIGHTS", "BIASES", "NORMALIZATION"):
                section = line
                continue

            if section == "WEIGHTS":
                house, dict_str = line.split(":", 1)
                weights[house] = ast.literal_eval(dict_str)

            elif section == "BIASES":
                house, bias_str = line.split(":", 1)
                biases[house] = float(bias_str)

            elif section == "NORMALIZATION":
                feature, values = line.split(":", 1)
                mu, sigma = values.split(",")
                stats[feature] = (float(mu), float(sigma))

    return weights, biases, stats


def calculate_z(x, w, bias):
    return sum(a * b for a, b in zip(x, w)) + bias

def calculate_sigmoid(z):
    return 1 / (1 + math.exp(-z))

def predict():
    with open("houses.csv", "w") as file:
        file.write("Index,Hogwarts House\n")
        for raw in data:
            x = [raw[feature] for feature in features]
            probabilities = {}
            for house in houses:
                w = weights[house]
                bias = biases[house]
                z = calculate_z(x, list(w.values()), bias)
                probabilities[house] = calculate_sigmoid(z)

            predicted_house = None
            max_prob = -1.0
            for house, prob in probabilities.items():
                if prob > max_prob:
                    max_prob = prob
                    predicted_house = house
            file.write(f"{raw['Index']},{predicted_house}\n")

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python logreg_predict.py dataset.csv")
        sys.exit(1)

    weights, biases, stats = read_weights()
    path = sys.argv[1]
    data = read_csv(path)
    features = get_numeric_columns(data)
    data = apply_normalization(data, features, stats)
    predict()
