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
    current_house = None

    with open("weights.txt", "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("house:"):
                current_house = line.split(":")[1].strip()
                weights[current_house] = {}
                
            elif line.startswith("weights") and current_house:
                weights_str = line.split("=")[1].strip()
                weights[current_house] = ast.literal_eval(weights_str)
                
            elif line.startswith("bias") and current_house:
                bias_str = line.split("=")[1].strip()
                biases[current_house] = float(bias_str)
                
    return weights, biases


def calculate_z(x, w, bias):
    return sum(a * b for a, b in zip(x, w)) + bias

def calculate_sigmoid(z):
    return 1 / (1 + math.exp(-z))

def predict():

    for raw in data:
        x = [raw[feature] for feature in features]
        probabilities = {}
        for house in houses:
            w = weights[house]
            bias = biases[house]
            z = calculate_z(x, list(w.values()), bias)
            probabilities[house] = calculate_sigmoid(z)

        predicted_house = max(probabilities, key=probabilities.get)
        with open("houses.csv", "a") as file:
            file.write(f"{raw['Index']} ,{predicted_house}\n")

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python logreg_predict.py dataset.csv")
        sys.exit(1)

    read_weights()
    path = sys.argv[1]
    data = read_csv(path)
    features = get_numeric_columns(data)
    data = apply_normalization(data, features, stats)
