import ast
import sys
import math
from utils.csv_reader import read_csv
from utils.normalization import apply_normalization
from utils.stats import get_numeric_columns

houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]


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


def predict_row(raw, features, weights, biases):
    x = [raw[feature] for feature in features]
    probabilities = {}
    for house in houses:
        w = weights[house]
        bias = biases[house]
        z = calculate_z(x, list(w.values()), bias)
        probabilities[house] = calculate_sigmoid(z)
    return max(probabilities, key=probabilities.get)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py dataset_train.csv")
        sys.exit(1)

    weights, biases, stats = read_weights()
    path = sys.argv[1]
    data = read_csv(path)
    features = get_numeric_columns(data)
    data = apply_normalization(data, features, stats)

    correct = 0
    total = 0
    errors = {"Gryffindor": 0, "Hufflepuff": 0, "Ravenclaw": 0, "Slytherin": 0}

    for row in data:
        actual = row.get("Hogwarts House")
        if actual is None:
            continue
        predicted = predict_row(row, features, weights, biases)
        if predicted == actual:
            correct += 1
        else:
            errors[actual] += 1
        total += 1

    accuracy = correct / total * 100 if total > 0 else 0

    print(f"\n{'='*40}")
    print(f"  Evaluation on: {path}")
    print(f"{'='*40}")
    print(f"  Total samples : {total}")
    print(f"  Correct       : {correct}")
    print(f"  Wrong         : {total - correct}")
    print(f"  Accuracy      : {accuracy:.2f}%")
    print(f"{'='*40}")
    print(f"  Misclassified per house:")
    for house, count in errors.items():
        print(f"    {house:<12}: {count} errors")
    print(f"{'='*40}\n")

    if accuracy >= 98.0:
        print("  ✅ Meets the 98% requirement!")
    else:
        print(f"  ❌ Below 98% — need {98.0 - accuracy:.2f}% more accuracy.")
