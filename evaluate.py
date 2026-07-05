import sys
import math
import time
from utils.csv_reader import read_csv
from utils.normalization import fit_normalization, apply_normalization
from utils.stats import get_numeric_columns
from logreg_train import train_one_vs_all, save_model

houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]


def calculate_z(x, w, bias):
    return sum(a * b for a, b in zip(x, w)) + bias


def calculate_sigmoid(z):
    # Numerically stable sigmoid function
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    else:
        return math.exp(z) / (1 + math.exp(z))


def predict_row(raw, features, weights, biases):
    x = [raw[feature] for feature in features]
    probabilities = {}
    for house in houses:
        w = weights[house]
        bias = biases[house]
        z = calculate_z(x, list(w.values()), bias)
        probabilities[house] = calculate_sigmoid(z)
    return max(probabilities, key=probabilities.get)


def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py dataset_train.csv")
        sys.exit(1)

    path = sys.argv[1]
    data = read_csv(path)
    features = get_numeric_columns(data)

    # Fit and apply normalization once for all methods
    stats = fit_normalization(data, features)
    data = apply_normalization(data, features, stats)

    methods = [
        {"name": "batch", "epochs": 150, "lr": 0.5, "batch_size": 32},
        {"name": "sgd", "epochs": 15, "lr": 0.01, "batch_size": 32},
        {"name": "mini-batch", "epochs": 30, "lr": 0.1, "batch_size": 32},
    ]

    results = []
    best_accuracy = -1.0
    best_weights = None
    best_biases = None
    best_method_name = ""

    print(f"\nTraining and evaluating all methods on: {path}\n")

    for m in methods:
        name = m["name"]
        epochs = m["epochs"]
        lr = m["lr"]
        batch_size = m["batch_size"]

        print(f"--- Running {name.upper()} Gradient Descent ---")
        start_time = time.time()
        weights, biases = train_one_vs_all(data, features, name, epochs, lr, batch_size)
        elapsed = time.time() - start_time

        correct = 0
        total = 0
        errors = {house: 0 for house in houses}

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

        accuracy = (correct / total * 100) if total > 0 else 0.0
        meets_requirement = accuracy >= 98.0

        results.append({
            "name": name,
            "epochs": epochs,
            "lr": lr,
            "time": elapsed,
            "accuracy": accuracy,
            "meets": meets_requirement,
            "errors": errors,
            "total": total,
            "correct": correct,
        })

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = weights
            best_biases = biases
            best_method_name = name

        print(f"Finished {name.upper()} in {elapsed:.4f}s with {accuracy:.2f}% accuracy.\n")

    # Print Comparative Summary Table
    print("=" * 80)
    print(f"{'EVALUATION OF ALL TRAINING METHODS':^80}")
    print("=" * 80)
    print(f"{'Method':<15} | {'Epochs':<6} | {'LR':<6} | {'Time (s)':<10} | {'Accuracy':<10} | {'Meets 98% Req?'}")
    print("-" * 80)
    for res in results:
        req_str = "Yes ✅" if res["meets"] else "No ❌"
        print(f"{res['name'].upper():<15} | {res['epochs']:<6} | {res['lr']:<6} | {res['time']:<10.4f} | {res['accuracy']:<9.2f}% | {req_str}")
    print("=" * 80)

    # Print Misclassification breakdown for each
    print("\nDetailed Misclassifications per House:")
    print("-" * 80)
    for res in results:
        err_str = ", ".join([f"{h}: {count}" for h, count in res["errors"].items()])
        print(f"{res['name'].upper():<12} -> Total Errors: {res['total'] - res['correct']} ({err_str})")
    print("-" * 80)

    # Save the best model
    if best_weights is not None:
        save_model("weights.txt", best_weights, best_biases, stats)
        print(f"\n🏆 Best model: {best_method_name.upper()} ({best_accuracy:.2f}% accuracy)")
        print("--> Saved the best performing model weights to weights.txt.\n")


if __name__ == "__main__":
    main()
