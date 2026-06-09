import sys
import matplotlib.pyplot as plt

from utils.csv_reader import read_csv


def extract_feature(data, feature):

    values = []

    for row in data:

        value = row[feature]

        if isinstance(value, float):
            values.append(value)

    return values


def main():

    if len(sys.argv) != 4:

        print("Usage: python scatter_plot.py dataset.csv feature1 feature2")
        return

    path = sys.argv[1]

    feature_x = sys.argv[2]
    feature_y = sys.argv[3]

    data = read_csv(path)

    x = extract_feature(data, feature_x)
    y = extract_feature(data, feature_y)

    size = min(len(x), len(y))

    x = x[:size]
    y = y[:size]

    plt.scatter(x, y)

    plt.xlabel(feature_x)
    plt.ylabel(feature_y)

    plt.show()


if __name__ == "__main__":
    main()