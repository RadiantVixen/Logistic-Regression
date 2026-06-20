import sys
import matplotlib.pyplot as plt
from utils.csv_reader import read_csv
from utils.stats import get_numeric_columns


def get_clean_pairs(data, x_feature, y_feature):

    x_values = []
    y_values = []

    for row in data:

        x = row[x_feature]
        y = row[y_feature]

        if isinstance(x, float) and isinstance(y, float):

            x_values.append(x)
            y_values.append(y)

    return x_values, y_values


def main():

    if len(sys.argv) != 2:

        print("Usage: python pair_plot.py dataset.csv")
        return

    path = sys.argv[1]

    data = read_csv(path)

    features = get_numeric_columns(data)

    size = len(features)

    fig, axes = plt.subplots(size, size, figsize=(15, 15))

    for i in range(size):

        for j in range(size):

            x_feature = features[j]
            y_feature = features[i]

            x, y = get_clean_pairs(data, x_feature, y_feature)

            axes[i][j].scatter(x, y, s=1)

            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])

            if i == size - 1:
                axes[i][j].set_xlabel(x_feature)

            if j == 0:
                axes[i][j].set_ylabel(y_feature)

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()

