import sys
import matplotlib.pyplot as plt

from utils.csv_reader import read_csv
from utils.stats import get_numeric_column


def main():

    if len(sys.argv) != 3:

        print("Usage: python histogram.py dataset.csv feature")
        return

    path = sys.argv[1]
    feature = sys.argv[2]

    data = read_csv(path)

    values = get_numeric_column(data, feature)

    plt.hist(values, bins=30)

    plt.title(feature)

    plt.xlabel(feature)
    plt.ylabel("Frequency")

    plt.show()


if __name__ == "__main__":
    main()