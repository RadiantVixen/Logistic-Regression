import sys

from utils.csv_reader import read_csv

from utils.stats import (
    mean,
    std,
    variance,
    minimum,
    maximum,
    percentile,
    get_numeric_columns,
    get_numeric_column
)


def main():

    if len(sys.argv) != 2:

        print("Usage: python describe.py dataset.csv")
        return

    path = sys.argv[1]

    data = read_csv(path)

    numeric_columns = get_numeric_columns(data)

    for column in numeric_columns:

        values = get_numeric_column(data, column)

        print(f"\nCOLUMN: {column}")

        print(f"Count : {len(values)}")
        print(f"Mean  : {mean(values)}")
        print(f"Std   : {std(values)}")
        print(f"Var   : {variance(values)}")
        print(f"Min   : {minimum(values)}")
        print(f"25%   : {percentile(values, 25)}")
        print(f"50%   : {percentile(values, 50)}")
        print(f"75%   : {percentile(values, 75)}")
        print(f"Max   : {maximum(values)}")
        print(f"Range : {maximum(values) - minimum(values)}")
        print(f"IQR   : {percentile(values, 75) - percentile(values, 25)}")


if __name__ == "__main__":
    main()
    