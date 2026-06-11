import csv


def to_float(value):

    try:
        return float(value)

    except:
        return None


def read_csv(path):

    data = []

    with open(path, "r") as file:

        reader = csv.DictReader(file)

        for row in reader:

            clean_row = {}

            for key, value in row.items():

                if value == "":
                    clean_row[key] = None

                else:

                    number = to_float(value)

                    if number is not None:
                        clean_row[key] = number

                    else:
                        clean_row[key] = value

            data.append(clean_row)

    return data
