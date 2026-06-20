import math


def mean(values):

    clean = [v for v in values if v is not None]

    return sum(clean) / len(clean)


def variance(values):

    clean = [v for v in values if v is not None]

    mu = mean(clean)

    total = 0

    for value in clean:

        total += (value - mu) ** 2

    return total / len(clean)


def std(values):

    return math.sqrt(variance(values))


def minimum(values):

    clean = [v for v in values if v is not None]

    return min(clean)


def maximum(values):

    clean = [v for v in values if v is not None]

    return max(clean)


def percentile(values, p):

    clean = sorted([v for v in values if v is not None])

    k = (len(clean) - 1) * (p / 100)

    f = int(k)
    c = f + 1

    if c >= len(clean):
        return clean[f]

    d0 = clean[f] * (c - k)
    d1 = clean[c] * (k - f)

    return d0 + d1


def get_numeric_columns(data):

    numeric_columns = []

    keys = data[0].keys()

    for key in keys:

        numeric_count = 0

        for row in data:

            if isinstance(row[key], float) and key != "Index" and key != "First Name":
                numeric_count += 1

        if numeric_count > 0:
            numeric_columns.append(key)

    return numeric_columns


def get_numeric_column(data, column_name):

    values = []

    for row in data:

        value = row[column_name]

        if isinstance(value, float):
            values.append(value)

    return values