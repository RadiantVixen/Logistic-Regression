from utils.stats import mean, std


def normalize(values):

    mu = mean(values)
    sigma = std(values)

    normalized = []

    for value in values:

        if sigma == 0:
            normalized.append(0)

        else:
            normalized.append((value - mu) / sigma)

    return normalized


def normalize_features(data, features):

    stats = {}

    for feature in features:

        values = []

        for row in data:

            value = row[feature]

            if isinstance(value, float):
                values.append(value)

        mu = mean(values)
        sigma = std(values)

        stats[feature] = (mu, sigma)

        for row in data:

            value = row[feature]

            if isinstance(value, float):

                if sigma == 0:
                    row[feature] = 0

                else:
                    row[feature] = (value - mu) / sigma

    return data, stats