from utils.stats import mean, std


def fit_normalization(data, features):
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

    return stats


def apply_normalization(data, features, stats):
    for row in data:
        for feature in features:
            value = row[feature]

            if not isinstance(value, float):
                continue

            mu, sigma = stats[feature]

            if sigma == 0:
                row[feature] = 0.0
            else:
                row[feature] = (value - mu) / sigma

    return data