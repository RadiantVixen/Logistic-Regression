import matplotlib.pyplot as plt


HOUSE_COLORS = {
    "Gryffindor": "red",
    "Ravenclaw": "blue",
    "Hufflepuff": "yellow",
    "Slytherin": "green"
}


def plot_histogram(values, title):

    plt.hist(values, bins=30)

    plt.title(title)

    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.show()


def plot_scatter(x, y, xlabel, ylabel):

    plt.scatter(x, y)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()