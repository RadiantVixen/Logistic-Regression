import sys
import matplotlib.pyplot as plt
from utils.csv_reader import read_csv

def main():
    if len(sys.argv) != 3:
        print("Usage: python histogram.py dataset.csv feature")
        return

    path = sys.argv[1]
    feature = sys.argv[2]

    data = read_csv(path)

    house_colors = {
        'Gryffindor': '#FF0000',
        'Slytherin': '#00FF00',
        'Ravenclaw': '#0000FF',
        'Hufflepuff': '#FFD700'
    }

    house_data = {house: [] for house in house_colors}

    for row in data:
        house = row.get('Hogwarts House')
        val = row.get(feature)
        
        if house in house_data and isinstance(val, (int, float)):
            house_data[house].append(val)

    plt.figure(figsize=(10, 6))

    for house, scores in house_data.items():
        if scores:
            plt.hist(scores, bins=25, color=house_colors[house], 
                     alpha=0.5, label=house, edgecolor='none')

    plt.title(f"Distribution of Scores: {feature}", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Scores", fontsize=11, labelpad=10)
    plt.ylabel("Frequency", fontsize=11, labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
