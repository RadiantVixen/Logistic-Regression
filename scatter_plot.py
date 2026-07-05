import sys
import matplotlib.pyplot as plt
from utils.csv_reader import read_csv

def main():
    if len(sys.argv) != 4:
        print("Usage: python scatter_plot.py dataset.csv feature1 feature2")
        return

    path = sys.argv[1]
    feature_x = sys.argv[2]
    feature_y = sys.argv[3]

    data = read_csv(path)

    house_colors = {
        'Gryffindor': '#FF0000',   
        'Slytherin': '#00FF00',    
        'Ravenclaw': '#0000FF',   
        'Hufflepuff': '#FFD700'   
    }
    default_color = '#808080'

    x_values = []
    y_values = []
    plot_colors = []

    for row in data:
        x = row.get(feature_x)
        y = row.get(feature_y)
        house = row.get('Hogwarts House')

        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            x_values.append(x)
            y_values.append(y)
            plot_colors.append(house_colors.get(house, default_color))

    if not x_values:
        print(f"Error: No valid numerical data found for pairs: {feature_x} vs {feature_y}")
        return

    plt.figure(figsize=(10, 7))
    
    scatter = plt.scatter(x_values, y_values, c=plot_colors, s=15, alpha=0.7, edgecolors='none')

    plt.xlabel(feature_x, fontsize=11, labelpad=10)
    plt.ylabel(feature_y, fontsize=11, labelpad=10)
    plt.title(f"Scatter Plot: {feature_x} vs {feature_y}", fontsize=14, weight='bold', pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=house) 
               for house, color in house_colors.items()]
    plt.legend(handles=handles, loc='lower left', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()