import sys
import matplotlib.pyplot as plt
from utils.csv_reader import read_csv
from utils.stats import get_numeric_columns

def main():
    if len(sys.argv) != 2:
        print("Usage: python pair_plot.py dataset.csv")
        return

    path = sys.argv[1]
    data = read_csv(path)
    features = get_numeric_columns(data)
    size = len(features)

    house_colors = {
        'Gryffindor': '#FF0000',   
        'Slytherin': '#00FF00',    
        'Ravenclaw': '#0000FF',   
        'Hufflepuff': '#FFD700'   
    }
    default_color = '#808080'

    colors = []
    for row in data:
        house = row.get('Hogwarts House')
        colors.append(house_colors.get(house, default_color))

    fig, axes = plt.subplots(size, size, figsize=(22, 22)) 

    for i in range(size):
        for j in range(size):
            x_feature = features[j]
            y_feature = features[i]
            ax = axes[i][j]

            x_values = []
            y_values = []
            plot_colors = []

            for k, row in enumerate(data):
                x = row.get(x_feature)
                y = row.get(y_feature)
                
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    x_values.append(x)
                    y_values.append(y)
                    plot_colors.append(colors[k])

            ax.scatter(x_values, y_values, c=plot_colors, s=1, alpha=0.6)

            ax.set_xticks([])
            ax.set_yticks([])

            if i == size - 1:
                ax.set_xlabel(x_feature, fontsize=7, rotation=45, ha='right')

            if j == 0:
                ax.set_ylabel(y_feature, fontsize=7, rotation=0, ha='right', labelpad=25)

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=house) 
               for house, color in house_colors.items()]
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.02, 0.02), fontsize=12)

    plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.15, wspace=0.4, hspace=0.4)

    plt.show()

if __name__ == "__main__":
    main()