import matplotlib.pyplot as plt
import numpy as np

def plot_histogram(y, label=None):
    bin_centers = sorted(y.unique())
    edges = [x - 0.5 for x in bin_centers] + [bin_centers[-1] + 0.5]
    y.hist(bins=edges, align='mid', density=True, rwidth=0.8)
    plt.xlabel(label if label else 'Value')
    plt.ylabel('Density')
    plt.title('Histogram of ' + (label if label else 'Histogram'))
    plt.show()


def main():
    import pandas as pd
    import numpy as np

    # Example data
    data = pd.Series(np.random.randint(0, 10, size=1000))
    plot_histogram(data, label='Random Integers')


if __name__ == "__main__":
    main()