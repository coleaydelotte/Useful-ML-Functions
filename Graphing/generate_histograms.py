import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

def generate_histograms(*, data, column_names=[], bins=30, figsize=(10, 5), ignore_warnings=False):
    if ignore_warnings:
        import warnings
        warnings.filterwarnings("ignore")
        
    _, ax = plt.subplots(len(column_names), 1, figsize=figsize)
    for i, column_name in enumerate(column_names):
        ax[i].hist(data[column_name], bins=bins)
        ax[i].set_title(column_name)
    plt.show()

def main():
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data["target"] = iris.target
    generate_histograms(data=data, column_names=iris.feature_names, bins=30, figsize=(10, 5), ignore_warnings=True)

if __name__ == "__main__":
    main()