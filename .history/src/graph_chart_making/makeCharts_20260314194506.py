import matplotlib.pyplot as plt

def make_chart( values, title, xlabel, ylabel):
    plt.scatter(range(len(values)), values, s=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()