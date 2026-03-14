import matplotlib.pyplot as plt

def make_chart( values, title, xlabel, ylabel):
    plt.plot(values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()