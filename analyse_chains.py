import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('results/ztf_train_2000/chains.pkl', 'rb') as file:
    chains = pickle.load(file)

print(chains.keys())

for i in range(chains['Rv'].shape[0]):
    plt.hist(chains['Rv'][i, :])
    plt.show()

