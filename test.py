import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances

# Load the saved data
with open('piece_embeddings_stockfish.pkl', 'rb') as f:
    embeddings = pickle.load(f)
    W = embeddings['embeddings']
    labels = embeddings['piece_labels']

# Compute pairwise distances
distance_matrix = pairwise_distances(W, metric='euclidean')/10

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(distance_matrix, xticklabels=labels, yticklabels=labels, cmap='viridis', annot=True, fmt=".2f")
plt.title('Pairwise Distance Heatmap')
plt.xlabel('Pieces')
plt.ylabel('Pieces')
plt.show()
