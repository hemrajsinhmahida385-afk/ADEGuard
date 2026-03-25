import numpy as np
from sklearn.decomposition import PCA

print("Loading embeddings...")
embeddings = np.load("ADEGuard/notebook/sbert_minilm_embeddings_split.npy")

print("Running PCA...")
pca = PCA(n_components=2)
emb2d = pca.fit_transform(embeddings)

print("Saving PCA...")
np.save("ADEGuard/notebook/pca.npy", emb2d)

print("✅ Done!")
