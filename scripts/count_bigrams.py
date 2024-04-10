import numpy as np
from scipy.sparse import coo_matrix
import pickle

# tokens = np.array([1, 2, 3], dtype=np.uint16)
tokens = np.memmap("/mnt/ssd-1/pile_preshuffled/standard/document.bin", dtype=np.uint16, mode="r")[:1_000_000]
bigrams = np.lib.stride_tricks.sliding_window_view(tokens, 2).view(np.uint32).squeeze()

counts = np.bincount(bigrams, minlength=2**32).reshape(2**16, 2**16)[:50277, :50277].T

es_bigrams = coo_matrix(counts)

with open('es_bigrams.pkl', 'wb') as f:
    pickle.dump(es_bigrams, f)


