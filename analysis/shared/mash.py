import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import polars as pl

def get_kmers_from_sequence(seq: str, kmer_size: int) -> set:
    seq_upper = seq.upper()
    return set(seq_upper[i:i+kmer_size] for i in range(len(seq_upper) - kmer_size + 1))

def compute_jaccard_index(kmers1: set, kmers2: set) -> float:
    intersection = len(kmers1.intersection(kmers2))
    union = len(kmers1.union(kmers2))
    return intersection / union if union > 0 else 0.0

def compute_mash_distance(jaccard_index: float, kmer_size: int) -> float:
    if jaccard_index <= 0:
        return float(1.0)  # Maximum distance if no shared kmers
    elif jaccard_index >= 1:
        return float(0.0)  # Minimum distance if all kmers are shared
    else:
        mash_distance = -1.0 / kmer_size * np.log(2 * jaccard_index / (1 + jaccard_index))
        return mash_distance

def compute_mash_matrix(sequences: dict, kmer_size=31):
    matrix = np.zeros((len(sequences), len(sequences)))
    for i, (id1, seq1) in enumerate(sequences.items()):
        kmers1 = get_kmers_from_sequence(seq1, kmer_size)
        jaccard_idx = compute_jaccard_index(kmers1, kmers1)  # Jaccard index with itself should be 1
        matrix[i, i] = compute_mash_distance(jaccard_idx, kmer_size)
        for j, (id2, seq2) in enumerate(sequences.items()):
            if i < j:  # Compute only for upper triangle (matrix is symmetric)
                kmers2 = get_kmers_from_sequence(seq2, kmer_size)
                jaccard_idx = compute_jaccard_index(kmers1, kmers2)
                mash_dist = compute_mash_distance(jaccard_idx, kmer_size)
                matrix[i, j] = mash_dist
                matrix[j, i] = mash_dist  # Symmetric entry
    return matrix, list(sequences.keys())

def cluster_by_mash(matrix: np.ndarray, labels: list, threshold: float):
    # Convert the distance matrix to a condensed format for clustering
    condensed_matrix = squareform(matrix)
    # Perform hierarchical clustering
    Z = linkage(condensed_matrix, method='average')
    # Assign cluster labels based on the specified threshold
    cluster_labels = fcluster(Z, t=threshold, criterion='distance')
    clusters = {}
    for label, cluster_id in zip(labels, cluster_labels):
        clusters.setdefault(cluster_id, []).append(label)
    return clusters

def melt_mash_matrix(matrix: np.ndarray, labels: list) -> pl.DataFrame:
    melted = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            melted.append((labels[i], labels[j], matrix[i, j]))
    return pl.DataFrame(melted, schema=["label1", "label2", "mash_distance"])
