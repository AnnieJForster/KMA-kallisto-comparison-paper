"""Real-data plotting functions from the final manuscript notebook."""
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from Bio import SeqIO
import json

def mash_tree_rows(matrix1, matrix2, labels1, labels2, titles=None):
    distance_matrix1 = np.asarray(matrix1, dtype=float)
    distance_matrix2 = np.asarray(matrix2, dtype=float)
    condensed_distance1 = squareform(distance_matrix1, checks=False)
    condensed_distance2 = squareform(distance_matrix2, checks=False)
    linkage_matrix1 = linkage(condensed_distance1, method="complete", optimal_ordering=True)
    linkage_matrix2 = linkage(condensed_distance2, method="complete", optimal_ordering=True)
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True, height_ratios=[len(labels1), len(labels2)])
    axs = axs.flatten()
    # invert the tree so bottom is top for matrix 1 for visualisation
    dendrogram(
        linkage_matrix1,
        labels=labels1,
        ax=axs[0],
        leaf_rotation=0,
        leaf_font_size=8,
        orientation='left',
        above_threshold_color='k',
        color_threshold=0.0,
    )
    axs[0].invert_yaxis()
    dendrogram(
        linkage_matrix2,
        labels=labels2,
        ax=axs[1],
        leaf_rotation=0,
        leaf_font_size=8,
        orientation='left',
        above_threshold_color='k',
        color_threshold=0.0,
    )
    # Use symlog scale to spread out values near 0 for better visibility
    axs[0].set_xscale('symlog', linthresh=0.01)
    axs[1].set_xscale('symlog', linthresh=0.01)
    axs[0].set_xlim(0.5, 0)
    # Set custom tick labels in normal decimal notation instead of scientific
    tick_positions = [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    tick_labels = ['0', '0.001', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5']
    for ax in axs:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
    if titles is not None:
        axs[0].set_title(titles[0], fontsize=12, weight="semibold")
        axs[1].set_title(titles[1], fontsize=12, weight="semibold")
    # if first axis, make the leaf labels italic
    axs[0].set_yticklabels(axs[0].get_yticklabels(), fontstyle="italic")
    for ax in axs:
        ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.45)
        sns.despine(ax=ax, left=True, bottom=True, right=True, top=True)
        # make leaf labels large
        ax.tick_params(axis="y", labelsize=14)
    axs[1].set_xlabel("MASH Distance", fontsize=18)
    return fig, axs

def compute_mash_distance(jaccard_index: float, kmer_size: int) -> float:
    if jaccard_index <= 0:
        return float(1.0)  # Maximum distance if no shared kmers
    elif jaccard_index >= 1:
        return float(0.0)  # Minimum distance if all kmers are shared
    else:
        mash_distance = -1.0 / kmer_size * np.log(2 * jaccard_index / (1 + jaccard_index))
        return mash_distance
    

def parse_fasta_file(fasta_file: str) -> dict:
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences

def short_genus(species_name: str) -> str:
    if "Culex" in species_name:
        return species_name.replace("Culex", "Cx.")
    elif "Anopheles" in species_name:
        return species_name.replace("Anopheles", "An.")
    elif "Aedes" in species_name:
        return species_name.replace("Aedes", "Ae.")
    elif "Mansonia" in species_name:
        return species_name.replace("Mansonia", "Ma.")
    elif "Ochlerotatus" in species_name:
        return species_name.replace("Ochlerotatus", "Ae.")
    elif "Haemagogus" in species_name:
        return species_name.replace("Haemagogus", "Hg.")
    elif "Toxorhynchites" in species_name:
        return species_name.replace("Toxorhynchites", "Tx.")
    elif "Coquillettidia" in species_name:
        return species_name.replace("Coquillettidia", "Cq.")
    elif "Psorophora" in species_name:
        return species_name.replace("Psorophora", "Ps.")
    elif "Sabethes" in species_name:
        return species_name.replace("Sabethes", "Sa.")
    elif "Tripteroides" in species_name:
        return species_name.replace("Tripteroides", "Tr.")
    elif "Uranotaenia" in species_name:
        return species_name.replace("Uranotaenia", "Ur.")
    elif "Verallina" in species_name:
        return species_name.replace("Verallina", "Ve.")
    else:
        return species_name
    
def load_json(json_file: str) -> dict:
    with open(json_file) as f:
        data = json.load(f)
    return data

def filter_seqs_by_json(fasta_file: str, json_file: str, json_key: str) -> dict:
    seqs = parse_fasta_file(fasta_file)
    data = load_json(json_file)
    selected_ids = set(data["target_sets"][json_key])
    # filter the sequences to only those in selected_ids
    filtered_seqs = {id: seq for id, seq in seqs.items() if id in selected_ids}
    return filtered_seqs

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

def compute_jaccard_index(kmers1: set, kmers2: set) -> float:
    intersection = len(kmers1.intersection(kmers2))
    union = len(kmers1.union(kmers2))
    return intersection / union if union > 0 else 0.0

def get_kmers_from_sequence(seq: str, kmer_size: int) -> set:
    seq_upper = seq.upper()
    return set(seq_upper[i:i+kmer_size] for i in range(len(seq_upper) - kmer_size + 1))


def violin_plots_split_vertical(df, x_col_suffix="", ylabel=""):
    plot_df = pl.concat([
        df.select(
            "dataset",
            pl.col(f"kallisto{x_col_suffix}").alias("data_to_plot"),
        ).with_columns(pl.lit("Kallisto").alias("method")),
        df.select(
            "dataset",
            pl.col(f"kma{x_col_suffix}").alias("data_to_plot"),
        ).with_columns(pl.lit("KMA").alias("method")),
    ])
    plot_df_pd = plot_df.to_pandas()
    order = sorted(plot_df_pd["dataset"].dropna().unique())

    fig, ax = plt.subplots(figsize=(2.8, 4))
    sns.violinplot(
        data=plot_df_pd,
        x="dataset",
        y="data_to_plot",
        hue="method",
        order=order,
        hue_order=["Kallisto", "KMA"],
        linewidth=1.0,
        cut=0,
        inner="quart",
        dodge=True,
        palette={"Kallisto": "tab:orange", "KMA": "tab:blue"},
        ax=ax,
    )

    # Overlay jittered points with explicit offsets so they align with dodged violins.
    method_order = ["Kallisto", "KMA"]
    method_offsets = {"Kallisto": -0.2, "KMA": 0.2}
    dataset_to_x = {dataset: idx for idx, dataset in enumerate(order)}
    rng = np.random.default_rng(42)
    for method in method_order:
        method_subset = plot_df_pd[plot_df_pd["method"] == method]
        x_base = method_subset["dataset"].map(dataset_to_x).astype(float) + method_offsets[method]
        x_jittered = x_base + rng.uniform(-0.05, 0.05, size=len(method_subset))
        ax.scatter(
            x_jittered,
            method_subset["data_to_plot"],
            color="black",
            alpha=0.22,
            s=14,
            linewidths=0,
            zorder=3,
        )

    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle
    ax.legend(unique.values(), unique.keys(), frameon=False, title=None, fontsize=8)

    ax.set_xlabel("", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.tick_params(axis="both", labelsize=9)
    # sns.despine(ax=ax, left=True, bottom=True)
    plt.tight_layout()
    return fig, ax
