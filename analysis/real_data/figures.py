"""Generate the manuscript real-data figure PDFs."""
from pathlib import Path

from .manuscript_plot import (
    pl, np, plt, sns, parse_fasta_file, load_json, compute_mash_matrix,
    short_genus, mash_tree_rows, violin_plots_split_vertical,
)
from ..shared.mosquito_ids import return_species_from_target_id

ROOT = Path(__file__).resolve().parents[2]


def main(show=False):
    FIGURES = ROOT / "figures"
    FIGURES.mkdir(exist_ok=True)
    kmer_size = 31
    mosquito_fasta_file = str(ROOT / "data/real_data/mosquito_references.fasta")
    covid_fasta_file = str(ROOT / "data/real_data/sars_cov2_references.fasta")
    samples_in_json = load_json(ROOT / "config/samples.updated.json")["mosquito"]["samples"].keys()

    seqs = parse_fasta_file(mosquito_fasta_file)
    mash_matrix_mos, mos_full_labels = compute_mash_matrix(seqs, kmer_size)
    print(mash_matrix_mos.shape, len(mos_full_labels))
    mosquito_species = [return_species_from_target_id(label).replace(" ", "_") for label in mos_full_labels]
    mosquito_short_labels = [short_genus(label).replace("_", " ") for label in mosquito_species]
    print(len(mos_full_labels), len(mosquito_short_labels))
    for i, (label, short_label) in enumerate(zip(mos_full_labels, mosquito_short_labels)):
        if "pipiens" in label:
            if "molestus" in label:
                mosquito_short_labels[i] = "Cx. pipiens molestus"
            elif "pipiens" in label:
                mosquito_short_labels[i] = "Cx. pipiens pipiens"
    print(len(mos_full_labels), len(mosquito_short_labels))

    seqs = parse_fasta_file(covid_fasta_file)
    mash_matrix_covid, labels = compute_mash_matrix(seqs, kmer_size)
    covid_short_labels = [label.split("_")[-1] for label in labels]

    # plot histograms of pairwise mash distances for mosquito and covid matrices
    fig, axs = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    axs = axs.flatten()
    mos_vals = mash_matrix_mos[np.triu_indices_from(mash_matrix_mos, k=1)]
    cov_vals = mash_matrix_covid[np.triu_indices_from(mash_matrix_covid, k=1)]
    all_vals = np.concatenate([mos_vals, cov_vals])
    positive_vals = all_vals[all_vals > 0]
    min_bin = max(positive_vals.min(), 1e-4)
    max_bin = positive_vals.max()
    bins = np.logspace(np.log10(min_bin), np.log10(max_bin), 20)
    axs[0].hist(mos_vals[mos_vals > 0], bins=bins, color="steelblue", edgecolor="black")
    axs[1].hist(cov_vals[cov_vals > 0], bins=bins, color="salmon", edgecolor="black")
    for ax in axs:
        ax.set_xscale("symlog", linthresh=0.01)
        ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.45)
        sns.despine(ax=ax, left=True, bottom=True, right=True, top=True)
    tick_positions = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1]
    tick_labels = ['0.001', '0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '1']
    for ax in axs:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
    axs[0].set_title("Mosquito COX1", fontsize=14, weight="semibold")
    axs[1].set_title("SARS-Cov-2", fontsize=14, weight="semibold")
    axs[1].set_xlabel("Pairwise inter-reference mash distance", fontsize=12)
    axs[0].set_ylabel("Frequency", fontsize=12)
    axs[1].set_ylabel("Frequency", fontsize=12)
    # # invert x axis so smaller distances are on the right
    # for ax in axs:
    #     ax.set_xlim(0.5, min_bin)
    plt.tight_layout()
    fig.savefig(FIGURES / "figure_6b.pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close("all")

    sample_json = ROOT / "config/samples.updated.json"
    fig, axs = mash_tree_rows(
            mash_matrix_mos, mash_matrix_covid, mosquito_short_labels, covid_short_labels
        )
    # make leaf labels smaller
    for ax in axs:
        ax.tick_params(axis="y", labelsize=8)
    # bold leaves that are in the JSON sample list
    mos_samples_in_json = load_json(sample_json)["mosquito"]["samples"].keys()
    covid_samples_in_json = load_json(sample_json)["SARS-Cov2"]["samples"].keys()
    mos_samples_in_json = [short_genus(sample).replace("_", " ") for sample in mosquito_species if sample in samples_in_json]
    for ax in axs:
        for label in ax.get_yticklabels():
            # make larger font
            label.set_fontsize(12)
            if label.get_text() in mos_samples_in_json or label.get_text() in covid_samples_in_json:
                label.set_fontweight("bold")
    # change x axis label to "Mash distance (symmetric log scale)"
    axs[1].set_xlabel("Mash distance\n(symmetric log scale)", fontsize=16)
    fig.savefig(FIGURES / "figure_6a.pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close("all")

    mixture_results = ROOT / "data/real_data/mixture_abundance_results.tsv"
    mixture_df = pl.read_csv(mixture_results, separator="\t")
    mixture_df = mixture_df.with_columns(
        pl.col("kallisto_log_error").abs().alias("kallisto_log_error_abs"),
        pl.col("kma_log_error").abs().alias("kma_log_error_abs"),
        ((pl.col("kallisto_total_estimated") - (pl.col("kallisto_count_A") + pl.col("kallisto_count_B")))/pl.col("kallisto_total_estimated")).alias("kallisto_total_estimated_error"),
        ((pl.col("kma_total_assigned") - (pl.col("kma_count_A") + pl.col("kma_count_B")))/pl.col("kma_total_assigned")).alias("kma_total_estimated_error")
    )
    # also fix dataset column for plotting
    dataset_label_expr = (
        pl.when(pl.col("dataset") == "covid").then(pl.lit("SARS-Cov2"))
        .when(pl.col("dataset") == "mosquito").then(pl.lit("Mosquito COX1"))
        .otherwise(pl.col("dataset"))
        .alias("dataset")
    )
    mixture_df = mixture_df.with_columns(dataset_label_expr)

    fig, ax = violin_plots_split_vertical(mixture_df, x_col_suffix="_total_estimated_error", ylabel="proportion of assigned reads incorrect")
    fig.savefig(FIGURES / "figure_7a.pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close("all")

    fig, ax = violin_plots_split_vertical(mixture_df, x_col_suffix="_log_error_abs", ylabel="absolute log-ratio error")
    fig.savefig(FIGURES / "figure_7b.pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
