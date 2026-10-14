from ..shared.mash import compute_mash_matrix, melt_mash_matrix
from .process import (
    report_mash_matrix_stats, filter_seqs_by_json, parse_fasta_file, 
    compute_mash_dists_between_samples_and_correct_ref, run_kallisto_and_kma_id_on_data,
    merge_mash_and_abundance_results, short_genus, load_json
)
from .plot import (
    mash_distance_heatmap, cluster_tree_and_heatmap, 
    box_plots, stacked_ID_bar_plot,
    ID_scatter, box_plots_split,
    mash_distance_heatmaps_multiple,
    violin_plots_split_horizontal,
    violin_plots_split_vertical,
    mash_tree_rows,
    example_stacked_bar_plot,
    example_stacked_bar_plot_assigned_only,
    ID_DM_2D_example
)
from .align import align_covid_samples_to_refs
from ..shared.mosquito_ids import return_species_from_target_id
from .mixture import make_mixtures_for_pairs_in_set_and_perform_abundance
import polars as pl
import numpy as np
import os
import seaborn as sns
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run the original real-data mixture analysis.")
    parser.add_argument("--samples", default="config/samples.updated.json")
    parser.add_argument("--output-dir", default="results/dynamics_poster")
    parser.add_argument("--mosquito-fasta", required=True)
    parser.add_argument("--covid-fasta", required=True)
    args = parser.parse_args()
    output_dir = args.output_dir
    sample_json = args.samples
    kmer_size = 31
    mosquito_fasta_file = args.mosquito_fasta
    covid_fasta_file = args.covid_fasta
    # make output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "align"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "mixture"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tmp"), exist_ok=True)

    # ## ALIGN COVID TO MAKE COMPARABLE TO MOSQUITO
    # align_covid_samples_to_refs(sample_json, covid_fasta_file)

    # ## MASH ANALYSIS
    # # mosquito cox1
    # sequences = filter_seqs_by_json(mosquito_fasta_file, selection_json, "clustered_0.995_pid_trimmed")
    # mash_matrix_mos, labels = compute_mash_matrix(sequences, kmer_size)
    # report_mash_matrix_stats(mash_matrix_mos, labels)
    # mosquito_species = [return_species_from_target_id(label).replace(" ", "_") for label in labels]
    # mosquito_mash_melted = melt_mash_matrix(mash_matrix_mos, mosquito_species)
    # mosquito_mash_melted = mosquito_mash_melted.with_columns(
    #     pl.lit("mosquito").alias("dataset")
    # )
    # # fig, ax = mash_distance_heatmap(mash_matrix_mos, labels)
    # # fig.savefig(os.path.join(output_dir, "plots/mosquito_mash_distance_heatmap.pdf"), dpi=300, bbox_inches="tight", format="pdf")
    # # covid
    # seqs = parse_fasta_file(covid_fasta_file)
    # mash_matrix_covid, labels = compute_mash_matrix(seqs, kmer_size)
    # covid_short_labels = [label.split("_")[-1] for label in labels]
    # report_mash_matrix_stats(mash_matrix_covid, labels)
    # covid_mash_melted = melt_mash_matrix(mash_matrix_covid, labels)
    # covid_mash_melted = covid_mash_melted.with_columns(
    #     pl.lit("covid").alias("dataset")
    # )
    # mash_melted = pl.concat([mosquito_mash_melted, covid_mash_melted])
    # # log transform the mash distance for better visualization in box plots
    # mash_melted = mash_melted.with_columns(
    #     pl.col("mash_distance").map_elements(lambda x: np.log10(x + 1e-10) if x > 0 else 0).alias("mash_distance_log")
    # )
    # mash_melted.write_csv(os.path.join(output_dir, "results/pairwise_mash_distances_within_refs.tsv"), separator="\t")
    # # fig, ax = mash_distance_heatmap(mash_matrix_covid, labels)
    # # fig.savefig(os.path.join(output_dir, "plots/covid_mash_distance_heatmap.pdf"), dpi=300, bbox_inches="tight", format="pdf")
    # os.makedirs(os.path.join(output_dir, "tmp"), exist_ok=True)
    # df = compute_mash_dists_between_samples_and_correct_ref([mosquito_fasta_file, covid_fasta_file], sample_json, 31, tmp_dir=os.path.join(output_dir, "tmp"))
    # df = df.with_columns(
    #     pl.when(pl.col("sample").str.contains("_")).then(pl.lit("mosquito")).otherwise(pl.lit("covid")).alias("dataset")
    # )
    # df.write_csv(os.path.join(output_dir, "results/mash_distance_sample_to_correct_ref.tsv"), separator="\t")
    # # plot a combined heatmap
    # samples_in_json = load_json(sample_json)["mosquito"]["samples"].keys()
    # mosquito_samples_in_json = [short_genus(sample).replace("_", " ") for sample in mosquito_species if sample in samples_in_json]
    # mosquito_short_labels = [short_genus(label).replace("_", " ") for label in mosquito_species]
    # print(mosquito_samples_in_json)
    # fig, axs = mash_distance_heatmaps_multiple(
    #     [mash_matrix_mos, mash_matrix_covid],
    #     [mosquito_short_labels, covid_short_labels],
    #     log_values=True, log_base=10,
    #     titles = ["Mosquito COX1", "SARS-Cov2"],
    #     tick_labels_to_show = [mosquito_samples_in_json, None]
    # )
    # fig.savefig(os.path.join(output_dir, "plots/mash_distance_heatmaps_multiple.pdf"), dpi=300, bbox_inches="tight", format="pdf")
    
    # # filter mos matrix and labels to just those in the json for the tree rows plot
    # indices_to_keep = [i for i, label in enumerate(mosquito_short_labels) if label in mosquito_samples_in_json]
    # mash_matrix_mos_filtered = mash_matrix_mos[np.ix_(indices_to_keep, indices_to_keep)]
    # mosquito_short_labels_filtered = [mosquito_short_labels[i] for i in indices_to_keep]
    # # add "subtype 1" and so on to mosquitoes if there are multiple samples with the same short label to make them distinguishable in the tree rows plot
    # label_counts = {}    
    # for i, label in enumerate(mosquito_short_labels_filtered):
    #     if label not in label_counts:
    #         label_counts[label] = 0
    #     label_counts[label] += 1
    #     if label_counts[label] > 1:
    #         mosquito_short_labels_filtered[i] = f"{label} subtype {label_counts[label]}"
    # # remove multiple subtypes from mosquito matrix and labels
    # indices_to_keep = [i for i, label in enumerate(mosquito_short_labels_filtered) if "subtype" not in label]
    # mash_matrix_mos_filtered = mash_matrix_mos_filtered[np.ix_(indices_to_keep, indices_to_keep)]
    # mosquito_short_labels_filtered = [mosquito_short_labels_filtered[i] for i in indices_to_keep]
    # fig, axs = mash_tree_rows(
    #     mash_matrix_mos_filtered, mash_matrix_covid, mosquito_short_labels_filtered, covid_short_labels
    # )
    # fig.savefig(os.path.join(output_dir, "plots/mash_tree_rows.pdf"), dpi=300, bbox_inches="tight", format="pdf")
    # # ## ABUNDANCE AND ID ANALYSIS
    ## ADDITION - make a version of the mosquito fasta file with just sequences in the JSON + Culex quinquefasciatus and Cx australicus
    mosquito_seqs = parse_fasta_file(mosquito_fasta_file)
    new_mosquito_fasta_file = os.path.join(output_dir, "tmp/mosquito_fasta_for_ID.fasta")
    samples_in_json = load_json(sample_json)["mosquito"]["samples"].keys()
    filtered_mosquito_seqs = {}
    for seq_id in mosquito_seqs.keys():
        if "consensus" in seq_id:
            continue
        species = return_species_from_target_id(seq_id).replace(" ", "_")
        if species in samples_in_json:
            filtered_mosquito_seqs[seq_id] = mosquito_seqs[seq_id]
    # add Culex quinquefasciatus and Cx australicus to the filtered mosquito seqs
    culex_quin_seq = [seq_id for seq_id in mosquito_seqs.keys() if "Culex_quinquefasciatus" in seq_id]
    culex_australicus_seq = [seq_id for seq_id in mosquito_seqs.keys() if "Culex-australicus" in seq_id]
    for seq_id in culex_quin_seq:
        filtered_mosquito_seqs[seq_id] = mosquito_seqs[seq_id]
    for seq_id in culex_australicus_seq:
        filtered_mosquito_seqs[seq_id] = mosquito_seqs[seq_id]
    print(f"Number of mosquito sequences in original fasta: {len(mosquito_seqs)}")
    print(f"Number of mosquito sequences in filtered fasta: {len(filtered_mosquito_seqs)}")
    with open(new_mosquito_fasta_file, "w") as f:
        for seq in filtered_mosquito_seqs:
            id = seq
            sequence = filtered_mosquito_seqs[seq]
            f.write(f">{id}\n{sequence}\n")
    # df = run_kallisto_and_kma_id_on_data([new_mosquito_fasta_file, covid_fasta_file], sample_json, tmp_dir=os.path.join(output_dir, "tmp"))
    # df.write_csv(os.path.join(output_dir, "results/ID_sample_to_correct_ref.tsv"), separator="\t")

    # ## MERGE MASH AND ABUNDANCE RESULTS FOR PLOTTING
    # df = pl.read_csv(os.path.join(output_dir, "results/ID_sample_to_correct_ref.tsv"), separator="\t")
    # df2 = pl.read_csv(os.path.join(output_dir, "results/mash_distance_sample_to_correct_ref.tsv"), separator="\t")
    # df3 = pl.read_csv(os.path.join(output_dir, "results/pairwise_mash_distances_within_refs.tsv"), separator="\t")
    # # Rename dataset labels for plotting.
    # dataset_label_expr = (
    #     pl.when(pl.col("dataset") == "covid").then(pl.lit("SARS-Cov2"))
    #     .when(pl.col("dataset") == "mosquito").then(pl.lit("Mosquito COX1"))
    #     .otherwise(pl.col("dataset"))
    #     .alias("dataset")
    # )
    # df = df.with_columns(dataset_label_expr)
    # df2 = df2.with_columns(dataset_label_expr)
    # df3 = df3.with_columns(dataset_label_expr)
    # df = merge_mash_and_abundance_results(df3, df2, df)

    # # # PLOT
    # df3 = df3.filter(pl.col("label1") != pl.col("label2")) # filter out same species in mosquitoes
    # fig, ax = box_plots(df3, hue_col="dataset", data_col="mash_distance", title="", xlabel="Mash (reference to reference (pairwise))", ylabel="", logx=True)
    # fig.savefig(os.path.join(output_dir, "plots/pairwise_mash_distances_within_refs_box_plot.pdf"), dpi=300, bbox_inches="tight", format="pdf")
    # fig, ax = box_plots(df, hue_col="dataset", data_col="mash_distance_to_correct_ref", title="", xlabel="Mash (sample to correct reference)", ylabel="")
    # fig.savefig(os.path.join(output_dir, "plots/mash_distance_sample_to_correct_ref_box_plot.pdf"), dpi=300, bbox_inches="tight", format="pdf")
    # fig, axs = stacked_ID_bar_plot(df)
    # fig.savefig(os.path.join(output_dir, "plots/ID_sample_to_correct_ref_stacked_bar_plot.pdf"), dpi=300, bbox_inches="tight", format="pdf")
    # # # fig, ax = ID_scatter(df)
    # # # fig.savefig(os.path.join(output_dir, "plots/ID_vs_mash_distance_sample_to_correct_ref_scatter_plot.pdf"), dpi=300, bbox_inches="tight", format="pdf")
    # fig, ax = box_plots_split(df, x_col_suffix="_not_assigned_to_correct_ref_prop", xlabel="identification error\n(1 - proportion assigned to correct ref)")
    # fig.savefig(os.path.join(output_dir, "plots/ID_sample_to_correct_ref_box_plots_split_by_dataset.pdf"), dpi=300, bbox_inches="tight", format="pdf")

    # MIXTURE ANALYSIS
    mixture_results = make_mixtures_for_pairs_in_set_and_perform_abundance(
        abundance_fastas=[new_mosquito_fasta_file, covid_fasta_file], 
        sample_json=sample_json, 
        output_dir=os.path.join(output_dir, "mixture")
    )
    mixture_results.write_csv(os.path.join(output_dir, "results/mixture_abundance_results.tsv"), separator="\t")
    # mixture_results = pl.read_csv(os.path.join(output_dir, "results/mixture_abundance_results.tsv"), separator="\t")
    # mixture_results = mixture_results.with_columns(
    #     pl.col("kallisto_log_error").abs().alias("kallisto_log_error_abs"),
    #     pl.col("kma_log_error").abs().alias("kma_log_error_abs"),
    #     ((pl.col("kallisto_total_estimated") - (pl.col("kallisto_count_A") + pl.col("kallisto_count_B")))/pl.col("kallisto_total_estimated")).alias("kallisto_total_estimated_error"),
    #     ((pl.col("kma_total_assigned") - (pl.col("kma_count_A") + pl.col("kma_count_B")))/pl.col("kma_total_assigned")).alias("kma_total_estimated_error")
    # )
    # # also fix dataset column for plotting
    # dataset_label_expr = (
    #     pl.when(pl.col("dataset") == "covid").then(pl.lit("SARS-Cov2"))
    #     .when(pl.col("dataset") == "mosquito").then(pl.lit("Mosquito COX1"))
    #     .otherwise(pl.col("dataset"))
    #     .alias("dataset")
    # )
    # mixture_results = mixture_results.with_columns(dataset_label_expr)
    # # print covid samples where kma log error is greater than 10
    # sample_A_where_error_high = mixture_results.filter((pl.col("dataset") == "SARS-Cov2") & (pl.col("kma_log_error_abs") > 8))["sample_A"].to_list()
    # sample_B_where_error_high = mixture_results.filter((pl.col("dataset") == "SARS-Cov2") & (pl.col("kma_log_error_abs") > 8))["sample_B"].to_list()
    # samples_kma_cant_handle = list(set(sample_A_where_error_high + sample_B_where_error_high))
    # print(samples_kma_cant_handle)
    # fig, ax = box_plots_split(mixture_results, x_col_suffix="_log_error_abs", xlabel="relative abundance error\n(|log(E)|)")
    # fig.savefig(os.path.join(output_dir, "plots/RA_log_error_in_mixtures.pdf"), dpi=300, bbox_inches="tight", format="pdf")
    # fig, ax = box_plots_split(mixture_results, x_col_suffix="_total_estimated_error", xlabel="proportion of assigned reads incorrect")
    # fig.savefig(os.path.join(output_dir, "plots/RA_total_estimated_error_in_mixtures.pdf"), dpi=300, bbox_inches="tight", format="pdf")   
    # fig, ax = violin_plots_split_horizontal(mixture_results, x_col_suffix="_log_error_abs", xlabel="relative abundance error\n(|log(E)|)")
    # fig.savefig(os.path.join(output_dir, "plots/RA_log_error_in_mixtures_violin_plots.pdf"), dpi=300, bbox_inches="tight", format="pdf")
    # fig, ax = violin_plots_split_horizontal(mixture_results, x_col_suffix="_total_estimated_error", xlabel="proportion of assigned reads incorrect")
    # fig.savefig(os.path.join(output_dir, "plots/RA_total_estimated_error_in_mixtures_violin_plots.pdf"), dpi=300, bbox_inches="tight", format="pdf")
    # fig, ax = violin_plots_split_vertical(mixture_results, x_col_suffix="_log_error_abs", ylabel="relative abundance error\n(|log(E)|)")
    # fig.savefig(os.path.join(output_dir, "plots/RA_log_error_in_mixtures_violin_plots_vertical.pdf"), dpi=300, bbox_inches="tight", format="pdf")
    # fig, ax = violin_plots_split_vertical(mixture_results, x_col_suffix="_total_estimated_error", ylabel="proportion of assigned reads incorrect")
    # fig.savefig(os.path.join(output_dir, "plots/RA_total_estimated_error_in_mixtures_violin_plots_vertical.pdf"), dpi=300, bbox_inches="tight", format="pdf")
    # fig, ax = example_stacked_bar_plot(mixture_results)
    # fig.savefig(os.path.join(output_dir, "plots/example_stacked_bar_plot_mixtures.pdf"), dpi=300, bbox_inches="tight", format="pdf")
    # fig, ax = example_stacked_bar_plot_assigned_only(mixture_results)
    # fig.savefig(os.path.join(output_dir, "plots/example_stacked_bar_plot_mixtures_assigned_only.pdf"), dpi=300, bbox_inches="tight", format="pdf")
    # # make vertical but just mosquito and just covid for total estimated and log error - 4 plots
    # fig, ax = violin_plots_split_vertical(mixture_results.filter(pl.col("dataset") == "Mosquito COX1"), x_col_suffix="_log_error_abs", ylabel="")
    # ax.set_ylim(0, 11)
    # ax.yaxis.tick_right()
    # # remove legend
    # ax.get_legend().remove()
    # fig.savefig(os.path.join(output_dir, "plots/RA_log_error_in_mixtures_violin_plots_vertical_mosquito_only.pdf"), dpi=300, bbox_inches="tight", format="pdf")
    # fig, ax = violin_plots_split_vertical(mixture_results.filter(pl.col("dataset") == "SARS-Cov2"), x_col_suffix="_log_error_abs", ylabel="")
    # ax.set_ylim(0, 11)
    # ax.yaxis.tick_right()
    # ax.get_legend().remove()
    # fig.savefig(os.path.join(output_dir, "plots/RA_log_error_in_mixtures_violin_plots_vertical_covid_only.pdf"), dpi=300, bbox_inches="tight", format="pdf")
    # fig, ax = violin_plots_split_vertical(mixture_results.filter(pl.col("dataset") == "Mosquito COX1"), x_col_suffix="_total_estimated_error", ylabel="")
    # ax.set_ylim(0, 1)
    # ax.get_legend().remove()
    # # put y axis on the right
    # fig.savefig(os.path.join(output_dir, "plots/RA_total_estimated_error_in_mixtures_violin_plots_vertical_mosquito_only.pdf"), dpi=300, bbox_inches="tight", format="pdf")
    # fig, ax = violin_plots_split_vertical(mixture_results.filter(pl.col("dataset") == "SARS-Cov2"), x_col_suffix="_total_estimated_error", ylabel="")
    # ax.set_ylim(0, 1)
    # ax.get_legend().remove()
    # fig.savefig(os.path.join(output_dir, "plots/RA_total_estimated_error_in_mixtures_violin_plots_vertical_covid_only.pdf"), dpi=300, bbox_inches="tight", format="pdf")
    # # plot a 2D scatter 
    # ID_DM_2D_example(os.path.join(output_dir, "plots"))

if __name__ == "__main__":
    main()
