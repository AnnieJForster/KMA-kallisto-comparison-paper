from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import squareform
import polars as pl
from ..simulations.plot import parse_results_for_plotting, plot_scatter_with_smooth_line
import os

def mash_distance_heatmap(matrix, labels):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap="viridis", square=True, ax=ax)
    ax.set_title("MASH Distance Heatmap")
    ax.set_xlabel("Sequences")
    ax.set_ylabel("Sequences")
    fig.tight_layout()
    return fig, ax

def mash_distance_heatmaps_multiple(matrices, labels_list, log_values=False, log_offset=1e-6, log_base=10, titles=None, tick_labels_to_show=[]):
    width_ratios = [1, 0.1]*len(matrices)
    width_ratios[-1] = 0.01
    fig, axs = plt.subplots(1, (len(matrices)*2) + 1, figsize=(12, 5.5), gridspec_kw={'width_ratios': width_ratios + [0.05], 'wspace': 0.04})
    matrices_to_plot = [np.asarray(matrix, dtype=float) for matrix in matrices]
    if log_values:
        if log_base <= 0 or log_base == 1:
            raise ValueError("log_base must be > 0 and not equal to 1")
        log_denom = np.log10(log_base)
        matrices_to_plot = [np.log10(matrix + log_offset) / log_denom for matrix in matrices_to_plot]
    mash_min = min(matrix.min() for matrix in matrices_to_plot)
    mash_max = max(matrix.max() for matrix in matrices_to_plot)
    for i, (matrix, labels) in enumerate(zip(matrices_to_plot, labels_list)):
        ax = axs[i*2]
        if tick_labels_to_show[i] is not None:
            labels = [label if label in tick_labels_to_show[i] else "" for label in labels]
        sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap="flare", square=False, ax=ax, vmin=mash_min, vmax=mash_max, cbar=False)
        if titles:
            ax.set_title(titles[i], fontsize=12, weight="semibold")
        # make bottom axis labels rotated 
        ax.set_xticklabels([])
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)
        # make empty axis in i+1 position for colorbar
        axs[(i*2)+1].axis("off")
    # add a single colorbar for all heatmaps
    norm = plt.Normalize(mash_min, mash_max)
    sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
    sm.set_array([])
    heatmap_pos = axs[0].get_position()
    cbar_pos = axs[-1].get_position()
    axs[-1].set_position([cbar_pos.x0, heatmap_pos.y0, cbar_pos.width, heatmap_pos.height])
    cbar = fig.colorbar(sm, cax=axs[-1], label=None)
    cbar.ax.set_position([cbar_pos.x0, heatmap_pos.y0, cbar_pos.width, heatmap_pos.height])
    cbar.outline.set_visible(False)
    if log_values:
        tick_values = [0, 0.001, 0.01, 0.1, 0.5, 1.0]
        log_tick_values = np.log10(np.array(tick_values) + log_offset) / np.log10(log_base)
        cbar.set_ticks(log_tick_values)
        cbar.set_ticklabels([str(tick) for tick in tick_values])
    fig.subplots_adjust(wspace=0.1)
    return fig, axs

def cluster_tree_and_heatmap(matrix, labels):
    distance_matrix = np.asarray(matrix, dtype=float)
    condensed_distance = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed_distance, method="average", optimal_ordering=True)
    g = sns.clustermap(
        distance_matrix,
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        figsize=(10, 10),
    )
    # make tick labels smaller
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=8)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)
    g.ax_heatmap.set_title("Clustered MASH Distance Heatmap")
    plt.show()

def box_plots(df, hue_col="dataset", data_col="mash_distance", title="", xlabel="", ylabel="", logx=False):
    linear_tick_values = None
    log_offset = 1e-3
    if logx:
        linear_tick_values = np.array([0.0, 0.001, 0.01, 0.1, 0.5, 1.0], dtype=float)
        df = df.with_columns(
            pl.col(data_col).map_elements(
                lambda x: np.log10(x + log_offset) if x >= 0 else np.nan
            ).alias(f"{data_col}_log")
        )
        data_col = f"{data_col}_log"
    plot_df = df.to_pandas()
    order = sorted(plot_df[hue_col].dropna().unique())
    fig, ax = plt.subplots(figsize=(6,2.5))
    sns.boxplot(
        data=plot_df,
        y=hue_col,
        x=data_col,
        order=order,
        orient="h",
        width=0.6,
        linewidth=1.1,
        fliersize=2,
        color='dimgray',
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        y=hue_col,
        x=data_col,
        order=order,
        orient="h",
        color="black",
        alpha=0.22,
        size=2.3,
        jitter=0.18,
        ax=ax,
    )
    ax.set_title(title, fontsize=13, weight="semibold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    # if log x, make sure ticks are labelled with linear values for better interpretability
    if logx and linear_tick_values is not None:
        tick_positions = np.log10(linear_tick_values + log_offset)
        x_min, x_max = ax.get_xlim()
        in_view = (tick_positions >= x_min) & (tick_positions <= x_max)
        shown_positions = tick_positions[in_view]
        shown_values = linear_tick_values[in_view]
        ax.set_xticks(shown_positions)
        label_map = {0.0: "0", 0.001: "0.001", 0.01: "0.01", 0.1: "0.1", 0.5: "0.5", 1.0: "1"}
        ax.set_xticklabels([label_map[float(tick)] for tick in shown_values])
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.tick_params(axis="both", labelsize=9)
    sns.despine(ax=ax, left=False, bottom=False)
    plt.tight_layout()
    return fig, ax

def stacked_ID_bar_plot(df):
    # sort ascneding mash distance to correct ref within each dataset for better visualization in the bar plot
    df = df.with_columns(
        pl.col("mash_distance_to_correct_ref").rank("dense").over("dataset").alias("mash_distance_rank")
    ).sort(["dataset", "mash_distance_rank"])
    # build plot
    n_mosquito_samples = df.filter(pl.col("dataset") == "Mosquito COX1").shape[0]
    n_covid_samples = df.filter(pl.col("dataset") == "SARS-Cov2").shape[0]
    # top row has two axes; bottom row is a single axis spanning both columns
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[2, 1],
        width_ratios=[n_covid_samples, n_mosquito_samples],
    )
    axs = np.empty((2, 2), dtype=object)
    axs[0, 0] = fig.add_subplot(gs[0, 0])
    axs[0, 1] = fig.add_subplot(gs[0, 1])
    axs[1, 0] = fig.add_subplot(gs[1, 0])
    axs[1, 1] = fig.add_subplot(gs[1, 1])
    # top plots are stacked bar plots - left mosquito, right covid
    for i, dataset in enumerate(["SARS-Cov2", "Mosquito COX1"]):
        subset = df.filter(pl.col("dataset") == dataset)
        # one bar for kallisto, one for kma - bars will be stacked: count on correct ref, count off target and count unassigned (as proportions)
        ax = axs[0, i]
        for x_tick, sample in enumerate(subset["sample"].to_list()):
            sample_row = subset.filter(pl.col("sample") == sample)
            kallisto_tick = x_tick - 0.24
            kma_tick = x_tick + 0.24
            bar_width = 0.34

            kallisto_correct = sample_row["kallisto_correct_ref_prop"].item()
            kallisto_off_target = sample_row["kallisto_off_target_prop"].item()
            kallisto_unassigned = sample_row["kallisto_unassigned_prop"].item()

            kma_correct = sample_row["kma_correct_ref_prop"].item()
            kma_off_target = sample_row["kma_off_target_prop"].item()
            kma_unassigned = sample_row["kma_unassigned_prop"].item()

            ax.bar(
                kallisto_tick,
                kallisto_correct,
                width=bar_width,
                color="tab:orange",
                edgecolor="black",
                linewidth=0.3,
                label="Kallisto Correct Ref" if x_tick == 0 else "",
            )
            ax.bar(
                kallisto_tick,
                kallisto_off_target,
                width=bar_width,
                bottom=kallisto_correct,
                color="tab:orange",
                hatch="///",
                edgecolor="black",
                linewidth=0.3,
                label="Kallisto Off Target" if x_tick == 0 else "",
            )
            ax.bar(
                kallisto_tick,
                kallisto_unassigned,
                width=bar_width,
                bottom=kallisto_correct + kallisto_off_target,
                color="tab:gray",
                edgecolor="black",
                linewidth=0.3,
                label="Kallisto Unassigned" if x_tick == 0 else "",
            )

            ax.bar(
                kma_tick,
                kma_correct,
                width=bar_width,
                color="tab:blue",
                edgecolor="black",
                linewidth=0.3,
                label="KMA Correct Ref" if x_tick == 0 else "",
            )
            ax.bar(
                kma_tick,
                kma_off_target,
                width=bar_width,
                bottom=kma_correct,
                color="tab:blue",
                hatch="///",
                edgecolor="black",
                linewidth=0.3,
                label="KMA Off Target" if x_tick == 0 else "",
            )
            ax.bar(
                kma_tick,
                kma_unassigned,
                width=bar_width,
                bottom=kma_correct + kma_off_target,
                color="tab:gray",
                edgecolor="black",
                linewidth=0.3,
                label="KMA Unassigned" if x_tick == 0 else "",
            )
    # now on bottom plots add a line bar plot where y axis is mash distance, left line is mash distance to correct ref, right line is min mash distance to any ref; one line for mosquito, one for covid
    for i, dataset in enumerate(["SARS-Cov2", "mosquito"]):
        subset = df.filter(pl.col("dataset") == dataset)
        ax = axs[1, i]
        ax.plot(
            subset["sample"].to_list(),
            subset["mash_distance_to_correct_ref"].to_list(),
            color="tab:green",
            marker="o",
            label="MASH Distance to Correct Ref",
        )
        ax.plot(
            subset["sample"].to_list(),
            subset["min_mash_distance_to_any_ref"].to_list(),
            color="tab:purple",
            marker="o",
            label="Min MASH Distance to Any Ref",
        )
        ax.set_xticks(range(len(subset)))
        ax.set_xticklabels(subset["sample"].to_list(), rotation=90, fontsize=8)
        ax.set_ylabel("MASH Distance", fontsize=10)
        ax.set_title(f"{dataset} Samples", fontsize=12, weight="semibold")
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    fig.tight_layout()
    return fig, axs
    
def ID_scatter(df):
    # two scatters - left is mash distance to correct ref and proportion unassigned, kallisto in orange, kma in blue; mosquito x, covid .
    # rigt is min mash distance to any ref and proportion off target, same colors and markers
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # add kallisto points in orange
    sns.scatterplot(
        data=df.to_pandas(),
        x="mash_distance_to_correct_ref",
        y="kallisto_correct_ref_prop",
        hue=None,
        style="dataset",
        color="tab:orange",
        ax=axs[0],
        s=100,
        edgecolor="black",
        linewidth=0.5,
    )
    # add kma points in blue
    sns.scatterplot(
        data=df.to_pandas(),
        x="mash_distance_to_correct_ref",
        y="kma_correct_ref_prop",
        hue=None,
        style="dataset",
        color="tab:blue",
        ax=axs[0],
        s=100,
        edgecolor="black",
        linewidth=0.5,
    )
    # kallisto points in orange
    sns.scatterplot(
        data=df.to_pandas(),
        x="min_mash_distance_to_any_ref",
        y="kallisto_correct_ref_prop",
        hue=None,
        style="dataset",
        color="tab:orange",
        ax=axs[1],
        s=100,
        edgecolor="black",
        linewidth=0.5,
    )
    # kma points in blue
    sns.scatterplot(
        data=df.to_pandas(),
        x="min_mash_distance_to_any_ref",
        y="kma_correct_ref_prop",
        hue=None,
        style="dataset",
        color="tab:blue",
        ax=axs[1],
        s=100,
        edgecolor="black",
        linewidth=0.5,
    )
    # log the x axes for better visualization
    axs[1].set_xscale("log")
    return fig, axs
                   
def box_plots_split(df, x_col_suffix="", xlabel=""):
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

    fig, ax = plt.subplots(figsize=(6, 2.5))
    sns.boxplot(
        data=plot_df_pd,
        y="dataset",
        x="data_to_plot",
        hue="method",
        order=order,
        hue_order=["Kallisto", "KMA"],
        orient="h",
        width=0.6,
        linewidth=1.1,
        fliersize=2,
        dodge=True,
        palette={"Kallisto": "tab:orange", "KMA": "tab:blue"},
        ax=ax,
    )
    # Use explicit y offsets so point overlays align exactly with dodged boxes.
    method_order = ["Kallisto", "KMA"]
    method_offsets = {"Kallisto": -0.15, "KMA": 0.15}
    dataset_to_y = {dataset: idx for idx, dataset in enumerate(order)}
    rng = np.random.default_rng(42)
    for method in method_order:
        method_subset = plot_df_pd[plot_df_pd["method"] == method]
        y_base = method_subset["dataset"].map(dataset_to_y).astype(float) + method_offsets[method]
        y_jittered = y_base + rng.uniform(-0.06, 0.06, size=len(method_subset))
        ax.scatter(
            method_subset["data_to_plot"],
            y_jittered,
            color="black",
            alpha=0.22,
            s=14,
            linewidths=0,
            zorder=3,
        )

    # Keep one legend entry per method after overlaying the stripplot.
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle
    ax.legend(unique.values(), unique.keys(), frameon=False, title=None, fontsize=8)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("", fontsize=11)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.tick_params(axis="both", labelsize=9)
    sns.despine(ax=ax, left=False, bottom=False)
    plt.tight_layout()
    return fig, ax

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

def violin_plots_split_horizontal(df, x_col_suffix="", xlabel=""):
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

    fig, ax = plt.subplots(figsize=(4.8, 2.5))
    sns.violinplot(
        data=plot_df_pd,
        y="dataset",
        x="data_to_plot",
        hue="method",
        order=order,
        hue_order=["Kallisto", "KMA"],
        linewidth=1.0,
        cut=0,
        inner="quart",
        dodge=True,
        orient="h",
        palette={"Kallisto": "tab:orange", "KMA": "tab:blue"},
        ax=ax,
    )

    # Overlay jittered points with explicit offsets so they align with dodged violins.
    method_order = ["Kallisto", "KMA"]
    method_offsets = {"Kallisto": -0.2, "KMA": 0.2}
    dataset_to_y = {dataset: idx for idx, dataset in enumerate(order)}
    rng = np.random.default_rng(42)
    for method in method_order:
        method_subset = plot_df_pd[plot_df_pd["method"] == method]
        y_base = method_subset["dataset"].map(dataset_to_y).astype(float) + method_offsets[method]
        y_jittered = y_base + rng.uniform(-0.05, 0.05, size=len(method_subset))
        ax.scatter(
            method_subset["data_to_plot"],
            y_jittered,
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

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("", fontsize=11)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.tick_params(axis="both", labelsize=9)
    sns.despine(ax=ax, left=False, bottom=False)
    plt.tight_layout()
    return fig, ax

def mash_tree_rows(matrix1, matrix2, labels1, labels2, titles=None):
    distance_matrix1 = np.asarray(matrix1, dtype=float)
    distance_matrix2 = np.asarray(matrix2, dtype=float)
    condensed_distance1 = squareform(distance_matrix1, checks=False)
    condensed_distance2 = squareform(distance_matrix2, checks=False)
    linkage_matrix1 = linkage(condensed_distance1, method="average", optimal_ordering=True)
    linkage_matrix2 = linkage(condensed_distance2, method="average", optimal_ordering=True)
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

def example_stacked_bar_plot(df):
    # select a random pair of mosquito and covid samples
    mosquito_pair = df.filter(pl.col("dataset") == "Mosquito COX1").filter(pl.col("proportion_A") == 0.5).sample(1, seed=42)
    covid_pair = df.filter(pl.col("dataset") == "SARS-Cov2").filter(pl.col("proportion_A") == 0.5).sample(1, seed=42)
    # print two members of each pair to verify they are different samples
    print("Mosquito sample:", mosquito_pair.select(["sample_A", "sample_B"]))
    print("COVID sample:", covid_pair.select(["sample_A", "sample_B"]))
    # for each, plot three stacked bar plots, expected proportion of A and B (50/50), kallisto assigned proportion of A and B, kma assigned proportion of A and B
    fig, axs = plt.subplots(1, 2, figsize=(5, 4))
    # Use spaced colors from the flare palette for clear, cohesive stacking.
    flare_palette = sns.color_palette("flare", 8).as_hex()
    A_colour = flare_palette[1]
    B_colour = flare_palette[3]
    incorrect_colour = "#8b1e3f"
    unassigned_colour = "#8e939b"
    for i, (dataset, subset) in enumerate(zip(["Mosquito COX1","SARS-Cov2"], [mosquito_pair, covid_pair])):
        expected_A = subset["proportion_A"].item()
        expected_B = 1 - expected_A
        total_reads_input = subset["expected_count_A"].item() + subset["expected_count_B"].item()
        kallisto_count_A = subset["kallisto_count_A"].item()
        kallisto_count_B = subset["kallisto_count_B"].item()
        kallisto_total_assigned = subset["kallisto_total_estimated"].item()
        kallisto_propA = kallisto_count_A / total_reads_input if total_reads_input > 0 else 0
        kallisto_propB = kallisto_count_B / total_reads_input if total_reads_input > 0 else 0
        kallisto_prop_assigned_incorrect = (kallisto_total_assigned - (kallisto_count_A + kallisto_count_B)) / total_reads_input if total_reads_input > 0 else 0
        kallisto_prop_unassigned = 1 - (kallisto_total_assigned / total_reads_input) if total_reads_input > 0 else 0
        kallisto_prop_unassigned = max(0, kallisto_prop_unassigned)
        kma_count_A = subset["kma_count_A"].item()
        kma_count_B = subset["kma_count_B"].item()
        kma_total_assigned = subset["kma_total_assigned"].item()
        kma_propA = kma_count_A / total_reads_input if total_reads_input > 0 else 0
        kma_propB = kma_count_B / total_reads_input if total_reads_input > 0 else 0
        kma_prop_assigned_incorrect = (kma_total_assigned - (kma_count_A + kma_count_B)) / total_reads_input if total_reads_input > 0 else 0
        kma_prop_unassigned = 1 - (kma_total_assigned / total_reads_input) if total_reads_input > 0 else 0
        kma_prop_unassigned = max(0, kma_prop_unassigned)
        ax = axs[i]
        ax.bar(0, expected_A, width=0.5, color=A_colour, edgecolor="black", linewidth=0.3, label="A")
        ax.bar(0, expected_B, width=0.5, bottom=expected_A, color=B_colour, edgecolor="black", linewidth=0.3, label="B")
        ax.bar(1, kallisto_propA, width=0.5, color=A_colour, edgecolor="black", linewidth=0.3, label="Kallisto A")
        ax.bar(1, kallisto_propB, width=0.5, bottom=kallisto_propA, color=B_colour, edgecolor="black", linewidth=0.3, label="Kallisto B")
        ax.bar(1, kallisto_prop_assigned_incorrect, width=0.5, bottom=kallisto_propA + kallisto_propB, color=incorrect_colour, edgecolor="black", linewidth=0.3, label="Kallisto Assigned Incorrect")
        ax.bar(1, kallisto_prop_unassigned, width=0.5, bottom=kallisto_propA + kallisto_propB + kallisto_prop_assigned_incorrect, color=unassigned_colour, edgecolor="black", linewidth=0.3, label="Kallisto Unassigned")
        ax.bar(2, kma_propA, width=0.5, color=A_colour, edgecolor="black", linewidth=0.3, label="KMA A")
        ax.bar(2, kma_propB, width=0.5, bottom=kma_propA, color=B_colour, edgecolor="black", linewidth=0.3, label="KMA B")
        ax.bar(2, kma_prop_assigned_incorrect, width=0.5, bottom=kma_propA + kma_propB, color=incorrect_colour, edgecolor="black", linewidth=0.3, label="KMA Assigned Incorrect")
        ax.bar(2, kma_prop_unassigned, width=0.5, bottom=kma_propA + kma_propB + kma_prop_assigned_incorrect, color=unassigned_colour, edgecolor="black", linewidth=0.3, label="KMA Unassigned")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Expected", "Kallisto", "KMA"])
        ax.set_ylim(0, 1)
        # ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
        ax.set_title(dataset, fontsize=12, weight="semibold")
        sns.despine(ax=ax, top=True, right=True, bottom=True, left=True)
    fig.tight_layout()
    return fig, axs

def example_stacked_bar_plot_assigned_only(df):
    # select a random pair of mosquito and covid samples
    mosquito_pair = df.filter(pl.col("dataset") == "Mosquito COX1").filter(pl.col("proportion_A") == 0.5).sample(1, seed=42)
    covid_pair = df.filter(pl.col("dataset") == "SARS-Cov2").filter(pl.col("proportion_A") == 0.5).sample(1, seed=42)
    # print two members of each pair to verify they are different samples
    print("Mosquito sample:", mosquito_pair.select(["sample_A", "sample_B"]))
    print("COVID sample:", covid_pair.select(["sample_A", "sample_B"]))
    # plot stacked bars showing A, B, and incorrectly assigned as proportions of assigned reads only
    fig, axs = plt.subplots(1, 2, figsize=(5, 4))
    flare_palette = sns.color_palette("flare", 8).as_hex()
    A_colour = flare_palette[1]
    B_colour = flare_palette[3]
    incorrect_colour = "#8b1e3f"
    for i, (dataset, subset) in enumerate(zip(["Mosquito COX1","SARS-Cov2"], [mosquito_pair, covid_pair])):
        expected_A = subset["proportion_A"].item()
        expected_B = 1 - expected_A
        kallisto_count_A = subset["kallisto_count_A"].item()
        kallisto_count_B = subset["kallisto_count_B"].item()
        kallisto_total_assigned = subset["kallisto_total_estimated"].item()
        kallisto_incorrect = kallisto_total_assigned - (kallisto_count_A + kallisto_count_B)
        # proportions out of assigned reads
        kallisto_propA = kallisto_count_A / kallisto_total_assigned if kallisto_total_assigned > 0 else 0
        kallisto_propB = kallisto_count_B / kallisto_total_assigned if kallisto_total_assigned > 0 else 0
        kallisto_prop_incorrect = kallisto_incorrect / kallisto_total_assigned if kallisto_total_assigned > 0 else 0
        kma_count_A = subset["kma_count_A"].item()
        kma_count_B = subset["kma_count_B"].item()
        kma_total_assigned = subset["kma_total_assigned"].item()
        kma_incorrect = kma_total_assigned - (kma_count_A + kma_count_B)
        # proportions out of assigned reads
        kma_propA = kma_count_A / kma_total_assigned if kma_total_assigned > 0 else 0
        kma_propB = kma_count_B / kma_total_assigned if kma_total_assigned > 0 else 0
        kma_prop_incorrect = kma_incorrect / kma_total_assigned if kma_total_assigned > 0 else 0
        ax = axs[i]
        # Expected bar
        ax.bar(0, expected_A, width=0.5, color=A_colour, edgecolor="black", linewidth=0.3, label="A")
        ax.bar(0, expected_B, width=0.5, bottom=expected_A, color=B_colour, edgecolor="black", linewidth=0.3, label="B")
        # Kallisto bar
        ax.bar(1, kallisto_propA, width=0.5, color=A_colour, edgecolor="black", linewidth=0.3, label="Kallisto A")
        ax.bar(1, kallisto_propB, width=0.5, bottom=kallisto_propA, color=B_colour, edgecolor="black", linewidth=0.3, label="Kallisto B")
        ax.bar(1, kallisto_prop_incorrect, width=0.5, bottom=kallisto_propA + kallisto_propB, color=incorrect_colour, edgecolor="black", linewidth=0.3, label="Kallisto Assigned Incorrect")
        # KMA bar
        ax.bar(2, kma_propA, width=0.5, color=A_colour, edgecolor="black", linewidth=0.3, label="KMA A")
        ax.bar(2, kma_propB, width=0.5, bottom=kma_propA, color=B_colour, edgecolor="black", linewidth=0.3, label="KMA B")
        ax.bar(2, kma_prop_incorrect, width=0.5, bottom=kma_propA + kma_propB, color=incorrect_colour, edgecolor="black", linewidth=0.3, label="KMA Assigned Incorrect")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Expected", "Kallisto", "KMA"])
        ax.set_ylim(0, 1)
        ax.set_title(dataset, fontsize=12, weight="semibold")
        sns.despine(ax=ax, top=True, right=True, bottom=True, left=True)
    fig.tight_layout()
    return fig, axs

def ID_DM_2D_example(out_folder: str, simulation_results_folder: str):
    # inputs
    mash_AB_range = (0.03, 0.04)
    seq_length = 10000
    min_mutations = 1
    max_mutations = 2000
    n_simulation_sets = 10
    # parse simulation results
    file_name = f"kmer_simulation.ID_DM.seq_length={seq_length}_num_reads=1000.min_mutations={min_mutations}.max_mutations={max_mutations}.tsv"
    df = parse_results_for_plotting(file_name, simulation_results_folder, seq_length, n_simulation_sets, None)
    # remove where mash AS is less than mash SB, as then reads should be put on A rather than B
    df = df.filter(pl.col("mash_AS") >= pl.col("mash_SB"))
    # limit mash SB to less than 1
    df = df.filter(pl.col("mash_SB") < 1)
    print(df.shape[0])
    # print num of mutations in S range for mash AB range for better interpretability of plots
    mutations_in_range = df.filter((pl.col("mash_AB") >= mash_AB_range[0]) & (pl.col("mash_AB") <= mash_AB_range[1]))["num_mutations_in_B"].to_list()
    print(len(mutations_in_range))
    print("Number of mutations in S range for mash AB range:", min(mutations_in_range), "to", max(mutations_in_range))
    # filter A-B to range
    df = df.filter((pl.col("mash_AB") >= mash_AB_range[0]) & (pl.col("mash_AB") <= mash_AB_range[1]))
    # filter to where both tools are assigning at least 10 reads
    df = df.filter(((pl.col("kallisto_count_A") + pl.col("kallisto_count_B")) >= 100) & ((pl.col("kma_count_A") + pl.col("kma_count_B")) >= 100))
    # plot scatter of proportion reads assigned to A or B
    fig = plot_scatter_with_smooth_line(df, x="mash_SB", y_kma="kma_proportion_assigned", y_kallisto="kallisto_proportion_assigned", 
                                        title="", x_label="mash B-S", y_label="Proportion input reads assigned", logx=False,
                                        fit_method="window", window_n=20, window_min_points=5)
    out_path = os.path.join(out_folder, f"ID_DM_2D_scatter_assigned.mash_AB_fixed_{mash_AB_range[0]}_{mash_AB_range[1]}.png")
    ax = fig.axes[0]
    ax.set_ylim(0, 1)
    xlim_min = ax.get_xlim()[0]
    ax.set_xlim(xlim_min, 0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    # plot scatter of reads assigned to B
    fig = plot_scatter_with_smooth_line(df, x="mash_SB", y_kma="kma_proportion_assigned_to_B", y_kallisto="kallisto_proportion_assigned_to_B", 
                                        title="", x_label="mash B-S", y_label="Proportion input reads assigned to B", logx=False,
                                        fit_method="window", window_n=20, window_min_points=5)
    ax = fig.axes[0]
    ax.set_ylim(0, 1)
    xlim_min = ax.get_xlim()[0]
    ax.set_xlim(xlim_min, 0.2)
    fig.tight_layout()
    out_path = os.path.join(out_folder, f"ID_DM_2D_scatter_assigned_B.mash_AB_fixed_{mash_AB_range[0]}_{mash_AB_range[1]}.png")
    fig.savefig(out_path, dpi=300)
    

