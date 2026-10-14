"""Generate the manuscript simulation figure PDFs."""
from pathlib import Path

from .manuscript_plot import (
    pl, np, plt, sns, plot_scatter_with_smooth_line, plot_joint_heatmap_with_bins,
)

ROOT = Path(__file__).resolve().parents[2]


def main(show=False):
    FIGURES = ROOT / "figures"
    FIGURES.mkdir(exist_ok=True)

    # input params
    results_dir = str(ROOT / "data/simulations/length=10000_simulation_sets=10_max_mutations=2000_2026-04-14")
    seq_length = 10000
    number_of_simulation_sets = 10
    mutation_counts = list(range(1, 101, 1)) + list(range(105, 251, 5)) + list(range(260, 501, 10)) + list(range(550, 1001, 50)) + list(range(1100, 2001, 100))
    min_mutations = min(mutation_counts)
    max_mutations = max(mutation_counts)
    # make dfs
    RAD_dfs = []
    for i in range(1, number_of_simulation_sets + 1):
        results_file = f"{results_dir}/simulation_set_{i}/kmer_simulation.RA_D.seq_length={seq_length}_num_reads=1000.min_mutations={min_mutations}.max_mutations={max_mutations}.tsv"
        df = pl.read_csv(results_file, separator="\t")
        df = df.with_columns(pl.lit(i).alias("simulation_set"))
        # make the kma_log_error and kallisto_log_error absolute values
        df = df.with_columns([pl.col("kma_log_error").abs().alias("kma_log_error_abs"), pl.col("kallisto_log_error").abs().alias("kallisto_log_error_abs")])
        RAD_dfs.append(df)
    RAD_df = pl.concat(RAD_dfs)

    RA_deltaM_dfs = []
    for i in range(1, number_of_simulation_sets + 1):
        results_file = f"{results_dir}/simulation_set_{i}/kmer_simulation.RA_deltaM.seq_length={seq_length}_num_reads=1000.min_mutations={min_mutations}.max_mutations={max_mutations}.tsv"
        df = pl.read_csv(results_file, separator="\t")
        df = df.with_columns(pl.lit(i).alias("simulation_set"))
        # make sure schemas work
        if i == 1:
            # save the schema of the first file to compare with the others
            first_schema = df.schema
        # match the schema of the first file
        for col in first_schema:
            dtype = first_schema[col]
            if col not in df.columns:
                # add the missing column with null values
                df = df.with_columns(pl.lit(None).alias(col))
            else:
                # cast the column to the same type as the first schema
                df = df.with_columns(pl.col(col).cast(dtype))
        RA_deltaM_dfs.append(df)
    RA_deltaM_df = pl.concat(RA_deltaM_dfs)

    RA_prop_dfs = []
    for i in range(1, number_of_simulation_sets + 1):
        results_file = f"{results_dir}/simulation_set_{i}/kmer_simulation.RA_prop.seq_length={seq_length}_num_reads=1000_with_250_mutations.tsv"
        df = pl.read_csv(results_file, separator="\t")
        df = df.with_columns(pl.lit(i).alias("simulation_set"))
        # make sure schemas work
        if i == 1:
            # save the schema of the first file to compare with the others
            first_schema = df.schema
        # match the schema of the first file
        for col in first_schema:
            dtype = first_schema[col]
            if col not in df.columns:
                # add the missing column with null values
                df = df.with_columns(pl.lit(None).alias(col))
            else:
                # cast the column to the same type as the first schema
                df = df.with_columns(pl.col(col).cast(dtype))
        RA_prop_dfs.append(df)
    RA_prop_df = pl.concat(RA_prop_dfs)

    IDM_dfs = []
    for i in range(1, number_of_simulation_sets + 1):
        results_file = f"{results_dir}/simulation_set_{i}/kmer_simulation.ID_M.seq_length={seq_length}_num_reads=1000.min_mutations={min_mutations}.max_mutations={max_mutations}.tsv"
        df = pl.read_csv(results_file, separator="\t")
        df = df.with_columns(pl.lit(i).alias("simulation_set"))
        # make sure schemas work
        if i == 1:
            # save the schema of the first file to compare with the others
            first_schema = df.schema
        # match the schema of the first file
        for col in first_schema:
            dtype = first_schema[col]
            if col not in df.columns:
                # add the missing column with null values
                df = df.with_columns(pl.lit(None).alias(col))
            else:
                # cast the column to the same type as the first schema
                df = df.with_columns(pl.col(col).cast(dtype))
        IDM_dfs.append(df)
    ID_M_df = pl.concat(IDM_dfs)

    ID_D_dfs = []
    for i in range(1, number_of_simulation_sets + 1):
        results_file = f"{results_dir}/simulation_set_{i}/kmer_simulation.ID_D.seq_length={seq_length}_num_reads=1000.min_mutations={min_mutations}.max_mutations={max_mutations}.tsv"
        df = pl.read_csv(results_file, separator="\t")
        df = df.with_columns(pl.lit(i).alias("simulation_set"))
        # make sure schemas work
        if i == 1:
            # save the schema of the first file to compare with the others
            first_schema = df.schema
        # match the schema of the first file
        for col in first_schema:
            dtype = first_schema[col]
            if col not in df.columns:
                # add the missing column with null values
                df = df.with_columns(pl.lit(None).alias(col))
            else:
                # cast the column to the same type as the first schema
                df = df.with_columns(pl.col(col).cast(dtype))
        ID_D_dfs.append(df)
    ID_D_df = pl.concat(ID_D_dfs)

    # is in a sep dir
    results_dir = str(ROOT / "data/simulations/length=10000_simulation_sets=10_max_mutations=2000_2026-04-17")

    IDDM_dfs = []
    for i in range(1, number_of_simulation_sets + 1):
        results_file = f"{results_dir}/simulation_set_{i}/kmer_simulation.ID_DM.seq_length={seq_length}_num_reads=1000.min_mutations={min_mutations}.max_mutations={max_mutations}.tsv"
        df = pl.read_csv(results_file, separator="\t")
        df = df.with_columns(pl.lit(i).alias("simulation_set"))
        # make sure schemas work
        if i == 1:
            # save the schema of the first file to compare with the others
            first_schema = df.schema
        # match the schema of the first file
        for col in first_schema:
            dtype = first_schema[col]
            if col not in df.columns:
                # add the missing column with null values
                df = df.with_columns(pl.lit(None).alias(col))
            else:
                # cast the column to the same type as the first schema
                df = df.with_columns(pl.col(col).cast(dtype))
        IDDM_dfs.append(df)
    ID_DM_df = pl.concat(IDDM_dfs)
    RA_deltaM_df = RA_deltaM_df.with_columns(pl.col("mash_where_differ").alias("mash_distance"))
    ID_DM_df = ID_DM_df.with_columns((pl.col("num_mutations_in_B") + int(0.01 * seq_length)).alias("num_mutations"), pl.col("mash_AB").alias("mash_distance"))

    window_n = 75
    window_min_points = 10
    banding = "percentile"
    # Figure 1A: sample-reference distance

    fig, ax = plot_scatter_with_smooth_line(ID_M_df, x="mash_distance", y_kma="kma_proportion_assigned", y_kallisto="kallisto_proportion_assigned", 
                                        title="", x_label="sample-reference mismatch", y_label="Proportion input reads assigned", logx=False,
                                        fit_method="window", window_n=window_n, window_min_points=window_min_points, banding=banding, percentile_low=25, percentile_high=75)
    # make axis labels bigger
    ax.set_xlabel("Sample-reference distance", fontsize=12)
    ax.text(0.5, -0.2, "(Mash)", 
            transform=ax.transAxes, 
            ha='center', 
            fontsize=8, 
            style='italic')
    ax.set_ylabel("Proportion of input\nreads assigned", fontsize=12, rotation=90)#, labelpad=10, ha="right", va="center")
    fig.savefig(FIGURES / "figure_1a.pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close("all")

    # Figure 1B: inter-reference distance

    fig, ax = plot_scatter_with_smooth_line(ID_D_df, x="mash_distance", y_kma="kma_proportion_assigned", y_kallisto="kallisto_proportion_assigned", title="", x_label="mash A-B", y_label="Proportion assigned reads correct", logx=False,
                                        fit_method="window", window_n=window_n, window_min_points=window_min_points, banding=banding, percentile_low=25, percentile_high=75)
    # make y axis tick labels to 3dp
    # first place y ticks at intervals that can be nicely labelled with 3 decimal places, then format the tick labels to show 3 decimal places
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.005))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))
    # ax.invert_xaxis()
    ax.set_xlabel("Inter-reference distance", fontsize=12)
    ax.text(0.5, -0.2, "(Mash)", 
            transform=ax.transAxes, 
            ha='center', 
            fontsize=8, 
            style='italic')
    ax.set_ylabel("Proportion of assigned\nreads correct", fontsize=12, rotation=90)#, labelpad=10, ha="right", va="center")
    ax.set_ylim(0.99, 1.0)

    fig.savefig(FIGURES / "figure_1b.pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close("all")

    # Figure 3: inter-reference distance and abundance error

    fig, ax = plot_scatter_with_smooth_line(RAD_df, x="mash_distance", y_kma="kma_log_error", y_kallisto="kallisto_log_error", 
                                            title="", x_label="mash A-B", y_label="absolute log error", logx=True, absy=True,
                                            fit_method="window", window_n=window_n, window_min_points=window_min_points, banding=banding, percentile_low=25, percentile_high=75)
    # ax.invert_xaxis()
    ax.set_xlabel("Inter-reference distance", fontsize=12)
    ax.text(0.5, -0.2, "(Mash)", 
            transform=ax.transAxes, 
            ha='center', 
            fontsize=8, 
            style='italic')
    ax.set_ylabel("Fold-change in relative\nabundance", fontsize=12, rotation=0, labelpad=10, ha="right", va="center")
    fig.savefig(FIGURES / "figure_3.pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close("all")

    # Figure 4: asymmetric sample-reference distance

    RA_deltaM_df = RA_deltaM_df.with_columns(
            pl.when(pl.col("mutated_sequence") == "second_original")
            .then(pl.col("mash_where_differ"))
            .otherwise(-pl.col("mash_where_differ"))
            .alias("delta_M")
        )
    fig, ax = plot_scatter_with_smooth_line(RA_deltaM_df, x="delta_M", y_kma="kma_log_error", y_kallisto="kallisto_log_error", 
                                            title="", x_label="mutation bias to B", y_label="log error", logx=False, absy=False,
                                            fit_method="window", window_n=window_n, window_min_points=window_min_points, banding=banding,
                                            percentile_low=25, percentile_high=75)

    ax.set_xlabel("Sample-reference distance bias to\ncomponent 1", fontsize=12)
    ax.text(0.5, -0.25, "($\Delta$Mash = Mash$_1$ − Mash$_2$)", 
            transform=ax.transAxes, 
            ha='center', 
            fontsize=8, 
            style='italic')
    ax.set_ylabel("Estimate fold-bias\nto component 2", fontsize=12, rotation=0, labelpad=10, ha="right", va="center")
    fig.savefig(FIGURES / "figure_4.pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close("all")

    # Figure 5: mixture proportions

    fig, ax = plot_scatter_with_smooth_line(RA_prop_df, x="proportion_original", y_kallisto="kallisto_log_error", y_kma="kma_log_error", 
                                            title="", x_label="Proportion of original sequence", y_label="log error", logx=False, absy=False,
                                            fit_method="window", window_n=window_n, window_min_points=window_min_points, banding=banding,
                                            percentile_low=25, percentile_high=75)

    ax.set_xlabel("Expected proportion of component 1", fontsize=12)

    ax.set_ylabel("Estimate fold-bias\nto component 2", fontsize=12, rotation=0, labelpad=10, ha="right", va="center")
    fig.savefig(FIGURES / "figure_5.pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close("all")

    # Figure 2A

    # filter ID_DM to mash_SB is 0.009-0.014 and plot a smoothed scatter of mash_AB vs proportion of reads assigned coloured by tool
    filtered_ID_DM = ID_DM_df.filter(pl.col("mash_SB").is_between(0.009, 0.014))
    fig, ax = plot_scatter_with_smooth_line(
        filtered_ID_DM,
        x="mash_AB",
        y_kma="kma_proportion_assigned_to_B",
        y_kallisto="kallisto_proportion_assigned_to_B",
        title="",
        x_label="Reference dissimilarity",
        y_label="Proportion of assigned reads correct",
        logx=False,
        logy=False,
        fit_method="window",
        window_n=75,
        window_min_points=10,
        banding="percentile",
        percentile_high=75,
        percentile_low=25,
    )
    ax.set_ylim(0, 1)
    # ax.invert_xaxis()
    ax.set_xlabel("Inter-reference distance", fontsize=12)
    ax.set_ylabel("Proportion of assigned\nreads correct", fontsize=12)

    ax.text(0.5, -0.2, "(Mash)", 
            transform=ax.transAxes, 
            ha='center', 
            fontsize=8, 
            style='italic')
    fig.savefig(FIGURES / "figure_2a.pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close("all")

    # Figure 2B

    filtered_ID_DM = ID_DM_df.filter(pl.col("mash_SB").is_between(0.059, 0.095))
    fig, ax = plot_scatter_with_smooth_line(
        filtered_ID_DM,
        x="mash_AB",
        y_kma="kma_proportion_assigned_to_B",
        y_kallisto="kallisto_proportion_assigned_to_B",
        title="",
        x_label="Reference dissimilarity",
        y_label="Proportion of assigned reads correct",
        logx=False,
        logy=False,
        fit_method="window",
        window_n=75,
        window_min_points=10,
        banding="percentile",
        percentile_high=75,
        percentile_low=25,
    )
    ax.set_ylim(0, 1)
    # ax.invert_xaxis()
    ax.set_xlabel("Inter-reference distance", fontsize=12)
    ax.set_ylabel("Proportion of assigned\nreads correct", fontsize=12)

    ax.text(0.5, -0.2, "(Mash)", 
            transform=ax.transAxes, 
            ha='center', 
            fontsize=8, 
            style='italic')
    fig.savefig(FIGURES / "figure_2b.pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close("all")

    # Figure 2E

    # Restrict inter-reference distance to 0.027–0.037, as in the manuscript.
    filtered_ID_DM = ID_DM_df.filter(pl.col("mash_AB").is_between(0.027, 0.037))
    fig, ax = plot_scatter_with_smooth_line(
        filtered_ID_DM,
        x="mash_SB",
        y_kma="kma_proportion_assigned_to_B",
        y_kallisto="kallisto_proportion_assigned_to_B",
        title="",
        x_label="Sample-reference distance",
        y_label="Proportion of assigned reads correct",
        logx=False,
        logy=False,
        fit_method="window",
        window_n=75,
        window_min_points=10,
        banding="percentile",
        percentile_low=25,
        percentile_high=75,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Sample-reference distance", fontsize=12)
    ax.set_ylabel("Proportion of assigned\nreads correct", fontsize=12)

    ax.text(0.5, -0.2, "(Mash)", 
            transform=ax.transAxes, 
            ha='center', 
            fontsize=8, 
            style='italic')
    fig.savefig(FIGURES / "figure_2e.pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close("all")

    # Figure 2F

    filtered_ID_DM = ID_DM_df.filter(pl.col("mash_AB").is_between(0.027, 0.037)).filter(pl.col("mash_SB") < 1)
    fig, ax = plot_scatter_with_smooth_line(
        filtered_ID_DM,
        x="mash_SB",
        y_kma="kma_proportion_assigned",
        y_kallisto="kallisto_proportion_assigned",
        title="",
        x_label="Sample-reference mismatch",
        y_label="Proportion of reads assigned",
        logx=False,
        logy=False,
        fit_method="window",
        window_n=75,
        window_min_points=10,
        banding="percentile",
        percentile_low=25,
        percentile_high=75,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Sample-reference distance", fontsize=12)
    ax.set_ylabel("Proportion of reads\nassigned", fontsize=12)

    ax.text(0.5, -0.2, "(Mash)", 
            transform=ax.transAxes, 
            ha='center', 
            fontsize=8, 
            style='italic')
    fig.savefig(FIGURES / "figure_2f.pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close("all")

    # Figure 2C: proportion of assigned reads correct

    ID_DM_df = ID_DM_df.filter(pl.col("mash_AS") >= pl.col("mash_SB"))
    # The highest log-distance bin (~0.624–1) contains only no-shared-kmer
    # sentinel values. Omit it in 2C/D, retaining the original lower bin edges.



    fig1, axs = plot_joint_heatmap_with_bins(ID_DM_df, x="mash_AB", y="mash_SB", colour_column1="kma_proportion_assigned_to_B", 
                                        colour_column2="kallisto_proportion_assigned_to_B", n_x_bins=10, n_y_bins=10, 
                                        title="", x_label="mash A-B", y_label="mash S-B", logx=True, logy=True, max_empty_cells_per_row=2, flip_y_axis=True, exclude_highest_y_bin=True)
    for ax in axs:
        # current_xlim = ax.get_xlim()
        # ax.set_xlim(current_xlim[1], current_xlim[0])
        # current_ylims = ax.get_ylim()
        # ax.set_ylim(current_ylims[1], current_ylims[0])
        ax.set_xlabel("Inter-reference distance", fontsize=12)
    axs[0].set_ylabel("Sample-reference\ndistance", fontsize=12)
    fig1.savefig(FIGURES / "figure_2c.pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close("all")

    # Figure 2D: input reads assigned

    fig, axs = plot_joint_heatmap_with_bins(ID_DM_df, x="mash_AB", y="mash_SB", colour_column1="kma_proportion_assigned", 
                                        colour_column2="kallisto_proportion_assigned", n_x_bins=10, n_y_bins=10, title="", x_label="mash A-B", 
                                        y_label="mash S-B", logx=True, logy=True, max_empty_cells_per_row=2, cmap="mako", flip_y_axis=True, exclude_highest_y_bin=True)
    for ax in axs:
        # current_xlim = ax.get_xlim()
        # ax.set_xlim(current_xlim[1], current_xlim[0])
        # current_ylims = ax.get_ylim()
        # ax.set_ylim(current_ylims[1], current_ylims[0])
        ax.set_xlabel("Inter-reference distance", fontsize=12)
    axs[0].set_ylabel("Sample-reference\ndistance", fontsize=12)
    plt.draw()
    fig.savefig(FIGURES / "figure_2d.pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close("all")

    # Figure S2: mutations and Mash distance

    # on the same plot, plot n_mutations vs mash distance, coloured by df
    fig, ax = plt.subplots(figsize=(10, 6))
    all_mash_df = pl.concat([RAD_df.select(["num_mutations", "mash_distance"]).with_columns(pl.lit("RA_D").alias("df")), RA_deltaM_df.select(["num_mutations", "mash_distance"]).with_columns(pl.lit("RA_deltaM").alias("df")), ID_M_df.select(["num_mutations", "mash_distance"]).with_columns(pl.lit("ID_M").alias("df")), ID_D_df.select(["num_mutations", "mash_distance"]).with_columns(pl.lit("ID_D").alias("df")), ID_DM_df.select(["num_mutations", "mash_distance"]).with_columns(pl.lit("ID_DM").alias("df"))])
    sns.scatterplot(data=all_mash_df.to_pandas(), x="num_mutations", y="mash_distance", color="k", edgecolor=None, alpha=0.5)
    ax.set_xlabel("Number of mutations")
    ax.set_ylabel("Mash distance")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    mash_levels_to_mark = [0.001, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
    for level in mash_levels_to_mark:
        # find range of num_mutations where mash_distance is around this level
        level_df = all_mash_df.filter(pl.col("mash_distance").is_between(level - 0.001, level + 0.001))
        if level_df.is_empty():
            continue
        min_mutations = level_df.select(pl.col("num_mutations").min()).item()
        max_mutations = level_df.select(pl.col("num_mutations").max()).item()
        ax.hlines(level, min_mutations, max_mutations, colors='r', linestyles='--')
        ax.vlines([min_mutations, max_mutations], 0, level, colors='r', linestyles='--')
        # add text annotation with semi-transparent white box behind it
        text_label = f"{min_mutations}-{max_mutations}"
        ax.text(
            (min_mutations + max_mutations) / 2,
            level + 0.005,
            text_label,
            color='r',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none')
        )
    # make y ticks every 0.01
    ax.set_yticks(np.arange(0, 0.25, 0.01))
    # only label every 0.05
    ax.set_yticklabels([f"{y:.2f}" if y % 0.05 == 0 else "" for y in np.arange(0, 0.25, 0.01)])
    # log the y axis
    # ax.set_yscale("log")
    sns.despine(ax=ax, top=True, right=True, trim=True)
    ax.set_xlim(0, 2010)
    if show:
        plt.show()
    else:
        plt.close("all")
    fig.savefig(FIGURES / "figure_s2.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
