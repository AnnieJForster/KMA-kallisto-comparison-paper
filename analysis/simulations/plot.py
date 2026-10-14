from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import polars as pl
import os
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator, LogLocator, NullFormatter

sns.set_theme(style="white")

def lowess_for_axis(x_values, y_values, logx=False, logy=False, frac=0.22):
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if logx:
        valid &= x_values > 0
    if logy:
        valid &= y_values > 0
    x_valid = x_values[valid]
    y_valid = y_values[valid]
    if len(x_valid) == 0:
        return np.array([]), np.array([])
    x_fit = np.log10(x_valid) if logx else x_valid
    y_fit = np.log10(y_valid) if logy else y_valid
    smoothed = lowess(y_fit, x_fit, frac=frac, return_sorted=True)
    x_smooth = (10 ** smoothed[:, 0]) if logx else smoothed[:, 0]
    y_smooth = (10 ** smoothed[:, 1]) if logy else smoothed[:, 1]
    return x_smooth, y_smooth

def poly_for_axis(x_values, y_values, order=3, logx=False, logy=False, n_points=300):
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if logx:
        valid &= x_values > 0
    if logy:
        valid &= y_values > 0

    x_valid = x_values[valid]
    y_valid = y_values[valid]
    if len(x_valid) == 0:
        return np.array([]), np.array([])

    x_fit = np.log10(x_valid) if logx else x_valid
    y_fit = np.log10(y_valid) if logy else y_valid

    # Guard against overfitting/failure when few unique x values are present.
    max_degree = max(1, min(order, len(np.unique(x_fit)) - 1, len(x_fit) - 1))
    coeffs = np.polyfit(x_fit, y_fit, deg=max_degree)

    x_line_fit = np.linspace(np.min(x_fit), np.max(x_fit), n_points)
    y_line_fit = np.polyval(coeffs, x_line_fit)

    x_line = (10 ** x_line_fit) if logx else x_line_fit
    y_line = (10 ** y_line_fit) if logy else y_line_fit
    return x_line, y_line

def windowed_average_for_axis(x_values, y_values, logx=False, logy=False, n_windows=40, window_size=None, min_points_per_window=1):
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if logx:
        valid &= x_values > 0
    if logy:
        valid &= y_values > 0

    x_valid = x_values[valid]
    y_valid = y_values[valid]
    if len(x_valid) == 0:
        return np.array([]), np.array([])

    x_fit = np.log10(x_valid) if logx else x_valid
    y_fit = np.log10(y_valid) if logy else y_valid

    x_min = np.min(x_fit)
    x_max = np.max(x_fit)
    if x_max == x_min:
        x_line = np.array([x_min])
        y_line = np.array([np.mean(y_fit)])
    else:
        if window_size is None:
            n_windows = max(1, int(n_windows))
            window_size = (x_max - x_min) / n_windows
        if window_size <= 0:
            raise ValueError("window_size must be greater than 0")

        window_ids = np.floor((x_fit - x_min) / window_size).astype(int)
        x_means = []
        y_means = []
        for window_id in np.unique(window_ids):
            in_window = window_ids == window_id
            if np.sum(in_window) < min_points_per_window:
                continue
            x_means.append(np.mean(x_fit[in_window]))
            y_means.append(np.mean(y_fit[in_window]))

        if len(x_means) == 0:
            return np.array([]), np.array([])

        x_line = np.asarray(x_means)
        y_line = np.asarray(y_means)
        order_idx = np.argsort(x_line)
        x_line = x_line[order_idx]
        y_line = y_line[order_idx]

    x_line = (10 ** x_line) if logx else x_line
    y_line = (10 ** y_line) if logy else y_line
    return x_line, y_line

def windowed_average_with_band_for_axis(
    x_values,
    y_values,
    logx=False,
    logy=False,
    n_windows=40,
    window_size=None,
    min_points_per_window=1,
    z_value=1.0,
):
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if logx:
        valid &= x_values > 0
    if logy:
        valid &= y_values > 0

    x_valid = x_values[valid]
    y_valid = y_values[valid]
    if len(x_valid) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    x_fit = np.log10(x_valid) if logx else x_valid
    y_fit = np.log10(y_valid) if logy else y_valid

    x_min = np.min(x_fit)
    x_max = np.max(x_fit)
    if x_max == x_min:
        count = len(y_fit)
        y_mean = np.mean(y_fit)
        y_std = np.std(y_fit, ddof=1) if count > 1 else 0.0
        y_se = y_std / np.sqrt(count) if count > 0 else 0.0
        x_line = np.array([x_min])
        y_line = np.array([y_mean])
        y_lower = np.array([y_mean - z_value * y_se])
        y_upper = np.array([y_mean + z_value * y_se])
    else:
        if window_size is None:
            n_windows = max(1, int(n_windows))
            window_size = (x_max - x_min) / n_windows
        if window_size <= 0:
            raise ValueError("window_size must be greater than 0")

        window_ids = np.floor((x_fit - x_min) / window_size).astype(int)
        x_means = []
        y_means = []
        y_lowers = []
        y_uppers = []

        for window_id in np.unique(window_ids):
            in_window = window_ids == window_id
            count = np.sum(in_window)
            if count < min_points_per_window:
                continue
            x_win = x_fit[in_window]
            y_win = y_fit[in_window]
            y_mean = np.mean(y_win)
            y_std = np.std(y_win, ddof=1) if count > 1 else 0.0
            y_se = y_std / np.sqrt(count)

            x_means.append(np.mean(x_win))
            y_means.append(y_mean)
            y_lowers.append(y_mean - z_value * y_se)
            y_uppers.append(y_mean + z_value * y_se)

        if len(x_means) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        x_line = np.asarray(x_means)
        y_line = np.asarray(y_means)
        y_lower = np.asarray(y_lowers)
        y_upper = np.asarray(y_uppers)

        order_idx = np.argsort(x_line)
        x_line = x_line[order_idx]
        y_line = y_line[order_idx]
        y_lower = y_lower[order_idx]
        y_upper = y_upper[order_idx]

    x_line = (10 ** x_line) if logx else x_line
    y_line = (10 ** y_line) if logy else y_line
    y_lower = (10 ** y_lower) if logy else y_lower
    y_upper = (10 ** y_upper) if logy else y_upper
    return x_line, y_line, y_lower, y_upper

def safe_log10_expr(column_name, alias):
    return pl.when(pl.col(column_name) > 0).then(pl.col(column_name).log10()).otherwise(None).alias(alias)

def parse_results_for_plotting(results_file, results_dir, seq_length, n_simulation_sets, mutation_counts, starting_set=1):
    dfs = []
    for i in range(starting_set, starting_set + n_simulation_sets):
        df = pl.read_csv(f"{results_dir}/simulation_set_{i}/{results_file}", separator="\t")
        df = df.with_columns(pl.lit(i).alias("simulation_set"))
        if "kma_log_error" in df.columns:
            df = df.with_columns([pl.col("kma_log_error").abs().alias("kma_log_error_abs"), pl.col("kallisto_log_error").abs().alias("kallisto_log_error_abs")])
        dfs.append(df)
    combined_df = pl.concat(dfs)
    return combined_df

def add_failure_masked_columns_for_id_dm(df):
    """Mask values from failed tool runs so they do not contribute to bin means.

    In this simulation setup, a tool failure/no-assignment is represented by
    proportion_assigned == 0 for that tool.
    """
    required = {
        "kallisto_proportion_assigned",
        "kma_proportion_assigned",
        "kallisto_proportion_assigned_to_B",
        "kma_proportion_assigned_to_B",
    }
    if not required.issubset(set(df.columns)):
        return df

    return df.with_columns(
        pl.when(pl.col("kallisto_proportion_assigned") > 0)
        .then(pl.col("kallisto_proportion_assigned_to_B"))
        .otherwise(None)
        .alias("kallisto_proportion_assigned_to_B_plot"),
        pl.when(pl.col("kma_proportion_assigned") > 0)
        .then(pl.col("kma_proportion_assigned_to_B"))
        .otherwise(None)
        .alias("kma_proportion_assigned_to_B_plot"),
        pl.when(pl.col("kallisto_proportion_assigned") > 0)
        .then(pl.col("kallisto_proportion_assigned"))
        .otherwise(None)
        .alias("kallisto_proportion_assigned_plot"),
        pl.when(pl.col("kma_proportion_assigned") > 0)
        .then(pl.col("kma_proportion_assigned"))
        .otherwise(None)
        .alias("kma_proportion_assigned_plot"),
    )

def axis_speed_up(ax, axis_to_speed_up, speed_up_factor, speed_up_start, speed_up_end):
    if speed_up_end <= speed_up_start:
        raise ValueError("speed_up_end must be greater than speed_up_start")
    if speed_up_factor <= 0:
        raise ValueError("speed_up_factor must be greater than 0")

    original_span = speed_up_end - speed_up_start
    compressed_span = original_span / speed_up_factor
    compressed_end = speed_up_start + compressed_span
    span_delta = original_span - compressed_span
    def forward(values):
        values = np.asarray(values)
        in_window = (values >= speed_up_start) & (values <= speed_up_end)
        above_window = values > speed_up_end
        transformed = values.copy()
        transformed[in_window] = speed_up_start + (values[in_window] - speed_up_start) / speed_up_factor
        transformed[above_window] = values[above_window] - span_delta
        return transformed
    def inverse(values):
        values = np.asarray(values)
        in_window = (values >= speed_up_start) & (values <= compressed_end)
        above_window = values > compressed_end
        transformed = values.copy()
        transformed[in_window] = speed_up_start + (values[in_window] - speed_up_start) * speed_up_factor
        transformed[above_window] = values[above_window] + span_delta
        return transformed
    if axis_to_speed_up == "x":
        ax.set_xscale("function", functions=(forward, inverse))
    elif axis_to_speed_up == "y":
        ax.set_yscale("function", functions=(forward, inverse))
    else:
        raise ValueError("axis_to_speed_up must be 'x' or 'y'")

def set_linear_x_ticks(ax, x_values, step=0.05, logx=False):
    x_values = np.asarray(x_values)
    valid = np.isfinite(x_values)
    if logx:
        valid &= x_values > 0
    x_valid = x_values[valid]
    if len(x_valid) == 0:
        return

    x_min = np.min(x_valid)
    x_max = np.max(x_valid)
    start = np.floor(x_min / step) * step
    end = np.ceil(x_max / step) * step
    if logx and start <= 0:
        start = step

    ticks = np.arange(start, end + (step * 0.5), step)
    if logx:
        ticks = ticks[ticks > 0]
    if len(ticks) == 0:
        return

    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick:.2f}" for tick in ticks])
    # Disable log minor ticks so grid/ticks stay on this linear sequence.
    ax.set_xticks([], minor=True)

def convert_log_error_axis_to_fold_change(ax, log_base=10):
    # Keep the axis in log-error space, but label ticks as fold-change.
    y_min, y_max = ax.get_ylim()
    fold_low = log_base ** min(y_min, y_max)
    fold_high = log_base ** max(y_min, y_max)

    span = fold_high - fold_low
    if span <= 1.0:
        # Narrow range: show dense, readable ticks every 0.1 fold-change.
        start = np.floor(fold_low * 10) / 10
        end = np.ceil(fold_high * 10) / 10
        visible = np.arange(start, end + 0.05, 0.1)
        visible = visible[(visible >= fold_low) & (visible <= fold_high) & (visible > 0)]
    else:
        # Middle and large ranges: keep the existing interpretable fold ticks.
        fold_ticks = np.array([
            1/50, 1/20, 1/10, 1/5, 1/2, 1, 2, 5, 10, 20, 50
        ])
        visible = fold_ticks[(fold_ticks >= fold_low) & (fold_ticks <= fold_high)]

    if len(visible) < 3:
        visible = np.linspace(fold_low, fold_high, 4)

    y_tick_positions = np.log(visible) / np.log(log_base)
    y_tick_labels = [f"{tick:.3g}" for tick in visible]

    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels)
    ax.set_yticks([], minor=True)
    ax.set_ylabel("Fold change", fontsize=8)


def format_plain_tick_label(value, _):
    """Format tick labels as plain decimals (avoid scientific/10^ notation)."""
    if not np.isfinite(value):
        return ""
    if np.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.10f}".rstrip("0").rstrip(".")

def plot_scatter_with_smooth_line(df, x, y_kma, y_kallisto, title="", x_label="", y_label="", logx=False, logy=False, absx=False, absy=False, fit_method="lowess", poly_order=3, window_n=40, window_min_points=1):
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if absx:
        df = df.with_columns(pl.col(x).abs().alias(x))
    if absy:
        df = df.with_columns(pl.col(y_kma).abs().alias(y_kma), pl.col(y_kallisto).abs().alias(y_kallisto))

    x_values = df[x].to_numpy()
    y_kma_values = df[y_kma].to_numpy()
    y_kallisto_values = df[y_kallisto].to_numpy()

    if fit_method == "poly":
        fit_kma_x, fit_kma_y = poly_for_axis(x_values, y_kma_values, order=poly_order, logx=logx, logy=logy)
        fit_kallisto_x, fit_kallisto_y = poly_for_axis(x_values, y_kallisto_values, order=poly_order, logx=logx, logy=logy)
        # Simple constant-width fallback band from residual spread.
        y_kma_interp = np.interp(x_values, fit_kma_x, fit_kma_y)
        y_kallisto_interp = np.interp(x_values, fit_kallisto_x, fit_kallisto_y)
        kma_sigma = np.std(y_kma_values - y_kma_interp)
        kallisto_sigma = np.std(y_kallisto_values - y_kallisto_interp)
        fit_kma_lower = fit_kma_y - kma_sigma
        fit_kma_upper = fit_kma_y + kma_sigma
        fit_kallisto_lower = fit_kallisto_y - kallisto_sigma
        fit_kallisto_upper = fit_kallisto_y + kallisto_sigma
    elif fit_method == "window":
        fit_kma_x, fit_kma_y, fit_kma_lower, fit_kma_upper = windowed_average_with_band_for_axis(
            x_values,
            y_kma_values,
            logx=logx,
            logy=logy,
            n_windows=window_n,
            min_points_per_window=window_min_points,
            z_value=1.96
        )
        fit_kallisto_x, fit_kallisto_y, fit_kallisto_lower, fit_kallisto_upper = windowed_average_with_band_for_axis(
            x_values,
            y_kallisto_values,
            logx=logx,
            logy=logy,
            n_windows=window_n,
            min_points_per_window=window_min_points,
            z_value=1.96
        )
    else:
        fit_kma_x, fit_kma_y = lowess_for_axis(x_values, y_kma_values, logx=logx, logy=logy)
        fit_kallisto_x, fit_kallisto_y = lowess_for_axis(x_values, y_kallisto_values, logx=logx, logy=logy)
        y_kma_interp = np.interp(x_values, fit_kma_x, fit_kma_y)
        y_kallisto_interp = np.interp(x_values, fit_kallisto_x, fit_kallisto_y)
        kma_sigma = np.std(y_kma_values - y_kma_interp)
        kallisto_sigma = np.std(y_kallisto_values - y_kallisto_interp)
        fit_kma_lower = fit_kma_y - kma_sigma
        fit_kma_upper = fit_kma_y + kma_sigma
        fit_kallisto_lower = fit_kallisto_y - kallisto_sigma
        fit_kallisto_upper = fit_kallisto_y + kallisto_sigma

    ax.fill_between(fit_kma_x, fit_kma_lower, fit_kma_upper, color="tab:blue", alpha=0.09, linewidth=0)
    ax.fill_between(fit_kallisto_x, fit_kallisto_lower, fit_kallisto_upper, color="tab:orange", alpha=0.09, linewidth=0)
    ax.plot(fit_kma_x, fit_kma_y, color="tab:blue", linestyle="-", linewidth=2.0, label="KMA")
    ax.plot(fit_kallisto_x, fit_kallisto_y, color="tab:orange", linestyle="-", linewidth=2.0, label="Kallisto")
    # set_linear_x_ticks(ax, x_values, step=0.2, logx=logx)
    # if y axis is log error, optionally enforce symmetric log-error range, then relabel as fold-change.
    ax.set_xlabel(x_label, fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    if y_kallisto == "kallisto_log_error" or y_kma == "kma_log_error":
        if not absy:
            ylims = ax.get_ylim()
            max_abs = max(abs(ylims[0]), abs(ylims[1]))
            ax.set_ylim(-max_abs, max_abs)
        convert_log_error_axis_to_fold_change(ax, log_base=10)
    if title:
        ax.set_title(title, fontsize=9.5, weight="semibold", pad=3)

    if logx:
        # Keep log spacing, but render labels as plain decimal numbers.
        ax.xaxis.set_major_locator(LogLocator(base=10.0))
        ax.xaxis.set_major_formatter(FuncFormatter(format_plain_tick_label))
        ax.xaxis.set_minor_formatter(NullFormatter())

    ax.tick_params(axis="both", which="major", labelsize=7, width=0.7, length=2.5, pad=1.5)
    ax.legend(frameon=False, fontsize=6.8, handlelength=2.0, borderaxespad=0.2, labelspacing=0.25)
    ax.grid(alpha=0.5, linestyle="--", linewidth=0.45)
    ax.margins(x=0.01, y=0.04)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    sns.despine(ax=ax, top=True, right=True)
    fig.tight_layout(pad=0.25)
    return fig

def format_bin_value(value):
    formatted = f"{value:.10f}".rstrip("0").rstrip(".")
    return formatted if formatted not in {"", "-0"} else "0"

def format_bin_range(low, high):
    for precision in range(1, 11):
        low_str = f"{low:.{precision}f}".rstrip("0").rstrip(".")
        high_str = f"{high:.{precision}f}".rstrip("0").rstrip(".")
        low_str = low_str if low_str not in {"", "-0"} else "0"
        high_str = high_str if high_str not in {"", "-0"} else "0"

        # Do not allow nonzero edges to collapse to literal 0.
        low_collapsed = (low > 0) and (low_str == "0")
        high_collapsed = (high > 0) and (high_str == "0")

        if low_str != high_str and not low_collapsed and not high_collapsed:
            return f"{low_str}-{high_str}"

    return f"{format_bin_value(low)}-{format_bin_value(high)}"

def format_bin_labels_from_edges(edges):
    """Format bin labels using a single precision for all edges on an axis."""
    edges = np.asarray(edges, dtype=float)
    if edges.size < 2:
        return []

    for precision in range(1, 11):
        edge_labels = []
        valid_precision = True
        for edge in edges:
            edge_str = f"{edge:.{precision}f}"
            if edge > 0 and float(edge_str) == 0.0:
                valid_precision = False
                break
            edge_labels.append(edge_str)

        if not valid_precision:
            continue
        if all(edge_labels[i] != edge_labels[i + 1] for i in range(len(edge_labels) - 1)):
            return [f"{edge_labels[i]}-{edge_labels[i + 1]}" for i in range(len(edge_labels) - 1)]

    return [f"{format_bin_value(edges[i])}-{format_bin_value(edges[i + 1])}" for i in range(len(edges) - 1)]

def fill_empty_heatmap_bins(pivot_df):
    """Fill NaN bins using simple 2D interpolation for smoother heatmaps.

    This is a visualization aid only; raw values are unchanged upstream.
    """
    if not pivot_df.isna().to_numpy().any():
        return pivot_df

    filled = pivot_df.copy()
    filled = filled.interpolate(axis=1, limit_direction="both")
    filled = filled.interpolate(axis=0, limit_direction="both")

    if filled.isna().to_numpy().any():
        mean_value = np.nanmean(pivot_df.to_numpy())
        if np.isnan(mean_value):
            mean_value = 0.0
        filled = filled.fillna(mean_value)
    return filled

def drop_sparse_joint_heatmap_rows(pivot_df1, pivot_df2, max_empty_cells_per_row):
    """Drop y-bin rows where either panel has too many empty bins.

    The threshold is strict: a row must have fewer than max_empty_cells_per_row
    empty cells in both panels to remain visible.
    """
    if max_empty_cells_per_row is None:
        return pivot_df1, pivot_df2, pivot_df1.index
    if max_empty_cells_per_row < 0:
        raise ValueError("max_empty_cells_per_row must be >= 0")

    empty_counts_1 = pivot_df1.isna().sum(axis=1)
    empty_counts_2 = pivot_df2.isna().sum(axis=1)
    rows_to_keep = (empty_counts_1 < max_empty_cells_per_row) & (empty_counts_2 < max_empty_cells_per_row)

    if not rows_to_keep.any():
        # Keep the least sparse row to avoid plotting an empty heatmap.
        combined_empty = (empty_counts_1 + empty_counts_2)
        best_row = combined_empty.idxmin()
        rows_to_keep = pivot_df1.index == best_row

    kept_index = pivot_df1.index[rows_to_keep]
    return pivot_df1.loc[kept_index], pivot_df2.loc[kept_index], kept_index

def filter_mutation_table_to_visible_heatmap_rows(mut_df, y_edges, kept_y_bin_positions, y_column="mash_SB", logy=False):
    """Remove mutation-table rows whose y values fall into dropped heatmap rows."""
    if mut_df is None:
        return None
    if mut_df.is_empty():
        return mut_df

    mut_pdf = mut_df.select([y_column]).to_pandas()
    y_values = mut_pdf[y_column].to_numpy(dtype=float)
    valid = np.isfinite(y_values)
    if logy:
        valid &= y_values > 0
        y_values = np.where(valid, np.log10(y_values), np.nan)

    y_bins = np.full(len(y_values), -1, dtype=int)
    if np.any(valid):
        y_bins[valid] = np.clip(np.digitize(y_values[valid], y_edges, right=False) - 1, 0, len(y_edges) - 2)

    keep_mask = valid & np.isin(y_bins, np.asarray(kept_y_bin_positions, dtype=int))
    return mut_df.filter(pl.Series(keep_mask))

def build_mutation_ranges_from_mash_bins(mut_df, mash_col, mutation_col, mash_bin_edges):
    """Map each mash-distance bin to the observed mutation-percentage range."""
    empty = pd.DataFrame(columns=["mash_low", "mash_high", "mash_mid", "mut_low", "mut_high", "count"])
    if mut_df is None or mut_df.is_empty():
        return empty

    pdf = mut_df.select([mash_col, mutation_col]).drop_nulls().to_pandas()
    if pdf.empty:
        return empty

    mash_values = pdf[mash_col].to_numpy(dtype=float)
    mut_values = pdf[mutation_col].to_numpy(dtype=float)
    valid = np.isfinite(mash_values) & np.isfinite(mut_values)
    if not np.any(valid):
        return empty

    edges = np.asarray(mash_bin_edges, dtype=float)
    if edges.size < 2:
        raise ValueError("mash_bin_edges must contain at least two values")

    mash_values = mash_values[valid]
    mut_values = mut_values[valid]
    bin_ids = np.clip(np.digitize(mash_values, edges, right=False) - 1, 0, len(edges) - 2)

    grouped = (
        pd.DataFrame({"bin": bin_ids, "mut": mut_values})
        .groupby("bin", as_index=False)
        .agg(mut_low=("mut", "min"), mut_high=("mut", "max"), count=("mut", "size"))
        .sort_values("bin")
    )
    if grouped.empty:
        return empty

    grouped["mash_low"] = grouped["bin"].map(lambda b: edges[int(b)])
    grouped["mash_high"] = grouped["bin"].map(lambda b: edges[int(b) + 1])
    grouped["mash_mid"] = 0.5 * (grouped["mash_low"] + grouped["mash_high"])
    return grouped[["mash_low", "mash_high", "mash_mid", "mut_low", "mut_high", "count"]]

def plot_heatmap_with_bins(df, x, y, colour_column, n_x_bins, n_y_bins, title="", x_label="", y_label="", logx=False, logy=False):
    fig, ax = plt.subplots(figsize=(6, 6))
    if logx:
        df = df.with_columns(safe_log10_expr(x, f"{x}_log"))
        x = f"{x}_log"
    if logy:
        df = df.with_columns(safe_log10_expr(y, f"{y}_log"))
        y = f"{y}_log"
    # Plot a 10x10 heatmap by binning D_AB and M_SB into bins
    heatmap_data = df.select([x, y, colour_column]).to_pandas()
    heatmap_data = heatmap_data.replace([np.inf, -np.inf], np.nan).dropna(subset=[x, y, colour_column])
    if heatmap_data.empty:
        raise ValueError(f"No valid rows remain for heatmap after filtering non-finite values in {x}, {y}, and {colour_column}.")
    x_edges = np.linspace(heatmap_data[x].min(), heatmap_data[x].max(), n_x_bins + 1)
    y_edges = np.linspace(heatmap_data[y].min(), heatmap_data[y].max(), n_y_bins + 1)
    heatmap_data["X_bin"] = np.clip(np.digitize(heatmap_data[x], x_edges, right=False) - 1, 0, n_x_bins - 1)
    heatmap_data["Y_bin"] = np.clip(np.digitize(heatmap_data[y], y_edges, right=False) - 1, 0, n_y_bins - 1)
    X_bin_labels = [f"{x_edges[i]:.2f}-{x_edges[i+1]:.2f}" for i in range(len(x_edges)-1)]
    Y_bin_labels = [f"{y_edges[i]:.2f}-{y_edges[i+1]:.2f}" for i in range(len(y_edges)-1)]
    if logx:
        X_bin_labels = format_bin_labels_from_edges(10 ** x_edges)
    if logy:
        Y_bin_labels = format_bin_labels_from_edges(10 ** y_edges)
    heatmap_data_pivot = (
        heatmap_data.groupby(["Y_bin", "X_bin"]) [colour_column]
        .mean()
        .unstack()
        .reindex(index=range(n_y_bins), columns=range(n_x_bins))
    )
    sns.heatmap(heatmap_data_pivot, ax=ax, cmap="viridis", cbar_kws={"shrink": 0.9, "pad": 0.02})#, vmin=0.5, vmax=1)
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=11, weight="semibold")
    ax.set_xticks(np.arange(n_x_bins) + 0.5)
    ax.set_xticklabels(X_bin_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_y_bins) + 0.5)
    ax.set_yticklabels(Y_bin_labels, rotation=0)
    ax.tick_params(axis="both", which="major", labelsize=8, width=0.8, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    return fig

def compute_mutation_percentage(fasta1, fasta2):
    # read the two fasta files and compute the percentage of positions that are different
    with open(fasta1) as f1, open(fasta2) as f2:
        seq1 = "".join([line.strip() for line in f1 if not line.startswith(">")])
        seq2 = "".join([line.strip() for line in f2 if not line.startswith(">")])
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be the same length")
    num_mutations = sum(1 for a, b in zip(seq1, seq2) if a != b)
    return num_mutations / len(seq1)

def make_table_of_mash_vs_mutation(results_dir, seq_length, n_simulation_sets, mutation_counts, starting_set=1):
    results = []
    min_mutations = min(mutation_counts)
    max_mutations = max(mutation_counts)
    for i in range(starting_set, n_simulation_sets + 1):
        results_file = f"{results_dir}/simulation_set_{i}/kmer_simulation.ID_DM.seq_length={seq_length}_num_reads=1000.min_mutations={min_mutations}.max_mutations={max_mutations}.tsv"
        df = pl.read_csv(results_file, separator="\t")
        df = df.filter(pl.col("mash_AS") >= pl.col("mash_SB"))
        df = add_failure_masked_columns_for_id_dm(df)
        n_mutations_to_add_to_B = int(0.01 * seq_length)
        A_fasta_file = os.path.join(results_dir, f"simulation_set_{i}", "references", f"original_simulated_sequence.length={seq_length}.fasta")
        for row in df.iter_rows(named=True):
            mash_AB2 = row["mash_AB"]
            mash_B2S = row["mash_SB"]
            num_mutations_i = row["num_mutations_in_B"]
            num_mutations_j = row["num_mutations_in_S"]
            S_fasta_file = os.path.join(results_dir, f"simulation_set_{i}", "references", f"simulated_sequence_with_{num_mutations_i}.length={seq_length}_with_{num_mutations_j}_mutations.fasta")
            B2_fasta_file = os.path.join(results_dir, f"simulation_set_{i}","references", f"simulated_sequence_with_{num_mutations_i}_B2.length={seq_length}_with_{n_mutations_to_add_to_B}_mutations.fasta")
            mutation_percentage_AB2 = compute_mutation_percentage(A_fasta_file, B2_fasta_file)
            mutation_percentage_SB2 = compute_mutation_percentage(S_fasta_file, B2_fasta_file)
            results.append({"mash_AB": mash_AB2, "mash_SB": mash_B2S, "mutation_percentage_AB": mutation_percentage_AB2, "mutation_percentage_SB": mutation_percentage_SB2})
    results_df = pl.DataFrame(results)
    return results_df

def plot_joint_heatmap_with_bins(df, x, y, colour_column1, colour_column2, n_x_bins, n_y_bins, title="", x_label="", y_label="", logx=False, logy=False, 
                                 fill_empty_bins=False, max_empty_cells_per_row=None, cmap="viridis", mut_df=None):
    width = 6
    height = 3
    fig = plt.figure(figsize=(width, height))
    # Layout: left heatmap | spacer | right heatmap | colorbar
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 0.05, 1, 0.05], wspace=0.04)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 2], sharey=ax_left)
    cax = fig.add_subplot(gs[0, 3])
    axs = [ax_left, ax_right]
    if logx:
        df = df.with_columns(safe_log10_expr(x, f"{x}_log"))
        x = f"{x}_log"
    if logy:
        df = df.with_columns(safe_log10_expr(y, f"{y}_log"))
        y = f"{y}_log"
    heatmap_data1 = df.select([x, y, colour_column1]).to_pandas()
    heatmap_data1 = heatmap_data1.replace([np.inf, -np.inf], np.nan).dropna(subset=[x, y, colour_column1])
    heatmap_data2 = df.select([x, y, colour_column2]).to_pandas()
    heatmap_data2 = heatmap_data2.replace([np.inf, -np.inf], np.nan).dropna(subset=[x, y, colour_column2])
    if heatmap_data1.empty or heatmap_data2.empty:
        raise ValueError(f"No valid rows remain for heatmap after filtering non-finite values in {x}, {y}, {colour_column1}, or {colour_column2}.")
    x_edges = np.linspace(min(heatmap_data1[x].min(), heatmap_data2[x].min()), max(heatmap_data1[x].max(), heatmap_data2[x].max()), n_x_bins + 1)
    y_edges = np.linspace(min(heatmap_data1[y].min(), heatmap_data2[y].min()), max(heatmap_data1[y].max(), heatmap_data2[y].max()), n_y_bins + 1)
    heatmap_data1["X_bin"] = np.clip(np.digitize(heatmap_data1[x], x_edges, right=False) - 1, 0, n_x_bins - 1)
    heatmap_data1["Y_bin"] = np.clip(np.digitize(heatmap_data1[y], y_edges, right=False) - 1, 0, n_y_bins - 1)
    heatmap_data2["X_bin"] = np.clip(np.digitize(heatmap_data2[x], x_edges, right=False) - 1, 0, n_x_bins - 1)
    heatmap_data2["Y_bin"] = np.clip(np.digitize(heatmap_data2[y], y_edges, right=False) - 1, 0, n_y_bins - 1)
    X_bin_labels = [f"{x_edges[i]:.2f}-{x_edges[i+1]:.2f}" for i in range(len(x_edges)-1)]
    Y_bin_labels = [f"{y_edges[i]:.2f}-{y_edges[i+1]:.2f}" for i in range(len(y_edges)-1)]
    if logx:
        X_bin_labels = format_bin_labels_from_edges(10 ** x_edges)
    if logy:
        Y_bin_labels = format_bin_labels_from_edges(10 ** y_edges)
    heatmap_data1_pivot = (
        heatmap_data1.groupby(["Y_bin", "X_bin"]) [colour_column1]
        .mean()
        .unstack()
        .reindex(index=range(n_y_bins), columns=range(n_x_bins))
    )
    heatmap_data2_pivot = (
        heatmap_data2.groupby(["Y_bin", "X_bin"]) [colour_column2]
        .mean()
        .unstack()
        .reindex(index=range(n_y_bins), columns=range(n_x_bins))
    )

    heatmap_data1_pivot, heatmap_data2_pivot, kept_y_bin_positions = drop_sparse_joint_heatmap_rows(
        heatmap_data1_pivot,
        heatmap_data2_pivot,
        max_empty_cells_per_row=max_empty_cells_per_row,
    )
    if heatmap_data1_pivot.shape[0] == 0:
        raise ValueError("No y-bin rows remain after dropping sparse heatmap rows.")

    y_bin_positions = heatmap_data1_pivot.index.to_list()
    y_bin_labels_to_plot = [Y_bin_labels[i] for i in y_bin_positions]

    if fill_empty_bins:
        heatmap_data1_pivot = fill_empty_heatmap_bins(heatmap_data1_pivot)
        heatmap_data2_pivot = fill_empty_heatmap_bins(heatmap_data2_pivot)

    # Use the same color scale across both panels and draw a single shared colorbar.
    combined_values = np.concatenate([
        heatmap_data1_pivot.to_numpy().ravel(),
        heatmap_data2_pivot.to_numpy().ravel(),
    ])
    finite_values = combined_values[np.isfinite(combined_values)]
    if len(finite_values) == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(np.min(finite_values)), float(np.max(finite_values))
        if vmin == vmax:
            vmax = vmin + 1e-9

    sns.heatmap(heatmap_data1_pivot, ax=axs[0], cmap=cmap, cbar=False, vmin=vmin, vmax=vmax)
    sns.heatmap(heatmap_data2_pivot, ax=axs[1], cmap=cmap, cbar=False, vmin=vmin, vmax=vmax)

    shared_mappable = axs[1].collections[0]
    cbar = fig.colorbar(shared_mappable, cax=cax)
    cbar.ax.tick_params(labelsize=6, width=0.7, length=2)
    cbar.outline.set_visible(False)
    cbar.outline.set_linewidth(0)
    for i, ax in enumerate(axs):
        ax.set_xlabel(x_label, fontsize=10)
        if i == 0:
            ax.set_ylabel(y_label, fontsize=10)
        else:
            ax.set_ylabel("")
        if title:
            ax.set_title(f"{title} - {'KMA' if i == 0 else 'Kallisto'}", fontsize=11, weight="semibold")
        else:
            ax.set_title("KMA" if i == 0 else "Kallisto", fontsize=10, weight="semibold")
        ax.set_xticks(np.arange(n_x_bins) + 0.5)
        ax.set_xticklabels(X_bin_labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(y_bin_labels_to_plot)) + 0.5)
        if i == 0:
            ax.set_yticklabels(y_bin_labels_to_plot, rotation=0)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.tick_params(axis="both", which="major", labelsize=6.5, width=0.8, length=3)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
    # Extra bottom room for rotated x tick labels.
    fig.subplots_adjust(left=0.15, right=0.96, top=0.93, bottom=0.28)

    # if there is mut_df, plot mash vs mutation fraction using the same bins as the heatmap
    if mut_df is not None:
        mut_df = filter_mutation_table_to_visible_heatmap_rows(
            mut_df,
            y_edges=y_edges,
            kept_y_bin_positions=kept_y_bin_positions,
            y_column="mash_SB",
            logy=logy,
        )
        if mut_df is None or mut_df.is_empty():
            raise ValueError("No mutation rows remain after filtering to visible heatmap rows.")

        mut_pdf = mut_df.select(["mutation_percentage_AB", "mutation_percentage_SB"]).to_pandas()
        mutation_edges_ab = np.linspace(mut_pdf["mutation_percentage_AB"].min(), mut_pdf["mutation_percentage_AB"].max(), n_x_bins + 1)
        mutation_edges_sb = np.linspace(mut_pdf["mutation_percentage_SB"].min(), mut_pdf["mutation_percentage_SB"].max(), n_y_bins + 1)

        mash_edges_for_ab = (10 ** x_edges) if logx else x_edges
        mash_edges_for_sb = (10 ** y_edges) if logy else y_edges
        mash_to_mut_ab = build_mutation_ranges_from_mash_bins(
            mut_df,
            mash_col="mash_AB",
            mutation_col="mutation_percentage_AB",
            mash_bin_edges=mash_edges_for_ab,
        )
        mash_to_mut_sb = build_mutation_ranges_from_mash_bins(
            mut_df,
            mash_col="mash_SB",
            mutation_col="mutation_percentage_SB",
            mash_bin_edges=mash_edges_for_sb,
        )

        fig2 = plot_mash_vs_mutation_fraction(
            mut_df,
            x_col="mutation_percentage_AB",
            y_col="mash_AB",
            title="",
            x_label="Percentage difference (A vs B)",
            y_label="Mash distance (A vs B)",
            range_edges=mutation_edges_ab,
            mash_to_mutation_ranges=mash_to_mut_ab,
            point_alpha=0.4
        )
        fig3 = plot_mash_vs_mutation_fraction(
            mut_df,
            x_col="mutation_percentage_SB",
            y_col="mash_SB",
            title="",
            x_label="Percentage difference (S vs B)",
            y_label="Mash distance (S vs B)",
            bin_width=0.05,
            range_edges=mutation_edges_sb,
            mash_to_mutation_ranges=mash_to_mut_sb,
            point_alpha=0.4
        )
        return fig, fig2, fig3
    else:
        return fig

def make_plots_using_D_and_M(results_dir, seq_length, n_simulation_sets, mutation_counts, results_dir_for_ID_DM):
    ## ID_M plot
    file_name = f"kmer_simulation.ID_M.seq_length={seq_length}_num_reads=1000.min_mutations={mutation_counts[0]}.max_mutations={mutation_counts[-1]}.tsv"
    df = parse_results_for_plotting(file_name, results_dir, seq_length, n_simulation_sets, mutation_counts)
    fig = plot_scatter_with_smooth_line(df, x="matching_M", y_kma="kma_proportion_assigned", y_kallisto="kallisto_proportion_assigned", 
                                        title="", x_label="M", y_label="Proportion input reads assigned", logx=False,
                                        fit_method="window", window_n=40, window_min_points=2)
    fig.savefig(f"{results_dir}/plots/ID_M_plot.png", dpi=300)
    ## ID_D plot
    file_name = f"kmer_simulation.ID_D.seq_length={seq_length}_num_reads=1000.min_mutations={mutation_counts[0]}.max_mutations={mutation_counts[-1]}.tsv"
    df = parse_results_for_plotting(file_name, results_dir, seq_length, n_simulation_sets, mutation_counts)
    fig = plot_scatter_with_smooth_line(df, x="distinguishibility_D", y_kma="kma_proportion_assigned", y_kallisto="kallisto_proportion_assigned", title="", x_label="D", y_label="Proportion assigned reads correct", logx=False)
    fig.savefig(f"{results_dir}/plots/ID_D_plot.png", dpi=300)
    # ## ID_DM plots
    # file_name=f"kmer_simulation.ID_DM.seq_length={seq_length}_num_reads=1000.min_mutations={mutation_counts[0]}.max_mutations={mutation_counts[-1]}.tsv"
    # df = parse_results_for_plotting(file_name, results_dir_for_ID_DM, seq_length, n_simulation_sets, mutation_counts,
    #                                 starting_set=1)
    # # remove where mash AS is less than mash SB, as then reads should be put on A rather than B
    # df = df.filter(pl.col("mash_AS") >= pl.col("mash_SB"))
    # df = add_failure_masked_columns_for_id_dm(df)
    # df_mut = make_table_of_mash_vs_mutation(results_dir_for_ID_DM, seq_length, n_simulation_sets, mutation_counts, starting_set=1)
    # # kallisto - proportion correctly assigned to B
    # # fig = plot_heatmap_with_bins(df, x="distinguishibility_D_AB", y="matching_M_SB", colour_column="kallisto_proportion_assigned_to_B", n_x_bins=10, n_y_bins=10, title="", x_label="D", y_label="M", logx=True, logy=True)
    # # fig.tight_layout()
    # # fig.savefig(f"{results_dir}/plots/ID_DM_plot_kallisto.png", dpi=300)
    # # ## KMA - proportion correctly assigned to B
    # # fig = plot_heatmap_with_bins(df, x="distinguishibility_D_AB", y="matching_M_SB", colour_column="kma_proportion_assigned_to_B", n_x_bins=10, n_y_bins=10, title="", x_label="D", y_label="M", logx=True, logy=True)
    # # fig.tight_layout()
    # # fig.savefig(f"{results_dir}/plots/ID_DM_plot_kma.png", dpi=300)
    # fig1, fig2, fig3 = plot_joint_heatmap_with_bins(df, x="distinguishibility_D_AB", y="matching_M_SB", colour_column1="kma_proportion_assigned_to_B_plot", colour_column2="kallisto_proportion_assigned_to_B_plot", 
    #                                     n_x_bins=10, n_y_bins=10, title="", x_label="D", y_label="M", logx=True, logy=True, max_empty_cells_per_row=2,
    #                                     mut_df=df_mut)
    # fig1.tight_layout()
    # fig1.savefig(f"{results_dir}/plots/ID_DM_plot-both.png", dpi=300)
    # fig2.tight_layout()
    # fig2.savefig(f"{results_dir}/plots/ID_DM_plot-mash_vs_mutation_AB.png", dpi=300)
    # fig3.tight_layout()
    # fig3.savefig(f"{results_dir}/plots/ID_DM_plot-mash_vs_mutation_SB.png", dpi=300)
    # # ## kallisto - proportion assigned
    # # fig = plot_heatmap_with_bins(df, x="distinguishibility_D_AB", y="matching_M_SB", colour_column="kallisto_proportion_assigned", n_x_bins=10, n_y_bins=10, title="", x_label="D", y_label="M", logx=True, logy=True)
    # # fig.tight_layout()
    # # fig.savefig(f"{results_dir}/plots/ID_DM_plot_kallisto_total_assigned.png", dpi=300)
    # # ## KMA - proportion assigned
    # # fig = plot_heatmap_with_bins(df, x="distinguishibility_D_AB", y="matching_M_SB", colour_column="kma_proportion_assigned", n_x_bins=10, n_y_bins=10, title="", x_label="D", y_label="M", logx=True, logy=True)
    # # fig.tight_layout()
    # # fig.savefig(f"{results_dir}/plots/ID_DM_plot_kma_total_assigned.png", dpi=300)
    # fig = plot_joint_heatmap_with_bins(df, x="distinguishibility_D_AB", y="matching_M_SB", colour_column1="kma_proportion_assigned_plot", 
    #                                     colour_column2="kallisto_proportion_assigned_plot", n_x_bins=10, n_y_bins=10, title="", x_label="D", 
    #                                     y_label="M", logx=True, logy=True, max_empty_cells_per_row=2, cmap="mako")
    # fig.tight_layout()
    # fig.savefig(f"{results_dir}/plots/ID_DM_plot_total_assigned-both.png", dpi=300)
    ## RA_D plot
    file_name = f"kmer_simulation.RA_D.seq_length={seq_length}_num_reads=1000.min_mutations={mutation_counts[0]}.max_mutations={mutation_counts[-1]}.tsv"
    df = parse_results_for_plotting(file_name, results_dir, seq_length, n_simulation_sets, mutation_counts)
    fig = plot_scatter_with_smooth_line(df, x="distinguishibility_D", y_kma="kma_log_error", y_kallisto="kallisto_log_error", 
                                        title="", x_label="D", y_label="absolute log error", logx=False, absy=True,
                                        fit_method="window", window_n=40, window_min_points=2)
    fig.tight_layout()
    fig.savefig(f"{results_dir}/plots/RA_D_plot.png", dpi=300)
    ## RA_prop plot
    file_name = f"kmer_simulation.RA_prop.seq_length={seq_length}_num_reads=1000_with_{int(0.025 * seq_length)}_mutations.tsv"
    df = parse_results_for_plotting(file_name, results_dir, seq_length, n_simulation_sets, mutation_counts)
    fig = plot_scatter_with_smooth_line(df, x="proportion_original", y_kallisto="kallisto_log_error", y_kma="kma_log_error", 
                                        title="", x_label="Proportion of original sequence", y_label="log error", logx=False, absy=False,
                                        fit_method="window", window_n=40, window_min_points=2)
    fig.tight_layout()
    ax = fig.axes[0]
    axis_speed_up(ax, axis_to_speed_up="x", speed_up_factor=5, speed_up_start=0.2, speed_up_end=0.8)
    fig.savefig(f"{results_dir}/plots/RA_prop_plot.png", dpi=300)
    ## RA_deltaM plot
    file_name = f"kmer_simulation.RA_deltaM.seq_length={seq_length}_num_reads=1000.min_mutations={mutation_counts[0]}.max_mutations={mutation_counts[-1]}.tsv"
    df = parse_results_for_plotting(file_name, results_dir, seq_length, n_simulation_sets, mutation_counts)
    df = df.with_columns(
        pl.when(pl.col("mutated_sequence") == "second_original")
        .then(-pl.col("delta_M"))
        .otherwise(pl.col("delta_M"))
        .alias("delta_M")
    )
    fig = plot_scatter_with_smooth_line(df, x="delta_M", y_kma="kma_log_error", y_kallisto="kallisto_log_error", 
                                        title="", x_label="delta M", y_label="log error", logx=False, absy=False,
                                        fit_method="window", window_n=100, window_min_points=2)
    ax = fig.axes[0]
    axis_speed_up(ax, axis_to_speed_up="x", speed_up_factor=5, speed_up_start=-0.8, speed_up_end=0.8)
    fig.tight_layout()
    fig.savefig(f"{results_dir}/plots/RA_deltaM_plot.png", dpi=300)

def plot_mash_vs_mutation_fraction(
    frame,
    x_col,
    y_col,
    title,
    x_label,
    y_label,
    bin_width=0.05,
    point_alpha=0.18,
    range_edges=None,
    mash_to_mutation_ranges=None,
):
    pdf = frame.select([x_col, y_col]).drop_nulls().to_pandas().sort_values(x_col)
    if pdf.empty:
        raise ValueError(f"No valid rows for {x_col} and {y_col}")

    x = pdf[x_col].astype(float)
    y = pdf[y_col].astype(float)

    if range_edges is None:
        max_x = float(np.nanmax(x))
        max_edge = max(bin_width, float(np.ceil(max_x / bin_width) * bin_width))
        bin_edges = np.arange(0.0, min(1.0, max_edge) + (bin_width * 0.5), bin_width)
        if len(bin_edges) < 2:
            bin_edges = np.array([0.0, max(1e-6, max_x)])
        if bin_edges[-1] < max_x:
            bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)
    else:
        bin_edges = np.asarray(range_edges, dtype=float)
        if len(bin_edges) < 2:
            raise ValueError("range_edges must contain at least two values")

    bin_ids = np.clip(np.digitize(x, bin_edges, right=False) - 1, 0, len(bin_edges) - 2)
    summary = (
        pd.DataFrame({"x": x, "y": y, "bin": bin_ids})
        .groupby("bin", as_index=False)
        .agg(
            x_left=("x", "min"),
            x_right=("x", "max"),
            x_mid=("x", "mean"),
            y_mean=("y", "mean"),
            y_median=("y", "median"),
            y_q25=("y", lambda s: s.quantile(0.25)),
            y_q75=("y", lambda s: s.quantile(0.75)),
            count=("y", "size"),
        )
        .sort_values("x_mid")
    )

    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    palette = sns.color_palette("deep")

    ax.scatter(
        x,
        y,
        s=16,
        alpha=point_alpha,
        color='lightgray',
        edgecolor="none",
        rasterized=True,
        label="Replicates",
    )

    # Keep the best-fit relation implicit: show range-to-fit connectors, not a continuous fit line.
    # for _, row in summary.iterrows():
    #     ax.hlines(row["y_median"], row["x_left"], row["x_right"], color=palette[1], linewidth=2.0, alpha=0.9)

    # Draw line ranges that map each heatmap mash bin to mutation-% range.
    if mash_to_mutation_ranges is not None and len(mash_to_mutation_ranges) > 0:
        for _, row in mash_to_mutation_ranges.iterrows():
            y_pos = float(row["mash_mid"])
            x_low = float(row["mut_low"])
            x_high = float(row["mut_high"])
            ax.hlines(y_pos, x_low, x_high, color="tab:red", linewidth=1.5, alpha=0.95, zorder=2)
            # ax.vlines([x_low, x_high], 0, y_pos, color="0.6", linewidth=0.9, alpha=0.9, zorder=1)

    ax.set_title(title, fontsize=14, weight="semibold", pad=8)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xlim(0, min(1.0, max(0.0, float(np.nanmax(x)) * 1.02)))

    # Fixed grid spacing: 1% on x (no % sign), and 0.01 on y.
    ax.xaxis.set_major_locator(MultipleLocator(0.01))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.0f}"))
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.tick_params(axis="x", which="major", length=4, width=0.8, direction="out")
    ax.tick_params(axis="y", which="major", length=4, width=0.8, direction="out")
    ax.grid(True, which="major", axis="both", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_box_aspect(1)
    sns.despine(ax=ax)
    return fig

def make_plots_using_mash_distance(results_dir, seq_length, n_simulation_sets, mutation_counts, results_dir_for_ID_DM, restrict_to_where_mash_not_1=False):
    ## ID_M plot
    file_name = f"kmer_simulation.ID_M.seq_length={seq_length}_num_reads=1000.min_mutations={mutation_counts[0]}.max_mutations={mutation_counts[-1]}.tsv"
    df = parse_results_for_plotting(file_name, results_dir, seq_length, n_simulation_sets, mutation_counts)
    if restrict_to_where_mash_not_1:
        df = df.filter(pl.col("mash_distance") < 1.0)
    fig = plot_scatter_with_smooth_line(df, x="mash_distance", y_kma="kma_proportion_assigned", y_kallisto="kallisto_proportion_assigned", 
                                        title="", x_label="mash A-S", y_label="Proportion input reads assigned", logx=False,
                                        fit_method="window", window_n=20, window_min_points=2)
    fig.savefig(f"{results_dir}/plots/ID_M_plot-mash.png", dpi=300)
    ## ID_D plot
    file_name = f"kmer_simulation.ID_D.seq_length={seq_length}_num_reads=1000.min_mutations={mutation_counts[0]}.max_mutations={mutation_counts[-1]}.tsv"
    df = parse_results_for_plotting(file_name, results_dir, seq_length, n_simulation_sets, mutation_counts)
    if restrict_to_where_mash_not_1:
        df = df.filter(pl.col("mash_distance") < 1.0)
    fig = plot_scatter_with_smooth_line(df, x="mash_distance", y_kma="kma_proportion_assigned", y_kallisto="kallisto_proportion_assigned", title="", x_label="mash A-B", y_label="Proportion assigned reads correct", logx=False)
    fig.savefig(f"{results_dir}/plots/ID_D_plot-mash.png", dpi=300)
    # ## ID_DM plots
    # file_name=f"kmer_simulation.ID_DM.seq_length={seq_length}_num_reads=1000.min_mutations={mutation_counts[0]}.max_mutations={mutation_counts[-1]}.tsv"
    # df = parse_results_for_plotting(file_name, results_dir_for_ID_DM, seq_length, n_simulation_sets, mutation_counts,
    #                                 starting_set=1)
    # # remove where mash AS is less than mash SB, as then reads should be put on A rather than B
    # df = df.filter(pl.col("mash_AS") >= pl.col("mash_SB"))
    # df = add_failure_masked_columns_for_id_dm(df)
    # df_mut = make_table_of_mash_vs_mutation(results_dir_for_ID_DM, seq_length, n_simulation_sets, mutation_counts, starting_set=1)
    # # filter where A and B counts are both 0
    # # # kallisto - proportion correctly assigned to B
    # # fig = plot_heatmap_with_bins(df, x="mash_AB", y="mash_SB", colour_column="kallisto_proportion_assigned_to_B", n_x_bins=10, n_y_bins=10, title="", x_label="mash A-B", y_label="mash S-B")#, logx=True, logy=True)
    # # fig.tight_layout()
    # # fig.savefig(f"{results_dir}/plots/ID_DM_plot_kallisto-mash.png", dpi=300)
    # # ## KMA - proportion correctly assigned to B
    # # fig = plot_heatmap_with_bins(df, x="mash_AB", y="mash_SB", colour_column="kma_proportion_assigned_to_B", n_x_bins=10, n_y_bins=10, title="", x_label="mash A-B", y_label="mash S-B")#, logx=True, logy=True)
    # # fig.tight_layout()
    # # fig.savefig(f"{results_dir}/plots/ID_DM_plot_kma-mash.png", dpi=300)
    # fig1, fig2, fig3 = plot_joint_heatmap_with_bins(df, x="mash_AB", y="mash_SB", colour_column1="kma_proportion_assigned_to_B_plot", 
    #                                     colour_column2="kallisto_proportion_assigned_to_B_plot", n_x_bins=10, n_y_bins=10, 
    #                                     title="", x_label="mash A-B", y_label="mash S-B", logx=True, logy=True, max_empty_cells_per_row=2,
    #                                     mut_df=df_mut)
    # fig1.tight_layout()
    # fig1.savefig(f"{results_dir}/plots/ID_DM_plot-both-mash.png", dpi=300)
    # fig2.tight_layout()
    # fig2.savefig(f"{results_dir}/plots/ID_DM_plot-mash_vs_mutation_AB-mash.png", dpi=300)
    # fig3.tight_layout()
    # fig3.savefig(f"{results_dir}/plots/ID_DM_plot-mash_vs_mutation_SB-mash.png", dpi=300)
    # # # ## kallisto - proportion assigned
    # # # fig = plot_heatmap_with_bins(df, x="mash_AB", y="mash_SB", colour_column="kallisto_proportion_assigned", n_x_bins=10, n_y_bins=10, title="", x_label="mash A-B", y_label="mash S-B")#, logx=True, logy=True)
    # # # fig.tight_layout()
    # # # fig.savefig(f"{results_dir}/plots/ID_DM_plot_kallisto_total_assigned-mash.png", dpi=300)
    # # # ## KMA - proportion assigned
    # # # fig = plot_heatmap_with_bins(df, x="mash_AB", y="mash_SB", colour_column="kma_proportion_assigned", n_x_bins=10, n_y_bins=10, title="", x_label="mash A-B", y_label="mash S-B")#, logx=True, logy=True)
    # # # fig.tight_layout()
    # # # fig.savefig(f"{results_dir}/plots/ID_DM_plot_kma_total_assigned-mash.png", dpi=300)
    # fig = plot_joint_heatmap_with_bins(df, x="mash_AB", y="mash_SB", colour_column1="kma_proportion_assigned_plot", 
    #                                    colour_column2="kallisto_proportion_assigned_plot", n_x_bins=10, n_y_bins=10, title="", x_label="mash A-B", 
    #                                    y_label="mash S-B", logx=True, logy=True, max_empty_cells_per_row=2, cmap="mako")
    # fig.tight_layout()
    # fig.savefig(f"{results_dir}/plots/ID_DM_plot_total_assigned_both-mash.png", dpi=300)
    ## RA_D plot
    file_name = f"kmer_simulation.RA_D.seq_length={seq_length}_num_reads=1000.min_mutations={mutation_counts[0]}.max_mutations={mutation_counts[-1]}.tsv"
    df = parse_results_for_plotting(file_name, results_dir, seq_length, n_simulation_sets, mutation_counts)
    if restrict_to_where_mash_not_1:
        df = df.filter(pl.col("mash_distance") < 1.0)
    fig = plot_scatter_with_smooth_line(df, x="mash_distance", y_kma="kma_log_error", y_kallisto="kallisto_log_error", 
                                        title="", x_label="mash A-B", y_label="absolute log error", logx=True, absy=True,
                                        fit_method="window", window_n=40, window_min_points=2)
    fig.tight_layout()
    fig.savefig(f"{results_dir}/plots/RA_D_plot-mash.png", dpi=300)
    ## RA_prop plot
    file_name = f"kmer_simulation.RA_prop.seq_length={seq_length}_num_reads=1000_with_{int(0.025 * seq_length)}_mutations.tsv"
    df = parse_results_for_plotting(file_name, results_dir, seq_length, n_simulation_sets, mutation_counts)
    fig = plot_scatter_with_smooth_line(df, x="proportion_original", y_kallisto="kallisto_log_error", y_kma="kma_log_error", 
                                        title="", x_label="Proportion of original sequence", y_label="log error", logx=False, absy=False,
                                        fit_method="window", window_n=40, window_min_points=2)
    fig.tight_layout()
    ax = fig.axes[0]
    # axis_speed_up(ax, axis_to_speed_up="x", speed_up_factor=5, speed_up_start=0.2, speed_up_end=0.8)
    fig.savefig(f"{results_dir}/plots/RA_prop_plot.png", dpi=300)
    ## RA_deltaM plot
    file_name = f"kmer_simulation.RA_deltaM.seq_length={seq_length}_num_reads=1000.min_mutations={mutation_counts[0]}.max_mutations={mutation_counts[-1]}.tsv"
    df = parse_results_for_plotting(file_name, results_dir, seq_length, n_simulation_sets, mutation_counts)
    if restrict_to_where_mash_not_1:
        df = df.filter(pl.col("mash_where_differ") < 1.0)
    df = df.with_columns(
        pl.when(pl.col("mutated_sequence") == "second_original")
        .then(pl.col("mash_where_differ"))
        .otherwise(-pl.col("mash_where_differ"))
        .alias("delta_M")
    )
    fig = plot_scatter_with_smooth_line(df, x="delta_M", y_kma="kma_log_error", y_kallisto="kallisto_log_error", 
                                        title="", x_label="mutation bias to B", y_label="log error", logx=False, absy=False,
                                        fit_method="window", window_n=25, window_min_points=2)
    ax = fig.axes[0]
    ax.set_ylabel("Fold change to A")
    # centre xaxis on 0
    x_min = min(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]))
    ax.set_xlim(-x_min, x_min)
    # axis_speed_up(ax, axis_to_speed_up="x", speed_up_factor=5, speed_up_start=-0.8, speed_up_end=0.8)
    fig.tight_layout()
    fig.savefig(f"{results_dir}/plots/RA_deltaM_plot-mash.png", dpi=300)

def convert_mash_to_jaccard(mash_distance, k):
    # mash_distance = - (1 / k) * log(2 * jaccard / (1 + jaccard))
    # rearranging gives:
    # jaccard = 1 / (1 + exp(mash_distance * k))
    return 1 / (1 + np.exp(mash_distance * k))

def make_plots_using_jaccard(results_dir, seq_length, n_simulation_sets, mutation_counts, results_dir_for_ID_DM, restrict_to_where_mash_not_1=False):
    ## ID_M plot
    file_name = f"kmer_simulation.ID_M.seq_length={seq_length}_num_reads=1000.min_mutations={mutation_counts[0]}.max_mutations={mutation_counts[-1]}.tsv"
    df = parse_results_for_plotting(file_name, results_dir, seq_length, n_simulation_sets, mutation_counts)
    df = df.with_columns(pl.col("mash_distance").map_elements(lambda x: convert_mash_to_jaccard(x, k=31)).alias("jaccard"))
    fig = plot_scatter_with_smooth_line(df, x="jaccard", y_kma="kma_proportion_assigned", y_kallisto="kallisto_proportion_assigned", 
                                        title="", x_label="Jaccard", y_label="Proportion input reads assigned", logx=False,
                                        fit_method="window", window_n=40, window_min_points=2)
    fig.savefig(f"{results_dir}/plots/ID_M_plot-jaccard.png", dpi=300)
    ## ID_D plot
    file_name = f"kmer_simulation.ID_D.seq_length={seq_length}_num_reads=1000.min_mutations={mutation_counts[0]}.max_mutations={mutation_counts[-1]}.tsv"
    df = parse_results_for_plotting(file_name, results_dir, seq_length, n_simulation_sets, mutation_counts)
    if restrict_to_where_mash_not_1:
        df = df.filter(pl.col("mash_distance") < 1.0)
    df = df.with_columns(pl.col("mash_distance").map_elements(lambda x: convert_mash_to_jaccard(x, k=31)).alias("jaccard"))
    fig = plot_scatter_with_smooth_line(df, x="jaccard", y_kma="kma_proportion_assigned", y_kallisto="kallisto_proportion_assigned", title="", x_label="Jaccard", y_label="Proportion assigned reads correct", logx=False)
    fig.savefig(f"{results_dir}/plots/ID_D_plot-jaccard.png", dpi=300)
    # ## ID_DM plots
    # file_name=f"kmer_simulation.ID_DM.seq_length={seq_length}_num_reads=1000.min_mutations={mutation_counts[0]}.max_mutations={mutation_counts[-1]}.tsv"
    # df = parse_results_for_plotting(file_name, results_dir_for_ID_DM, seq_length, n_simulation_sets, mutation_counts,
    #                                 starting_set=1)
    # # remove where mash AS is less than mash SB, as then reads should be put on A rather than B
    # df = df.filter(pl.col("mash_AS") >= pl.col("mash_SB"))
    # df = add_failure_masked_columns_for_id_dm(df)
    # df = df.with_columns(
    #     pl.col("mash_AB").map_elements(lambda x: convert_mash_to_jaccard(x, k=31)).alias("jaccard_AB"),
    #     pl.col("mash_SB").map_elements(lambda x: convert_mash_to_jaccard(x, k=31)).alias("jaccard_SB"),
    # )
    # # # kallisto - proportion correctly assigned to B
    # # fig = plot_heatmap_with_bins(df, x="jaccard_AB", y="jaccard_SB", colour_column="kallisto_proportion_assigned_to_B", n_x_bins=10, n_y_bins=10, title="", x_label="Jaccard A-B", y_label="Jaccard S-B")#, logx=True, logy=True)
    # # fig.tight_layout()
    # # fig.savefig(f"{results_dir}/plots/ID_DM_plot_kallisto-jaccard.png", dpi=300)
    # # ## KMA - proportion correctly assigned to B
    # # fig = plot_heatmap_with_bins(df, x="jaccard_AB", y="jaccard_SB", colour_column="kma_proportion_assigned_to_B", n_x_bins=10, n_y_bins=10, title="", x_label="Jaccard A-B", y_label="Jaccard S-B")#, logx=True, logy=True)
    # # fig.tight_layout()
    # # fig.savefig(f"{results_dir}/plots/ID_DM_plot_kma-jaccard.png", dpi=300)
    # fig = plot_joint_heatmap_with_bins(df, x="jaccard_AB", y="jaccard_SB", colour_column1="kma_proportion_assigned_to_B_plot", colour_column2="kallisto_proportion_assigned_to_B_plot", n_x_bins=10, n_y_bins=10, title="", x_label="Jaccard A-B", y_label="Jaccard S-B", max_empty_cells_per_row=2)#, logx=True, logy=True)
    # fig.tight_layout()
    # fig.savefig(f"{results_dir}/plots/ID_DM_plot-both-jaccard.png", dpi=300)
    # # ## kallisto - proportion assigned
    # # fig = plot_heatmap_with_bins(df, x="jaccard_AB", y="jaccard_SB", colour_column="kallisto_proportion_assigned", n_x_bins=10, n_y_bins=10, title="", x_label="Jaccard A-B", y_label="Jaccard S-B")#, logx=True, logy=True)
    # # fig.tight_layout()
    # # fig.savefig(f"{results_dir}/plots/ID_DM_plot_kallisto_total_assigned-jaccard.png", dpi=300)
    # # ## KMA - proportion assigned
    # # fig = plot_heatmap_with_bins(df, x="jaccard_AB", y="jaccard_SB", colour_column="kma_proportion_assigned", n_x_bins=10, n_y_bins=10, title="", x_label="Jaccard A-B", y_label="Jaccard S-B")#, logx=True, logy=True)
    # # fig.tight_layout()
    # # fig.savefig(f"{results_dir}/plots/ID_DM_plot_kma_total_assigned-jaccard.png", dpi=300)
    # fig = plot_joint_heatmap_with_bins(df, x="jaccard_AB", y="jaccard_SB", colour_column1="kma_proportion_assigned_plot", colour_column2="kallisto_proportion_assigned_plot", n_x_bins=10, n_y_bins=10, title="", x_label="Jaccard A-B", y_label="Jaccard S-B", max_empty_cells_per_row=2)#, logx=True, logy=True)
    # fig.tight_layout()
    # fig.savefig(f"{results_dir}/plots/ID_DM_plot_total_assigned_both-jaccard.png", dpi=300)
    # RA_D plot
    file_name = f"kmer_simulation.RA_D.seq_length={seq_length}_num_reads=1000.min_mutations={mutation_counts[0]}.max_mutations={mutation_counts[-1]}.tsv"
    df = parse_results_for_plotting(file_name, results_dir, seq_length, n_simulation_sets, mutation_counts)
    if restrict_to_where_mash_not_1:
        df = df.filter(pl.col("mash_distance") < 1.0)
    df = df.with_columns(pl.col("mash_distance").map_elements(lambda x: convert_mash_to_jaccard(x, k=31)).alias("jaccard"))
    fig = plot_scatter_with_smooth_line(df, x="jaccard", y_kma="kma_log_error", y_kallisto="kallisto_log_error", 
                                        title="", x_label="Jaccard A-B", y_label="absolute log error", logx=False, absy=True,
                                        fit_method="window", window_n=40, window_min_points=2)
    fig.tight_layout()
    fig.savefig(f"{results_dir}/plots/RA_D_plot-jaccard.png", dpi=300)
    ## RA_prop plot
    file_name = f"kmer_simulation.RA_prop.seq_length={seq_length}_num_reads=1000_with_{int(0.025 * seq_length)}_mutations.tsv"
    df = parse_results_for_plotting(file_name, results_dir, seq_length, n_simulation_sets, mutation_counts)
    fig = plot_scatter_with_smooth_line(df, x="proportion_original", y_kallisto="kallisto_log_error", y_kma="kma_log_error", 
                                        title="", x_label="Proportion of original sequence", y_label="log error", logx=False, absy=False,
                                        fit_method="window", window_n=40, window_min_points=2)
    fig.tight_layout()
    ax = fig.axes[0]
    # axis_speed_up(ax, axis_to_speed_up="x", speed_up_factor=5, speed_up_start=0.2, speed_up_end=0.8)
    fig.savefig(f"{results_dir}/plots/RA_prop_plot.png", dpi=300)
    ## RA_deltaM plot
    file_name = f"kmer_simulation.RA_deltaM.seq_length={seq_length}_num_reads=1000.min_mutations={mutation_counts[0]}.max_mutations={mutation_counts[-1]}.tsv"
    df = parse_results_for_plotting(file_name, results_dir, seq_length, n_simulation_sets, mutation_counts)
    if restrict_to_where_mash_not_1:
        df = df.filter(pl.col("mash_where_differ") < 1.0)
    df = df.with_columns(
        pl.when(pl.col("mutated_sequence") == "second_original")
        .then(1-convert_mash_to_jaccard(pl.col("mash_where_differ"), k=31))
        .otherwise(convert_mash_to_jaccard(pl.col("mash_where_differ"), k=31) -1)
        .alias("delta_M")
    )
    fig = plot_scatter_with_smooth_line(df, x="delta_M", y_kma="kma_log_error", y_kallisto="kallisto_log_error", 
                                        title="", x_label="delta jaccard", y_label="log error", logx=False, absy=False,
                                        fit_method="window", window_n=100, window_min_points=2)
    ax = fig.axes[0]
    axis_speed_up(ax, axis_to_speed_up="x", speed_up_factor=5, speed_up_start=-0.8, speed_up_end=0.8)
    fig.tight_layout()
    fig.savefig(f"{results_dir}/plots/RA_deltaM_plot-jaccard.png", dpi=300)

def plot_figures(results_dir, seq_length, n_simulation_sets, mutation_counts, results_dir_for_ID_DM):
    # make a folder to store plots
    os.makedirs(f"{results_dir}/plots", exist_ok=True)
    # make figures mash
    make_plots_using_mash_distance(results_dir, seq_length, n_simulation_sets, mutation_counts, results_dir_for_ID_DM, restrict_to_where_mash_not_1=True)
    # # make figures D and M
    make_plots_using_D_and_M(results_dir, seq_length, n_simulation_sets, mutation_counts, results_dir_for_ID_DM)
    # # plot using jaccard
    make_plots_using_jaccard(results_dir, seq_length, n_simulation_sets, mutation_counts, results_dir_for_ID_DM, restrict_to_where_mash_not_1=True)
