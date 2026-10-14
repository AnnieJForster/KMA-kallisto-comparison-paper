import polars as pl
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# functions from plot
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.ticker import MultipleLocator, LogLocator, NullFormatter, FuncFormatter

def format_plain_tick_label(value, _):
    """Format tick labels as plain decimals (avoid scientific/10^ notation)."""
    if not np.isfinite(value):
        return ""
    if np.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.10f}".rstrip("0").rstrip(".")

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

# def windowed_average_with_band_for_axis(
#     x_values,
#     y_values,
#     logx=False,
#     logy=False,
#     n_windows=40,
#     window_size=None,
#     min_points_per_window=1,
#     z_value=1.0,
# ):
#     valid = np.isfinite(x_values) & np.isfinite(y_values)
#     if logx:
#         valid &= x_values > 0
#     if logy:
#         valid &= y_values > 0

#     x_valid = x_values[valid]
#     y_valid = y_values[valid]
#     if len(x_valid) == 0:
#         return np.array([]), np.array([]), np.array([]), np.array([])

#     x_fit = np.log10(x_valid) if logx else x_valid
#     y_fit = np.log10(y_valid) if logy else y_valid

#     x_min = np.min(x_fit)
#     x_max = np.max(x_fit)
#     if x_max == x_min:
#         count = len(y_fit)
#         y_mean = np.mean(y_fit)
#         y_std = np.std(y_fit, ddof=1) if count > 1 else 0.0
#         y_se = y_std / np.sqrt(count) if count > 0 else 0.0
#         x_line = np.array([x_min])
#         y_line = np.array([y_mean])
#         y_lower = np.array([y_mean - z_value * y_se])
#         y_upper = np.array([y_mean + z_value * y_se])
#     else:
#         if window_size is None:
#             n_windows = max(1, int(n_windows))
#             window_size = (x_max - x_min) / n_windows
#         if window_size <= 0:
#             raise ValueError("window_size must be greater than 0")

#         window_ids = np.floor((x_fit - x_min) / window_size).astype(int)
#         x_means = []
#         y_means = []
#         y_lowers = []
#         y_uppers = []

#         for window_id in np.unique(window_ids):
#             in_window = window_ids == window_id
#             count = np.sum(in_window)
#             if count < min_points_per_window:
#                 continue
#             x_win = x_fit[in_window]
#             y_win = y_fit[in_window]
#             y_mean = np.mean(y_win)
#             y_std = np.std(y_win, ddof=1) if count > 1 else 0.0
#             y_se = y_std / np.sqrt(count)

#             x_means.append(np.mean(x_win))
#             y_means.append(y_mean)
#             y_lowers.append(y_mean - z_value * y_se)
#             y_uppers.append(y_mean + z_value * y_se)

#         if len(x_means) == 0:
#             return np.array([]), np.array([]), np.array([]), np.array([])

#         x_line = np.asarray(x_means)
#         y_line = np.asarray(y_means)
#         y_lower = np.asarray(y_lowers)
#         y_upper = np.asarray(y_uppers)

#         order_idx = np.argsort(x_line)
#         x_line = x_line[order_idx]
#         y_line = y_line[order_idx]
#         y_lower = y_lower[order_idx]
#         y_upper = y_upper[order_idx]

#     x_line = (10 ** x_line) if logx else x_line
#     y_line = (10 ** y_line) if logy else y_line
#     y_lower = (10 ** y_lower) if logy else y_lower
#     y_upper = (10 ** y_upper) if logy else y_upper
#     return x_line, y_line, y_lower, y_upper


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

def plot_scatter_no_smooth(df, x, y_kma, y_kallisto, title="", x_label="", y_label="", logx=False, logy=False, absx=False, absy=False):
    fig, ax = plt.subplots(figsize=(4.5, 3))
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if absx:
        df = df.with_columns(pl.col(x).abs().alias(x))
    if absy:
        df = df.with_columns(pl.col(y_kma).abs().alias(y_kma), pl.col(y_kallisto).abs().alias(y_kallisto))

    sns.scatterplot(data=df.to_pandas(), x=x, y=y_kma, color='none', edgecolor="tab:blue", alpha=0.7, label="KMA", ax=ax)
    sns.scatterplot(data=df.to_pandas(), x=x, y=y_kallisto, color='none', edgecolor="tab:orange", alpha=0.7
                    , label="Kallisto", ax=ax)

    ax.set_xlabel(x_label, fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    sns.despine(ax=ax, top=True, right=True, trim=True)
    ax.legend(fontsize=8)
    return fig, ax

def plot_scatter_with_smooth_line(df, x, y_kma, y_kallisto, title="", x_label="", y_label="", logx=False, logy=False, absx=False, absy=False, fit_method="lowess", poly_order=3, window_n=40, window_min_points=1):
    fig, ax = plt.subplots(figsize=(4.5, 3))
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
    return fig, ax


def safe_log10_expr(column_name, alias):
    return pl.when(pl.col(column_name) > 0).then(pl.col(column_name).log10()).otherwise(None).alias(alias)

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

def format_bin_value(value):
    formatted = f"{value:.10f}".rstrip("0").rstrip(".")
    return formatted if formatted not in {"", "-0"} else "0"

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

def plot_joint_heatmap_with_bins(df, x, y, colour_column1, colour_column2, n_x_bins, n_y_bins, title="", x_label="", y_label="", logx=False, logy=False, 
                                 fill_empty_bins=False, max_empty_cells_per_row=None, cmap="viridis", flip_y_axis=False,
                                 exclude_highest_y_bin=False):
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

    # Omit the highest original distance bin without changing the other edges.
    # Apply before sparse-row filtering so the cutoff is independent of occupancy.
    if exclude_highest_y_bin:
        heatmap_data1_pivot = heatmap_data1_pivot.drop(index=n_y_bins - 1)
        heatmap_data2_pivot = heatmap_data2_pivot.drop(index=n_y_bins - 1)

    heatmap_data1_pivot, heatmap_data2_pivot, kept_y_bin_positions = drop_sparse_joint_heatmap_rows(
        heatmap_data1_pivot,
        heatmap_data2_pivot,
        max_empty_cells_per_row=max_empty_cells_per_row,
    )
    if heatmap_data1_pivot.shape[0] == 0:
        raise ValueError("No y-bin rows remain after dropping sparse heatmap rows.")

    y_bin_positions = heatmap_data1_pivot.index.to_list()
    y_bin_labels_to_plot = [Y_bin_labels[i] for i in y_bin_positions]

    if flip_y_axis:
        heatmap_data1_pivot = heatmap_data1_pivot.iloc[::-1]
        heatmap_data2_pivot = heatmap_data2_pivot.iloc[::-1]
        y_bin_labels_to_plot = y_bin_labels_to_plot[::-1]

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
    
    return fig, axs

def windowed_average_with_band_for_axis(
    x_values,
    y_values,
    logx=False,
    logy=False,
    n_windows=40,
    window_size=None,
    min_points_per_window=1,
    z_value=1.0,
    binning="quantile",
    banding="se",
    percentile_low=25,
    percentile_high=75,
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
        if banding == "percentile":
            y_lower = np.array([np.percentile(y_fit, percentile_low)])
            y_upper = np.array([np.percentile(y_fit, percentile_high)])
        else:
            y_lower = np.array([y_mean - z_value * y_se])
            y_upper = np.array([y_mean + z_value * y_se])
    else:
        n_windows = max(1, int(n_windows))
        if binning == "quantile":
            window_edges = np.quantile(x_fit, np.linspace(0, 1, n_windows + 1))
            window_edges = np.unique(window_edges)
            if len(window_edges) < 2:
                window_edges = np.array([x_min, x_max])
        else:
            if window_size is None:
                window_size = (x_max - x_min) / n_windows
            if window_size <= 0:
                raise ValueError("window_size must be greater than 0")
            window_edges = np.linspace(x_min, x_max, n_windows + 1)

        x_means = []
        y_means = []
        y_lowers = []
        y_uppers = []

        for left_edge, right_edge in zip(window_edges[:-1], window_edges[1:]):
            if right_edge == window_edges[-1]:
                in_window = (x_fit >= left_edge) & (x_fit <= right_edge)
            else:
                in_window = (x_fit >= left_edge) & (x_fit < right_edge)
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
            if banding == "percentile":
                y_lowers.append(np.percentile(y_win, percentile_low))
                y_uppers.append(np.percentile(y_win, percentile_high))
            else:
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


def plot_scatter_with_smooth_line(df, x, y_kma, y_kallisto, title="", x_label="", y_label="", logx=False, logy=False, absx=False, absy=False, fit_method="lowess", poly_order=3, window_n=40, window_min_points=1, banding="se", percentile_low=2.5, percentile_high=97.5):
    fig, ax = plt.subplots(figsize=(4.5, 3))
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
            z_value=1.96,
            binning="quantile",
            banding=banding,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
        )
        fit_kallisto_x, fit_kallisto_y, fit_kallisto_lower, fit_kallisto_upper = windowed_average_with_band_for_axis(
            x_values,
            y_kallisto_values,
            logx=logx,
            logy=logy,
            n_windows=window_n,
            min_points_per_window=window_min_points,
            z_value=1.96,
            binning="quantile",
            banding=banding,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
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
    return fig, ax

