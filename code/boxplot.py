import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_method_boxplot(
    df,
    value_cols,
    id_vars,
    method_name_map=None,
    palette=None,
    ylabel="Metric",
    xlabel="Condition",
    ylim=None,
    yscale=None,
    figsize=(7, 4),
    output_path=None,
):
    """
    Generic function to plot boxplots comparing methods across conditions.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    value_cols : list
        Columns containing metrics for different methods.
    id_vars : tuple/list
        Columns defining experiment conditions.
    method_name_map : dict
        Optional mapping from column names to method display names.
    palette : dict or seaborn palette
        Colors for methods.
    ylabel : str
    xlabel : str
    ylim : tuple
    figsize : tuple
    output_path : str
    """

    df_plot = df.copy()

    df_melted = df_plot.melt(
        id_vars=list(id_vars),
        value_vars=value_cols,
        var_name="method",
        value_name="value"
    )

    if method_name_map:
        df_melted["method"] = df_melted["method"].replace(method_name_map)

    # convert grouping vars to string
    for c in id_vars:
        df_melted[c] = df_melted[c].astype(str)

    df_melted["group"] = " x ".join(id_vars)
    df_melted["group"] = df_melted[list(id_vars)].agg(" x ".join, axis=1)

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "legend.title_fontsize": 14,
        "axes.linewidth": 1,  # thin axes
        "lines.linewidth": 1, # thin box lines
        "grid.color": "none",  # no grid
        "xtick.major.width": 1,
        "ytick.major.width": 1,
    })

    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=df_melted,
        x="group",
        y="value",
        hue="method",
        palette=palette,
        linewidth=1.2,
        fliersize=4,
        ax=ax
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if ylim:
        ax.set_ylim(*ylim)
    
    if yscale:
        ax.set_yscale(yscale)
    
    # for spine in ["left", "bottom"]:
    #     ax.spines[spine].set_alpha(0.3)

    ax.tick_params(axis="x", rotation=60)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(title="Method", frameon=False, loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)

    plt.show()
