import os, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings
from matplotlib_venn import venn2, venn3
import matplotlib.patheffects as path_effects
from itertools import combinations
import scipy.cluster.hierarchy as sch
import gseapy as gp
import seaborn as sns
import matplotlib.ticker as ticker

sns.set_context("poster", font_scale=1)
sns.set(style="darkgrid")

def trim_gene_sets(dge_results):
    """
    Trim the DataFrames in dge_results so that all gene sets are consistent across clusters.

    Parameters:
        dge_results (dict): Dictionary of DataFrames containing DGE results for each cluster.

    Returns:
        dict: A new dictionary with trimmed DataFrames containing only common genes.
        list: A list of gene names that were removed.
    """
    # Ensure 'feature_id' and 'gene_name_biomart' columns exist in all DataFrames
    required_cols = {'feature_id', 'gene_name_biomart'}
    if not all(required_cols.issubset(df.columns) for df in dge_results.values()):
        raise ValueError("Missing required columns ('feature_id', 'gene_name_biomart') in one or more DataFrames.")
    
    # Extract gene sets based on Ensembl IDs (feature_id)
    all_gene_ids = [set(df['feature_id'].dropna()) for df in dge_results.values()]
    common_gene_ids = set.intersection(*all_gene_ids)
    all_genes = set.union(*all_gene_ids)
    trimmed_gene_ids = all_genes.difference(common_gene_ids)
    
    # Map Ensembl IDs to gene names for human-readable output
    trimmed_gene_names = set()
    for df in dge_results.values():
        trimmed_gene_names.update(df.loc[df['feature_id'].isin(trimmed_gene_ids), 'gene_name_biomart'].dropna().unique())
    
    # Debugging output
    print("All genes (Ensembl IDs):", len(all_genes))
    print("Common genes (Ensembl IDs):", len(common_gene_ids))
    print("Trimmed genes (Ensembl IDs):", len(trimmed_gene_ids))
    print("Trimmed gene names:", trimmed_gene_names)
    
    # Ensure all DataFrames have the same ordered gene set after trimming
    trimmed_dge_results = {
        cluster: df[df['feature_id'].isin(common_gene_ids)].set_index('feature_id').reindex(sorted(common_gene_ids))
        for cluster, df in dge_results.items()
    }
    
    return trimmed_dge_results, list(trimmed_gene_names)

def validate_and_count_genes(dge_results):
    """
    Validate the consistency of gene sets across all clusters and count the total number of unique genes.

    Parameters:
        dge_results (dict): Dictionary of DataFrames containing DGE results for each cluster.

    Returns:
        int: Total number of unique genes.

    Raises:
        ValueError: If the gene sets are not consistent across clusters.
    """
    # Extract only feature IDs for comparison from index
    all_gene_ids = [set(df.index.dropna().unique()) for df in dge_results.values()]
    common_gene_ids = set.intersection(*all_gene_ids)
    
    # Validate consistency after trimming
    for genes in all_gene_ids:
        if genes != common_gene_ids:
            raise ValueError("Gene sets are not consistent across clusters after trimming.")
    
    return len(common_gene_ids)


def extract_significant_genes(dge_results, effect_size_col, p_value_col, gene_name_col, effect_size_threshold, p_value_threshold):
    """
    Extracts significant genes based on effect size and adjusted p-value thresholds.

    Parameters:
        dge_results (pd.DataFrame): DataFrame containing differential gene expression results.
        effect_size_col (str): Column name for effect size.
        p_value_col (str): Column name for adjusted p-values.
        gene_name_col (str): Column name for gene names.
        effect_size_threshold (float): Threshold for absolute effect size.
        p_value_threshold (float): Threshold for adjusted p-value.

    Returns:
        tuple: A set of significant gene names and a DataFrame containing effect size, p-value, and gene name.
    """
    filtered_df = dge_results[
        (dge_results[effect_size_col].abs() > effect_size_threshold) & 
        (dge_results[p_value_col] < p_value_threshold)
    ][[gene_name_col, effect_size_col, p_value_col]].copy()
    
    significant_gene_set = set(filtered_df[gene_name_col])
    
    return significant_gene_set, filtered_df


def count_unique_genes(significant_genes):
    """
    Computes the total number of unique genes (by feature_id) across all clusters in significant_genes.
    
    Parameters:
        significant_genes (dict): Dictionary where keys are cluster labels and values are Pandas DataFrames
                                  indexed by feature_id.
    
    Returns:
        int: Total number of unique feature_id values.
    """
    unique_genes = set()
    
    for df in significant_genes.values():
        unique_genes.update(df.index)
    
    return len(unique_genes)


# Function to determine overall up or downregulation trend for each cluster
def determine_regulation_trend(significant_genes):
    """
    Determines whether each cluster has an overall trend of upregulation or downregulation,
    taking into account both the number of genes and their effect sizes.
    
    Parameters:
        significant_genes (dict): Dictionary where keys are cluster labels and values are DataFrames 
                                  containing significant genes with an 'effect_size' column.
    
    Returns:
        pd.DataFrame: A summary table indicating the proportion of upregulated and downregulated genes per cluster,
                      including weighted effect size contributions.
    """
    cluster_trends = {}
    
    for cluster, df in significant_genes.items():
        total_genes = len(df)
        upregulated = df[df['effect_size'] > 0]
        downregulated = df[df['effect_size'] < 0]
        
        up_count = len(upregulated)
        down_count = len(downregulated)
        
        up_ratio = up_count / total_genes if total_genes > 0 else np.nan
        down_ratio = down_count / total_genes if total_genes > 0 else np.nan
        
        mean_up_effect = upregulated['effect_size'].mean() if not upregulated.empty else 0
        mean_down_effect = downregulated['effect_size'].mean() if not downregulated.empty else 0
        
        weighted_up = up_count * mean_up_effect
        weighted_down = down_count * mean_down_effect
        
        combined_weighted_score = weighted_up + weighted_down  # Weighing effect size by count
        
        cluster_trends[cluster] = {
            'Total Genes': total_genes,
            'Upregulated': up_count,
            'Downregulated': down_count,
            'Upregulated Ratio': up_ratio,
            'Downregulated Ratio': down_ratio,
            'Mean Upregulated Effect Size': mean_up_effect,
            'Mean Downregulated Effect Size': mean_down_effect,
            'Weighted Upregulated Score': weighted_up,
            'Weighted Downregulated Score': weighted_down,
            'Combined Weighted Score': combined_weighted_score,
            'Trend': 'Upregulated' if combined_weighted_score > 0 else ('Downregulated' if combined_weighted_score < 0 else 'Balanced')
        }
    
    return pd.DataFrame.from_dict(cluster_trends, orient='index')

def visualize_general_trend(dge_results, effect_size_threshold, p_value_threshold, output_dir, input_type='Genes'):
    """
    Visualizes regulation trends across multiple metrics using separate upregulation and downregulation scores.

    Args:
        dge_results (dict): Dictionary of DataFrames containing DGE results for each cluster or experiment, indexed by labels.
        input_type (str, optional): A term used in the title and filename to indicate the input data type
                                (e.g., "Genes", "Proteins"). Defaults to "Genes".
    """
    sns.set_context("paper")
    
    trends = []

    if input_type == 'Genes':
        title = f"Transcriptomic Signal by Metric"
    if input_type == 'Proteins':
        title = f"Proteomic Signal by Metric"
    
    # Iterate through dge_results to calculate scores
    for label, data in dge_results.items():
        # Check for required columns
        if 'effect_size' not in data.columns or 'adj.P.Val' not in data.columns:
            raise KeyError(f"Dataset '{label}' must contain 'effect_size' and 'adj.P.Val' columns.")

        # Calculate upregulation and downregulation scores separately
        upregulation_score = data.loc[(data['effect_size'] > effect_size_threshold) & (data['adj.P.Val'] <= p_value_threshold), 'effect_size'].sum()
        downregulation_score = data.loc[(data['effect_size'] < -effect_size_threshold) & (data['adj.P.Val'] <= p_value_threshold), 'effect_size'].sum()

        trends.append({
            'Upregulation': upregulation_score,
            'Downregulation': downregulation_score  # Keep negative for proper y-axis positioning
        })

    # Create DataFrame and use keys as index
    trends_df = pd.DataFrame(trends)
    trends_df.index = list(dge_results.keys())  # Use keys as index
    
    # Plotting the regulation scores
    fig, ax = plt.subplots(figsize=(6, 3))
    bar_width = 0.6  # Ensure bars align

    # Define colors based on input_type
    if input_type.lower().startswith('protein'):
        up_color = '#e47cb4'  # Purple for proteins upregulated
        down_color = '#9cc288'  # Green for proteins downregulated
    else:
        up_color = '#ea7d63'  # Coral for genes upregulated
        down_color = 'lightsteelblue'  # Light blue for genes downregulated
    
    # Plot upregulation bars
    ax.bar(trends_df.index, trends_df['Upregulation'], color=up_color, width=bar_width, label=f'Upregulated {input_type}')
    # Plot downregulation bars in the same position
    ax.bar(trends_df.index, trends_df['Downregulation'], color=down_color, width=bar_width, label=f'Downregulated {input_type}')

    ax.set_xlabel("Fibrosis Metric", fontsize=16)  # X-axis label
    ax.set_ylabel('Sum of Effect Sizes', fontsize=16)

    ax.set_title(title, fontsize=18, loc='center')
    
    ax.axhline(0, color='black', linewidth=1)  # Add a horizontal line at y=0 for clarity
    ax.set_xticks(range(len(trends_df.index)))
    ax.set_xticklabels(trends_df.index, fontsize=14, ha='center')

    # Format the y-axis labels with a 'k' suffix
    #ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.1f}k'))
    if input_type == 'Genes':
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5000))
    if input_type == 'Proteins':
        ax.yaxis.set_major_locator(ticker.MultipleLocator(500))
    ax.tick_params(axis='y', labelsize=14)
    
    plt.legend(facecolor='white', framealpha=0.8, loc='upper left', fontsize=11)
    plt.tight_layout()
    
    file_name = f"{output_dir}/{title.replace(' ', '_')}.png"
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    
    print(f"Figure saved as {file_name}")


def plot_gene_overlap(gene_dict):
    """
    Plots a Venn diagram showing the overlap of significant genes between different metrics.

    Parameters:
    gene_dict (dict): A dictionary where keys are metric names and values are lists or sets of significant genes.

    Example:
    gene_dict = {
        "Metric A": {"Gene1", "Gene2", "Gene3"},
        "Metric B": {"Gene2", "Gene3", "Gene4"},
        "Metric C": {"Gene1", "Gene5"}
    }
    plot_gene_overlap(gene_dict)
    """
    from venn import venn
    if len(gene_dict) > 6:
        raise ValueError("The venn library supports up to 6 sets. Please provide 6 or fewer lists.")

    # Convert values to sets (if they aren't already)
    gene_sets = {k: set(v) for k, v in gene_dict.items()}

    # Plot the Venn diagram
    venn(gene_sets)
    plt.title("Gene Overlap Between Metrics")
    plt.show()


def create_cluster_venn(significant_genes, effect_size_threshold, p_value_threshold, selected_labels, output_dir, colors=None, text_color='white', input_type='Genes', file_name=None):
    """
    Creates and saves a Venn diagram for specified clusters from the significant genes dictionary.

    Parameters:
        significant_genes (dict): Dictionary of significant genes with cluster labels as keys (e.g., 'C4', 'C5', 'C6').
        effect_size_threshold (float): Effect size threshold for filtering.
        p_value_threshold (float): P-value threshold for filtering.
        selected_labels (list): List of two or three cluster labels to include in the Venn diagram.
        colors (list, optional): List of colors corresponding to each selected label.
        text_color (str, optional): Color for the text inside the circles. Default is white.
        input_type (str, optional): Whether we're visualising genes or proteins. Formats the title.
        file_name (str, optional): Path to save the Venn diagram image. If None, a meaningful filename is generated.
    """
    if len(selected_labels) not in [2, 3]:
        raise ValueError("Please provide exactly two or three valid cluster labels.")
    
    if colors and len(colors) != len(selected_labels):
        raise ValueError(f"Please provide exactly {len(selected_labels)} colors corresponding to the selected labels.")
    
    # Generate a meaningful filename if not provided
    if file_name is None:
        labels_str = "_".join(selected_labels)
        file_name = f"{output_dir}/venn_{labels_str}_ES{effect_size_threshold}_P{p_value_threshold}.png"
    
    # Extract the gene sets and labels dynamically
    gene_sets = [set(significant_genes[label]) for label in selected_labels]
    
    # Create the Venn diagram
    plt.figure(figsize=(8, 6))
    
    if len(selected_labels) == 3:
        venn = venn3(gene_sets, selected_labels, set_colors=colors if colors else ('r', 'g', 'b'))
    else:
        venn = venn2(gene_sets, selected_labels, set_colors=colors if colors else ('r', 'g'))

    # Adjust alpha (transparency) to 1 for individual circles, and reduce for overlapping areas
    for subset in ['100', '010', '001']:  # Individual sets
        if venn.get_patch_by_id(subset) and colors:
            venn.get_patch_by_id(subset).set_alpha(1)  # Make the colors fully opaque
    
    for subset in ['110', '101', '011', '111']:  # Overlapping regions
        if venn.get_patch_by_id(subset):
            venn.get_patch_by_id(subset).set_alpha(0.9)  # Reduce transparency for overlaps

    # Adjust font size and boldness for set labels
    for label in venn.set_labels:
        if label:
            label.set_fontsize(28)
            label.set_path_effects([path_effects.withStroke(linewidth=1, foreground='white')])
    
    # Change text color inside circles, make bold, and add a grey outline
    for subset in venn.subset_labels:
        if subset:
            subset.set_fontsize(18)
            subset.set_color(text_color)
            subset.set_fontweight('bold')
            subset.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='dimgrey')])

    if input_type == "Genes":
        plt.title(f'Transcript Overlap by Fibrosis Metric', fontsize=24)
    if input_type == "Proteins":
        plt.title(f'Protein Overlap by Fibrosis Metric', fontsize=24)
    
    # Save the plot
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Venn diagram saved as '{file_name}'")


def create_upset_plot(significant_genes):
    """
    Creates and saves an UpSet plot for more than 3 selected clusters.

    Parameters:
    - significant_genes (dict): Dictionary where keys are cluster names and values are sets of genes.
    """
    import upsetplot

    # Create a list of all unique genes across the significant gene sets
    gene_list = list(set.union(*significant_genes.values()))  # Ensure gene_list is a list

    # Remove any non-string entries from the gene list (e.g., NaN, floats)
    gene_list = [gene for gene in gene_list if isinstance(gene, str)]

    # Create a DataFrame where each row is a gene and columns represent the labeled gene sets (C0, C1, ..., C6)
    gene_membership = pd.DataFrame(
        {cluster_label: [1 if gene in significant_genes[cluster_label] else 0 for gene in gene_list] 
         for cluster_label in significant_genes},
        index=gene_list
    )

    # Convert this DataFrame to an appropriate format for the UpSet plot
    upset_data = gene_membership.groupby(list(gene_membership.columns)).size()

    # Suppress FutureWarning
    warnings.filterwarnings("ignore", category=FutureWarning, module="upsetplot")

    # Convert to an UpSet-compatible format
    upset = upsetplot.UpSet(upset_data, sort_categories_by='-input')
    upset.plot()

    # Save the figure
    plt.title('UpSet Plot of Significant Gene Overlap')
    file_name = "./fibrosis_dge_interpretation/plots/upset_plot.png"
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    plt.show()


def filter_significant_results(df, adj_p_value_threshold=0.05):
    """
    Filters the enrichment results based on a significance threshold.

    Parameters:
        df (pd.DataFrame): The enrichment results dataframe.
        adj_p_value_threshold (float): The significance threshold for filtering (default 0.05).

    Returns:
        pd.DataFrame: Filtered dataframe with significant results.
    """
    # Check for the column name dynamically
    adj_p_value_column = 'Adjusted P-value' if 'Adjusted P-value' in df.columns else 'adj.P.Val'
    
    if adj_p_value_column not in df.columns:
        raise ValueError("The dataframe does not contain a recognized adjusted p-value column ('Adjusted P-value' or 'adj.P.Val').")
    
    filtered_df = df[df[adj_p_value_column] <= adj_p_value_threshold]
    return filtered_df


# Function to compute hypergeometric test for all pairs of clusters
def compute_hypergeometric_tests(significant_genes, total_gene_universe):
    """
    Computes hypergeometric p-values for the overlap of DEGs between all pairs of clusters.
    
    Parameters:
        significant_genes (dict): Dictionary where keys are cluster labels and values are DataFrames 
                                  containing significant genes with 'feature_id' as the index.
        total_gene_universe (int): Total number of genes in the background set (e.g., genome or dataset size).
    
    Returns:
        pd.DataFrame: Pairwise matrix of hypergeometric p-values for DEG overlap.
    """
    clusters = list(significant_genes.keys())
    p_values = pd.DataFrame(index=clusters, columns=clusters, dtype=float)
    
    for i, cluster1 in enumerate(clusters):
        genes1 = set(significant_genes[cluster1].index.dropna())
        for j, cluster2 in enumerate(clusters):
            if j <= i:
                continue  # Avoid redundant comparisons
            genes2 = set(significant_genes[cluster2].index.dropna())
            
            overlap = len(genes1 & genes2)
            K = len(genes1)  # Size of first set
            n = len(genes2)  # Size of second set
            N = total_gene_universe  # Background universe size
            
            # Hypergeometric test
            p_value = stats.hypergeom.sf(overlap - 1, N, K, n)  # Survival function to get p(X >= overlap)
            
            p_values.loc[cluster1, cluster2] = p_value
            p_values.loc[cluster2, cluster1] = p_value  # Mirror the matrix
    
    return p_values

# Function to compare DEG counts between clusters using a Chi-squared test with pairwise comparisons
def compare_deg_counts(significant_genes, total_gene_universe):
    """
    Compares the number of DEGs identified across different clusters using a Chi-squared test,
    and performs pairwise comparisons between clusters.
    
    Parameters:
        significant_genes (dict): Dictionary where keys are cluster labels and values are DataFrames 
                                  containing significant genes with 'feature_id' as the index.
        total_gene_universe (int): Total number of genes in the dataset (background gene universe).
    
    Returns:
        pd.DataFrame: Pairwise matrix of Chi-squared p-values for DEG comparisons.
    """
    clusters = list(significant_genes.keys())
    deg_counts = {cluster: len(df) for cluster, df in significant_genes.items()}
    
    # Identify clusters with zero DEGs
    zero_clusters = [cluster for cluster, count in deg_counts.items() if count == 0]
    
    if zero_clusters:
        print(f"Warning: The following clusters have zero DEGs and will be excluded from the test: {', '.join(zero_clusters)}")
        deg_counts = {cluster: count for cluster, count in deg_counts.items() if count > 0}
    
    if len(deg_counts) < 2:
        print("Not enough clusters with DEGs to perform a Chi-squared test.")
        return pd.DataFrame(index=[], columns=[])
    
    non_deg_counts = {cluster: total_gene_universe - count for cluster, count in deg_counts.items()}
    
    # Create a contingency table
    contingency_table = [list(deg_counts.values()), list(non_deg_counts.values())]
    
    # Global Chi-squared test
    chi2, global_p_value, _, _ = stats.chi2_contingency(contingency_table)
    print(f"Global Chi-squared p-value: {global_p_value}")
    
    # Pairwise Chi-squared tests matrix
    p_values = pd.DataFrame(index=deg_counts.keys(), columns=deg_counts.keys(), dtype=float)
    
    for cluster1, cluster2 in combinations(deg_counts.keys(), 2):
        contingency = [[deg_counts[cluster1], non_deg_counts[cluster1]],
                       [deg_counts[cluster2], non_deg_counts[cluster2]]]
        _, pairwise_p_value, _, _ = stats.chi2_contingency(contingency)
        p_values.loc[cluster1, cluster2] = pairwise_p_value
        p_values.loc[cluster2, cluster1] = pairwise_p_value  # Mirror the matrix
    
    return p_values

# Function to compare absolute effect sizes between clusters using pairwise t-tests
def compare_absolute_effect_sizes(significant_genes):
    """
    Compares the absolute effect sizes of DEGs across different clusters using pairwise t-tests and
    computes both the mean and median absolute effect size differences for each comparison.
    """
    clusters = list(significant_genes.keys())
    
    # Initialize DataFrames
    p_values = pd.DataFrame(index=clusters, columns=clusters, dtype=float)
    abs_mean_differences = pd.DataFrame(index=clusters, columns=clusters, dtype=float)
    abs_median_differences = pd.DataFrame(index=clusters, columns=clusters, dtype=float)
    
    abs_mean_effect_sizes = {
        cluster: significant_genes[cluster]['effect_size'].abs().dropna().mean()
        if not significant_genes[cluster]['effect_size'].abs().dropna().empty else 0
        for cluster in clusters
    }
    
    abs_median_effect_sizes = {
        cluster: significant_genes[cluster]['effect_size'].abs().dropna().median()
        if not significant_genes[cluster]['effect_size'].abs().dropna().empty else 0
        for cluster in clusters
    }
    
    for cluster1, cluster2 in combinations(clusters, 2):
        abs_effect_sizes1 = significant_genes[cluster1]['effect_size'].abs().dropna()
        abs_effect_sizes2 = significant_genes[cluster2]['effect_size'].abs().dropna()
        
        # Check if variance is zero (all values are identical)
        if abs_effect_sizes1.nunique() <= 1 or abs_effect_sizes2.nunique() <= 1:
            p_value = None
            abs_mean_diff = None
            abs_median_diff = None
        else:
            _, p_value = stats.ttest_ind(abs_effect_sizes1, abs_effect_sizes2, equal_var=False)
            abs_mean_diff = abs_mean_effect_sizes[cluster1] - abs_mean_effect_sizes[cluster2]
            abs_median_diff = abs_median_effect_sizes[cluster1] - abs_median_effect_sizes[cluster2]
        
        # Store values
        p_values.loc[cluster1, cluster2] = p_value
        p_values.loc[cluster2, cluster1] = p_value
        abs_mean_differences.loc[cluster1, cluster2] = abs_mean_diff
        abs_mean_differences.loc[cluster2, cluster1] = abs_mean_diff if abs_mean_diff is not None else None
        abs_median_differences.loc[cluster1, cluster2] = abs_median_diff
        abs_median_differences.loc[cluster2, cluster1] = abs_median_diff if abs_median_diff is not None else None
    
    return p_values, abs_mean_differences, abs_median_differences


# Function to compare upregulated and downregulated effect sizes separately
def compare_up_down_effect_sizes(significant_genes):
    """
    Separates effect sizes into upregulated and downregulated categories and compares them using pairwise t-tests.
    Returns the p-values along with mean and median effect sizes for each category.
    """
    clusters = list(significant_genes.keys())
    up_p_values = pd.DataFrame(index=clusters, columns=clusters, dtype=float)
    down_p_values = pd.DataFrame(index=clusters, columns=clusters, dtype=float)
    up_mean_differences = pd.DataFrame(index=clusters, columns=clusters, dtype=float)
    down_mean_differences = pd.DataFrame(index=clusters, columns=clusters, dtype=float)
    up_median_differences = pd.DataFrame(index=clusters, columns=clusters, dtype=float)
    down_median_differences = pd.DataFrame(index=clusters, columns=clusters, dtype=float)
    
    for cluster1, cluster2 in combinations(clusters, 2):
        effect_sizes1 = significant_genes[cluster1]['effect_size'].dropna()
        effect_sizes2 = significant_genes[cluster2]['effect_size'].dropna()
        
        up_effect_sizes1 = effect_sizes1[effect_sizes1 > 0]
        up_effect_sizes2 = effect_sizes2[effect_sizes2 > 0]
        down_effect_sizes1 = effect_sizes1[effect_sizes1 < 0]
        down_effect_sizes2 = effect_sizes2[effect_sizes2 < 0]
        
        # Compute mean and median effect sizes safely
        up_mean_diff = up_effect_sizes1.mean() - up_effect_sizes2.mean() if not up_effect_sizes1.empty and not up_effect_sizes2.empty else np.nan
        down_mean_diff = down_effect_sizes1.mean() - down_effect_sizes2.mean() if not down_effect_sizes1.empty and not down_effect_sizes2.empty else np.nan
        up_median_diff = up_effect_sizes1.median() - up_effect_sizes2.median() if not up_effect_sizes1.empty and not up_effect_sizes2.empty else np.nan
        down_median_diff = down_effect_sizes1.median() - down_effect_sizes2.median() if not down_effect_sizes1.empty and not down_effect_sizes2.empty else np.nan
        
        # Check for zero variance before t-tests
        if up_effect_sizes1.nunique() <= 1 or up_effect_sizes2.nunique() <= 1:
            up_p_value = np.nan
        else:
            _, up_p_value = stats.ttest_ind(up_effect_sizes1, up_effect_sizes2, equal_var=False)
        
        if down_effect_sizes1.nunique() <= 1 or down_effect_sizes2.nunique() <= 1:
            down_p_value = np.nan
        else:
            _, down_p_value = stats.ttest_ind(down_effect_sizes1, down_effect_sizes2, equal_var=False)
        
        # Store values
        up_p_values.loc[cluster1, cluster2] = up_p_value
        up_p_values.loc[cluster2, cluster1] = up_p_value
        down_p_values.loc[cluster1, cluster2] = down_p_value
        down_p_values.loc[cluster2, cluster1] = down_p_value
        up_mean_differences.loc[cluster1, cluster2] = up_mean_diff
        down_mean_differences.loc[cluster1, cluster2] = down_mean_diff
        up_median_differences.loc[cluster1, cluster2] = up_median_diff
        down_median_differences.loc[cluster1, cluster2] = down_median_diff
    
    return up_p_values, down_p_values, up_mean_differences, down_mean_differences, up_median_differences, down_median_differences


def plot_metric_correlations(correlation_matrix, p_value_matrix, metric_labels, p=0.05):
    """
    Plots a heatmap of the correlation matrix between different metrics, marking significant correlations.
    
    Parameters:
        correlation_matrix (pd.DataFrame): Correlation matrix between metrics.
        p_value_matrix (pd.DataFrame): Matrix of p-values for correlation significance testing.
        p (float): Significance threshold for marking correlations.
    """
    plt.figure(figsize=(8, 6))
    mask = p_value_matrix >= p  # Mask non-significant correlations
    ax = sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f", mask=mask)
    plt.title("Correlation Matrix of Metrics (Significant Correlations Marked)")
    plt.xlabel("Metrics")
    plt.ylabel("Metrics")
    
    # Set tick labels from metric_labels and make them horizontal
    ax.set_xticklabels(metric_labels[:len(correlation_matrix.columns)], rotation=0)
    ax.set_yticklabels(metric_labels[:len(correlation_matrix.index)], rotation=0)
    
    plt.show()


def plot_pca_of_significant_genes(significant_genes, top_n=10):
    """
    Creates a PCA plot of significant genes based on their effect sizes across clusters.

    Parameters:
        significant_genes (dict): Dictionary where keys are cluster names and values are DataFrames of significant genes.
        top_n (int): Number of top genes to retrieve based on highest PC2 and PC1 values.
    
    Returns:
        tuple: (List of top n genes based on PC2, DataFrame of PCA component loadings)
    """
    # Combine effect sizes into a single DataFrame
    combined_df = []
    for cluster, df in significant_genes.items():
        df = df[['effect_size']].copy()
        df.columns = [f"{cluster}_effect_size"]
        combined_df.append(df)
    
    merged_df = pd.concat(combined_df, axis=1, join="outer").fillna(0)

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(merged_df)

    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    variance_explained_pc1 = pca.explained_variance_ratio_[0] * 100  # Percentage
    variance_explained_pc2 = pca.explained_variance_ratio_[1] * 100  # Percentage

    print(f"PC1 explains {variance_explained_pc1:.2f}% of the variance")
    print(f"PC2 explains {variance_explained_pc2:.2f}% of the variance")
    
    # Get PCA component compositions
    components_df = pd.DataFrame(pca.components_, columns=merged_df.columns, index=["PC1", "PC2"])
    #print("PCA Component Loadings:")
    #print(components_df)

    # Create a DataFrame for visualization
    pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"], index=merged_df.index)
    
    # Apply symmetric log transformation to PC2
    pca_df["PC2"] = np.sign(pca_df["PC2"]) * np.log1p(np.abs(pca_df["PC2"]))

    # Identify top n genes with highest and lowest PC2 values
    top_pc2_genes = pca_df.nlargest(top_n, "PC2").index.tolist()
    bottom_pc2_genes = pca_df.nsmallest(top_n, "PC2").index.tolist()
    
    # Identify top n genes with highest and lowest PC1 values
    top_pc1_genes = pca_df.nlargest(top_n, "PC1").index.tolist()
    bottom_pc1_genes = pca_df.nsmallest(top_n, "PC1").index.tolist()
    
    # Retrieve gene names from significant_genes
    unique_genes = list(set(top_pc2_genes + top_pc1_genes + bottom_pc1_genes))
    gene_labels = {}
    for gene in unique_genes:
        for cluster, df in significant_genes.items():
            if gene in df.index:
                gene_labels[gene] = df.loc[gene, "gene_name_biomart"]
                break
    
    # Plot PCA
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.7)
    
    # Annotate unique top/bottom n points with gene names with better spacing
    for i, gene in enumerate(unique_genes):
        if gene in gene_labels:
            x, y = pca_df.loc[gene, "PC1"], pca_df.loc[gene, "PC2"]
            plt.annotate(gene_labels[gene], (x, y), fontsize=9, alpha=0.8, ha='right', 
                         xytext=(0,0), textcoords='offset points') # xytext=(10 * (-1)**(i % 2), 10 * (-1)**((i // 2) % 2))
    
    plt.xlabel("Principal Component 1")
    plt.ylabel("SymLog(PC2)")
    plt.title("PCA of Significant Genes Based on Effect Sizes")
    plt.show()
    
    return [top_pc2_genes, bottom_pc2_genes, top_pc1_genes, bottom_pc1_genes], components_df

def plot_pca_loadings_heatmap(components_df, xtick_labels, output_dir):
    """
    Plots a heatmap of PCA component loadings to visualize how clusters contribute to each principal component.
    
    Parameters:
        components_df (pd.DataFrame): DataFrame of PCA component loadings.
        xtick_labels (list, optional): Custom labels for x-axis ticks.
    """
    plt.figure(figsize=(10, 2))  # Adjusted to be less tall
    ax = sns.heatmap(components_df, cmap="coolwarm", annot=True, fmt=".2f", center=0, vmin=-1)  # Center at 0, bottom limit -1
    plt.title("PCA Component Loadings Heatmap")
    plt.xlabel("Metrics")
    plt.ylabel("Principal Components")
    
    # Rotate tick labels to be horizontal
    if xtick_labels:
        ax.set_xticklabels(xtick_labels, rotation=0)
    else:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    file_name = f'{output_dir}/PCA_component_loadings_heatmap.png'
    
    # Save the plot
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Heatmap saved as '{file_name}'")


def create_regulation_data(dge_results, enrichment_results):
    import pandas as pd
    """
    Creates a regulation_data DataFrame that links pathway terms to genes and their regulation status.

    Parameters:
        dge_results (pd.DataFrame): DataFrame containing differential gene expression results, including 'Gene' and 'log2FoldChange'.
        enrichment_results (pd.DataFrame): DataFrame containing enrichment results with 'Term' and 'Genes' columns.

    Returns:
        pd.DataFrame: A DataFrame with columns ['Term', 'Gene', 'Regulation'].
    """
    # Expand the enrichment results to link pathways to individual genes
    enrichment_expanded = enrichment_results[['Term', 'Genes']].copy()
    enrichment_expanded['Genes'] = enrichment_expanded['Genes'].str.split(';')
    enrichment_expanded = enrichment_expanded.explode('Genes')
    enrichment_expanded.rename(columns={'Genes': 'Gene'}, inplace=True)

    # Merge with DGE results to get log2FoldChange
    dge_results.rename(columns={'gene_name_biomart': 'Gene', 'effect_size': 'log2FoldChange'}, inplace=True)
    regulation_data = pd.merge(enrichment_expanded, dge_results[['Gene', 'log2FoldChange']], on='Gene', how='left')

    # Determine regulation status based on log2FoldChange
    regulation_data['Regulation'] = regulation_data['log2FoldChange'].apply(
        lambda x: 'upregulated' if x > 0 else ('downregulated' if x < 0 else 'neutral')
    )

    return regulation_data[['Term', 'Gene', 'Regulation']]


def plot_enrichment_results_combined_with_regulation(enrichment_results, regulation_data, title, output_dir, sort_by='overlap', max_terms=20):
    """
    Plots overlap proportion as bars and displays stars for p-value significance, with proportions of upregulated and downregulated genes.
    
    Args:
        enrichment_results (DataFrame): Must include 'Term', 'Overlap', and 'Adjusted P-value' columns.
        regulation_data (DataFrame): Must include 'Term', 'Gene', and 'Regulation' columns ('upregulated' or 'downregulated').
        title (str): Title for the plot.
        output_dir (str): The directory where the plot should be saved.
        sort_by (str): Column to sort by ('overlap', 'upregulated', 'downregulated').
        max_terms (int): Maximum number of terms to display (default is 20).
    """
    # Copy the enrichment_results DataFrame to avoid modifying the original
    enrichment_results = enrichment_results.copy()

    # Calculate Overlap Proportion as a percentage
    enrichment_results['Overlap Proportion (%)'] = enrichment_results['Overlap'].apply(
        lambda x: (int(x.split('/')[0]) / int(x.split('/')[1])) * 100
    )

    # Merge with regulation_data to calculate proportions
    regulation_summary = regulation_data.groupby(['Term', 'Regulation']).size().unstack(fill_value=0)
    regulation_summary['Total Genes'] = regulation_summary.sum(axis=1)
    regulation_summary['Upregulated Proportion'] = regulation_summary.get('upregulated', 0) / regulation_summary['Total Genes']
    regulation_summary['Downregulated Proportion'] = regulation_summary.get('downregulated', 0) / regulation_summary['Total Genes']

    # Merge the calculated proportions back into the enrichment_results DataFrame
    enrichment_results = enrichment_results.merge(
        regulation_summary[['Upregulated Proportion', 'Downregulated Proportion']],
        left_on='Term', right_index=True, how='left'
    )

    # Fill NaN values with 0 for terms without regulation data
    enrichment_results.fillna({'Upregulated Proportion': 0, 'Downregulated Proportion': 0}, inplace=True)

    # Scale upregulated and downregulated proportions to total overlap proportion
    enrichment_results['Upregulated Proportion (%)'] = enrichment_results['Upregulated Proportion'] * enrichment_results['Overlap Proportion (%)']
    enrichment_results['Downregulated Proportion (%)'] = enrichment_results['Downregulated Proportion'] * enrichment_results['Overlap Proportion (%)']

    # Determine sorting column
    sort_columns = {
        'overlap': 'Overlap Proportion (%)',
        'upregulated': 'Upregulated Proportion (%)',
        'downregulated': 'Downregulated Proportion (%)'
    }
    if sort_by not in sort_columns:
        raise ValueError(f"Invalid sort_by value: {sort_by}. Choose from {list(sort_columns.keys())}.")

    # Sort enrichment_results by the selected column (descending)
    enrichment_results = enrichment_results.sort_values(by=sort_columns[sort_by], ascending=False)

    # Trim the DataFrame to the top max_terms rows
    enrichment_results = enrichment_results.head(max_terms)

    # Define significance levels for stars
    def significance_stars(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return ''

    enrichment_results['Significance'] = enrichment_results['Adjusted P-value'].apply(significance_stars)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plotting the stacked bars for upregulated and downregulated proportions
    ax.barh(enrichment_results['Term'], enrichment_results['Upregulated Proportion (%)'], color='coral', label='Upregulated')
    ax.barh(enrichment_results['Term'], enrichment_results['Downregulated Proportion (%)'], color='skyblue', 
            left=enrichment_results['Upregulated Proportion (%)'], label='Downregulated')

    # Dynamically scale the font size based on the number of terms
    num_terms = len(enrichment_results)
    base_font_size = 14
    if num_terms <= 5:
        font_size = base_font_size * 2  # Increase font size for fewer terms
    elif num_terms <= 20:
        font_size = base_font_size  # Keep base font size for moderate terms
    else:
        font_size = base_font_size * 0.8  # Decrease font size for many terms

    # Add labels and titles with dynamic font size
    ax.set_xlabel('Overlap Proportion (%)', fontsize=font_size)
    ax.set_ylabel('Terms', fontsize=font_size)
    ax.set_xlim(0, 80)  # Set x-axis limit to 80
    ax.invert_yaxis()  # Invert y-axis for better readability
    ax.set_title(title, fontsize=font_size + 2)

    # Annotate stars for significance with dynamic font size
    for i, (overlap, significance) in enumerate(zip(enrichment_results['Overlap Proportion (%)'], enrichment_results['Significance'])):
        if significance:  # Add stars only if significant
            ax.text(overlap + 1, i, significance, va='center', ha='left', fontsize=font_size * 0.9, color='orange')

    # Increase tick label font sizes
    ax.tick_params(axis='x', labelsize=font_size * 0.8)
    ax.tick_params(axis='y', labelsize=font_size * 0.8)

    # Add legend
    ax.legend(fontsize=font_size, loc='lower right')

    # Make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the plot in the given directory
    file_name = os.path.join(output_dir, '{}.png'.format(title.replace(' ', '_')))
    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')

    print(f"Plot saved to {file_name}")


def plot_dendrogram(effect_size_dict, threshold=130):
    """
    Generates a hierarchical clustering dendrogram based on effect sizes across metrics
    and extracts clusters based on a specified distance threshold.
    
    Parameters:
        effect_size_dict (dict): Dictionary where keys are metric names and values are Pandas DataFrames
                                 indexed by gene names with an 'effect_size' column.
        threshold (float): The distance threshold for cutting the dendrogram to define clusters.
    
    Returns:
        pd.DataFrame: A DataFrame mapping each gene to its assigned cluster.
    """
    combined_dfs = []
    
    for metric, df in effect_size_dict.items():
        if 'effect_size' not in df.columns:
            raise KeyError(f"Expected 'effect_size' column in DataFrame for {metric}, but it was not found.")
        
        combined_dfs.append(df[['effect_size']].rename(columns={'effect_size': metric}))
    
    # Merge all DataFrames on index (genes)
    merged_df = pd.concat(combined_dfs, axis=1, join='outer').fillna(0)
    
    # Perform hierarchical clustering
    linkage_matrix = sch.linkage(merged_df, method='ward')
    
    plt.figure(figsize=(10, 6))
    dendrogram = sch.dendrogram(linkage_matrix, labels=merged_df.index, no_labels=True)
    
    # Draw threshold line
    plt.axhline(y=threshold, color='red', linestyle='--')
    
    # Formatting
    plt.title("Hierarchical Clustering of Genes Based on Effect Sizes")
    plt.xlabel("Genes")
    plt.ylabel("Distance")
    plt.show()
    
    # Extract clusters based on the threshold
    cluster_assignments = sch.fcluster(linkage_matrix, threshold, criterion='distance')
    
    # Create a DataFrame mapping genes to clusters
    cluster_df = pd.DataFrame({'Gene': merged_df.index, 'Cluster': cluster_assignments})
    cluster_df.set_index('Gene', inplace=True)
    
    return cluster_df


def extract_gene_metadata(significant_genes):
    """
    Extracts gene metadata from the significant_genes dictionary.
    
    Parameters:
        significant_genes (dict): Dictionary of DataFrames containing significant genes.

    Returns:
        pd.DataFrame: DataFrame with gene_name_biomart indexed by feature_id.
    """
    gene_metadata = pd.concat(
        [df[['gene_name_biomart']] for df in significant_genes.values()],
        axis=0
    ).drop_duplicates()
    
    return gene_metadata

def query_enrichr_pathways(gene_list, library="Reactome_Pathways_2024"):
    """
    Queries Enrichr to retrieve enriched pathways for a given list of genes.
    
    Parameters:
        gene_list (list): List of gene symbols.
        library (str): Enrichr library to use (default: Reactome_2022).
    
    Returns:
        pd.DataFrame: DataFrame containing enriched pathways with p-values.
    """
    try:
        enr = gp.enrichr(gene_list=gene_list, gene_sets=library, organism='human')
        return enr.results.sort_values(by="Adjusted P-value")
    except Exception as e:
        print(f"Error querying Enrichr: {e}")
        return None

def perform_pathway_enrichment(gene_clusters, gene_metadata, library="Reactome_Pathways_2024"):
    """
    Performs pathway enrichment for each gene cluster using Enrichr.
    Extracts gene symbols from gene_metadata and manages novel genes by keeping Ensembl IDs if 'novel' is present.
    
    Parameters:
        gene_clusters (pd.DataFrame): DataFrame mapping genes to their assigned cluster.
        gene_metadata (pd.DataFrame): DataFrame containing gene_name_biomart (symbols) for Ensembl IDs.
        library (str): Enrichr library to use (default: Reactome_2022).
    
    Returns:
        dict: Dictionary where keys are cluster numbers and values are DataFrames of enriched pathways.
    """
    cluster_pathways = {}
    
    for cluster in gene_clusters["Cluster"].unique():
        cluster_genes = gene_clusters[gene_clusters["Cluster"] == cluster].index.tolist()
        
        # Extract gene symbols; keep Ensembl IDs if the name contains 'novel'
        symbol_genes = [
            gene_metadata.loc[gene, "gene_name_biomart"] if gene in gene_metadata.index and 
            "novel" not in str(gene_metadata.loc[gene, "gene_name_biomart"]).lower() else gene
            for gene in cluster_genes
        ]
        
        print(f"Querying Enrichr for Cluster {cluster} ({len(symbol_genes)} genes)...")
        
        enriched_pathways = query_enrichr_pathways(symbol_genes, library=library)
        
        if enriched_pathways is not None:
            cluster_pathways[cluster] = enriched_pathways
    
    return cluster_pathways

# Function to count significant genes containing 'novel' in their names and return filtered dataframes
def count_novel_genes(significant_genes):
    """
    Counts the number of significant genes that contain 'novel' in their names per cluster and 
    returns a dictionary of dataframes containing only the novel significant genes.
    
    Parameters:
        significant_genes (dict): Dictionary where keys are cluster labels and values are DataFrames 
                                  containing significant genes with a column 'gene_name_biomart'.
    
    Returns:
        tuple:
            - dict: Dictionary where keys are cluster labels and values are the count of genes containing 'novel'.
            - dict: Dictionary where keys are cluster labels and values are DataFrames containing only 'novel' genes.
    """
    novel_counts = {}
    novel_gene_dfs = {}
    
    for cluster, df in significant_genes.items():
        novel_mask = df['gene_name_biomart'].str.contains('novel', case=False, na=False)
        novel_counts[cluster] = novel_mask.sum()
        novel_gene_dfs[cluster] = df[novel_mask].copy()
    
    for cluster, count in novel_counts.items():
        print(f"Cluster {cluster}: {count} significant genes containing 'novel'")
    
    return novel_counts, novel_gene_dfs

# Function to count the number of unique novel gene IDs across all clusters
def count_unique_novel_genes(novel_gene_dfs):
    """
    Counts the number of unique gene IDs across all clusters in novel_gene_dfs.
    
    Parameters:
        novel_gene_dfs (dict): Dictionary where keys are cluster labels and values are DataFrames 
                               containing only 'novel' significant genes with 'feature_id' as the index.
    
    Returns:
        int: Number of unique novel gene IDs across all clusters.
    """
    unique_genes = set()
    
    for df in novel_gene_dfs.values():
        unique_genes.update(df.index.dropna().unique())
    
    unique_count = len(unique_genes)
    print(f"Total unique novel genes across all clusters: {unique_count}")
    
    return unique_count

def filter_and_sort_significant_genes(significant_genes, include_metrics=None, sort_by_metric=None):
    """
    Filters significant_genes to keep only specified metrics and sorts genes by a chosen effect size metric.
    
    Parameters:
        significant_genes (dict): Dictionary with metrics as keys and DataFrames as values.
        include_metrics (list): List of metric names to keep.
        sort_by_metric (str): Metric to use for sorting. Defaults to max effect size if None.
    
    Returns:
        dict: Filtered and sorted dictionary.
    """
    if include_metrics is None:
        include_metrics = list(significant_genes.keys())  # Default to all metrics  
    
    filtered_genes = {metric: df for metric, df in significant_genes.items() if metric in include_metrics}
    
    # Sort by the selected metric's effect size
    sorted_genes = filtered_genes[sort_by_metric].sort_values(by='effect_size', ascending=False).index
    
    # Reorder genes in each metric DataFrame explicitly
    for metric in filtered_genes:
        filtered_genes[metric] = filtered_genes[metric].reindex(sorted_genes)
    
    return filtered_genes

def plot_gene_effect_sizes(effect_size_dict, metric_colours, output_dir, input_type='Genes', ylims=None):
    """
    Plots effect sizes for each gene across different metrics using a visually optimized seaborn figure.
    
    Parameters:
        effect_size_dict (dict): Dictionary where keys are metric names and values are Pandas DataFrames
                                 indexed by gene names with an 'effect_size' column.
        metric_colours (dict): Dictionary mapping metric names to colors.
        input_type (str, optional): A term used in the title and filename to indicate the input data type
                                (e.g., "Genes", "Proteins"). Defaults to "Genes".
        ylims (tuple): Tuple of y axis lims.
    """

    sns.set_context("paper")
    combined_dfs = []
    
    for metric, df in effect_size_dict.items():
        if 'effect_size' not in df.columns:
            raise KeyError(f"Expected 'effect_size' column in DataFrame for {metric}, but it was not found.")
        
        combined_dfs.append(df[['effect_size']].rename(columns={'effect_size': metric}))
    
    # Merge all DataFrames on index (genes)
    merged_df = pd.concat(combined_dfs, axis=1, join='outer')
    
    fig, ax = plt.subplots(figsize=(7, 3))
    if ylims:
        plt.ylim(ylims)
    
    # Plot each metric as a separate line using the specified colors
    for metric in merged_df.columns:
        color = metric_colours.get(metric, 'black')  # Default to black if metric not found in dict
        sns.scatterplot(x=range(len(merged_df.index)), y=merged_df[metric], label=metric, alpha=0.6, color=color, s=15, linewidth=0.1)
    
    # Labels and formatting
    plt.ylabel("Effect Size", fontsize=16)
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    ax.tick_params(axis='y', labelsize=14)

    if input_type == 'Genes':
        plt.title("Effect Sizes for Common Transcripts", fontsize=18)
        plt.xlabel("Genes", fontsize=16)
    elif input_type == 'Proteins':
        plt.title("Effect Sizes for Common Targets", fontsize=18)
        plt.xlabel("Proteins", fontsize=16)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Horizontal line at 0
    
    # Remove x-axis ticks
    plt.xticks([], [])
    
    # Adjust legend marker size
    legend = plt.legend(facecolor='white', framealpha=0.8, fontsize=12, loc='upper right')
    for handle in legend.legendHandles:
        handle.set_sizes([200])  # Increase legend marker size
        handle.set_linewidth(4)  # Increase legend line width for visibility
    
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Generate filename based on metrics
    metrics_str = "_".join(sorted(effect_size_dict.keys()))
    filename = f"{output_dir}/effect_sizes_{metrics_str}.png"
    
    # Save the plot
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {filename}")
    
    # Show the plot
    plt.show()

def filter_named_genes(dge_results, padj_threshold=0.05):
    """
    Removes novel genes from the significant genes dictionary and maps Ensembl IDs to gene names.
    
    Args:
        dge_results (dict): Dictionary of DataFrames containing DGE results for each cluster or experiment.
    
    Returns:
        dict: A new dictionary with novel genes removed and Ensembl IDs mapped to gene names.
    """
    
    filtered_results = {}
    for label, data in dge_results.items():
        if 'gene_name_biomart' in data.columns:
            # Remove novel genes
            filtered_data = data[~data['gene_name_biomart'].str.contains('novel', case=False, na=False)]
            
            # Map Ensembl IDs to gene names
            filtered_data = filtered_data[filtered_data['adj.P.Val'] <= padj_threshold].dropna(subset=['gene_name_biomart']).set_index('gene_name_biomart')
        filtered_results[label] = filtered_data
    
    return filtered_results

def query_reactome_pathways(named_genes, output_dir, padj_threshold=0.05, min_overlap=3):
    """
    Queries Reactome pathways for a dictionary of named genes structured like significant_genes using gprofiler enrichr.
    
    Args:
        named_genes (dict): Dictionary of DataFrames containing named genes for each cluster or experiment.
    
    Returns:
        dict: Dictionary with Reactome pathway results per cluster.
    """
    import gseapy as gp
    reactome_results = {}
    
    for label, data in named_genes.items():
        genes = list(data.index)  # Gene names are now the index
        if not genes:
            reactome_results[label] = []
            continue
        
        # Query Reactome via Enrichr
        try:
            enrichr_results = gp.enrichr(gene_list=genes, gene_sets='Reactome_Pathways_2024', organism='human')
            if enrichr_results is not None:
                results_df = enrichr_results.results
                results_df = results_df[(results_df['Adjusted P-value'] <= padj_threshold) & (results_df['Overlap'].apply(lambda x: int(x.split('/')[0]) >= min_overlap))]
                reactome_results[label] = results_df if not results_df.empty else []
            else:
                reactome_results[label] = []
        except Exception as e:
            reactome_results[label] = []
            print(f"Error querying Reactome for {label}: {str(e)}")

    if not os.path.exists(f'{output_dir}/reactome_results'):
        os.makedirs(f'{output_dir}/reactome_results')
    
    for label, pathways in reactome_results.items():
        num_pathways = len(pathways) if isinstance(pathways, list) else pathways.shape[0]
        print(f"{label}: {num_pathways} significant pathways")
        
        # Save non-empty DataFrames to CSV
        if not isinstance(pathways, list) and not pathways.empty:
            file_path = f"{output_dir}/reactome_results/{label}_reactome_results.csv"
            pathways.to_csv(file_path, index=False)
            print(f"Saved {label} results to {file_path}")
    
    return reactome_results

def filter_pathways_by_keywords(significant_pathways, keywords, output_dir, domain="Fibrosis"):
    """
    Filters significant pathways based on a list of keywords and saves results to CSV.

    Args:
        significant_pathways (dict): Dictionary of DataFrames containing pathway enrichment results per cluster.
        keywords (list): List of keywords to filter pathways.
        output_dir (str or Path): Base output directory.
        domain (str, optional): Label for file names (e.g., "Fibrosis", "Metabolism").
                                Defaults to "Fibrosis".
    
    Returns:
        dict: Dictionary mapping cluster label -> filtered DataFrame (or empty list if none).
    """
    output_dir = Path(output_dir) / "filtered_pathways"
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered_results = {}
    
    for label, pathways in significant_pathways.items():
        if isinstance(pathways, list) or pathways.empty:
            filtered_results[label] = []
            continue

        # keyword filter
        filtered_df = pathways[pathways['Term'].str.contains('|'.join(keywords), case=False, na=False)]
        filtered_results[label] = filtered_df if not filtered_df.empty else []
        print(f"{label}: {len(filtered_df)} pathways after filtering")

        # save to CSV if non-empty
        if not filtered_df.empty:
            file_path = output_dir / f"{label}_{domain}_pathways.csv"
            filtered_df.to_csv(file_path, index=False)
            print(f"Saved {label} selected pathways to {file_path}")

    return filtered_results

def compute_pathway_weighted_effect(filtered_pathways, significant_genes):
    """
    Computes the weighted effect size for pathways based on gene effect sizes and provides a breakdown
    of the number of upregulated and downregulated genes.
    
    Args:
        filtered_pathways (dict): Dictionary of DataFrames containing filtered pathway enrichment results per cluster.
        significant_genes (dict): Dictionary of DataFrames containing significant genes and their effect sizes.
    
    Returns:
        dict: Dictionary of DataFrames with pathways, their weighted effect sizes, up/downregulated gene counts,
              total pathway genes, summed positive and negative effect sizes, and weighted positive/negative effects.
    """
    import pandas as pd
    pathway_weighted_effects = {}
    
    for label, pathways in filtered_pathways.items():
        if isinstance(pathways, list) or pathways.empty:
            pathway_weighted_effects[label] = pd.DataFrame(columns=['Pathway', 'Weighted Effect', 'Upregulated Genes', 'Downregulated Genes', 'Total Pathway Genes', 'Sum Positive Effect', 'Sum Negative Effect', 'Weighted Positive Effect', 'Weighted Negative Effect'])
            continue
        
        pathway_data = []
        for _, row in pathways.iterrows():
            pathway_name = row['Term']
            associated_genes = [gene.strip().upper() for gene in row['Genes'].split(';')]  # Normalize gene names
            
            # Get total pathway genes from filtered_pathways
            total_pathway_genes = int(row['Overlap'].split('/')[1]) if 'Overlap' in row else len(associated_genes)
            
            # Extract effect sizes from significant_genes
            effect_sizes = []
            sum_positive_effect = 0
            sum_negative_effect = 0
            upregulated_count = 0
            downregulated_count = 0
            if label in significant_genes:
                gene_index = significant_genes[label].index.str.upper()
                for gene in associated_genes:
                    if gene in gene_index:
                        effect_size = significant_genes[label].loc[gene, 'effect_size']
                        effect_sizes.append(effect_size)
                        if effect_size > 0:
                            upregulated_count += 1
                            sum_positive_effect += effect_size
                        elif effect_size < 0:
                            downregulated_count += 1
                            sum_negative_effect += effect_size
            
            if not effect_sizes:
                continue
            
            weighted_effect = ((upregulated_count * sum_positive_effect) + (downregulated_count * sum_negative_effect)) / total_pathway_genes if total_pathway_genes > 0 else 0
            weighted_positive_effect = (upregulated_count * sum_positive_effect) / total_pathway_genes if total_pathway_genes > 0 else 0
            weighted_negative_effect = (downregulated_count * sum_negative_effect) / total_pathway_genes if total_pathway_genes > 0 else 0
            
            pathway_data.append((pathway_name, weighted_effect, upregulated_count, downregulated_count, total_pathway_genes, sum_positive_effect, sum_negative_effect, weighted_positive_effect, weighted_negative_effect))
        
        pathway_weighted_effects[label] = pd.DataFrame(pathway_data, columns=['Pathway', 'Weighted Effect', 'Upregulated Genes', 'Downregulated Genes', 'Total Pathway Genes', 'Sum Positive Effect', 'Sum Negative Effect', 'Weighted Positive Effect', 'Weighted Negative Effect'])
    
    return pathway_weighted_effects


def break_before_and(label):
    return label.replace(' and ', '\nand ')

def plot_pathway_weighted_enrichment(pathway_effects, metric_colours, output_dir, ordered_pathways=None, pathways_to_rename=None, pathways_to_bold=None, domain="ECM Organisation", formatting='full'):
    """
    Plots the weighted positive and negative enrichment for each pathway across all metrics.
    
    Args:
        pathway_effects (dict): Dictionary containing DataFrames with weighted pathway effects.
        metric_colours (dict): Dictionary mapping metrics to colors.
        ordered_pathways (list, optional): List of pathway names defining the order on the x-axis.
                                           If None, pathways are sorted alphabetically.
        pathways_to_rename (dict, optional): Dictionary mapping pathway names to shortened versions.
        pathways_to_bold (list, optional): List of pathway names to boldface in x-axis labels.
        domain (str, optional): A term used in the title and filename to indicate the biological domain 
                                (e.g., "ECM Organisation", "Metabolism"). Defaults to "ECM Organisation".
                                                   If None, pathways are sorted alphabetically.
        formatting (str, optional): A term used in the title and filename to indicate the purpose of the figure 
                                (e.g., "full", "paper"). Defaults to "full".
    """
    
    sns.set_context("poster")
    sns.set_style("white")

    fig, ax = plt.subplots(figsize=(14, 8))
    width = 0.15  # Thinner bars
    
    # Collect all pathways from the results if ordered_pathways is not provided
    if ordered_pathways is None:
        all_pathways = set()
        for df in pathway_effects.values():
            if not df.empty:
                all_pathways.update(df['Pathway'])
        ordered_pathways = sorted(all_pathways)

    # Apply pathway renaming if provided
    if pathways_to_rename:
        ordered_pathways = [pathways_to_rename.get(p, p) for p in ordered_pathways]
    
    x = np.arange(len(ordered_pathways))
    
    # Spread bars apart for each metric
    bar_positions = np.linspace(-3 * width, 3 * width, num=len(pathway_effects), endpoint=True)
    
    for i, (label, df) in enumerate(pathway_effects.items()):
        if df.empty:
            continue  # Skip empty clusters
        
        # Reindex using the provided order, fill missing with 0
        if pathways_to_rename:
            df['Pathway'] = df['Pathway'].map(lambda p: pathways_to_rename.get(p, p))
        df = df.set_index('Pathway').reindex(ordered_pathways).fillna(0)
        
        ax.bar(x + bar_positions[i], df['Weighted Positive Effect'], width=width, 
               color=metric_colours.get(label, 'gray'), label=label, alpha=0.8)
        ax.bar(x + bar_positions[i], df['Weighted Negative Effect'], width=width, 
               color=metric_colours.get(label, 'gray'), alpha=0.4)

    plt.axhline(y=0, color='grey', linestyle='-')  # horizontal line at 0

    # Define the center positions of each group
    group_centers = x + np.max(bar_positions)
    ax.set_xticks(group_centers)
    
    wrapped_labels = []
    for label in ordered_pathways:
        display_label = break_before_and(label)
        if pathways_to_bold and label in pathways_to_bold:
            wrapped_labels.append(plt.Text(0, 0, display_label, fontweight='bold'))
        else:
            wrapped_labels.append(plt.Text(0, 0, display_label))
    
    # Apply wrapped labels with appropriate formatting
    ax.set_xticklabels([lbl.get_text() for lbl in wrapped_labels], rotation=30, ha='right', fontsize=17)
    for tick, lbl in zip(ax.get_xticklabels(), wrapped_labels):
        tick.set_fontweight(lbl.get_fontweight())
        
    ax.set_ylabel('Enrichment Score')
    if domain == "ECM Organisation":
        pathways_title = f"{domain} Pathways"
    if domain == "Selected":
        pathways_title = "Major Liver Functional Pathways"
    #ax.set_xlabel(pathways_title)

    if domain == 'ECM Organisation' and formatting == 'paper':   
        ax.set_ylim(-20)

    if formatting == 'full':
        ax.legend(title='Fibrosis metrics', facecolor='white', framealpha=0.8)

        # Generate dynamic title and filename using the domain argument
        title = f"Enrichment of {domain} Pathways Across Fibrosis Metrics"
        ax.set_title(title, fontsize=16)
    
    sns.despine()
    plt.tight_layout()
    
    filename = f"{output_dir}/{domain.lower()}_pathway_enrichment_{formatting}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Plot saved as {filename}")
    
    plt.show()


def plot_only_legend(metric_colours, output_dir, filename="metrics_legend.png"):
    """
    Plots only the legend for the metrics, excluding the first four items.

    Args:
        metric_colours (dict): Dictionary mapping metrics to colors.
        output_dir (str): Directory to save the output plot.
        filename (str, optional): Name of the file to save. Defaults to "legend_only.png".
    """
    sns.set_context("poster")
    sns.set_style("white")

    fig, ax = plt.subplots(figsize=(4, 4))
    handles = []
    labels = []

    # Exclude the first four items
    filtered_items = list(metric_colours.items())[4:]

    for label, color in filtered_items:
        handles.append(plt.Line2D([0], [0], linestyle='-', linewidth=12, color=color, label=label))
        labels.append(label)

    # Create a separate legend
    leg = ax.legend(handles=handles, labels=labels, title="Fibrosis Metrics", loc='center')
    ax.axis('off')  # Turn off the axis

    sns.despine()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filepath = f"{output_dir}/{filename}"
    plt.savefig(filepath, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Legend saved as {filepath}")

    plt.show()


def get_unique_pathways(fibrosis_pathways):
    """
    Extracts and returns a sorted list of unique pathway names from fibrosis_pathways.

    Args:
        fibrosis_pathways (dict): Dictionary where keys are labels (e.g., clusters)
                                  and values are either DataFrames or lists of DataFrames/dictionaries
                                  containing pathway enrichment results.
                                  Each DataFrame/dictionary should contain a column/key named 'Term'.

    Returns:
        list: Sorted list of unique pathway names.
    """
    unique_pathways = set()
    
    for key, value in fibrosis_pathways.items():
        if isinstance(value, pd.DataFrame):
            if not value.empty and 'Term' in value.columns:
                unique_pathways.update(value['Term'].dropna().unique())
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, pd.DataFrame):
                    if not item.empty and 'Term' in item.columns:
                        unique_pathways.update(item['Term'].dropna().unique())
                elif isinstance(item, dict):
                    if 'Term' in item:
                        unique_pathways.add(item['Term'])
                elif isinstance(item, str):
                    unique_pathways.add(item)
    return sorted(unique_pathways)


# Define a function to get the top N genes based on effect size
def get_top_genes(dataframe, top_n=None, downregulated=False):
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame, but got {type(dataframe)}")
    if dataframe.empty:
        return dataframe  # Return an empty DataFrame if no genes exist
    
    # Ensure required columns exist
    required_cols = {'gene_name_biomart', 'effect_size', 'adj.P.Val'}
    missing_cols = required_cols - set(dataframe.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in dataframe: {missing_cols}")
    
    # Sort by effect size and select the top N genes
    ascending = downregulated  # If downregulated, sort in ascending order to get the most negative values
    return dataframe.sort_values(by='effect_size', ascending=ascending).head(top_n) if top_n else dataframe

def create_summary_table(significant_genes, top_n, downregulated=False):
    top_genes = {}
    for key, df in significant_genes.items():
        if isinstance(df, pd.DataFrame):
            top_genes[key] = get_top_genes(df, top_n, downregulated)
        else:
            raise TypeError(f"Expected DataFrame for cluster {key}, got {type(df)}")
    
    # Handle empty top_genes case
    if not top_genes:
        return pd.DataFrame()  # Return an empty DataFrame if no data exists
    
    # Create a unique feature_id list for the summary table
    summary_table = pd.DataFrame(
        index=pd.Index(
            pd.concat([df.index.to_series() for df in top_genes.values() if not df.empty]).unique(),
            name='Feature ID'
        )
    )
    
    # Merge effect size, P-value, and gene name data for each experiment
    for key, df in top_genes.items():
        if not df.empty:
            df = df.rename_axis('Feature ID')  # Ensure index is 'Feature ID'
            summary_table = summary_table.join(
                df.drop(columns=['gene_name_biomart'], errors='ignore').rename(
                    columns={'effect_size': f'{key} Effect Size', 'adj.P.Val': f'{key} adj.P.Val'}
                )
            )
        
        if 'Gene Name' not in summary_table.columns and 'gene_name_biomart' in df.columns:
            summary_table = summary_table.join(df[['gene_name_biomart']].rename(columns={'gene_name_biomart': 'Gene Name'}))
    
    # Sort the summary table by the strongest effect size across experiments
    effect_size_cols = [f'{key} Effect Size' for key in significant_genes.keys() if f'{key} Effect Size' in summary_table.columns]
    
    if downregulated:
        summary_table['Min Effect Size'] = summary_table[effect_size_cols].min(axis=1)
        summary_table = summary_table.sort_values('Min Effect Size')
    else:
        summary_table['Max Effect Size'] = summary_table[effect_size_cols].max(axis=1)
        summary_table = summary_table.sort_values('Max Effect Size', ascending=False)
    
    return summary_table

# Function to fill missing values in summary table from significant_genes
def fill_missing_values(summary_table, dge_results):
    summary_table = summary_table.rename_axis('Feature ID')
    for key, df in dge_results.items():
        effect_size_col = f'{key} Effect Size'
        pval_col = f'{key} adj.P.Val'
        
        if effect_size_col not in summary_table.columns:
            summary_table[effect_size_col] = None
        if pval_col not in summary_table.columns:
            summary_table[pval_col] = None
        
        df = df.rename_axis('Feature ID')  # Ensure index is 'Feature ID'
        feature_ids = set(summary_table.index) & set(df.index)
        for feature_id in feature_ids:
            summary_table.loc[feature_id, effect_size_col] = df.loc[feature_id, 'effect_size']
            summary_table.loc[feature_id, pval_col] = df.loc[feature_id, 'adj.P.Val']
            if 'Gene Name' in summary_table.columns and 'gene_name_biomart' in df.columns:
                summary_table.loc[feature_id, 'Gene Name'] = df.loc[feature_id, 'gene_name_biomart']
    
    return summary_table


def plot_top_genes_heatmap(
    summary_table,
    effect_size_cols,
    pval_cols,
    output_path,
    title="Top Genes",
    cmap="coolwarm",
    bold_genes=None,
    rename_genes=None
):

    sns.set_context("paper")

    # Prepare data
    heatmap_data = summary_table[effect_size_cols].copy().round(0).astype(int)
    heatmap_data.index = summary_table.index
    pval_data = summary_table[pval_cols].copy()
    pval_data.index = summary_table.index

    # if rename_genes:
    #     heatmap_data.rename(index=rename_genes, inplace=True)
    #     pval_data.rename(index=rename_genes, inplace=True)

    column_mapping = dict(zip(effect_size_cols, pval_cols))

    annot = pd.DataFrame("", index=heatmap_data.index, columns=heatmap_data.columns)
    stars_only = pd.DataFrame("", index=heatmap_data.index, columns=heatmap_data.columns)
    mask = pd.DataFrame(False, index=heatmap_data.index, columns=heatmap_data.columns)

    for effect_col, pval_col in column_mapping.items():
        # Vectorized masking of non-significant values
        if pval_col in pval_data.columns:
            mask[effect_col] = pval_data[pval_col] >= 0.05

        for gene in heatmap_data.index:
            if gene in pval_data.index and pval_col in pval_data.columns:
                pval = pval_data.loc[gene, pval_col]
                effect = heatmap_data.loc[gene, effect_col]
            else:
                continue

            if isinstance(pval, (float, int)) and pd.notnull(pval):
                if pval < 0.001:
                    stars = '***'
                elif pval < 0.01:
                    stars = '**'
                elif pval < 0.05:
                    stars = '*'
                else:
                    stars = ''
                    mask.loc[gene, effect_col] = True
                annot.loc[gene, effect_col] = f"{int(effect)}" if stars else ""
                stars_only.loc[gene, effect_col] = stars

    def choose_label(index, row):
        name = row['Gene Name']
        if isinstance(name, str) and 'novel' in name.lower():
            return index  # Feature ID from index
        return name

    gene_labels = [
        rename_genes.get(choose_label(idx, row), choose_label(idx, row))
        if rename_genes else choose_label(idx, row)
        for idx, row in summary_table.loc[heatmap_data.index].iterrows()
    ]

    # Plotting
    plt.figure(figsize=(6, 8))
    ax = sns.heatmap(
        heatmap_data,
        annot=annot,
        fmt='s',
        cmap=cmap if isinstance(cmap, str) else cmap,
        center=0,
        linewidths=0.5,
        annot_kws={"size": 12},
        cbar=False,
        xticklabels=[col.split()[0] for col in effect_size_cols],
        yticklabels=gene_labels
    )

    for y_index, gene in enumerate(heatmap_data.index):
        for x_index, metric in enumerate(heatmap_data.columns):
            star_text = stars_only.loc[gene, metric]
            if isinstance(star_text, str) and star_text.strip() != "":
                value = heatmap_data.loc[gene, metric]
                star_color = 'white' if abs(value) > 25 else 'black'
                ax.text(
                    x_index + 0.95, y_index + 0.1,
                    star_text,
                    ha='right', va='top', fontsize=9, color=star_color
                )

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    if bold_genes:
        for label in ax.get_yticklabels():
            if label.get_text() in bold_genes:
                label.set_fontweight('bold')

    ax.set_title(title, fontsize=20, pad=20)
    plt.text(0.5, 1.005, "Effect size", ha='center', va='bottom', transform=plt.gca().transAxes)
    plt.xlabel('Fibrosis Metric', fontsize=18)
    label = 'Protein' if 'protein' in title.lower() else 'Gene'
    plt.ylabel(label, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# Example usage:
# plot_top_genes_heatmap(
#     summary_table_upregulated,
#     ['C4 Effect Size','C5 Effect Size','C6 Effect Size','F3-F4 Effect Size','CPA Effect Size'],
#     ['C4 adj.P.Val','C5 adj.P.Val','C6 adj.P.Val','F3-F4 adj.P.Val','CPA adj.P.Val'],
#     output_path=f'{output_dir}/plots/top_{top_n}_upregulated_genes_heatmap.png',
#     title="Top Upregulated Genes",
#     cmap='coolwarm',
#     bold_genes=["COL10A1"],
#     rename_genes={"novel transcript": "KRT222"}
# )


def extract_pathway_genes(significant_pathways, cluster_key='C5', delimiter=';'):
    """
    Extracts a dictionary mapping pathway names to their gene sets from the Genes column 
    of the specified cluster in significant_pathways.

    Args:
        significant_pathways (dict): Dictionary of DataFrames containing pathway enrichment results per cluster.
        cluster_key (str): Key to select the specific cluster (e.g., 'C5'). Defaults to 'C5'.
        delimiter (str): Delimiter used to split the Genes column if it is a string. Defaults to ';'.
    
    Returns:
        dict: Dictionary where keys are pathway names (from 'Term') and values are sets of genes.
    """
    if cluster_key not in significant_pathways:
        raise KeyError(f"Cluster key '{cluster_key}' not found in significant_pathways.")
    
    df = significant_pathways[cluster_key]
    
    # Verify that the required columns exist
    if 'Term' not in df.columns or 'Genes' not in df.columns:
        raise ValueError("DataFrame must contain 'Term' and 'Genes' columns.")
    
    pathway_genes = {}
    for idx, row in df.iterrows():
        pathway = row['Term']
        genes = row['Genes']
        
        # If the genes are stored as a string, split them by the delimiter;
        # otherwise assume they are already list-like.
        if isinstance(genes, str):
            genes_set = set(g.strip() for g in genes.split(delimiter) if g.strip())
        elif isinstance(genes, (list, set)):
            genes_set = set(genes)
        else:
            # Fallback: convert to string and split.
            genes_set = set(g.strip() for g in str(genes).split(delimiter) if g.strip())
        
        pathway_genes[pathway] = genes_set
    
    return pathway_genes

# Example usage:
# Assuming significant_pathways is defined and significant_pathways['C5'] is a valid DataFrame:
# pathway_genes = extract_pathway_genes(significant_pathways, cluster_key='C5')
# print(pathway_genes)


def compute_overlap_scores(pathway_genes, pathway_order):
    """
    Computes pairwise overlap scores between pathways.
    
    The overlap score is defined as the number of shared genes between two pathways.
    
    Args:
        pathway_genes (dict): Dictionary mapping pathway names to a list or set of genes.
                              Example: {'Extracellular Matrix Organization': {'gene1', 'gene2', ...}, ...}
        pathway_order (list): Ordered list of pathway names to compute the overlap scores.
                              Example: ['Extracellular Matrix Organization', ...]
    
    Returns:
        pd.DataFrame: A DataFrame where each cell (i, j) contains the count of shared genes between 
                      pathway i and pathway j.
    """
    # Create an empty DataFrame with pathway names as both index and columns
    overlap_df = pd.DataFrame(index=pathway_order, columns=pathway_order)
    
    # Convert gene lists to sets (if they aren't already)
    gene_sets = {path: set(genes) for path, genes in pathway_genes.items()}
    
    # Compute pairwise overlap scores
    for i, path1 in enumerate(pathway_order):
        for j, path2 in enumerate(pathway_order):
            # If either pathway is not present in gene_sets, treat overlap as 0
            if path1 not in gene_sets or path2 not in gene_sets:
                overlap_score = 0
            else:
                overlap_score = round(len(gene_sets[path1].intersection(gene_sets[path2])))
            overlap_df.at[path1, path2] = overlap_score
            overlap_df = overlap_df.astype("Int64")
            
    return overlap_df


def compute_metric_correlations(
    significant_genes: dict,
    metric_labels: list | None = None,
    method: str = "pearson",
    min_overlap: int = 3,
):
    """
    Compute correlations of effect sizes across clusters/metrics using pairwise-complete observations.
    
    Parameters
    ----------
    significant_genes : dict[str, pd.DataFrame]
        Mapping cluster/metric name -> DataFrame with an 'effect_size' column and gene IDs as index.
        Gene indices need not be identical across DataFrames.
    metric_labels : list[str] | None, optional
        Optional labels to use for rows/cols of the outputs. If provided, length must match the
        number of clusters in `significant_genes`. If None, cluster keys are used.
    method : {"pearson", "spearman"}, default "pearson"
        Correlation type.
    min_overlap : int, default 3
        Minimum number of shared (non-NaN) genes required to compute a correlation; otherwise NaN.

    Returns
    -------
    corr_df : pd.DataFrame
        Correlation matrix (float) between clusters/metrics.
    pval_df : pd.DataFrame
        Matrix of two-sided p-values corresponding to `corr_df`.
    n_shared_df : pd.DataFrame
        Matrix of the number of genes shared (non-NaN in both) for each pair.
    """
    # --- Build a union index of all genes (preserving Ensembl IDs or whatever index you use)
    all_genes = set()
    for df in significant_genes.values():
        if isinstance(df, pd.DataFrame) and not df.empty:
            all_genes.update(df.index)
    all_genes = sorted(all_genes)

    # --- Assemble effect size matrix with outer join semantics (no zero-filling!)
    effect_size_matrix = pd.DataFrame(index=all_genes, dtype=float)
    for label, df in significant_genes.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            effect_size_matrix[label] = np.nan
            continue
        s = df['effect_size']
        # In case of duplicated gene indices, average them
        if not s.index.is_unique:
            s = s.groupby(level=0).mean()
        effect_size_matrix[label] = s.reindex(effect_size_matrix.index)

    cols = list(effect_size_matrix.columns)
    n = len(cols)

    # --- Initialize outputs
    corr_df   = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    pval_df   = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    n_shared_df = pd.DataFrame(0,     index=cols, columns=cols, dtype=int)

    # --- Pairwise correlations with pairwise-complete observations
    method = method.lower()
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")

    for i in range(n):
        xi = effect_size_matrix.iloc[:, i]
        for j in range(i, n):
            yj = effect_size_matrix.iloc[:, j]
            mask = xi.notna() & yj.notna()
            k = int(mask.sum())
            n_shared_df.iat[i, j] = n_shared_df.iat[j, i] = k

            if i == j:
                corr, p = 1.0, 0.0
            elif k >= min_overlap:
                if method == "pearson":
                    corr, p = stats.pearsonr(xi[mask], yj[mask])
                else:  # spearman
                    corr, p = stats.spearmanr(xi[mask], yj[mask])
            else:
                corr, p = np.nan, np.nan

            corr_df.iat[i, j] = corr_df.iat[j, i] = corr
            pval_df.iat[i, j] = pval_df.iat[j, i] = p

    # --- Optional relabeling
    if metric_labels is not None:
        if len(metric_labels) != n:
            raise ValueError("metric_labels length must match the number of metrics.")
        corr_df.index = corr_df.columns = metric_labels
        pval_df.index = pval_df.columns = metric_labels
        n_shared_df.index = n_shared_df.columns = metric_labels

    return corr_df, pval_df, n_shared_df


def jaccard_top_effect_genes(
    significant_genes: dict,
    reference_key: str = "C5",
    compare_keys: list | None = None,
    n: int = 100,
    direction: str = "up",          # "up" or "down"
    include_counts: bool = False,   # store counts in df.attrs if True
) -> pd.DataFrame:
    """
    Compute Jaccard index between top-N up/down-regulated genes of each metric and a reference metric.

    Parameters
    ----------
    significant_genes : dict[str, pd.DataFrame]
        metric/cluster -> DataFrame with 'effect_size' column and gene IDs as index.
    reference_key : str, default "C5"
        Reference metric.
    compare_keys : list[str] | None, default None
        Metrics to compare vs reference. If None, uses all except the reference.
    n : int, default 100
        Number of top genes to select per metric.
    direction : {"up","down"}, default "up"
        Use top upregulated (largest positive) or top downregulated (most negative) genes.
    include_counts : bool, default False
        If True, stores counts in jaccard_df.attrs (no extra columns).

    Returns
    -------
    pd.DataFrame
        Index: metric; Columns: ['Jaccard_index','intersection','union'].
        If include_counts=True, adds:
            - jaccard_df.attrs['n_top_ref'] = int
            - jaccard_df.attrs['n_top_per_key'] = dict[str,int]
    """
    if reference_key not in significant_genes:
        raise KeyError(f"reference_key '{reference_key}' not found in significant_genes.")
    if direction not in {"up", "down"}:
        raise ValueError("direction must be 'up' or 'down'.")

    if compare_keys is None:
        compare_keys = [k for k in significant_genes.keys() if k != reference_key]

    def _top_index(df: pd.DataFrame) -> list:
        if not isinstance(df, pd.DataFrame) or df.empty or "effect_size" not in df.columns:
            return []
        s = df["effect_size"].dropna()
        if direction == "up":
            s = s[s > 0].nlargest(n)
        else:  # "down"
            s = s[s < 0].nsmallest(n)
        return list(s.index)

    # Build sets (reference + compare keys only)
    ref_set = set(_top_index(significant_genes.get(reference_key, pd.DataFrame())))
    rows = {}
    n_top_per_key = {}

    for key in compare_keys:
        other_set = set(_top_index(significant_genes.get(key, pd.DataFrame())))
        n_top_per_key[key] = len(other_set)
        inter = len(ref_set & other_set)
        union = len(ref_set | other_set)
        jaccard = inter / union if union else np.nan
        rows[key] = {"Jaccard_index": jaccard, "intersection": inter, "union": union}

    jaccard_df = pd.DataFrame.from_dict(rows, orient="index").sort_values("Jaccard_index", ascending=False)

    if include_counts:
        jaccard_df.attrs["n_top_ref"] = len(ref_set)
        jaccard_df.attrs["n_top_per_key"] = n_top_per_key

    return jaccard_df