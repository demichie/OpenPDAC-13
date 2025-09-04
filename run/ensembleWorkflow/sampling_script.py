import pandas as pd
import numpy as np
from pathlib import Path
from linecache import getline
from typing import List, Tuple, Dict, Any
from scipy.stats import qmc
from scipy.stats import truncnorm  # For the truncated normal distribution
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Global or configuration constants
OUTPUT_CSV_FILE = 'samples.csv'
OUTPUT_PLOT_BASE_FILENAME = 'samples_plot'


def read_asc_file(asc_file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Reads an ESRI ASCII grid file (.asc) and returns its data and metadata.

    Args:
        asc_file_path (Path): The path to the .asc file to be read.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]: A tuple containing:
            - A numpy array with the grid data, flipped vertically to have its
              origin at the bottom-left corner.
            - A dictionary with the header metadata (e.g., 'ncols', 'nrows',
              'xllcorner', 'yllcorner', 'cellsize').

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    print(f"Reading .asc file: {asc_file_path}")
    if not asc_file_path.is_file():
        raise FileNotFoundError(f"File {asc_file_path} was not found.")
    header_lines = [getline(str(asc_file_path), i) for i in range(1, 7)]
    header = {
        key.lower(): float(value)
        for key, value in (line.split() for line in header_lines)
    }
    header['ncols'], header['nrows'] = int(header['ncols']), int(
        header['nrows'])
    data = pd.read_table(asc_file_path,
                         delim_whitespace=True,
                         header=None,
                         skiprows=6,
                         dtype=np.float64).values
    return np.flipud(data), header


def generate_samples(parameter_config: List[Dict],
                     n_samples: int,
                     method: str = 'lhs') -> Dict[str, Any]:
    """Generates a set of parameter samples using a specified sampling method.

    This is the core engine that transforms a uniform Latin Hypercube sample
    into samples for various continuous and discrete distributions using the
    Inverse Transform Sampling technique.

    Args:
        parameter_config (List[Dict]):
            A list where each dictionary defines a parameter to be sampled.
            Each dictionary must contain specific keys depending on the 'type'
            and 'distribution'.

            For 'type': 'continuous':
                - 'name' (str): The parameter's name.
                - 'distribution' (str): One of 'linear', 'log', 'truncnorm',
                  'powerlaw', or 'trunclognorm'.
                - 'range' (List[float]): The [min, max] bounds for the sample.

                Additional keys for specific distributions:
                - for 'truncnorm':
                    - 'mean' (float): The mean of the underlying normal distribution.
                    - 'std_dev' (float): The standard deviation of the underlying
                      normal distribution.
                - for 'powerlaw':
                    - 'exponent' (float): The exponent 'k' for the p(x) ~ x^k relation.
                - for 'trunclognorm':
                    - 'log_mean' (float): The mean of the *logarithm* of the variable.
                    - 'log_std_dev' (float): The standard deviation of the
                      *logarithm* of the variable.

            For 'type': 'discrete':
                - 'name' (str) or 'names' (List[str]): The output column name(s).
                - 'values' (List): The list of possible discrete values.
                - 'weights' (List[float], optional): Relative weights for each value.
                - 'unpack' (bool, optional): If True and 'names' is used, unpacks
                  paired values (like coordinates) into separate columns.

        n_samples (int): The number of samples to generate for each parameter.
        method (str): The sampling method to use. Currently supports 'lhs'.

    Returns:
        Dict[str, Any]: A dictionary where keys are the parameter names and
        values are lists or numpy arrays of the generated samples.

    Raises:
        KeyError: If a parameter's configuration is missing a required key.
        ValueError: If an unknown sampling method or an invalid value (e.g.,
                    non-positive std_dev, non-positive range for log/powerlaw)
                    is provided.
    """
    samples = {}
    if method.lower() == 'lhs':
        print("Using Latin Hypercube Sampling for all configured parameters.")
        lhs_params = parameter_config
        n_dims = len(lhs_params)
        sampler = qmc.LatinHypercube(d=n_dims, seed=np.random.default_rng())
        unit_samples = sampler.random(n=n_samples)

        for i, param in enumerate(lhs_params):
            param_type = param.get('type', 'continuous')
            if param_type == 'continuous':
                distribution = param.get('distribution', 'linear')
                if distribution == 'linear':
                    min_val, max_val = param['range']
                    samples[param['name']] = min_val + \
                        unit_samples[:, i] * (max_val - min_val)
                elif distribution == 'log':
                    min_val, max_val = param['range']
                    if min_val == max_val:
                        samples[param['name']] = np.full(n_samples, min_val)
                    else:
                        samples[param['name']] = np.exp(
                            np.log(min_val) + unit_samples[:, i] *
                            (np.log(max_val) - np.log(min_val)))
                elif distribution == 'truncnorm':
                    try:
                        mean, std_dev, (
                            min_val, max_val
                        ) = param['mean'], param['std_dev'], param['range']
                    except KeyError as e:
                        raise KeyError(
                            f"Parameter '{param['name']}' with 'truncnorm' is missing key: {e}"
                        )
                    if std_dev <= 0:
                        raise ValueError(
                            f"Std dev for '{param['name']}' must be positive.")
                    a, b = (min_val - mean) / \
                        std_dev, (max_val - mean) / std_dev
                    dist = truncnorm(a, b, loc=mean, scale=std_dev)
                    samples[param['name']] = dist.ppf(unit_samples[:, i])
                elif distribution == 'trunclognorm':
                    try:
                        log_mean, log_std_dev, (min_val, max_val) = param[
                            'log_mean'], param['log_std_dev'], param['range']
                    except KeyError as e:
                        raise KeyError(
                            f"Parameter '{param['name']}' with 'trunclognorm' is missing key: {e}"
                        )
                    if log_std_dev <= 0:
                        raise ValueError(
                            f"Log std dev for '{param['name']}' must be positive."
                        )
                    if min_val <= 0:
                        raise ValueError(
                            f"Range for trunclognorm must be positive. Got min_val={min_val}"
                        )

                    # Work in log-space: Y = ln(X)
                    log_min, log_max = np.log(min_val), np.log(max_val)

                    # Standardize the log-space bounds for scipy's truncnorm
                    a = (log_min - log_mean) / log_std_dev
                    b = (log_max - log_mean) / log_std_dev

                    # Create a truncated normal distribution in log-space
                    dist_log = truncnorm(a, b, loc=log_mean, scale=log_std_dev)

                    # Sample from it using the LHS samples
                    log_samples = dist_log.ppf(unit_samples[:, i])

                    # Convert samples back to the original space by taking the exponential
                    samples[param['name']] = np.exp(log_samples)
                elif distribution == 'powerlaw':
                    try:
                        exponent, (min_val,
                                   max_val) = param['exponent'], param['range']
                    except KeyError as e:
                        raise KeyError(
                            f"Parameter '{param['name']}' with 'powerlaw' is missing key: {e}"
                        )
                    if min_val <= 0:
                        raise ValueError(
                            f"The range for a power-law distribution must be positive. Got min_val={min_val}"
                        )
                    if exponent == -1:
                        raise ValueError(
                            "Exponent of -1 is a special case. Use 'log' distribution instead."
                        )
                    y = unit_samples[:, i]
                    k = exponent + 1.0
                    samples[param['name']] = (y * (max_val**k - min_val**k) +
                                              min_val**k)**(1.0 / k)

            elif param_type == 'discrete':
                values, weights = param['values'], param.get('weights', None)
                if weights is None:
                    indices = np.floor(unit_samples[:, i] *
                                       len(values)).astype(int)
                else:
                    cdf = np.cumsum(weights) / np.sum(weights)
                    indices = np.searchsorted(cdf,
                                              unit_samples[:, i],
                                              side='right')
                sampled_values = [values[idx] for idx in indices]
                if param.get('unpack', False):
                    unpacked = list(zip(*sampled_values))
                    for j, name in enumerate(param['names']):
                        samples[name] = unpacked[j]
                else:
                    samples[param['name']] = sampled_values
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    return samples


def plot_pair_grid(df: pd.DataFrame, axis_vars: List, category_vars: List,
                   log_scale_vars: set, config: Dict):
    """Generates the main pair plot with unified histograms and a correct, manual legend.

    Args:
        df (pd.DataFrame): The DataFrame containing the sample data.
        axis_vars (List[str]): A list of column names to be used as plot axes.
        category_vars (List[str]): A list of column names to be used for styling (hue/style).
        log_scale_vars (set): A set of column names that require a logarithmic scale.
        config (Dict): A dictionary containing plot configuration, like 'base_filename'.

    Returns:
        None. The plot is saved to both PNG and PDF files.
    """
    plot_df = df.copy()
    hue_var, style_var, ordered_cat_vars = None, None, []
    if category_vars:
        cat_uniqueness = sorted([(var, plot_df[var].nunique())
                                 for var in category_vars],
                                key=lambda x: x[1],
                                reverse=True)
        hue_var = cat_uniqueness[0][0]
        ordered_cat_vars.append(hue_var)
        if len(cat_uniqueness) > 1:
            style_var = cat_uniqueness[1][0]
            ordered_cat_vars.append(style_var)

    print("\n--- Generating Main Pair Plot ---")
    print(f"Plotting axes: {axis_vars}")
    if hue_var:
        print(
            f"Variable for color (hue): '{hue_var}' ({plot_df[hue_var].nunique()} unique values)"
        )
    if style_var:
        print(
            f"Variable for marker (style): '{style_var}' ({plot_df[style_var].nunique()} unique values)"
        )

    hue_map, marker_map = None, None
    if hue_var:
        hue_categories = sorted(plot_df[hue_var].unique())
        palette = sns.color_palette("viridis", n_colors=len(hue_categories))
        hue_map = {cat: color for cat, color in zip(hue_categories, palette)}
    if style_var:
        style_categories = sorted(plot_df[style_var].unique())
        markers = ['o', 'X', 's', '^', 'P', 'D'][:len(style_categories)]
        marker_map = {
            cat: marker
            for cat, marker in zip(style_categories, markers)
        }

    n_vars = len(axis_vars)
    fig, axes = plt.subplots(n_vars,
                             n_vars,
                             figsize=(n_vars * 2.5, n_vars * 2.5))
    for i, j in np.ndindex(axes.shape):
        ax = axes[i, j]
        x_var, y_var = axis_vars[j], axis_vars[i]
        if i == j:
            sns.histplot(data=plot_df,
                         x=x_var,
                         ax=ax,
                         legend=False,
                         log_scale=(x_var in log_scale_vars),
                         color="#555555")
        else:
            sns.scatterplot(data=plot_df,
                            x=x_var,
                            y=y_var,
                            hue=hue_var,
                            style=style_var,
                            ax=ax,
                            legend=False,
                            alpha=0.8,
                            s=40,
                            palette=hue_map,
                            markers=marker_map)
        if i != j and x_var in log_scale_vars:
            ax.set_xscale('log')
        if i != j and y_var in log_scale_vars:
            ax.set_yscale('log')
        if j == 0:
            ax.set_ylabel(y_var)
        else:
            ax.set_ylabel('')
        if i == n_vars - 1:
            ax.set_xlabel(x_var)
        else:
            ax.set_xlabel('')

    legend_elements, legend_title = [], ", ".join(ordered_cat_vars)
    if hue_var and style_var:
        for style_cat, marker in marker_map.items():
            for hue_cat, color in hue_map.items():
                label = f"{str(hue_cat)}, {str(style_cat)}"
                legend_elements.append(
                    Line2D([0], [0],
                           marker=marker,
                           color='w',
                           label=label,
                           markerfacecolor=color,
                           markersize=8))
    elif hue_var:
        for cat, color in hue_map.items():
            legend_elements.append(
                Line2D([0], [0],
                       marker='o',
                       color='w',
                       label=cat,
                       markerfacecolor=color,
                       markersize=8))

    fig.subplots_adjust(right=0.85)
    fig.legend(handles=legend_elements,
               loc='center left',
               bbox_to_anchor=(0.88, 0.5),
               title=legend_title)
    fig.suptitle('Pair Plot of Sampled Parameters', fontsize=16)

    base_path = Path(config['base_filename'])
    png_path = base_path.with_suffix('.png')
    pdf_path = base_path.with_suffix('.pdf')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(
        f"Main pair plot successfully saved to '{png_path}' and '{pdf_path}'")
    plt.close()


def plot_category_counts(df: pd.DataFrame, category_vars: List):
    """Generates a separate frequency bar chart for each categorical variable, saving as both PNG and PDF.

    Args:
        df (pd.DataFrame): The DataFrame containing the sample data.
        category_vars (List[str]): A list of categorical column names to plot.

    Returns:
        None. One or more plot files are saved to disk.
    """
    if not category_vars:
        return
    print("\n--- Generating Category Frequency Plots ---")
    for cat_var in category_vars:
        plt.figure(figsize=(8, 5))
        ax = sns.countplot(data=df,
                           x=cat_var,
                           palette="viridis",
                           order=sorted(df[cat_var].unique()))
        ax.set_title(f"Frequency of Categories for '{cat_var}'")
        ax.set_ylabel("Count")
        ax.set_xlabel("Category")
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        safe_cat_var = "".join(c for c in cat_var
                               if c.isalnum() or c in (' ', '_')).rstrip()
        base_filename = f"counts_of_{safe_cat_var.replace(' ', '_')}"
        png_path = Path(f"{base_filename}.png")
        pdf_path = Path(f"{base_filename}.pdf")
        plt.savefig(png_path, dpi=150)
        plt.savefig(pdf_path)
        print(f"Category count plot saved to '{png_path}' and '{pdf_path}'")
        plt.close()


def main():
    """Main script to generate, save, and visualize a parameter set for simulations.

    This script orchestrates the entire process:
    1. Defines the parameter space in a flexible configuration dictionary.
    2. Uses Latin Hypercube Sampling (LHS) to generate an efficient sample set.
    3. Saves the samples to a CSV file.
    4. Generates a comprehensive set of plots to visualize the samples,
       including a main pair plot and frequency counts for categorical variables.
    """
    N_SAMPLES = 100
    SAMPLING_METHOD = 'lhs'
    GENERATE_PLOT = True

    parameter_config = [
    {
            'name': 'Permeability',
            'type': 'continuous',
            'distribution': 'trunclognorm',
            'log_mean': -12.0,       # Mean of ln(Permeability)
            'log_std_dev': 1.5,      # Std dev of ln(Permeability)
            'range': [1e-7, 1e-4]  # Truncation range for Permeability itself
    }, {
        'name': 'Volume',
        'type': 'continuous',
        'distribution': 'log',
        'range': [1000, 1e6]
    }, {
        'name': 'Pressure',
        'type': 'continuous',
        'distribution': 'truncnorm',
        'mean': 50.0,
        'std_dev': 15.0,
        'range': [30.0, 90.0]
    }, {
        'name': 'GrainSize',
        'type': 'continuous',
        'distribution': 'powerlaw',
        'exponent': -1.5,
        'range': [0.1, 100.0]
    }, {
        'names': ['x', 'y'],
        'type': 'discrete',
        'values': [[500360, 4177600], [500650, 4177913]],
        'weights': [2, 1],
        'unpack': True,
        'plot_as': 'category'
    }, {
        'name': 'Velocity',
        'type': 'discrete',
        'values': [0.0, 30.0, 60.0, 90.0],
        'plot_as': 'category'
    }]

    samples = generate_samples(parameter_config, N_SAMPLES, SAMPLING_METHOD)
    df = pd.DataFrame(samples)
    df.to_csv(OUTPUT_CSV_FILE, index=False)
    print(
        f"\nSuccessfully generated '{OUTPUT_CSV_FILE}' with {N_SAMPLES} samples."
    )

    if GENERATE_PLOT:
        plot_df = df.copy()
        axis_vars, category_vars = [], []
        log_scale_vars = set()
        for param in parameter_config:
            if param.get('plot_as') == 'category':
                if param.get('unpack'):
                    source_name = f"({param['names'][0]},{param['names'][1]})"
                    plot_df[source_name] = plot_df.apply(
                        lambda r:
                        f"P({r[param['names'][0]]},{r[param['names'][1]]})",
                        axis=1)
                    category_vars.append(source_name)
                else:
                    category_vars.append(param['name'])
            else:
                if param.get('unpack'):
                    axis_vars.extend(param['names'])
                else:
                    axis_vars.append(param['name'])
            if param.get('distribution') in ['log', 'powerlaw','trunclognorm']:
                log_scale_vars.add(param['name'])

        plot_config = {'base_filename': OUTPUT_PLOT_BASE_FILENAME}
        plot_pair_grid(plot_df, axis_vars, category_vars, log_scale_vars,
                       plot_config)
        plot_category_counts(plot_df, category_vars)

    print("\nScript finished.")


if __name__ == '__main__':
    main()
