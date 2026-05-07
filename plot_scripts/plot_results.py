"""
Script to plot figure 2 in the paper.
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_event_files(base_dirs):
    """
    Recursively find all TensorBoard event files in the given directories.

    Args:
        base_dirs: List of base directory paths to search

    Returns:
        List of tuples: (method_name, seed_number, run_number, file_path)
    """
    event_files = []

    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"Warning: Directory '{base_dir}' not found, skipping...")
            continue

        # Walk through all subdirectories
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                # Look for TensorBoard event files
                if file.startswith('events.out.tfevents'):
                    file_path = os.path.join(root, file)

                    # Extract method name from base directory
                    method_name = os.path.basename(base_dir)
                    if method_name.startswith('results_'):
                        method_name = method_name.replace('results_', '')

                    # Try to extract seed number from path
                    seed_match = re.search(r'seed[_-]?(\d+)', root, re.IGNORECASE)
                    seed_num = int(seed_match.group(1)) if seed_match else None

                    # Try to extract run number from path
                    run_match = re.search(r'run(\d+)', root, re.IGNORECASE)
                    run_num = int(run_match.group(1)) if run_match else 1

                    event_files.append((method_name, seed_num, run_num, file_path))

    return event_files


def load_tensorboard_data(file_path, tag='average_episode_rewards'):
    """
    Load data from a TensorBoard event file.

    Args:
        file_path: Path to the event file
        tag: Name of the metric to extract

    Returns:
        Tuple of (steps, values) numpy arrays
    """
    try:
        # Create EventAccumulator and load the event file
        ea = EventAccumulator(file_path)
        ea.Reload()

        # Try to find the tag (try multiple common variations)
        available_tags = ea.Tags().get('scalars', [])

        # Try exact match first
        if tag in available_tags:
            target_tag = tag
        else:
            # Try to find a tag containing the key words
            matching_tags = [t for t in available_tags if 'average' in t.lower() and 'reward' in t.lower()]
            if matching_tags:
                target_tag = matching_tags[0]
            elif available_tags:
                # Use first available tag
                target_tag = available_tags[0]
                print(f"Warning: Tag '{tag}' not found in {file_path}, using '{target_tag}'")
            else:
                print(f"Warning: No scalar data found in {file_path}")
                return None, None

        # Extract the data
        events = ea.Scalars(target_tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])

        return steps, values

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None


def plot_results(event_files, output_file='training_results.png', smooth_window=10):
    """
    Plot all results on a single figure.

    Args:
        event_files: List of tuples (method_name, seed_num, run_num, file_path)
        output_file: Path to save the plot
        smooth_window: Window size for smoothing curves (moving average)
    """
    # Set professional style
    # Set Times New Roman font globally
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        try:
            plt.style.use('seaborn-paper')
        except:
            # Fallback to default if seaborn not available
            plt.rcParams.update({
                'font.size': 11,
                'axes.labelsize': 12,
                'axes.titlesize': 14,
                'xtick.labelsize': 11,
                'ytick.labelsize': 11,
                'legend.fontsize': 11,
                'figure.titlesize': 14
            })

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    # Define colors for different methods (publication-quality colors)
    method_colors = {
        'mad': '#2ecc71',  # Green
        'informarl': '#9b59b6',  # Violet
        'mappo': '#3498db',  # Blue
    }

    method_names = {
        'mad': 'Ours',
        'informarl': 'InforMARL',
        'mappo': 'MAPPO'
    }

    # Group data by method
    data_by_method = defaultdict(list)

    for method_name, seed_num, run_num, file_path in event_files:
        print(f"Loading: {method_name}, seed {seed_num}")
        steps, values = load_tensorboard_data(file_path)

        if steps is not None and values is not None and len(values) > 0:
            data_by_method[method_name].append({
                'seed': seed_num,
                'run': run_num,
                'steps': steps,
                'values': values,
                'file': file_path
            })

    if not data_by_method:
        print("Error: No data found to plot!")
        return

    # Find the minimum length across all runs
    min_length = float('inf')
    for method_name, runs in data_by_method.items():
        for run_data in runs:
            min_length = min(min_length, len(run_data['values']))

    print(f"\nMinimum data length found: {min_length}")
    print(f"Truncating all data to {min_length} points for consistent comparison\n")

    # Truncate all data to the minimum length
    for method_name, runs in data_by_method.items():
        for run_data in runs:
            run_data['steps'] = run_data['steps'][:min_length]
            run_data['values'] = run_data['values'][:min_length]

    # Calculate mean and std for each method
    for method_name, runs in data_by_method.items():
        if not runs:
            continue

        color = method_colors.get(method_name.lower(), None)

        # Stack all values for this method
        all_values = np.array([run_data['values'] for run_data in runs])
        steps = runs[0]['steps']  # All have same steps after truncation

        # Apply smoothing if window > 1
        if smooth_window > 1 and len(steps) >= smooth_window:
            # Smooth each run separately
            smoothed_all_values = []
            for values in all_values:
                smoothed = np.convolve(values,
                                      np.ones(smooth_window)/smooth_window,
                                      mode='valid')
                smoothed_all_values.append(smoothed)
            smoothed_all_values = np.array(smoothed_all_values)
            smoothed_steps = steps[smooth_window-1:]
        else:
            smoothed_all_values = all_values
            smoothed_steps = steps

        # Calculate mean and std across runs
        mean_values = np.mean(smoothed_all_values, axis=0)
        # Use ddof=1 for sample standard deviation (standard practice for small samples)
        std_values = np.std(smoothed_all_values, axis=0, ddof=1)

        # Get nice display name
        display_name = method_names.get(method_name.lower(), method_name)

        # Plot mean trajectory
        ax.plot(smoothed_steps, mean_values,
                label=f"{display_name}",
                color=color, linewidth=2.5, zorder=2)

        # Plot variance band (mean ± std)
        ax.fill_between(smoothed_steps,
                        mean_values - std_values,
                        mean_values + std_values,
                        color=color, alpha=0.25, linewidth=0, zorder=1)

    # Professional styling
    ax.set_xlabel('Steps', fontsize=25)
    ax.set_ylabel('Rewards', fontsize=25)

    # Legend inside plot (upper left corner)
    ax.legend(loc='upper left', fontsize=17, frameon=True,
             fancybox=True, shadow=True, framealpha=0.95)

    # Grid styling - dotted background
    ax.grid(True, alpha=0.5, linestyle=':', linewidth=1.0)

    # Add minor ticks for better readability
    ax.minorticks_on()
    ax.grid(True, which='minor', alpha=0.3, linestyle=':', linewidth=0.7)

    # Improve tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=20, length=6, width=1.5)
    ax.tick_params(axis='both', which='minor', length=3, width=1)

    # Spine styling
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Format x-axis in scientific notation if values are large
    max_step = max([run_data['steps'][-1] for runs in data_by_method.values()
                    for run_data in runs])
    if max_step > 100000:
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))

    # Tight layout with some padding
    plt.tight_layout(pad=0.5)

    # Save figure with high DPI for publication
    # Save both PNG and PDF (PDF is preferred for LaTeX papers)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nPlot saved to: {output_file}")

    # Also save as PDF for publication
    pdf_file = output_file.replace('.png', '.pdf')
    plt.savefig(pdf_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"PDF version saved to: {pdf_file}")

    # Show plot
    plt.show()


def main():
    """Main function to run the plotting script."""
    # Define base directories to search
    base_dirs = ['results_mad', 'results_informarl']

    print("Searching for TensorBoard event files...")
    event_files = find_event_files(base_dirs)

    if not event_files:
        print("No event files found! Please check the directory structure.")
        print(f"Searched in: {base_dirs}")
        return

    print(f"\nFound {len(event_files)} event file(s):")
    for method, seed, run, path in event_files:
        print(f"  - {method}, seed {seed}, run {run}: {path}")

    print("\nGenerating plot...")
    plot_results(event_files, smooth_window=10)


if __name__ == '__main__':
    main()
