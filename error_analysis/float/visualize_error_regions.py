#!/usr/bin/env python3
"""
Visualize 10D input space to identify regions with high error (low precise digits)

This script loads the precise digits data and raw input parameters to visualize
which regions of the input space (4 masses + 6 momenta) lead to high error.

Usage:
    python visualize_error_regions.py --integral BIN0 --precision 24bit_a100
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from collections import defaultdict
import argparse

# Import the csv parser from the current directory
from csv_parser import QCDLoopCSVParser

def load_data():
    """Load all precision data and raw input data"""
    data_file = 'precision_analysis_data/all_precision_data.pkl'
    
    print("Loading precision data from pickle...")
    with open(data_file, 'rb') as f:
        all_data = pickle.load(f)
    
    print(f"Found {len(all_data['precise_digits'])} integrals")
    
    # Load raw input data to get the actual input values
    parser = QCDLoopCSVParser()
    raw_dir = 'raw'
    
    # Get baseline file to extract input parameters
    baseline_path = os.path.join(raw_dir, 'box_cpu_fp128.csv')
    print("Loading baseline inputs...")
    baseline_data = parser.parse_box_run_0(baseline_path)
    
    return all_data, baseline_data

def create_input_error_dataset(all_data, baseline_data, integral_type='BIN0', 
                              precision='24bit_a100', coeff='Coeff1', part='real'):
    """
    Create a dataset matching inputs to their error values for a specific integral
    
    Args:
        integral_type: Which integral (e.g., 'BIN0')
        precision: Which precision level (e.g., '24bit_a100')
        coeff: Which coefficient ('Coeff1', 'Coeff2', 'Coeff3')
        part: 'real' or 'imag'
    
    Returns:
        inputs: (N, 10) array of [ms[0], ms[1], ms[2], ms[3], ps[0], ps[1], ps[2], ps[3], ps[4], ps[5]]
        precise_digits: (N,) array of precise digit values
    """
    print(f"\nCreating dataset for:")
    print(f"  Integral: {integral_type}")
    print(f"  Precision: {precision}")
    print(f"  Coefficient: {coeff} ({part})")
    
    # Check if integral exists
    if integral_type not in all_data['precise_digits']:
        raise ValueError(f"Integral {integral_type} not found in data")
    
    # Check if precision exists
    if precision not in all_data['precise_digits'][integral_type]:
        available = list(all_data['precise_digits'][integral_type].keys())
        raise ValueError(f"Precision {precision} not found for {integral_type}. Available: {available}")
    
    # Get precise digits
    try:
        precise_digits = all_data['precise_digits'][integral_type][precision][coeff][part]
    except KeyError as e:
        raise ValueError(f"Could not access data: {e}")
    
    print(f"  Found {len(precise_digits)} data points")
    
    # Group baseline by integral
    baseline_rows = []
    for row in baseline_data:
        if row.target_integral == integral_type:
            baseline_rows.append(row)
    
    print(f"  Found {len(baseline_rows)} baseline rows")
    
    # Match up inputs with error values
    all_inputs = []
    matched_digits = []
    
    for i, row in enumerate(baseline_rows):
        if i < len(precise_digits):
            # Input: [ms[0], ms[1], ms[2], ms[3], ps[0], ps[1], ps[2], ps[3], ps[4], ps[5]]
            input_vector = list(row.ms) + list(row.ps)
            all_inputs.append(input_vector)
            matched_digits.append(precise_digits[i])
    
    inputs = np.array(all_inputs)
    precise_digits_array = np.array(matched_digits)
    
    print(f"  Matched {len(inputs)} input-output pairs")
    
    return inputs, precise_digits_array




def plot_pairwise_scatter_matrix(inputs, precise_digits, integral_type='BIN0',
                                 precision='24bit_a100', output_dir='.'):
    """
    Create pairwise scatter plot matrix

    Args:
        inputs: (N, 10) array
        precise_digits: (N,) array
        integral_type: Name of integral
        precision: Precision level
        output_dir: Where to save plots
    """
    print(f"\nGenerating pairwise scatter matrix for {integral_type}...")
    print("  Using only momentum dimensions (p0-p5, indices 4-9)")

    # Extract only momentum values (p0-p5, indices 4-9)
    momentum_inputs = inputs[:, 4:10]
    n_dims = 6
    # Feature names: only 6 momenta
    feature_names = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5']

    # Plot all points (no subsampling)
    N = len(momentum_inputs)
    print(f"  Plotting all {N} points...")

    # Color range logic
    data_min, data_max = float(precise_digits.min()), float(precise_digits.max())
    vmin = 12.0
    vmax = min(15.0, data_max)
    print(f"  Color range: {vmin:.1f} to {vmax:.1f} precise digits (everything <12 gets same color)")

    # Clip values below vmin so they map to the same color
    color_values = np.clip(precise_digits, vmin, vmax)

    # 6x6 figure and use constrained_layout to reduce manual fiddling
    fig, axes = plt.subplots(n_dims, n_dims, figsize=(18, 18), constrained_layout=True)

    cmap = plt.get_cmap('plasma')
    norm = Normalize(vmin=vmin, vmax=vmax)

    # scatter settings tuned for many points
    scatter_s = 1.0
    scatter_alpha = 0.8

    for i in range(n_dims):
        for j in range(n_dims):
            ax = axes[i, j]

            if i == j:
                # Diagonal: histogram
                ax.hist(momentum_inputs[:, i], bins=30, color='steelblue', alpha=0.6, edgecolor='black')
                ax.set_xlabel(feature_names[i], fontsize=10)
                ax.set_ylabel('Count', fontsize=9)
            else:
                # Off-diagonal: scatter colored by clipped precise_digits
                sc = ax.scatter(momentum_inputs[:, j], momentum_inputs[:, i],
                                c=color_values, cmap=cmap, norm=norm,
                                s=scatter_s, alpha=scatter_alpha, marker='.', linewidths=0)

                # Only label outer axes to avoid overlap
                if i < n_dims - 1:
                    ax.set_xlabel('')
                    ax.set_xticks([])
                else:
                    ax.set_xlabel(feature_names[j], fontsize=10)

                if j > 0:
                    ax.set_ylabel('')
                    ax.set_yticks([])
                else:
                    ax.set_ylabel(feature_names[i], fontsize=10)

            # Style
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.12, linestyle='--', linewidth=0.4)

    # Add a single colorbar to the right of all subplots, referencing the last scatter (sc)
    # Use the axes list so constrained_layout reserves space correctly.
    cb = fig.colorbar(sc, ax=axes.ravel().tolist(), orientation='vertical',
                      fraction=0.02, pad=0.02)
    cb.set_label('Precise Digits', rotation=270, labelpad=18, fontsize=11)
    cb.ax.tick_params(labelsize=9)

    # Title above the figure (constrained_layout handles spacing)
    fig.suptitle(f'Pairwise Scatter Matrix (Momenta Only) - {integral_type} ({precision})', fontsize=16, fontweight='bold')

    # Save with tight bounding box so nothing gets clipped
    filename = f'error_regions_pairwise_{integral_type}_{precision}.png'
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight', pad_inches=0.3)
    print(f"  Saved: {filepath}")
    plt.close(fig)


def plot_pairwise_scatter_matrix_highlight(inputs, precise_digits, integral_type='BIN0',
                                           precision='24bit_a100', threshold=11, output_dir='.'):
    """
    Create pairwise scatter plot matrix with binary highlighting (low vs high precision)
    
    Args:
        threshold: Threshold for precise digits (below = error, above = good)
    """
    print(f"\nGenerating pairwise scatter matrix (highlight mode) for {integral_type}...")
    print(f"  Threshold: {threshold} precise digits")
    
    n_dims = inputs.shape[1]
    feature_names = ['m0', 'm1', 'm2', 'm3', 'p0', 'p1', 'p2', 'p3', 'p4', 'p5']
    
    # No subsampling - plot all points
    print(f"  Plotting all {len(inputs)} points...")
    
    # Binary classification: low vs high precision
    high_error_mask = precise_digits < threshold
    
    # Stats
    n_low = np.sum(high_error_mask)
    n_high = np.sum(~high_error_mask)
    print(f"  Low precision (<{threshold}): {n_low} points ({100*n_low/len(inputs):.1f}%)")
    print(f"  High precision (≥{threshold}): {n_high} points ({100*n_high/len(inputs):.1f}%)")
    
    # Create subplot matrix
    fig, axes = plt.subplots(n_dims, n_dims, figsize=(20, 20))
    
    for i in range(n_dims):
        for j in range(n_dims):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histogram showing distribution
                ax.hist(inputs[:, i], bins=50, color='blue', alpha=0.5, edgecolor='black')
                ax.set_xlabel(feature_names[i], fontsize=10)
                ax.set_ylabel('Count', fontsize=8)
            else:
                # Off-diagonal: scatter plot
                # First plot all points in gray with low opacity
                ax.scatter(inputs[:, j], inputs[:, i], 
                          c='gray', s=0.5, alpha=0.05, label='All points')
                
                # Then overlay error points in red
                mask_low = high_error_mask
                if np.any(mask_low):
                    ax.scatter(inputs[mask_low, j], inputs[mask_low, i], 
                              c='red', s=2, alpha=0.9, label='Low precision' if i==0 and j==1 else '')
                
                ax.set_xlabel(feature_names[j], fontsize=9)
                ax.set_ylabel(feature_names[i], fontsize=9)
            
            # Style
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    plt.suptitle(f'Pairwise Scatter Matrix - {integral_type} ({precision})\n'
                f'Red: <{threshold} precise digits, Gray: all points', 
                fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    # Save
    filename = f'error_regions_pairwise_highlight_{integral_type}_{precision}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close()

def plot_per_dimension_analysis(inputs, precise_digits, integral_type='BIN0',
                                 precision='24bit_a100', threshold=11, output_dir='.'):
    """Plot histogram of inputs per dimension, highlighting high error regions"""
    print(f"\nGenerating per-dimension analysis for {integral_type}...")
    print(f"  Threshold: {threshold} precise digits")
    
    n_dims = inputs.shape[1]
    feature_names = ['m0', 'm1', 'm2', 'm3', 'p0', 'p1', 'p2', 'p3', 'p4', 'p5']
    
    # Binary classification: low vs high precision
    high_error_mask = precise_digits < threshold
    n_low = np.sum(high_error_mask)
    n_high = np.sum(~high_error_mask)
    print(f"  Low precision (<{threshold}): {n_low} points ({100*n_low/len(inputs):.1f}%)")
    print(f"  High precision (≥{threshold}): {n_high} points ({100*n_high/len(inputs):.1f}%)")
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for dim in range(n_dims):
        ax = axes[dim]
        
        # All data histogram
        ax.hist(inputs[:, dim], bins=50, alpha=0.5, color='gray', 
               label='All points', density=True, edgecolor='black')
        
        # High error regions histogram
        if np.any(high_error_mask):
            ax.hist(inputs[high_error_mask, dim], bins=50, alpha=0.8, 
                   color='red', label=f'Low precision (n={n_low})', 
                   density=True, edgecolor='darkred', linewidth=1.5)
        
        ax.set_xlabel(feature_names[dim], fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.suptitle(f'Input Distribution by Dimension - {integral_type} ({precision})\n'
                f'Red regions indicate high error (precise digits < {threshold})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f'error_regions_per_dimension_{integral_type}_{precision}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close()

def plot_umap_reduction(inputs, precise_digits, integral_type='BIN0',
                        precision='24bit_a100', threshold=11, output_dir='.',
                        use_momenta_only=False, custom_dims=None, **kwargs):
    """
    Use UMAP to reduce dimensionality and visualize error regions
    
    Args:
        inputs: (N, 10) array of inputs
        precise_digits: (N,) array of precise digits
        use_momenta_only: If True, use only the 6 momentum dimensions (indices 4-9)
    """
    try:
        import umap
    except ImportError:
        print("Error: umap-learn not installed. Install with: pip install umap-learn")
        return
    
    print(f"\nGenerating UMAP visualization for {integral_type}...")
    print(f"  Threshold: {threshold} precise digits")
    
    # Select which dimensions to use
    if use_momenta_only:
        print("  Using only momentum dimensions (p0-p5, indices 4-9)")
        input_data = inputs[:, 4:10]  # Only momenta
        suffix = "momenta"
    else:
        print("  Using all 10 dimensions")
        input_data = inputs
        suffix = "all_dims"
    
    # Allow custom dimension selection
    # If custom dimensions are specified, use those instead
    if custom_dims:
        dim_indices = [int(x.strip()) for x in custom_dims.split(',')]
        input_data = inputs[:, dim_indices]
        dim_names = ['m0', 'm1', 'm2', 'm3', 'p0', 'p1', 'p2', 'p3', 'p4', 'p5']
        selected_dims = [dim_names[i] for i in dim_indices]
        print(f"  Using custom dimensions: {selected_dims} (indices {dim_indices})")
        suffix = f"custom_{'_'.join(selected_dims)}"
    
    print(f"  Using all {len(input_data)} points (no subsampling)")
    
    # Update title based on what we're actually using
    title_suffix = f"Using: {', '.join(selected_dims)}" if custom_dims else ("Momenta only" if use_momenta_only else "All 10 dimensions")
    
    # Standardize the data
    input_data_normalized = (input_data - input_data.mean(axis=0)) / (input_data.std(axis=0) + 1e-10)
    
    # Fit UMAP
    print("  Fitting UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
    embedding = reducer.fit_transform(input_data_normalized)
    
    # Binary classification
    high_error_mask = precise_digits < threshold
    n_low = np.sum(high_error_mask)
    n_high = np.sum(~high_error_mask)
    print(f"  Low precision (<{threshold}): {n_low} points ({100*n_low/len(precise_digits):.1f}%)")
    print(f"  High precision (≥{threshold}): {n_high} points ({100*n_high/len(precise_digits):.1f}%)")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Continuous color scale
    scatter1 = ax1.scatter(embedding[:, 0], embedding[:, 1], 
                          c=precise_digits, s=1, alpha=0.3, 
                          cmap='RdYlGn')
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Precise Digits', rotation=270, labelpad=20, fontsize=12)
    ax1.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax1.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax1.set_title('UMAP Embedding - Continuous Scale', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Binary highlighting
    # Plot high precision first (gray background)
    mask_high = ~high_error_mask
    ax2.scatter(embedding[mask_high, 0], embedding[mask_high, 1], 
               c='gray', s=1, alpha=0.05, label=f'High precision (n={n_high})')
    
    # Plot low precision on top (red, more visible)
    if np.any(high_error_mask):
        mask_low = high_error_mask
        ax2.scatter(embedding[mask_low, 0], embedding[mask_low, 1], 
                   c='red', s=5, alpha=0.9, label=f'Low precision (n={n_low})')
    
    ax2.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax2.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax2.set_title('UMAP Embedding - Binary Highlight', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'UMAP Visualization - {integral_type} ({precision})\n{title_suffix}', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    filename = f'error_regions_umap_{suffix}_{integral_type}_{precision}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize error regions in 10D input space')
    parser.add_argument('--integral', type=str, default='BIN0', 
                       help='Integral type (e.g., BIN0, B1, etc.)')
    parser.add_argument('--precision', type=str, default='24bit_a100',
                       help='Precision level (e.g., 24bit_a100, 24bit_cpu, 53bit_cpu)')
    parser.add_argument('--coeff', type=str, default='Coeff1',
                       choices=['Coeff1', 'Coeff2', 'Coeff3'],
                       help='Which coefficient')
    parser.add_argument('--part', type=str, default='real',
                       choices=['real', 'imag'],
                       help='Real or imaginary part')
    parser.add_argument('--threshold', type=float, default=11,
                       help='Threshold for low vs high precision')
    parser.add_argument('--skip-pairwise', action='store_true',
                       help='Skip pairwise scatter matrix plots')
    parser.add_argument('--per-dimension-only', action='store_true',
                       help='Generate only per-dimension analysis')
    parser.add_argument('--skip-per-dimension', action='store_true',
                       help='Skip per-dimension analysis plots')
    parser.add_argument('--all-integrals', action='store_true',
                       help='Generate plots for all integrals (B1-B16 and BIN0-BIN4)')
    parser.add_argument('--umap', action='store_true',
                       help='Generate UMAP dimensionality reduction plots')
    parser.add_argument('--momenta-only', action='store_true',
                       help='Use only momentum dimensions for UMAP (indices 4-9)')
    parser.add_argument('--custom-dims', type=str, default=None,
                       help='Custom dimension indices to use (comma-separated, e.g., "4,5,7,8" for p1,p2,p4,p5)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("10D Input Space Error Region Visualization")
    print("=" * 70)
    
    # Load data once
    all_data, baseline_data = load_data()
    
    # If --all-integrals flag, process all integrals
    if args.all_integrals:
        # Get all available integrals from the data
        all_available_integrals = sorted(all_data['precise_digits'].keys())
        
        # Filter for B1-B16 and BIN0-BIN4
        target_integrals = []
        for integral in all_available_integrals:
            if integral.startswith('B') and len(integral) <= 3:  # B1-B16
                target_integrals.append(integral)
            elif integral.startswith('BIN'):  # BIN0-BIN4
                target_integrals.append(integral)
        
        print(f"Processing {len(target_integrals)} integrals: {target_integrals}\n")
        
        # Create output directory
        output_dir = 'error_region_analysis'
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each integral
        for integral in target_integrals:
            print("\n" + "=" * 70)
            print(f"Processing {integral}...")
            print("=" * 70)
            
            # Create combined dataset for this integral
            try:
                inputs, precise_digits = create_input_error_dataset(
                    all_data, baseline_data, 
                    integral_type=integral,
                    precision=args.precision,
                    coeff=args.coeff,
                    part=args.part
                )
                
                # Generate per-dimension analysis (unless skipped)
                if not args.skip_per_dimension:
                    plot_per_dimension_analysis(inputs, precise_digits,
                                                integral_type=integral,
                                                precision=args.precision,
                                                threshold=args.threshold,
                                                output_dir=output_dir)
                
                # Generate pairwise scatter matrix if not skipped
                if not args.skip_pairwise:
                    plot_pairwise_scatter_matrix(inputs, precise_digits,
                                                integral_type=integral,
                                                precision=args.precision,
                                                output_dir=output_dir)
                
            except ValueError as e:
                print(f"  Error processing {integral}: {e}")
                continue
        
        print("\n" + "=" * 70)
        print("Batch processing complete!")
        print("=" * 70)
        print(f"\nPlots saved to: {output_dir}/")
        return
    
    # Single integral processing (original behavior)
    # Create combined dataset
    try:
        inputs, precise_digits = create_input_error_dataset(
            all_data, baseline_data, 
            integral_type=args.integral,
            precision=args.precision,
            coeff=args.coeff,
            part=args.part
        )
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    print(f"\nDataset summary:")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Precise digits range: [{precise_digits.min():.2f}, {precise_digits.max():.2f}]")
    print(f"  Mean precise digits: {precise_digits.mean():.2f}")
    print(f"  Median precise digits: {np.median(precise_digits):.2f}")
    print(f"  Low precision (<{args.threshold}): {np.sum(precise_digits < args.threshold)} "
          f"({100*np.sum(precise_digits < args.threshold)/len(precise_digits):.1f}%)")
    
    # Create output directory
    output_dir = 'error_region_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    print("\n" + "=" * 70)
    print("Generating visualizations...")
    print("=" * 70)
    
    generated_plots = []
    
    if not args.per_dimension_only and not args.skip_pairwise:
        # Pairwise scatter matrix with continuous color scale
        plot_pairwise_scatter_matrix(inputs, precise_digits, 
                                    integral_type=args.integral,
                                    precision=args.precision,
                                    output_dir=output_dir)
        generated_plots.append("Pairwise scatter matrix (continuous color)")
    
    # Generate per-dimension analysis (unless skipped)
    if not args.skip_per_dimension:
        plot_per_dimension_analysis(inputs, precise_digits,
                                    integral_type=args.integral,
                                    precision=args.precision,
                                    threshold=args.threshold,
                                    output_dir=output_dir)
        generated_plots.append("Per-dimension analysis (histogram for each input variable)")
    
    # Generate UMAP visualization if requested
    if args.umap:
        plot_umap_reduction(inputs, precise_digits,
                            integral_type=args.integral,
                            precision=args.precision,
                            threshold=args.threshold,
                            output_dir=output_dir,
                            use_momenta_only=args.momenta_only,
                            custom_dims=args.custom_dims)
        if args.custom_dims:
            plot_type = f"custom dimensions {args.custom_dims}"
        elif args.momenta_only:
            plot_type = "momenta only"
        else:
            plot_type = "all dimensions"
        generated_plots.append(f"UMAP visualization ({plot_type})")
    
    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70)
    print(f"\nPlots saved to: {output_dir}/")
    print("\nGenerated plots:")
    for i, plot_name in enumerate(generated_plots, 1):
        print(f"  {i}. {plot_name}")

if __name__ == "__main__":
    main()

