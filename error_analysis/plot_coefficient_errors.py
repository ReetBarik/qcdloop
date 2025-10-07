#!/usr/bin/env python3
"""
Script to plot coefficient differences between fp64 implementations and quad precision baseline
Creates 3x3 grid plots for each target integral with rows for implementations and columns for coefficients
"""

from csv_parser import QCDLoopCSVParser
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def plot_coefficient_errors():
    parser = QCDLoopCSVParser()
    
    print("=== Coefficient Error Box Plot Analysis ===")
    print("Baseline: box_cpu_fp128.csv (quad precision)")
    print("Comparing against: box_cpu_fp64.csv, box_a100_fp64.csv, box_mi250_fp64.csv\n")
    
    # Create output directory
    import os
    output_dir = 'coeff_difference_plots'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}\n")
    
    # Parse all files
    print("Loading data...")
    baseline_data = parser.parse_box_run_0('box_cpu_fp128.csv')
    cpu_fp64_data = parser.parse_box_run_1('box_cpu_fp64.csv')
    a100_fp64_data = parser.parse_box_run_1('box_a100_fp64.csv')
    mi250_fp64_data = parser.parse_box_run_1('box_mi250_fp64.csv')
    
    datasets = {
        'CPU FP64': cpu_fp64_data,
        'A100 FP64': a100_fp64_data,
        'MI250 FP64': mi250_fp64_data
    }
    
    print(f"Loaded {len(baseline_data)} rows from baseline (quad precision)")
    for name, data in datasets.items():
        print(f"Loaded {len(data)} rows from {name}")
    print()
    
    # Group data by target integral
    baseline_by_integral = defaultdict(list)
    cpu_by_integral = defaultdict(list)
    a100_by_integral = defaultdict(list)
    mi250_by_integral = defaultdict(list)
    
    for i, (baseline_row, cpu_row, a100_row, mi250_row) in enumerate(zip(baseline_data, cpu_fp64_data, a100_fp64_data, mi250_fp64_data)):
        integral_type = baseline_row.target_integral
        baseline_by_integral[integral_type].append(baseline_row)
        cpu_by_integral[integral_type].append(cpu_row)
        a100_by_integral[integral_type].append(a100_row)
        mi250_by_integral[integral_type].append(mi250_row)
    
    print(f"Found {len(baseline_by_integral)} different target integrals:")
    for integral_type in sorted(baseline_by_integral.keys()):
        print(f"  {integral_type}: {len(baseline_by_integral[integral_type])} rows")
    
    print("\nGenerating 3x3 grid plots...")
    
    # Create plots for each target integral
    for integral_type in sorted(baseline_by_integral.keys()):
        baseline_rows = baseline_by_integral[integral_type]
        cpu_rows = cpu_by_integral[integral_type]
        a100_rows = a100_by_integral[integral_type]
        mi250_rows = mi250_by_integral[integral_type]
        
        # Check if all datasets have the same number of rows
        min_rows = min(len(baseline_rows), len(cpu_rows), len(a100_rows), len(mi250_rows))
        if len(baseline_rows) != min_rows or len(cpu_rows) != min_rows or len(a100_rows) != min_rows or len(mi250_rows) != min_rows:
            print(f"Warning: Mismatch in row counts for {integral_type}")
            continue
        
        # Create the 3x3 grid plot
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f'Coefficient Error Analysis - {integral_type}', fontsize=16, fontweight='bold')
        
        # Define implementation names and colors
        implementations = ['CPU FP64', 'A100 FP64', 'MI250 FP64']
        implementation_rows = [cpu_rows, a100_rows, mi250_rows]
        row_colors = ['lightblue', 'lightgreen', 'lightcoral']
        
        # Define coefficient names
        coefficients = ['Coeff1', 'Coeff2', 'Coeff3']
        
        # Create plots for each implementation (row) and coefficient (column)
        for row_idx, (impl_name, impl_rows, row_color) in enumerate(zip(implementations, implementation_rows, row_colors)):
            for col_idx, coeff_name in enumerate(coefficients):
                ax = axes[row_idx, col_idx]
                
                # Calculate differences for this implementation and coefficient
                real_diffs = []
                imag_diffs = []
                
                for baseline_row, impl_row in zip(baseline_rows, impl_rows):
                    if coeff_name == 'Coeff1':
                        real_diffs.append(abs(impl_row.coeff1_real - baseline_row.coeff1_real))
                        imag_diffs.append(abs(impl_row.coeff1_imag - baseline_row.coeff1_imag))
                    elif coeff_name == 'Coeff2':
                        real_diffs.append(abs(impl_row.coeff2_real - baseline_row.coeff2_real))
                        imag_diffs.append(abs(impl_row.coeff2_imag - baseline_row.coeff2_imag))
                    elif coeff_name == 'Coeff3':
                        real_diffs.append(abs(impl_row.coeff3_real - baseline_row.coeff3_real))
                        imag_diffs.append(abs(impl_row.coeff3_imag - baseline_row.coeff3_imag))
                
                # Create box plot
                coeff_data = [real_diffs, imag_diffs]
                bp = ax.boxplot(coeff_data, labels=['Real', 'Imaginary'], patch_artist=True)
                
                # Set title and labels
                ax.set_title(f'{impl_name} - {coeff_name}', fontsize=12, fontweight='bold')
                if row_idx == 2:  # Bottom row
                    ax.set_xlabel('Component', fontsize=10)
                if col_idx == 0:  # Left column
                    ax.set_ylabel('|Difference|', fontsize=10)
                
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3)
                
                # Color the boxes
                bp['boxes'][0].set_facecolor('lightblue')
                bp['boxes'][1].set_facecolor('lightcoral')
                
                # Add statistics text
                real_mean = np.mean(real_diffs)
                imag_mean = np.mean(imag_diffs)
                real_max = np.max(real_diffs)
                imag_max = np.max(imag_diffs)
                
                ax.text(0.02, 0.98, f'Real Mean: {real_mean:.2e}\nReal Max: {real_max:.2e}\nImag Mean: {imag_mean:.2e}\nImag Max: {imag_max:.2e}', 
                        transform=ax.transAxes, verticalalignment='top', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save the plot
        filename = f'coeff_errors_{integral_type.lower()}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {filepath}")
        
        # Close the figure to free memory
        plt.close(fig)
    
    print(f"\nGenerated {len(baseline_by_integral)} 3x3 grid plot figures total.")
    print("All plots saved as PNG files with 300 DPI resolution.")

def create_summary_plot():
    """Create a summary plot showing mean differences across all implementations and integrals"""
    parser = QCDLoopCSVParser()
    
    print("\n=== Creating Summary Plot ===\n")
    
    # Create output directory
    import os
    output_dir = 'coeff_difference_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse all files
    baseline_data = parser.parse_box_run_0('box_cpu_fp128.csv')
    cpu_fp64_data = parser.parse_box_run_1('box_cpu_fp64.csv')
    a100_fp64_data = parser.parse_box_run_1('box_a100_fp64.csv')
    mi250_fp64_data = parser.parse_box_run_1('box_mi250_fp64.csv')
    
    datasets = {
        'CPU FP64': cpu_fp64_data,
        'A100 FP64': a100_fp64_data,
        'MI250 FP64': mi250_fp64_data
    }
    
    # Group data by target integral
    baseline_by_integral = defaultdict(list)
    cpu_by_integral = defaultdict(list)
    a100_by_integral = defaultdict(list)
    mi250_by_integral = defaultdict(list)
    
    for i, (baseline_row, cpu_row, a100_row, mi250_row) in enumerate(zip(baseline_data, cpu_fp64_data, a100_fp64_data, mi250_fp64_data)):
        integral_type = baseline_row.target_integral
        baseline_by_integral[integral_type].append(baseline_row)
        cpu_by_integral[integral_type].append(cpu_row)
        a100_by_integral[integral_type].append(a100_row)
        mi250_by_integral[integral_type].append(mi250_row)
    
    # Prepare data for summary plot
    all_integrals = sorted(baseline_by_integral.keys())
    implementations = ['CPU FP64', 'A100 FP64', 'MI250 FP64']
    implementation_data = [cpu_by_integral, a100_by_integral, mi250_by_integral]
    
    # Pre-calculate all mean differences to determine global y-axis range
    all_real_means = []
    all_imag_means = []
    mean_data = {}  # Store calculated means for plotting
    
    for impl_idx, (impl_name, impl_data) in enumerate(zip(implementations, implementation_data)):
        mean_data[impl_name] = {}
        for coeff_idx, coeff_name in enumerate(['Coeff1', 'Coeff2', 'Coeff3']):
            mean_data[impl_name][coeff_name] = {'real': [], 'imag': []}
            
            for integral_type in all_integrals:
                baseline_rows = baseline_by_integral[integral_type]
                impl_rows = impl_data[integral_type]
                
                if len(baseline_rows) != len(impl_rows):
                    continue
                
                # Calculate differences for this integral
                real_diffs = []
                imag_diffs = []
                
                for baseline_row, impl_row in zip(baseline_rows, impl_rows):
                    if coeff_name == 'Coeff1':
                        real_diffs.append(abs(impl_row.coeff1_real - baseline_row.coeff1_real))
                        imag_diffs.append(abs(impl_row.coeff1_imag - baseline_row.coeff1_imag))
                    elif coeff_name == 'Coeff2':
                        real_diffs.append(abs(impl_row.coeff2_real - baseline_row.coeff2_real))
                        imag_diffs.append(abs(impl_row.coeff2_imag - baseline_row.coeff2_imag))
                    elif coeff_name == 'Coeff3':
                        real_diffs.append(abs(impl_row.coeff3_real - baseline_row.coeff3_real))
                        imag_diffs.append(abs(impl_row.coeff3_imag - baseline_row.coeff3_imag))
                
                real_mean = np.mean(real_diffs)
                imag_mean = np.mean(imag_diffs)
                
                mean_data[impl_name][coeff_name]['real'].append(real_mean)
                mean_data[impl_name][coeff_name]['imag'].append(imag_mean)
                
                all_real_means.append(real_mean)
                all_imag_means.append(imag_mean)
    
    # Calculate global y-axis range
    all_means = all_real_means + all_imag_means
    all_means = [x for x in all_means if x > 0]  # Filter out zeros for log scale
    if all_means:
        y_min = min(all_means) * 0.5  # Add some margin
        y_max = max(all_means) * 2.0  # Add some margin
    else:
        y_min, y_max = 1e-20, 1e10  # Default range if no data
    
    # Create summary plot with 3x3 grid (implementations x coefficients)
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Mean Coefficient Differences Across All Target Integrals', fontsize=16, fontweight='bold')
    
    for impl_idx, (impl_name, impl_data) in enumerate(zip(implementations, implementation_data)):
        for coeff_idx, coeff_name in enumerate(['Coeff1', 'Coeff2', 'Coeff3']):
            ax = axes[impl_idx, coeff_idx]
            
            # Get pre-calculated means
            real_means = mean_data[impl_name][coeff_name]['real']
            imag_means = mean_data[impl_name][coeff_name]['imag']
            
            # Create bar plot
            x_pos = np.arange(len(all_integrals))
            width = 0.35
            
            ax.bar(x_pos - width/2, real_means, width, label='Real', alpha=0.8, color='lightblue')
            ax.bar(x_pos + width/2, imag_means, width, label='Imaginary', alpha=0.8, color='lightcoral')
            
            ax.set_title(f'{impl_name} - {coeff_name}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Mean |Difference|', fontsize=10)
            ax.set_yscale('log')
            ax.set_ylim(y_min, y_max)  # Set consistent y-axis range
            ax.set_xticks(x_pos)
            ax.set_xticklabels(all_integrals, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add labels only to outer subplots
            if impl_idx == 2:  # Bottom row
                ax.set_xlabel('Target Integral', fontsize=10)
            if coeff_idx == 0:  # Left column
                ax.set_ylabel('Mean |Difference|', fontsize=10)
    
    plt.tight_layout()
    summary_filepath = os.path.join(output_dir, 'coeff_errors_summary.png')
    plt.savefig(summary_filepath, dpi=300, bbox_inches='tight')
    print(f"Saved summary plot: {summary_filepath}")
    plt.close(fig)

def main():
    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Error: matplotlib and numpy are required for plotting.")
        print("Install them with: pip install matplotlib numpy")
        return
    
    # Generate individual 3x3 grid plots for each target integral
    plot_coefficient_errors()
    
    # Generate summary plot
    create_summary_plot()
    
    print("\n=== Plotting Complete ===")
    print("All plots have been generated and saved as PNG files.")
    print("Each target integral has its own 3x3 grid plot:")
    print("  - Rows: CPU FP64, A100 FP64, MI250 FP64")
    print("  - Columns: Coeff1, Coeff2, Coeff3")
    print("  - Each subplot shows box plots for real and imaginary part differences.")
    print("Summary plot shows mean differences across all target integrals.")

if __name__ == "__main__":
    main()