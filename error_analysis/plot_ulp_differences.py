#!/usr/bin/env python3
"""
Script to plot ULP differences between fp64 implementations and quad precision baseline
Creates 3x3 grid plots for each target integral with rows for implementations and columns for coefficients

Workflow:
1. First run: Generates ULP data from CSV files and saves to ulp_data/ulp_data.pkl
2. Subsequent runs: Loads pre-generated data from pickle file for fast plotting
3. To regenerate data: Delete ulp_data/ directory or call regenerate_data() function

This separation allows for fast iteration on plotting logic without regenerating data each time.
"""

from csv_parser import QCDLoopCSVParser
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import struct
import math
import json
import pickle

def ulp_double(x):
    """Calculate ULP for double precision float"""
    if x == 0.0:
        return 2.2250738585072014e-308  # Smallest positive double
    if math.isnan(x) or math.isinf(x):
        return float('nan')
    
    # Get the bit representation
    packed = struct.pack('d', x)
    bits = struct.unpack('Q', packed)[0]
    
    # Extract exponent
    exponent = (bits >> 52) & 0x7FF
    if exponent == 0:
        # Denormalized number
        return 2.2250738585072014e-308
    else:
        # Normalized number
        return 2.0 ** (exponent - 1023 - 52)

def generate_ulp_data():
    """Generate ULP data and save to text files"""
    parser = QCDLoopCSVParser()
    
    print("=== Generating ULP Data ===")
    print("Baseline: box_cpu_fp128.csv (quad precision)")
    print("Comparing against: box_cpu_fp64.csv, box_a100_fp64.csv, box_mi250_fp64.csv\n")
    
    # Create output directory
    import os
    data_dir = 'ulp_data'
    os.makedirs(data_dir, exist_ok=True)
    print(f"Created data directory: {data_dir}\n")
    
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
    
    print("\nGenerating ULP data for each target integral...")
    
    # Generate ULP data for each target integral
    ulp_data = {}
    
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
        
        # Store ULP data for this integral
        integral_ulp_data = {
            'CPU FP64': {'Coeff1': {'real': [], 'imag': []}, 'Coeff2': {'real': [], 'imag': []}, 'Coeff3': {'real': [], 'imag': []}},
            'A100 FP64': {'Coeff1': {'real': [], 'imag': []}, 'Coeff2': {'real': [], 'imag': []}, 'Coeff3': {'real': [], 'imag': []}},
            'MI250 FP64': {'Coeff1': {'real': [], 'imag': []}, 'Coeff2': {'real': [], 'imag': []}, 'Coeff3': {'real': [], 'imag': []}}
        }
        
        implementation_rows = [cpu_rows, a100_rows, mi250_rows]
        implementation_names = ['CPU FP64', 'A100 FP64', 'MI250 FP64']
        
        for impl_name, impl_rows in zip(implementation_names, implementation_rows):
            for coeff_name in ['Coeff1', 'Coeff2', 'Coeff3']:
                real_ulps = []
                imag_ulps = []
                
                for baseline_row, impl_row in zip(baseline_rows, impl_rows):
                    # Calculate absolute differences with ULP thresholding
                    # Set to zero if FP64 ULP > |Baseline| to prevent ULP explosion near zero
                    fp64_ulp_c_real = ulp_double(impl_row.coeff1_real if coeff_name == 'Coeff1' else impl_row.coeff2_real if coeff_name == 'Coeff2' else impl_row.coeff3_real)
                    fp64_ulp_c_imag = ulp_double(impl_row.coeff1_imag if coeff_name == 'Coeff1' else impl_row.coeff2_imag if coeff_name == 'Coeff2' else impl_row.coeff3_imag)
                    
                    baseline_real = baseline_row.coeff1_real if coeff_name == 'Coeff1' else baseline_row.coeff2_real if coeff_name == 'Coeff2' else baseline_row.coeff3_real
                    baseline_imag = baseline_row.coeff1_imag if coeff_name == 'Coeff1' else baseline_row.coeff2_imag if coeff_name == 'Coeff2' else baseline_row.coeff3_imag
                    
                    impl_real = impl_row.coeff1_real if coeff_name == 'Coeff1' else impl_row.coeff2_real if coeff_name == 'Coeff2' else impl_row.coeff3_real
                    impl_imag = impl_row.coeff1_imag if coeff_name == 'Coeff1' else impl_row.coeff2_imag if coeff_name == 'Coeff2' else impl_row.coeff3_imag
                    
                    real_abs = 0.0 if fp64_ulp_c_real > abs(baseline_real) else abs(impl_real - baseline_real)
                    imag_abs = 0.0 if fp64_ulp_c_imag > abs(baseline_imag) else abs(impl_imag - baseline_imag)
                    
                    # Calculate ULP differences using pre-calculated ULP values
                    # If abs_diff is 0 (due to thresholding), ULP is also 0
                    real_ulp = real_abs / fp64_ulp_c_real if impl_real != 0 and fp64_ulp_c_real > 0 else 0
                    imag_ulp = imag_abs / fp64_ulp_c_imag if impl_imag != 0 and fp64_ulp_c_imag > 0 else 0
                    
                    # Filter out NaN and infinite values
                    if not (math.isnan(real_ulp) or math.isinf(real_ulp)):
                        real_ulps.append(real_ulp)
                    if not (math.isnan(imag_ulp) or math.isinf(imag_ulp)):
                        imag_ulps.append(imag_ulp)
                
                integral_ulp_data[impl_name][coeff_name]['real'] = real_ulps
                integral_ulp_data[impl_name][coeff_name]['imag'] = imag_ulps
        
        ulp_data[integral_type] = integral_ulp_data
        print(f"Generated ULP data for {integral_type}")
    
    # Save ULP data to pickle file for fast loading
    ulp_data_file = os.path.join(data_dir, 'ulp_data.pkl')
    with open(ulp_data_file, 'wb') as f:
        pickle.dump(ulp_data, f)
    print(f"\nSaved ULP data to: {ulp_data_file}")
    
    # Also save a human-readable summary
    summary_file = os.path.join(data_dir, 'ulp_data_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("ULP Data Summary\n")
        f.write("================\n\n")
        for integral_type in sorted(ulp_data.keys()):
            f.write(f"{integral_type}:\n")
            for impl_name in ['CPU FP64', 'A100 FP64', 'MI250 FP64']:
                f.write(f"  {impl_name}:\n")
                for coeff_name in ['Coeff1', 'Coeff2', 'Coeff3']:
                    real_count = len(ulp_data[integral_type][impl_name][coeff_name]['real'])
                    imag_count = len(ulp_data[integral_type][impl_name][coeff_name]['imag'])
                    f.write(f"    {coeff_name}: {real_count} real, {imag_count} imag ULP values\n")
            f.write("\n")
    print(f"Saved ULP data summary to: {summary_file}")
    
    return ulp_data

def load_ulp_data():
    """Load ULP data from pickle file"""
    import os
    data_dir = 'ulp_data'
    ulp_data_file = os.path.join(data_dir, 'ulp_data.pkl')
    
    if not os.path.exists(ulp_data_file):
        print(f"ULP data file not found: {ulp_data_file}")
        print("Generating ULP data first...")
        return generate_ulp_data()
    
    print(f"Loading ULP data from: {ulp_data_file}")
    with open(ulp_data_file, 'rb') as f:
        ulp_data = pickle.load(f)
    
    print(f"Loaded ULP data for {len(ulp_data)} target integrals:")
    for integral_type in sorted(ulp_data.keys()):
        print(f"  {integral_type}")
    
    return ulp_data

def plot_ulp_differences(ulp_data=None):
    """Generate 3x3 ULP grid plots using pre-generated data"""
    
    # Load ULP data if not provided
    if ulp_data is None:
        ulp_data = load_ulp_data()
    
    print("=== ULP Difference Box Plot Analysis ===")
    print("Using pre-generated ULP data\n")
    
    # Create output directory
    import os
    output_dir = 'ulp_difference_plots'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}\n")
    
    print(f"Found {len(ulp_data)} different target integrals:")
    for integral_type in sorted(ulp_data.keys()):
        print(f"  {integral_type}")
    
    print("\nGenerating 3x3 ULP grid plots...")
    
    # Create plots for each target integral
    for integral_type in sorted(ulp_data.keys()):
        integral_ulp_data = ulp_data[integral_type]
        
        # Create the 3x3 grid plot
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f'ULP Difference Analysis - {integral_type}', fontsize=16, fontweight='bold')
        
        # Define implementation names
        implementations = ['CPU FP64', 'A100 FP64', 'MI250 FP64']
        
        # Define coefficient names
        coefficients = ['Coeff1', 'Coeff2', 'Coeff3']
        
        # Create plots for each implementation (row) and coefficient (column)
        for row_idx, impl_name in enumerate(implementations):
            for col_idx, coeff_name in enumerate(coefficients):
                ax = axes[row_idx, col_idx]
                
                # Get pre-calculated ULP data
                real_ulps = integral_ulp_data[impl_name][coeff_name]['real']
                imag_ulps = integral_ulp_data[impl_name][coeff_name]['imag']
                
                # Create box plot
                coeff_data = [real_ulps, imag_ulps]
                
                if coeff_data[0] or coeff_data[1]:  # Only plot if we have data
                    bp = ax.boxplot(coeff_data, labels=['Real', 'Imaginary'], patch_artist=True)
                    
                    # Set title and labels
                    ax.set_title(f'{impl_name} - {coeff_name}', fontsize=12, fontweight='bold')
                    if row_idx == 2:  # Bottom row
                        ax.set_xlabel('Component', fontsize=10)
                    if col_idx == 0:  # Left column
                        ax.set_ylabel('ULP Difference', fontsize=10)
                    
                    ax.set_yscale('log')
                    ax.grid(True, alpha=0.3)
                    
                    # Color the boxes
                    bp['boxes'][0].set_facecolor('lightblue')
                    bp['boxes'][1].set_facecolor('lightcoral')
                    
                    # Add statistics text
                    real_mean = np.mean(real_ulps) if real_ulps else 0
                    imag_mean = np.mean(imag_ulps) if imag_ulps else 0
                    real_max = np.max(real_ulps) if real_ulps else 0
                    imag_max = np.max(imag_ulps) if imag_ulps else 0
                    
                    ax.text(0.02, 0.98, f'Real Mean: {real_mean:.2f}\nReal Max: {real_max:.2f}\nImag Mean: {imag_mean:.2f}\nImag Max: {imag_max:.2f}', 
                            transform=ax.transAxes, verticalalignment='top', fontsize=8,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                else:
                    ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(f'{impl_name} - {coeff_name}', fontsize=12, fontweight='bold')
                    if row_idx == 2:  # Bottom row
                        ax.set_xlabel('Component', fontsize=10)
                    if col_idx == 0:  # Left column
                        ax.set_ylabel('ULP Difference', fontsize=10)
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save the plot
        filename = f'ulp_differences_{integral_type.lower()}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {filepath}")
        
        # Close the figure to free memory
        plt.close(fig)
    
    print(f"\nGenerated {len(ulp_data)} 3x3 ULP grid plot figures total.")
    print("All plots saved as PNG files with 300 DPI resolution.")

def create_ulp_summary_plot(ulp_data=None):
    """Create a summary plot showing mean ULP differences across all implementations and integrals"""
    
    # Load ULP data if not provided
    if ulp_data is None:
        ulp_data = load_ulp_data()
    
    print("\n=== Creating ULP Summary Plot ===")
    print("Using pre-generated ULP data\n")
    
    # Create output directory
    import os
    output_dir = 'ulp_difference_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for summary plot
    all_integrals = sorted(ulp_data.keys())
    implementations = ['CPU FP64', 'A100 FP64', 'MI250 FP64']
    
    # Pre-calculate all mean ULP differences to determine global y-axis range
    all_real_means = []
    all_imag_means = []
    mean_data = {}  # Store calculated means for plotting
    
    for impl_name in implementations:
        mean_data[impl_name] = {}
        for coeff_name in ['Coeff1', 'Coeff2', 'Coeff3']:
            mean_data[impl_name][coeff_name] = {'real': [], 'imag': []}
            
            for integral_type in all_integrals:
                # Get pre-calculated ULP data
                real_ulps = ulp_data[integral_type][impl_name][coeff_name]['real']
                imag_ulps = ulp_data[integral_type][impl_name][coeff_name]['imag']
                
                real_mean = np.mean(real_ulps) if real_ulps else 0
                imag_mean = np.mean(imag_ulps) if imag_ulps else 0
                
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
        y_min, y_max = 1e-10, 1e10  # Default range if no data
    
    # Create summary plot with 3x3 grid (implementations x coefficients)
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Mean ULP Differences Across All Target Integrals', fontsize=16, fontweight='bold')
    
    for impl_idx, impl_name in enumerate(implementations):
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
            ax.set_ylabel('Mean ULP Difference', fontsize=10)
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
                ax.set_ylabel('Mean ULP Difference', fontsize=10)
    
    plt.tight_layout()
    summary_filepath = os.path.join(output_dir, 'ulp_differences_summary.png')
    plt.savefig(summary_filepath, dpi=300, bbox_inches='tight')
    print(f"Saved ULP summary plot: {summary_filepath}")
    plt.close(fig)

def regenerate_data():
    """Force regenerate ULP data (useful for testing or when data changes)"""
    import os
    import shutil
    
    data_dir = 'ulp_data'
    if os.path.exists(data_dir):
        print(f"Removing existing data directory: {data_dir}")
        shutil.rmtree(data_dir)
    
    print("Generating fresh ULP data...")
    return generate_ulp_data()

def main():
    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Error: matplotlib and numpy are required for plotting.")
        print("Install them with: pip install matplotlib numpy")
        return
    
    # Check if ULP data exists, if not generate it
    import os
    data_dir = 'ulp_data'
    ulp_data_file = os.path.join(data_dir, 'ulp_data.pkl')
    
    if not os.path.exists(ulp_data_file):
        print("ULP data not found. Generating data first...")
        ulp_data = generate_ulp_data()
    else:
        print("Loading existing ULP data...")
        ulp_data = load_ulp_data()
    
    # Generate individual 3x3 ULP grid plots for each target integral
    plot_ulp_differences(ulp_data)
    
    # Generate ULP summary plot
    create_ulp_summary_plot(ulp_data)
    
    print("\n=== ULP Plotting Complete ===")
    print("All ULP plots have been generated and saved as PNG files.")
    print("Each target integral has its own 3x3 ULP grid plot:")
    print("  - Rows: CPU FP64, A100 FP64, MI250 FP64")
    print("  - Columns: Coeff1, Coeff2, Coeff3")
    print("  - Each subplot shows box plots for real and imaginary part ULP differences.")
    print("Summary plot shows mean ULP differences across all target integrals.")
    print(f"\nULP data is cached in: {ulp_data_file}")
    print("To regenerate data, delete the ulp_data directory and run again.")

if __name__ == "__main__":
    main()
