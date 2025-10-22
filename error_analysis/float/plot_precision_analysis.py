#!/usr/bin/env python3
"""
Unified script for precision analysis plotting

Generates plots for different precision analysis types:
- 0: Precise decimal digits
- 1: Absolute differences  
- 2: ULP differences

Command line usage:
  python plot_precision_analysis.py <plot_type> <plot_mode>
  
Arguments:
  plot_type: 0=precise digits, 1=absolute differences, 2=ULP differences
  plot_mode: 0=summary plots only, 1=individual plots per integral + summary plots

Features:
- Single pickle file containing all three data types
- CSV data loaded from /home/rbarik/qcdloop/error_analysis/float/raw/
- Separate output directories for each plot type
- Command line interface for easy switching between plot types
"""

import sys
import os
import math
import struct
import pickle
import argparse
from collections import defaultdict

# Import csv_parser from the same directory
from csv_parser import QCDLoopCSVParser
import matplotlib.pyplot as plt
import numpy as np

def ulp_float(x):
    """Calculate ULP for single precision float (FP32)"""
    if x == 0.0:
        return 1.1754943508222875e-38  # Smallest positive float
    if math.isnan(x) or math.isinf(x):
        return float('nan')
    
    # Get the bit representation (pack as float, unpack as unsigned int)
    packed = struct.pack('f', x)
    bits = struct.unpack('I', packed)[0]
    
    # Extract exponent (8 bits for FP32)
    exponent = (bits >> 23) & 0xFF
    if exponent == 0:
        # Denormalized number
        return 1.1754943508222875e-38
    else:
        # Normalized number (FP32: bias=127, mantissa=23 bits)
        return 2.0 ** (exponent - 127 - 23)

def ulp_double(x):
    """Calculate ULP for double precision float (FP64)"""
    if x == 0.0:
        return 2.2250738585072014e-308  # Smallest positive double
    if math.isnan(x) or math.isinf(x):
        return float('nan')
    
    # Get the bit representation
    packed = struct.pack('d', x)
    bits = struct.unpack('Q', packed)[0]
    
    # Extract exponent (11 bits for FP64)
    exponent = (bits >> 52) & 0x7FF
    if exponent == 0:
        # Denormalized number
        return 2.2250738585072014e-308
    else:
        # Normalized number (FP64: bias=1023, mantissa=52 bits)
        return 2.0 ** (exponent - 1023 - 52)

def calculate_precise_digits(true_value, absolute_error, precision_level):
    """Calculate the number of precise decimal digits from absolute error with proper edge case handling"""
    # Use the same precision detection as ULP calculation
    if '24bit' in precision_level:
        max_precise = 7   # FP32: ~7 decimal digits
    else:  # 53bit (FP64)
        max_precise = 15  # FP64: ~15 decimal digits
    
    # Handle invalid inputs
    if math.isnan(true_value) or math.isnan(absolute_error) or \
       math.isinf(true_value) or math.isinf(absolute_error):
        return 0
    
    # Error larger than true value → zero precision
    if absolute_error > abs(true_value):
        return 0
    
    # Both zero → max precision
    if true_value == 0 and absolute_error == 0:
        return max_precise
    
    # Error is zero → max precision
    if absolute_error == 0:
        return max_precise
    
    # True value is zero → zero precision
    if true_value == 0:
        return 0
    
    # Determine the minimum representable error for this precision level
    if '24bit' in precision_level:
        # FP32: minimum representable difference for 7 decimal digits
        min_representable_error = abs(true_value) * 1e-7
    else:  # 53bit (FP64)
        # FP64: minimum representable difference for 15 decimal digits  
        min_representable_error = abs(true_value) * 1e-15
    
    # If error is smaller than what the precision level can represent,
    # set to maximum precision for that level
    if absolute_error < min_representable_error:
        return max_precise
    
    # Normal case
    return -math.log10(abs(absolute_error) / abs(true_value))

def generate_all_data():
    """Generate all three data types and save to single pickle file"""
    parser = QCDLoopCSVParser()
    
    print("=== Generating All Precision Analysis Data ===")
    print("Baseline: /home/rbarik/qcdloop/error_analysis/float/raw/box_cpu_fp128.csv (quad precision)")
    print("Comparing precision levels: CPU FP64, CPU FP32, Simulated FP32, A100 FP64, A100 FP32, MI250 FP64, MI250 FP32\n")
    
    # Create output directory
    data_dir = 'precision_analysis_data'
    os.makedirs(data_dir, exist_ok=True)
    print(f"Created data directory: {data_dir}\n")
    
    # Define precision levels
    precision_levels = ['53bit_cpu', '24bit_cpu', '24bit_simulated', 
                       '53bit_a100', '24bit_a100', '53bit_mi250', '24bit_mi250']
    
    # CSV files in raw directory
    raw_data_dir = '/home/rbarik/qcdloop/error_analysis/float/raw'
    precision_files = {
        '53bit_cpu': os.path.join(raw_data_dir, 'box_cpu_fp64.csv'),
        '24bit_cpu': os.path.join(raw_data_dir, 'box_cpu_fp32.csv'),
        '24bit_a100': os.path.join(raw_data_dir, 'box_a100_fp32.csv'),
        '24bit_mi250': os.path.join(raw_data_dir, 'box_mi250_fp32.csv'),
        '53bit_a100': os.path.join(raw_data_dir, 'box_a100_fp64.csv'),
        '53bit_mi250': os.path.join(raw_data_dir, 'box_mi250_fp64.csv'),
        '24bit_simulated': os.path.join(raw_data_dir, 'box_simulated_fp32.csv')
    }
    
    # Parse baseline file
    print("Loading baseline data...")
    baseline_path = os.path.join(raw_data_dir, 'box_cpu_fp128.csv')
    baseline_data = parser.parse_box_run_0(baseline_path)
    print(f"Loaded {len(baseline_data)} rows from baseline (quad precision)")
    
    # Parse all precision level files
    precision_datasets = {}
    for precision in precision_levels:
        filepath = precision_files[precision]
        print(f"Loading {precision} data from {filepath}...")
        try:
            data = parser.parse_box_run_1(filepath)
            precision_datasets[precision] = data
            print(f"  Loaded {len(data)} rows from {precision}")
        except Exception as e:
            print(f"  Error loading {precision}: {e}")
            precision_datasets[precision] = []
    
    print()
    
    # Group data by target integral
    baseline_by_integral = defaultdict(list)
    precision_by_integral = {precision: defaultdict(list) for precision in precision_levels}
    
    # First, group baseline data
    for baseline_row in baseline_data:
        integral_type = baseline_row.target_integral
        baseline_by_integral[integral_type].append(baseline_row)
    
    # Then group precision data, ensuring alignment with baseline
    for precision in precision_levels:
        if precision not in precision_datasets or not precision_datasets[precision]:
            continue
        
        for i, row in enumerate(precision_datasets[precision]):
            if i < len(baseline_data):
                integral_type = baseline_data[i].target_integral
                precision_by_integral[precision][integral_type].append(row)
    
    print(f"Found {len(baseline_by_integral)} different target integrals:")
    for integral_type in sorted(baseline_by_integral.keys()):
        print(f"  {integral_type}: {len(baseline_by_integral[integral_type])} rows")
    
    print("\nGenerating all data types for each target integral...")
    
    # Initialize data structures for all three types
    all_data = {
        'precise_digits': {},
        'absolute_differences': {},
        'ulp_differences': {}
    }
    
    for integral_type in sorted(baseline_by_integral.keys()):
        baseline_rows = baseline_by_integral[integral_type]
        
        # Initialize data structures for this integral
        integral_data = {
            'precise_digits': {},
            'absolute_differences': {},
            'ulp_differences': {}
        }
        
        for precision in precision_levels:
            for data_type in ['precise_digits', 'absolute_differences', 'ulp_differences']:
                integral_data[data_type][precision] = {
                    'Coeff1': {'real': [], 'imag': []},
                    'Coeff2': {'real': [], 'imag': []},
                    'Coeff3': {'real': [], 'imag': []}
                }
        
        for precision in precision_levels:
            if precision not in precision_by_integral or integral_type not in precision_by_integral[precision]:
                continue
            
            precision_rows = precision_by_integral[precision][integral_type]
            
            # Check if row counts match
            if len(baseline_rows) != len(precision_rows):
                print(f"Warning: Mismatch in row counts for {integral_type}, {precision}")
                continue
            
            for coeff_name in ['Coeff1', 'Coeff2', 'Coeff3']:
                # Initialize lists for all data types
                real_abs_diffs = []
                imag_abs_diffs = []
                real_precise_digits = []
                imag_precise_digits = []
                real_ulp_diffs = []
                imag_ulp_diffs = []
                
                # Select appropriate ULP function based on precision level
                ulp_func = ulp_float if '24bit' in precision else ulp_double
                
                for baseline_row, precision_row in zip(baseline_rows, precision_rows):
                    # Get values
                    precision_real = precision_row.coeff1_real if coeff_name == 'Coeff1' else precision_row.coeff2_real if coeff_name == 'Coeff2' else precision_row.coeff3_real
                    precision_imag = precision_row.coeff1_imag if coeff_name == 'Coeff1' else precision_row.coeff2_imag if coeff_name == 'Coeff2' else precision_row.coeff3_imag
                    
                    baseline_real = baseline_row.coeff1_real if coeff_name == 'Coeff1' else baseline_row.coeff2_real if coeff_name == 'Coeff2' else baseline_row.coeff3_real
                    baseline_imag = baseline_row.coeff1_imag if coeff_name == 'Coeff1' else baseline_row.coeff2_imag if coeff_name == 'Coeff2' else baseline_row.coeff3_imag
                    
                    # Calculate absolute differences
                    real_abs_diff = abs(precision_real - baseline_real)
                    imag_abs_diff = abs(precision_imag - baseline_imag)
                    
                    # Calculate precise digits with precision level detection
                    real_precise_digit = calculate_precise_digits(baseline_real, real_abs_diff, precision)
                    imag_precise_digit = calculate_precise_digits(baseline_imag, imag_abs_diff, precision)
                    
                    # Calculate ULP differences
                    ulp_c_real = ulp_func(precision_real)
                    ulp_c_imag = ulp_func(precision_imag)
                    
                    # ULP thresholding (set to zero if ULP > |Baseline|)
                    real_abs_thresholded = 0.0 if ulp_c_real > abs(baseline_real) else real_abs_diff
                    imag_abs_thresholded = 0.0 if ulp_c_imag > abs(baseline_imag) else imag_abs_diff
                    
                    real_ulp_diff = real_abs_thresholded / ulp_c_real if precision_real != 0 and ulp_c_real > 0 else 0
                    imag_ulp_diff = imag_abs_thresholded / ulp_c_imag if precision_imag != 0 and ulp_c_imag > 0 else 0
                    
                    # Filter out invalid values for each data type
                    if not (math.isnan(real_abs_diff) or math.isinf(real_abs_diff)):
                        real_abs_diffs.append(real_abs_diff)
                    if not (math.isnan(imag_abs_diff) or math.isinf(imag_abs_diff)):
                        imag_abs_diffs.append(imag_abs_diff)
                    
                    # Precise digits now returns finite values, so just check for NaN
                    if not math.isnan(real_precise_digit):
                        real_precise_digits.append(real_precise_digit)
                    if not math.isnan(imag_precise_digit):
                        imag_precise_digits.append(imag_precise_digit)
                    
                    if not (math.isnan(real_ulp_diff) or math.isinf(real_ulp_diff)):
                        real_ulp_diffs.append(real_ulp_diff)
                    if not (math.isnan(imag_ulp_diff) or math.isinf(imag_ulp_diff)):
                        imag_ulp_diffs.append(imag_ulp_diff)
                
                # Store all data types
                integral_data['absolute_differences'][precision][coeff_name]['real'] = real_abs_diffs
                integral_data['absolute_differences'][precision][coeff_name]['imag'] = imag_abs_diffs
                
                integral_data['precise_digits'][precision][coeff_name]['real'] = real_precise_digits
                integral_data['precise_digits'][precision][coeff_name]['imag'] = imag_precise_digits
                
                integral_data['ulp_differences'][precision][coeff_name]['real'] = real_ulp_diffs
                integral_data['ulp_differences'][precision][coeff_name]['imag'] = imag_ulp_diffs
        
        # Store data for this integral
        for data_type in ['precise_digits', 'absolute_differences', 'ulp_differences']:
            all_data[data_type][integral_type] = integral_data[data_type]
        
        print(f"Generated all data types for {integral_type}")
    
    # Save all data to single pickle file
    all_data_file = os.path.join(data_dir, 'all_precision_data.pkl')
    with open(all_data_file, 'wb') as f:
        pickle.dump(all_data, f)
    print(f"\nSaved all precision analysis data to: {all_data_file}")
    
    # Save summary
    summary_file = os.path.join(data_dir, 'all_data_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("All Precision Analysis Data Summary\n")
        f.write("===================================\n\n")
        for data_type in ['precise_digits', 'absolute_differences', 'ulp_differences']:
            f.write(f"{data_type.upper()}:\n")
            for integral_type in sorted(all_data[data_type].keys()):
                f.write(f"  {integral_type}:\n")
                for precision in precision_levels:
                    f.write(f"    {precision}:\n")
                    for coeff_name in ['Coeff1', 'Coeff2', 'Coeff3']:
                        real_count = len(all_data[data_type][integral_type][precision][coeff_name]['real'])
                        imag_count = len(all_data[data_type][integral_type][precision][coeff_name]['imag'])
                        f.write(f"      {coeff_name}: {real_count} real, {imag_count} imag values\n")
            f.write("\n")
    print(f"Saved data summary to: {summary_file}")
    
    return all_data

def load_all_data():
    """Load all precision analysis data from pickle file"""
    data_dir = 'precision_analysis_data'
    all_data_file = os.path.join(data_dir, 'all_precision_data.pkl')
    
    if not os.path.exists(all_data_file):
        print(f"Precision analysis data file not found: {all_data_file}")
        print("Generating all data types first...")
        return generate_all_data()
    
    print(f"Loading precision analysis data from: {all_data_file}")
    with open(all_data_file, 'rb') as f:
        all_data = pickle.load(f)
    
    print(f"Loaded data for {len(all_data['precise_digits'])} target integrals")
    print("Available data types: precise_digits, absolute_differences, ulp_differences")
    
    return all_data

def create_individual_plots(data, data_type, plot_config):
    """Create individual 3x2 grid plots for each target integral"""
    print(f"\n=== Creating Individual {plot_config['name']} Plots ===")
    
    # Create output directory
    output_dir = plot_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}\n")
    
    precision_levels = ['53bit_cpu', '24bit_cpu', 
                       '53bit_a100', '24bit_a100', '53bit_mi250', '24bit_mi250']
    # precision_levels = ['53bit_cpu', '24bit_cpu', '24bit_simulated', 
    #                    '53bit_a100', '24bit_a100', '53bit_mi250', '24bit_mi250']
    precision_labels = ['FP64\nCPU', 'FP32\nCPU', 
                       'FP64\nA100', 'FP32\nA100', 'FP64\nMI250', 'FP32\nMI250']
    # precision_labels = ['FP64\nCPU', 'FP32\nCPU', 'FP32\nSimulated', 
    #                    'FP64\nA100', 'FP32\nA100', 'FP64\nMI250', 'FP32\nMI250']
    
    for integral_type in sorted(data.keys()):
        integral_data = data[integral_type]
        
        # Create the 3x2 grid plot
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f'{plot_config["title"]} - {integral_type}', fontsize=16, fontweight='bold')
        
        coefficients = ['Coeff1', 'Coeff2', 'Coeff3']
        
        for row_idx, coeff_name in enumerate(coefficients):
            ax_real = axes[row_idx, 0]
            ax_imag = axes[row_idx, 1]
            
            # Collect data for all precision levels
            real_data_all = []
            imag_data_all = []
            
            for precision in precision_levels:
                real_values = integral_data[precision][coeff_name]['real']
                imag_values = integral_data[precision][coeff_name]['imag']
                real_data_all.append(real_values)
                imag_data_all.append(imag_values)
            
            # Plot real part
            if any(real_data_all):
                bp_real = ax_real.boxplot(real_data_all, labels=precision_labels, patch_artist=True)
                
                # Color the boxes with gradient
                colors = plt.cm.viridis(np.linspace(0, 1, len(precision_levels)))
                for patch, color in zip(bp_real['boxes'], colors):
                    patch.set_facecolor(color)
                
                ax_real.set_title(f'{coeff_name} - Real Part', fontsize=12, fontweight='bold')
                ax_real.set_xlabel('Precision / Hardware', fontsize=10)
                ax_real.set_ylabel(plot_config['y_label'], fontsize=10)
                ax_real.set_yscale(plot_config['y_scale'])
                ax_real.grid(True, alpha=0.3)
                ax_real.tick_params(axis='x', rotation=45)
            else:
                ax_real.text(0.5, 0.5, 'No data', transform=ax_real.transAxes, ha='center', va='center')
                ax_real.set_title(f'{coeff_name} - Real Part', fontsize=12, fontweight='bold')
            
            # Plot imaginary part
            if any(imag_data_all):
                bp_imag = ax_imag.boxplot(imag_data_all, labels=precision_labels, patch_artist=True)
                
                # Color the boxes with gradient
                colors = plt.cm.viridis(np.linspace(0, 1, len(precision_levels)))
                for patch, color in zip(bp_imag['boxes'], colors):
                    patch.set_facecolor(color)
                
                ax_imag.set_title(f'{coeff_name} - Imaginary Part', fontsize=12, fontweight='bold')
                ax_imag.set_xlabel('Precision / Hardware', fontsize=10)
                ax_imag.set_ylabel(plot_config['y_label'], fontsize=10)
                ax_imag.set_yscale(plot_config['y_scale'])
                ax_imag.grid(True, alpha=0.3)
                ax_imag.tick_params(axis='x', rotation=45)
            else:
                ax_imag.text(0.5, 0.5, 'No data', transform=ax_imag.transAxes, ha='center', va='center')
                ax_imag.set_title(f'{coeff_name} - Imaginary Part', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        filename = f'{plot_config["filename_prefix"]}_{integral_type.lower()}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {filepath}")
        
        plt.close(fig)
    
    print(f"\nGenerated {len(data)} individual {plot_config['name']} plot figures.")

def create_summary_plot(data, data_type, plot_config):
    """Create full summary plot (4x6 grid)"""
    print(f"\n=== Creating {plot_config['name']} Summary Plot ===")
    
    # Create output directory
    output_dir = plot_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    all_integrals = sorted(data.keys())
    precision_levels = ['53bit_cpu', '24bit_cpu', 
                       '53bit_a100', '24bit_a100', '53bit_mi250', '24bit_mi250']
    # precision_levels = ['53bit_cpu', '24bit_cpu', '24bit_simulated', 
    #                    '53bit_a100', '24bit_a100', '53bit_mi250', '24bit_mi250']
    precision_labels = ['FP64 CPU', 'FP32 CPU', 
                       'FP64 A100', 'FP32 A100', 'FP64 MI250', 'FP32 MI250']
    # precision_labels = ['FP64 CPU', 'FP32 CPU', 'FP32 Simulated', 
    #                    'FP64 A100', 'FP32 A100', 'FP64 MI250', 'FP32 MI250']
    
    # Pre-calculate y-axis limits
    all_values = []
    for coeff_name in ['Coeff1', 'Coeff2', 'Coeff3']:
        for precision in precision_levels:
            for integral_type in all_integrals:
                real_values = data[integral_type][precision][coeff_name]['real']
                imag_values = data[integral_type][precision][coeff_name]['imag']
                # Filter based on data type
                if data_type == 'precise_digits':
                    # Precise digits now returns finite values, so no need to filter infinities
                    all_values.extend(real_values)
                    all_values.extend(imag_values)
                else:
                    all_values.extend([v for v in real_values if v > 0])
                    all_values.extend([v for v in imag_values if v > 0])
    
    if all_values:
        if data_type == 'precise_digits':
            y_min = max(0, min(all_values) - 1)
            y_max = min(20, max(all_values) + 2)
        else:
            y_min = min(all_values) * 0.5
            y_max = max(all_values) * 2.0
    else:
        if data_type == 'precise_digits':
            y_min, y_max = 0, 20
        else:
            y_min, y_max = 1e-20, 1e10
    
    y_limits = (y_min, y_max)
    
    # Create summary plot with 6x6 grid
    fig, axes = plt.subplots(6, 6, figsize=(30, 24))
    
    for prec_idx, precision in enumerate(precision_levels):
        for coeff_idx, coeff_name in enumerate(['Coeff1', 'Coeff2', 'Coeff3']):
            col_real = coeff_idx * 2
            col_imag = coeff_idx * 2 + 1
            
            ax_real = axes[prec_idx, col_real]
            ax_imag = axes[prec_idx, col_imag]
            
            # Collect box plot data for all integrals - REAL PART
            box_data_real = []
            labels_real = []
            
            for integral_type in all_integrals:
                real_values = data[integral_type][precision][coeff_name]['real']
                if real_values:
                    box_data_real.append(real_values)
                    labels_real.append(integral_type)
            
            # Plot real part
            if box_data_real:
                bp_real = ax_real.boxplot(box_data_real, labels=labels_real, patch_artist=True,
                                           flierprops=dict(marker='o', markerfacecolor='black', markersize=3, 
                                                          linestyle='none', markeredgewidth=0.5))
                
                for patch in bp_real['boxes']:
                    patch.set_facecolor('yellow')
                
                for median in bp_real['medians']:
                    median.set_color('red')
                
                label = precision_labels[prec_idx]
                ax_real.set_title(f'{coeff_name} Real ({label})', fontsize=11, fontweight='bold')
                ax_real.set_yscale(plot_config['y_scale'])
                ax_real.set_ylim(y_limits)
                ax_real.grid(True, alpha=0.3)
                ax_real.tick_params(axis='x', rotation=45)
                
                if prec_idx == 5:
                    ax_real.set_xlabel('Target Integral', fontsize=10)
                if col_real == 0:
                    ax_real.set_ylabel(plot_config['y_label'], fontsize=10)
            else:
                ax_real.text(0.5, 0.5, 'No data', transform=ax_real.transAxes, ha='center', va='center')
                label = precision_labels[prec_idx]
                ax_real.set_title(f'{coeff_name} Real ({label})', fontsize=11, fontweight='bold')
            
            # Collect box plot data for all integrals - IMAGINARY PART
            box_data_imag = []
            labels_imag = []
            
            for integral_type in all_integrals:
                imag_values = data[integral_type][precision][coeff_name]['imag']
                if imag_values:
                    box_data_imag.append(imag_values)
                    labels_imag.append(integral_type)
            
            # Plot imaginary part
            if box_data_imag:
                bp_imag = ax_imag.boxplot(box_data_imag, labels=labels_imag, patch_artist=True,
                                           flierprops=dict(marker='o', markerfacecolor='black', markersize=3, 
                                                          linestyle='none', markeredgewidth=0.5))
                
                for patch in bp_imag['boxes']:
                    patch.set_facecolor('lightgreen')
                
                for median in bp_imag['medians']:
                    median.set_color('red')
                
                label = precision_labels[prec_idx]
                ax_imag.set_title(f'{coeff_name} Imag ({label})', fontsize=11, fontweight='bold')
                ax_imag.set_yscale(plot_config['y_scale'])
                ax_imag.set_ylim(y_limits)
                ax_imag.grid(True, alpha=0.3)
                ax_imag.tick_params(axis='x', rotation=45)
                
                if prec_idx == 5:
                    ax_imag.set_xlabel('Target Integral', fontsize=10)
            else:
                ax_imag.text(0.5, 0.5, 'No data', transform=ax_imag.transAxes, ha='center', va='center')
                label = precision_labels[prec_idx]
                ax_imag.set_title(f'{coeff_name} Imag ({label})', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    summary_filepath = os.path.join(output_dir, f'{plot_config["filename_prefix"]}_summary.png')
    plt.savefig(summary_filepath, dpi=300, bbox_inches='tight')
    print(f"Saved {plot_config['name']} summary plot: {summary_filepath}")
    plt.close(fig)

def create_summary_subset_plot(data, data_type, plot_config):
    """Create subset summary plot (2x4 grid, Coeff1 only)"""
    print(f"\n=== Creating {plot_config['name']} Subset Summary Plot ===")
    
    # Create output directory
    output_dir = plot_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    all_integrals = sorted(data.keys())
    precision_levels = ['53bit_cpu', '24bit_cpu', 
                       '53bit_a100', '24bit_a100', '53bit_mi250', '24bit_mi250']
    # precision_levels = ['53bit_cpu', '24bit_cpu', '24bit_simulated', 
    #                    '53bit_a100', '24bit_a100', '53bit_mi250', '24bit_mi250']
    precision_names = ['FP64 CPU', 'FP32 CPU', 
                       'FP64 A100', 'FP32 A100', 'FP64 MI250', 'FP32 MI250']
    # precision_names = ['FP64 CPU', 'FP32 CPU', 'FP32 Simulated', 
    #                    'FP64 A100', 'FP32 A100', 'FP64 MI250', 'FP32 MI250']
    
    # Pre-calculate y-axis limits for Coeff1 only
    all_values = []
    coeff_name = 'Coeff1'
    for precision in precision_levels:
        for integral_type in all_integrals:
            real_values = data[integral_type][precision][coeff_name]['real']
            imag_values = data[integral_type][precision][coeff_name]['imag']
            if data_type == 'precise_digits':
                # Precise digits now returns finite values, so no need to filter infinities
                all_values.extend(real_values)
                all_values.extend(imag_values)
            else:
                all_values.extend([v for v in real_values if v > 0])
                all_values.extend([v for v in imag_values if v > 0])
    
    if all_values:
        if data_type == 'precise_digits':
            y_min = max(0, min(all_values) - 1)
            y_max = min(20, max(all_values) + 2)
        else:
            y_min = min(all_values) * 0.5
            y_max = max(all_values) * 2.0
    else:
        if data_type == 'precise_digits':
            y_min, y_max = 0, 20
        else:
            y_min, y_max = 1e-20, 1e10
    
    y_limits = (y_min, y_max)
    
    # Create summary plot with 2x3 grid (FP64 row, FP32 row)
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    
    # Define precision levels for each row
    fp64_precisions = ['53bit_cpu', '53bit_a100', '53bit_mi250']
    fp32_precisions = ['24bit_cpu', '24bit_a100', '24bit_mi250']
    fp64_labels = ['FP64 CPU', 'FP64 A100', 'FP64 MI250']
    fp32_labels = ['FP32 CPU', 'FP32 A100', 'FP32 MI250']
    
    # First row: FP64 plots
    for col_idx, precision in enumerate(fp64_precisions):
        ax = axes[0, col_idx]
        
        # Collect box plot data for all integrals - REAL PART
        box_data_real = []
        labels_real = []
        
        for integral_type in all_integrals:
            real_values = data[integral_type][precision][coeff_name]['real']
            if real_values:
                box_data_real.append(real_values)
                labels_real.append(integral_type)
        
        # Plot real part
        if box_data_real:
            bp_real = ax.boxplot(box_data_real, labels=labels_real, patch_artist=True,
                                 flierprops=dict(marker='o', markerfacecolor='black', markersize=3, 
                                                linestyle='none', markeredgewidth=0.5))
            
            for patch in bp_real['boxes']:
                patch.set_facecolor('yellow')
            
            for median in bp_real['medians']:
                median.set_color('red')
            
            precision_name = fp64_labels[col_idx]
            ax.set_title(f'{coeff_name} ({precision_name})', fontsize=12, fontweight='bold')
            ax.set_yscale(plot_config['y_scale'])
            ax.set_ylim(y_limits)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            if col_idx == 0:
                ax.set_ylabel(plot_config['y_label'], fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
            precision_name = fp64_labels[col_idx]
            ax.set_title(f'{coeff_name} ({precision_name})', fontsize=12, fontweight='bold')
    
    # Second row: FP32 plots
    for col_idx, precision in enumerate(fp32_precisions):
        ax = axes[1, col_idx]
        
        # Collect box plot data for all integrals - REAL PART
        box_data_real = []
        labels_real = []
        
        for integral_type in all_integrals:
            real_values = data[integral_type][precision][coeff_name]['real']
            if real_values:
                box_data_real.append(real_values)
                labels_real.append(integral_type)
        
        # Plot real part
        if box_data_real:
            bp_real = ax.boxplot(box_data_real, labels=labels_real, patch_artist=True,
                                 flierprops=dict(marker='o', markerfacecolor='black', markersize=3, 
                                                linestyle='none', markeredgewidth=0.5))
            
            for patch in bp_real['boxes']:
                patch.set_facecolor('lightgreen')
            
            for median in bp_real['medians']:
                median.set_color('red')
            
            precision_name = fp32_labels[col_idx]
            ax.set_title(f'{coeff_name} ({precision_name})', fontsize=12, fontweight='bold')
            ax.set_yscale(plot_config['y_scale'])
            ax.set_ylim(y_limits)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            ax.set_xlabel('Target Integral', fontsize=10)
            
            if col_idx == 0:
                ax.set_ylabel(plot_config['y_label'], fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
            precision_name = fp32_labels[col_idx]
            ax.set_title(f'{coeff_name} ({precision_name})', fontsize=12, fontweight='bold')
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='yellow', edgecolor='black', label='FP64 (Double Precision)'),
        Patch(facecolor='lightgreen', edgecolor='black', label='FP32 (Single Precision)')
    ]
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0), 
               ncol=2, fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    summary_filepath = os.path.join(output_dir, f'{plot_config["filename_prefix"]}_summary_subset.png')
    plt.savefig(summary_filepath, dpi=300, bbox_inches='tight')
    print(f"Saved {plot_config['name']} subset summary plot: {summary_filepath}")
    plt.close(fig)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate precision analysis plots')
    parser.add_argument('plot_type', type=int, choices=[0, 1, 2], 
                       help='Plot type: 0=precise digits, 1=absolute differences, 2=ULP differences')
    parser.add_argument('plot_mode', type=int, choices=[0, 1],
                       help='Plot mode: 0=summary plots only, 1=individual plots + summary plots')
    
    args = parser.parse_args()
    
    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Error: matplotlib and numpy are required for plotting.")
        print("Install them with: pip install matplotlib numpy")
        return
    
    # Load or generate all data
    all_data = load_all_data()
    
    # Define plot configurations
    plot_configs = {
        0: {  # Precise digits
            'name': 'Precise Decimal Digits',
            'title': 'Precision Level Precise Decimal Digits Analysis',
            'y_label': 'Precise Decimal Digits',
            'y_scale': 'linear',
            'output_dir': 'precision_precise_digits_plots',
            'filename_prefix': 'precision_precise_digits'
        },
        1: {  # Absolute differences
            'name': 'Absolute Differences',
            'title': 'Precision Level Absolute Difference Analysis',
            'y_label': 'Absolute Difference',
            'y_scale': 'log',
            'output_dir': 'precision_abs_plots',
            'filename_prefix': 'precision_abs_differences'
        },
        2: {  # ULP differences
            'name': 'ULP Differences',
            'title': 'Precision Level ULP Difference Analysis',
            'y_label': 'ULP Difference',
            'y_scale': 'log',
            'output_dir': 'precision_ulp_plots',
            'filename_prefix': 'precision_ulp_differences'
        }
    }
    
    # Get data type and plot config
    data_types = ['precise_digits', 'absolute_differences', 'ulp_differences']
    data_type = data_types[args.plot_type]
    plot_config = plot_configs[args.plot_type]
    
    print(f"\n=== Generating {plot_config['name']} Plots ===")
    print(f"Plot mode: {'Individual + Summary' if args.plot_mode == 1 else 'Summary only'}")
    
    # Get the data for the selected type
    data = all_data[data_type]
    
    # Generate plots based on mode
    if args.plot_mode == 1:
        # Individual plots + summary plots
        create_individual_plots(data, data_type, plot_config)
        create_summary_plot(data, data_type, plot_config)
        create_summary_subset_plot(data, data_type, plot_config)
    else:
        # Summary plots only
        create_summary_plot(data, data_type, plot_config)
        create_summary_subset_plot(data, data_type, plot_config)
    
    print(f"\n=== {plot_config['name']} Plotting Complete ===")
    print(f"All plots saved as PNG files with 300 DPI resolution.")
    print(f"Output directory: {plot_config['output_dir']}")

if __name__ == "__main__":
    main()


