#!/usr/bin/env python3
"""
Script to calculate differences between different precision levels and quad precision baseline in ULP
Analyzes precision degradation across 11bit, 24bit, 40bit, 45bit, 50bit, 51bit, 52bit, and 53bit mantissa precision

Workflow:
1. First run: Generates ULP data from CSV files and saves to precision_ulp_data/ulp_data.pkl
2. Subsequent runs: Loads pre-generated data from pickle file for fast analysis
3. To regenerate data: Delete precision_ulp_data/ directory or call regenerate_data() function

This separation allows for fast iteration on analysis logic without regenerating data each time.

The analysis is performed per target integral, providing granular statistics for each specific
integral type rather than aggregating across all data points.
"""

import sys
import os

# Add the gpu_arch directory to path to import csv_parser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gpu_arch'))

from csv_parser import QCDLoopCSVParser
import statistics
import struct
import math
import pickle
from collections import defaultdict

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

def generate_ulp_data():
    """Generate ULP data and save to pickle file"""
    parser = QCDLoopCSVParser()
    
    print("=== Generating ULP Data for Precision Analysis ===")
    print("Baseline: error_analysis/float/box_cpu_fp128.csv (quad precision)")
    print("Comparing precision levels: CPU FP32, CPU FP64, A100 FP32, MI250 FP32\n")
    
    # Create output directory
    data_dir = 'precision_ulp_data'
    os.makedirs(data_dir, exist_ok=True)
    print(f"Created data directory: {data_dir}\n")
    
    # Define precision levels (CPU, A100, MI250)
    # Format: 'bits_hardware'
    precision_levels = ['53bit_cpu', '24bit_cpu', '24bit_a100', '24bit_mi250']
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    precision_files = {
        '53bit_cpu': os.path.join(script_dir, 'box_cpu_fp64.csv'),
        '24bit_cpu': os.path.join(script_dir, 'box_cpu_fp32.csv'),
        '24bit_a100': os.path.join(script_dir, 'box_a100_fp32.csv'),
        '24bit_mi250': os.path.join(script_dir, 'box_mi250_fp32.csv')
    }
    
    # Parse baseline file
    print("Loading baseline data...")
    baseline_path = os.path.join(script_dir, 'box_cpu_fp128.csv')
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
    
    print("\nGenerating ULP data for each target integral...")
    
    # Generate ULP data for each target integral
    ulp_data = {}
    
    for integral_type in sorted(baseline_by_integral.keys()):
        baseline_rows = baseline_by_integral[integral_type]
        
        # Store ULP data for this integral
        integral_ulp_data = {}
        for precision in precision_levels:
            integral_ulp_data[precision] = {
                'Coeff1': {'real': [], 'imag': [], 'test_ids': [], 'thresholding_stats': {}},
                'Coeff2': {'real': [], 'imag': [], 'test_ids': [], 'thresholding_stats': {}},
                'Coeff3': {'real': [], 'imag': [], 'test_ids': [], 'thresholding_stats': {}}
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
                real_ulps = []
                imag_ulps = []
                test_ids = []
                
                # Select appropriate ULP function based on precision level
                # Check if it's a 24-bit (FP32) or 53-bit (FP64) precision
                ulp_func = ulp_float if '24bit' in precision else ulp_double
                
                # Track thresholding statistics
                thresholding_stats = {'real_thresholded': 0, 'imag_thresholded': 0, 'total_comparisons': 0}
                
                for idx, (baseline_row, precision_row) in enumerate(zip(baseline_rows, precision_rows)):
                    # Calculate absolute differences with ULP thresholding
                    # Set to zero if ULP > |Baseline| to prevent ULP explosion near zero
                    precision_real = precision_row.coeff1_real if coeff_name == 'Coeff1' else precision_row.coeff2_real if coeff_name == 'Coeff2' else precision_row.coeff3_real
                    precision_imag = precision_row.coeff1_imag if coeff_name == 'Coeff1' else precision_row.coeff2_imag if coeff_name == 'Coeff2' else precision_row.coeff3_imag
                    
                    # Use precision-appropriate ULP calculation
                    ulp_c_real = ulp_func(precision_real)
                    ulp_c_imag = ulp_func(precision_imag)
                    
                    baseline_real = baseline_row.coeff1_real if coeff_name == 'Coeff1' else baseline_row.coeff2_real if coeff_name == 'Coeff2' else baseline_row.coeff3_real
                    baseline_imag = baseline_row.coeff1_imag if coeff_name == 'Coeff1' else baseline_row.coeff2_imag if coeff_name == 'Coeff2' else baseline_row.coeff3_imag
                    
                    # Check if thresholding is applied
                    real_thresholded = ulp_c_real > abs(baseline_real)
                    imag_thresholded = ulp_c_imag > abs(baseline_imag)
                    
                    # Track thresholding statistics
                    thresholding_stats['total_comparisons'] += 2  # real and imaginary
                    if real_thresholded:
                        thresholding_stats['real_thresholded'] += 1
                    if imag_thresholded:
                        thresholding_stats['imag_thresholded'] += 1
                    
                    real_abs = 0.0 if real_thresholded else abs(precision_real - baseline_real)
                    imag_abs = 0.0 if imag_thresholded else abs(precision_imag - baseline_imag)
                    
                    # Calculate ULP differences using pre-calculated ULP values
                    # If abs_diff is 0 (due to thresholding), ULP is also 0
                    real_ulp = real_abs / ulp_c_real if precision_real != 0 and ulp_c_real > 0 else 0
                    imag_ulp = imag_abs / ulp_c_imag if precision_imag != 0 and ulp_c_imag > 0 else 0
                    
                    # Get test ID (use row id if available, otherwise use index)
                    test_id = getattr(baseline_row, 'id', idx)
                    
                    # Store both ULP values and test ID together
                    # Filter out NaN and infinite values
                    if not (math.isnan(real_ulp) or math.isinf(real_ulp)):
                        real_ulps.append(real_ulp)
                    else:
                        real_ulps.append(0.0)  # Keep list sizes aligned
                    
                    if not (math.isnan(imag_ulp) or math.isinf(imag_ulp)):
                        imag_ulps.append(imag_ulp)
                    else:
                        imag_ulps.append(0.0)  # Keep list sizes aligned
                    
                    test_ids.append(test_id)
                
                integral_ulp_data[precision][coeff_name]['real'] = real_ulps
                integral_ulp_data[precision][coeff_name]['imag'] = imag_ulps
                integral_ulp_data[precision][coeff_name]['test_ids'] = test_ids
                integral_ulp_data[precision][coeff_name]['thresholding_stats'] = thresholding_stats
        
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
        f.write("ULP Data Summary - Precision Analysis\n")
        f.write("======================================\n\n")
        for integral_type in sorted(ulp_data.keys()):
            f.write(f"{integral_type}:\n")
            for precision in precision_levels:
                f.write(f"  {precision}:\n")
                for coeff_name in ['Coeff1', 'Coeff2', 'Coeff3']:
                    real_count = len(ulp_data[integral_type][precision][coeff_name]['real'])
                    imag_count = len(ulp_data[integral_type][precision][coeff_name]['imag'])
                    f.write(f"    {coeff_name}: {real_count} real, {imag_count} imag ULP values\n")
            f.write("\n")
    print(f"Saved ULP data summary to: {summary_file}")
    
    return ulp_data

def load_ulp_data():
    """Load ULP data from pickle file"""
    data_dir = 'precision_ulp_data'
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

def check_data_completeness(ulp_data):
    """Check if cached ULP data includes thresholding statistics and test IDs"""
    precision_levels = ['53bit_cpu', '24bit_cpu', '24bit_a100', '24bit_mi250']
    for integral_type in ulp_data.keys():
        for precision in precision_levels:
            if precision not in ulp_data[integral_type]:
                continue
            for coeff_name in ['Coeff1', 'Coeff2', 'Coeff3']:
                if 'thresholding_stats' not in ulp_data[integral_type][precision][coeff_name]:
                    return False
                if 'test_ids' not in ulp_data[integral_type][precision][coeff_name]:
                    return False
    return True

def regenerate_data():
    """Force regenerate ULP data (useful for testing or when data changes)"""
    import shutil
    
    data_dir = 'precision_ulp_data'
    if os.path.exists(data_dir):
        print(f"Removing existing data directory: {data_dir}")
        shutil.rmtree(data_dir)
    
    print("Generating fresh ULP data...")
    return generate_ulp_data()

def calculate_ulp_differences(ulp_data=None):
    """Calculate ULP differences using pre-generated data"""
    
    # Load ULP data if not provided
    if ulp_data is None:
        ulp_data = load_ulp_data()
    
    print("=== Precision Level ULP Analysis ===")
    print("Using pre-generated ULP data\n")
    
    # Create output directory for analysis results
    output_dir = 'precision_ulp_analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}\n")
    
    precision_levels = ['53bit_cpu', '24bit_cpu', '24bit_a100', '24bit_mi250']
    
    # Calculate ULP differences for each target integral
    for integral_type in sorted(ulp_data.keys()):
        print(f"Analyzing {integral_type}...")
        
        # Create output file for this target integral
        integral_filename = f"{integral_type.lower()}_precision_analysis.txt"
        integral_filepath = os.path.join(output_dir, integral_filename)
        
        with open(integral_filepath, 'w') as f:
            f.write(f"=== Precision Level ULP Analysis for Target Integral: {integral_type} ===\n")
            f.write("Generated from pre-calculated ULP data\n")
            f.write("Baseline: Quad Precision (128-bit float)\n")
            f.write(f"Precision Levels: {', '.join(precision_levels)}\n\n")
            
            # Analyze each precision level for this integral
            for precision in precision_levels:
                integral_ulp_data = ulp_data[integral_type]
                
                if precision not in integral_ulp_data:
                    f.write(f"--- Precision Level: {precision} ---\n")
                    f.write(f"No data available for {precision}\n\n")
                    continue
                
                dataset_ulp_data = integral_ulp_data[precision]
                
                f.write(f"--- Precision Level: {precision} ---\n")
                
                # Collect ULP differences for this integral and precision level
                ulp_differences = {
                    'coeff1_real': dataset_ulp_data['Coeff1']['real'],
                    'coeff1_imag': dataset_ulp_data['Coeff1']['imag'],
                    'coeff2_real': dataset_ulp_data['Coeff2']['real'],
                    'coeff2_imag': dataset_ulp_data['Coeff2']['imag'],
                    'coeff3_real': dataset_ulp_data['Coeff3']['real'],
                    'coeff3_imag': dataset_ulp_data['Coeff3']['imag']
                }
                
                total_data_points = sum(len(ulps) for ulps in ulp_differences.values())
                f.write(f"Analyzing {total_data_points} data points for {precision}\n")
                f.write("\n")
                
                # Calculate and display thresholding statistics
                f.write(f"Thresholding Statistics for {precision} - {integral_type}\n")
                f.write("Coefficient | Real Thresholded | Imag Thresholded | Total Thresholded | Threshold %\n")
                f.write("-" * 80 + "\n")
                
                total_thresholded_all = 0
                total_comparisons_all = 0
                
                for coeff_name in ['Coeff1', 'Coeff2', 'Coeff3']:
                    thresholding_stats = dataset_ulp_data[coeff_name]['thresholding_stats']
                    
                    real_thresholded = thresholding_stats['real_thresholded']
                    imag_thresholded = thresholding_stats['imag_thresholded']
                    total_thresholded = real_thresholded + imag_thresholded
                    total_comparisons = thresholding_stats['total_comparisons']
                    threshold_percentage = (total_thresholded / total_comparisons * 100) if total_comparisons > 0 else 0
                    
                    f.write(f"{coeff_name:11} | {real_thresholded:15d} | {imag_thresholded:15d} | {total_thresholded:16d} | {threshold_percentage:10.1f}%\n")
                    
                    total_thresholded_all += total_thresholded
                    total_comparisons_all += total_comparisons
                
                overall_threshold_percentage = (total_thresholded_all / total_comparisons_all * 100) if total_comparisons_all > 0 else 0
                f.write("-" * 80 + "\n")
                f.write(f"{'Overall':11} | {'':15} | {'':15} | {total_thresholded_all:16d} | {overall_threshold_percentage:10.1f}%\n")
                f.write("\n")
                
                # Calculate statistics for each coefficient for this integral
                f.write(f"Statistical Summary (ULP) for {precision} - {integral_type}\n")
                f.write("Coefficient | Min ULP  | Max ULP  | Mean ULP | Median ULP | Std Dev\n")
                f.write("-" * 70 + "\n")
                
                for coeff_name, ulp_diffs in ulp_differences.items():
                    # Filter out NaN and infinite values
                    valid_ulps = [x for x in ulp_diffs if not (math.isnan(x) or math.isinf(x))]
                    
                    if valid_ulps:
                        min_ulp = min(valid_ulps)
                        max_ulp = max(valid_ulps)
                        mean_ulp = statistics.mean(valid_ulps)
                        median_ulp = statistics.median(valid_ulps)
                        std_ulp = statistics.stdev(valid_ulps) if len(valid_ulps) > 1 else 0
                        
                        f.write(f"{coeff_name:11} | {min_ulp:8.2f} | {max_ulp:8.2f} | {mean_ulp:8.2f} | {median_ulp:9.2f} | {std_ulp:8.2f}\n")
                    else:
                        f.write(f"{coeff_name:11} | All zeros\n")
                
                f.write("\n" + "="*70 + "\n\n")
                
                # Special handling for BIN1: Print test IDs with ULP error > 10^11
                if integral_type == 'BIN1':
                    f.write(f"Test IDs with ULP error > 10^11 for {precision} - {integral_type}\n")
                    f.write("-" * 80 + "\n")
                    
                    high_error_ids = []
                    threshold = 1e11
                    
                    for coeff_name in ['Coeff1', 'Coeff2', 'Coeff3']:
                        real_ulps = dataset_ulp_data[coeff_name]['real']
                        imag_ulps = dataset_ulp_data[coeff_name]['imag']
                        test_ids = dataset_ulp_data[coeff_name].get('test_ids', [])
                        
                        # Check real part
                        for i, (ulp_val, test_id) in enumerate(zip(real_ulps, test_ids)):
                            if ulp_val > threshold:
                                msg = f"Test ID {test_id}: {coeff_name} (real) = {ulp_val:.2e} ULP"
                                f.write(msg + "\n")
                                high_error_ids.append(msg)
                        
                        # Check imaginary part
                        for i, (ulp_val, test_id) in enumerate(zip(imag_ulps, test_ids)):
                            if ulp_val > threshold:
                                msg = f"Test ID {test_id}: {coeff_name} (imag) = {ulp_val:.2e} ULP"
                                f.write(msg + "\n")
                                high_error_ids.append(msg)
                    
                    if high_error_ids:
                        print(f"\n  *** BIN1 - {precision}: Found {len(high_error_ids)} test cases with ULP > 10^11 ***")
                        for msg in high_error_ids:
                            print(f"      {msg}")
                    else:
                        f.write("No test IDs found with ULP error > 10^11\n")
                    
                    f.write("\n" + "="*80 + "\n\n")
                
                # Find largest ULP differences for this integral
                f.write(f"Largest ULP Differences for {precision} - {integral_type}\n")
                
                # Find maximum ULP for each coefficient
                max_ulps = {}
                for coeff_name, ulp_diffs in ulp_differences.items():
                    valid_ulps = [x for x in ulp_diffs if not (math.isnan(x) or math.isinf(x))]
                    max_ulps[coeff_name] = max(valid_ulps) if valid_ulps else 0
                
                f.write("Maximum ULP differences by coefficient:\n")
                f.write("Coefficient | Max ULP\n")
                f.write("-" * 30 + "\n")
                for coeff_name, max_ulp in max_ulps.items():
                    f.write(f"{coeff_name:11} | {max_ulp:8.2f}\n")
                
                f.write("\n" + "="*30 + "\n\n")
                
                # Count how many values are within certain ULP thresholds for this integral
                f.write(f"ULP Threshold Analysis for {precision} - {integral_type}\n")
                thresholds = [0.5, 1.0, 2.0, 5.0, 10.0]
                for threshold in thresholds:
                    count_within = 0
                    total_count = 0
                    for coeff_name, ulp_diffs in ulp_differences.items():
                        valid_ulps = [x for x in ulp_diffs if not (math.isnan(x) or math.isinf(x))]
                        count_within += sum(1 for x in valid_ulps if x <= threshold)
                        total_count += len(valid_ulps)
                    
                    percentage = (count_within / total_count * 100) if total_count > 0 else 0
                    f.write(f"Values within {threshold:4.1f} ULP: {count_within:5d}/{total_count:5d} ({percentage:5.1f}%)\n")
                
                f.write("\n" + "="*120 + "\n\n")
        
        print(f"  Saved analysis to: {integral_filepath}")
    
    # Create a summary file with interpretation
    summary_filepath = os.path.join(output_dir, 'precision_analysis_summary.txt')
    with open(summary_filepath, 'w') as f:
        f.write("=== Precision Level ULP Analysis Summary ===\n")
        f.write("Generated analysis files for each target integral:\n")
        for integral_type in sorted(ulp_data.keys()):
            integral_filename = f"{integral_type.lower()}_precision_analysis.txt"
            f.write(f"  - {integral_filename}\n")
        f.write("\n")
        
        f.write("=== Precision Levels Analyzed ===\n")
        f.write("Baseline: Quad Precision (128-bit floating point)\n")
        f.write("Precision Levels:\n")
        f.write("  - 53bit_cpu: CPU FP64 (double precision)\n")
        f.write("  - 24bit_cpu: CPU FP32 (single precision)\n")
        f.write("  - 24bit_a100: A100 GPU FP32\n")
        f.write("  - 24bit_mi250: MI250 GPU FP32\n")
        f.write("\n")
        
        f.write("=== ULP Interpretation ===\n")
        f.write("ULP (Units in the Last Place) measures the spacing between consecutive floating-point numbers.\n")
        f.write("For double precision:\n")
        f.write("- 1 ULP = smallest representable difference at that magnitude\n")
        f.write("- Values within 0.5 ULP are considered 'exactly representable'\n")
        f.write("- Values within 1-2 ULP are considered 'very close'\n")
        f.write("- Values > 10 ULP may indicate significant precision loss\n")
        f.write("\n")
        f.write("=== ULP Thresholding Applied ===\n")
        f.write("To prevent ULP explosion near zero, differences are set to 0 when:\n")
        f.write("FP64 ULP > |Baseline value|\n")
        f.write("This occurs when the baseline value is below the noise floor of double precision.\n")
        f.write("The thresholding statistics show the percentage of comparisons where this protection\n")
        f.write("was applied, helping identify integrals with many values near zero.\n")
        f.write("\n")
        f.write("=== Analysis Structure ===\n")
        f.write("Each target integral file contains analysis for all precision levels:\n")
        f.write("1. Thresholding Statistics - Shows how often values near zero required protection\n")
        f.write("2. Statistical Summary - Min, Max, Mean, Median, Std Dev for each coefficient\n")
        f.write("3. Largest ULP Differences - Maximum ULP values by coefficient\n")
        f.write("4. ULP Threshold Analysis - Percentage of values within various ULP thresholds\n")
        f.write("\n")
        f.write("=== Expected Trends ===\n")
        f.write("As precision changes, we expect:\n")
        f.write("- FP32 (24bit CPU/A100/MI250): Higher ULP differences due to lower precision\n")
        f.write("- FP64 (53bit CPU): Lower ULP differences, closer to quad precision baseline\n")
        f.write("- GPU implementations may show additional differences due to hardware-specific optimizations\n")
    
    print(f"Created summary file: {summary_filepath}")
    print("\n=== Precision Level ULP Analysis Complete ===")
    print("All analysis results have been saved to text files in the 'precision_ulp_analysis_results' directory.")
    print("Each target integral has its own detailed analysis file with all precision levels, plus a summary file with interpretation.")

def main():
    # Check if ULP data exists, if not generate it
    data_dir = 'precision_ulp_data'
    ulp_data_file = os.path.join(data_dir, 'ulp_data.pkl')
    
    if not os.path.exists(ulp_data_file):
        print("ULP data not found. Generating data first...")
        ulp_data = generate_ulp_data()
    else:
        print("Loading existing ULP data...")
        ulp_data = load_ulp_data()
        
        # Check if cached data includes thresholding statistics
        if not check_data_completeness(ulp_data):
            print("Warning: Cached ULP data is missing thresholding statistics.")
            print("This indicates the data was generated before thresholding statistics were added.")
            print("Regenerating data to include thresholding statistics...")
            ulp_data = regenerate_data()
    
    # Calculate ULP differences using cached data
    calculate_ulp_differences(ulp_data)
    
    print(f"\nULP data is cached in: {ulp_data_file}")
    print("To regenerate data, delete the precision_ulp_data directory and run again.")

if __name__ == "__main__":
    main()




