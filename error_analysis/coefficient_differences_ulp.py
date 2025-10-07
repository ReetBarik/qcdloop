#!/usr/bin/env python3
"""
Script to calculate differences between fp64 implementations and quad precision baseline in ULP
Simplified version that focuses on double precision ULP analysis per target integral

Workflow:
1. First run: Generates ULP data from CSV files and saves to ulp_data/ulp_data.pkl
2. Subsequent runs: Loads pre-generated data from pickle file for fast analysis
3. To regenerate data: Delete ulp_data/ directory or call regenerate_data() function

This separation allows for fast iteration on analysis logic without regenerating data each time.

The analysis is performed per target integral, providing granular statistics for each specific
integral type rather than aggregating across all data points.
"""

from csv_parser import QCDLoopCSVParser
import statistics
import struct
import math
import os
import pickle
from collections import defaultdict

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
    """Generate ULP data and save to pickle file"""
    parser = QCDLoopCSVParser()
    
    print("=== Generating ULP Data ===")
    print("Baseline: box_cpu_fp128.csv (quad precision)")
    print("Comparing against: box_cpu_fp64.csv, box_a100_fp64.csv, box_mi250_fp64.csv\n")
    
    # Create output directory
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
            'CPU FP64': {'Coeff1': {'real': [], 'imag': [], 'thresholding_stats': {}}, 'Coeff2': {'real': [], 'imag': [], 'thresholding_stats': {}}, 'Coeff3': {'real': [], 'imag': [], 'thresholding_stats': {}}},
            'A100 FP64': {'Coeff1': {'real': [], 'imag': [], 'thresholding_stats': {}}, 'Coeff2': {'real': [], 'imag': [], 'thresholding_stats': {}}, 'Coeff3': {'real': [], 'imag': [], 'thresholding_stats': {}}},
            'MI250 FP64': {'Coeff1': {'real': [], 'imag': [], 'thresholding_stats': {}}, 'Coeff2': {'real': [], 'imag': [], 'thresholding_stats': {}}, 'Coeff3': {'real': [], 'imag': [], 'thresholding_stats': {}}}
        }
        
        implementation_rows = [cpu_rows, a100_rows, mi250_rows]
        implementation_names = ['CPU FP64', 'A100 FP64', 'MI250 FP64']
        
        for impl_name, impl_rows in zip(implementation_names, implementation_rows):
            for coeff_name in ['Coeff1', 'Coeff2', 'Coeff3']:
                real_ulps = []
                imag_ulps = []
                
                # Track thresholding statistics
                thresholding_stats = {'real_thresholded': 0, 'imag_thresholded': 0, 'total_comparisons': 0}
                
                for baseline_row, impl_row in zip(baseline_rows, impl_rows):
                    # Calculate absolute differences with ULP thresholding
                    # Set to zero if FP64 ULP > |Baseline| to prevent ULP explosion near zero
                    fp64_ulp_c_real = ulp_double(impl_row.coeff1_real if coeff_name == 'Coeff1' else impl_row.coeff2_real if coeff_name == 'Coeff2' else impl_row.coeff3_real)
                    fp64_ulp_c_imag = ulp_double(impl_row.coeff1_imag if coeff_name == 'Coeff1' else impl_row.coeff2_imag if coeff_name == 'Coeff2' else impl_row.coeff3_imag)
                    
                    baseline_real = baseline_row.coeff1_real if coeff_name == 'Coeff1' else baseline_row.coeff2_real if coeff_name == 'Coeff2' else baseline_row.coeff3_real
                    baseline_imag = baseline_row.coeff1_imag if coeff_name == 'Coeff1' else baseline_row.coeff2_imag if coeff_name == 'Coeff2' else baseline_row.coeff3_imag
                    
                    impl_real = impl_row.coeff1_real if coeff_name == 'Coeff1' else impl_row.coeff2_real if coeff_name == 'Coeff2' else impl_row.coeff3_real
                    impl_imag = impl_row.coeff1_imag if coeff_name == 'Coeff1' else impl_row.coeff2_imag if coeff_name == 'Coeff2' else impl_row.coeff3_imag
                    
                    # Check if thresholding is applied
                    real_thresholded = fp64_ulp_c_real > abs(baseline_real)
                    imag_thresholded = fp64_ulp_c_imag > abs(baseline_imag)
                    
                    # Track thresholding statistics
                    thresholding_stats['total_comparisons'] += 2  # real and imaginary
                    if real_thresholded:
                        thresholding_stats['real_thresholded'] += 1
                    if imag_thresholded:
                        thresholding_stats['imag_thresholded'] += 1
                    
                    real_abs = 0.0 if real_thresholded else abs(impl_real - baseline_real)
                    imag_abs = 0.0 if imag_thresholded else abs(impl_imag - baseline_imag)
                    
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
                integral_ulp_data[impl_name][coeff_name]['thresholding_stats'] = thresholding_stats
        
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

def check_data_completeness(ulp_data):
    """Check if cached ULP data includes thresholding statistics"""
    for integral_type in ulp_data.keys():
        for impl_name in ['CPU FP64', 'A100 FP64', 'MI250 FP64']:
            for coeff_name in ['Coeff1', 'Coeff2', 'Coeff3']:
                if 'thresholding_stats' not in ulp_data[integral_type][impl_name][coeff_name]:
                    return False
    return True

def regenerate_data():
    """Force regenerate ULP data (useful for testing or when data changes)"""
    import shutil
    
    data_dir = 'ulp_data'
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
    
    print("=== Coefficient Differences Analysis in ULP ===")
    print("Using pre-generated ULP data\n")
    
    # Create output directory for analysis results
    output_dir = 'ulp_analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}\n")
    
    # Calculate ULP differences for each target integral
    for integral_type in sorted(ulp_data.keys()):
        print(f"Analyzing {integral_type}...")
        
        # Create output file for this target integral
        integral_filename = f"{integral_type.lower()}_analysis.txt"
        integral_filepath = os.path.join(output_dir, integral_filename)
        
        with open(integral_filepath, 'w') as f:
            f.write(f"=== ULP Analysis for Target Integral: {integral_type} ===\n")
            f.write("Generated from pre-calculated ULP data\n\n")
            
            # Analyze each implementation for this integral
            for dataset_name in ['CPU FP64', 'A100 FP64', 'MI250 FP64']:
                integral_ulp_data = ulp_data[integral_type]
                dataset_ulp_data = integral_ulp_data[dataset_name]
                
                f.write(f"--- Implementation: {dataset_name} ---\n")
                
                # Collect ULP differences for this integral and implementation
                ulp_differences = {
                    'coeff1_real': dataset_ulp_data['Coeff1']['real'],
                    'coeff1_imag': dataset_ulp_data['Coeff1']['imag'],
                    'coeff2_real': dataset_ulp_data['Coeff2']['real'],
                    'coeff2_imag': dataset_ulp_data['Coeff2']['imag'],
                    'coeff3_real': dataset_ulp_data['Coeff3']['real'],
                    'coeff3_imag': dataset_ulp_data['Coeff3']['imag']
                }
                
                total_data_points = sum(len(ulps) for ulps in ulp_differences.values())
                f.write(f"Analyzing {total_data_points} data points for {dataset_name}\n")
                f.write("\n")
                
                # Calculate and display thresholding statistics
                f.write(f"Thresholding Statistics for {dataset_name} - {integral_type}\n")
                f.write("Coefficient | Real Thresholded | Imag Thresholded | Total Thresholded | Threshold %\n")
                f.write("-" * 80 + "\n")
                
                total_thresholded_all = 0
                total_comparisons_all = 0
                
                for coeff_name in ['Coeff1', 'Coeff2', 'Coeff3']:
                    dataset_ulp_data = integral_ulp_data[dataset_name]
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
                f.write(f"Statistical Summary (ULP) for {dataset_name} - {integral_type}\n")
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
                
                # Find largest ULP differences for this integral
                f.write(f"Largest ULP Differences for {dataset_name} - {integral_type}\n")
                
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
                f.write(f"ULP Threshold Analysis for {dataset_name} - {integral_type}\n")
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
    summary_filepath = os.path.join(output_dir, 'analysis_summary.txt')
    with open(summary_filepath, 'w') as f:
        f.write("=== ULP Analysis Summary ===\n")
        f.write("Generated analysis files for each target integral:\n")
        for integral_type in sorted(ulp_data.keys()):
            integral_filename = f"{integral_type.lower()}_analysis.txt"
            f.write(f"  - {integral_filename}\n")
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
        f.write("Each target integral file contains analysis for all implementations:\n")
        f.write("1. Thresholding Statistics - Shows how often values near zero required protection\n")
        f.write("2. Statistical Summary - Min, Max, Mean, Median, Std Dev for each coefficient\n")
        f.write("3. Largest ULP Differences - Maximum ULP values by coefficient\n")
        f.write("4. ULP Threshold Analysis - Percentage of values within various ULP thresholds\n")
    
    print(f"Created summary file: {summary_filepath}")
    print("\n=== ULP Analysis Complete ===")
    print("All analysis results have been saved to text files in the 'ulp_analysis_results' directory.")
    print("Each target integral has its own detailed analysis file with all implementations, plus a summary file with interpretation.")

def main():
    # Check if ULP data exists, if not generate it
    data_dir = 'ulp_data'
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
    print("To regenerate data, delete the ulp_data directory and run again.")

if __name__ == "__main__":
    main()
