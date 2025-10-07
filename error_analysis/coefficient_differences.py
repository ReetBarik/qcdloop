#!/usr/bin/env python3
"""
Script to calculate differences between fp64 implementations and quad precision baseline
"""

from csv_parser import QCDLoopCSVParser
import statistics

def calculate_differences():
    parser = QCDLoopCSVParser()
    
    print("=== Coefficient Differences Analysis ===")
    print("Baseline: box_cpu_fp128.csv (quad precision)")
    print("Comparing against: box_cpu_fp64.csv, box_a100_fp64.csv, box_mi250_fp64.csv\n")
    
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
    
    # Ensure all datasets have the same number of rows
    min_rows = min(len(baseline_data), min(len(data) for data in datasets.values()))
    if len(baseline_data) != min_rows or any(len(data) != min_rows for data in datasets.values()):
        print(f"Warning: Different number of rows. Using first {min_rows} rows from each file.\n")
    
    # Calculate differences for each dataset
    for dataset_name, dataset in datasets.items():
        print(f"=== Analysis for {dataset_name} ===")
        
        differences = {
            'coeff1_real': [],
            'coeff1_imag': [],
            'coeff2_real': [],
            'coeff2_imag': [],
            'coeff3_real': [],
            'coeff3_imag': []
        }
        
        print("Calculating differences for each row...")
        print("Row | Coeff1 Real Diff | Coeff1 Imag Diff | Coeff2 Real Diff | Coeff2 Imag Diff | Coeff3 Real Diff | Coeff3 Imag Diff")
        print("-" * 120)
        
        for i in range(min_rows):
            baseline_row = baseline_data[i]
            dataset_row = dataset[i]
            
            # Calculate differences
            coeff1_real_diff = abs(dataset_row.coeff1_real - baseline_row.coeff1_real)
            coeff1_imag_diff = abs(dataset_row.coeff1_imag - baseline_row.coeff1_imag)
            coeff2_real_diff = abs(dataset_row.coeff2_real - baseline_row.coeff2_real)
            coeff2_imag_diff = abs(dataset_row.coeff2_imag - baseline_row.coeff2_imag)
            coeff3_real_diff = abs(dataset_row.coeff3_real - baseline_row.coeff3_real)
            coeff3_imag_diff = abs(dataset_row.coeff3_imag - baseline_row.coeff3_imag)
            
            # Store differences
            differences['coeff1_real'].append(coeff1_real_diff)
            differences['coeff1_imag'].append(coeff1_imag_diff)
            differences['coeff2_real'].append(coeff2_real_diff)
            differences['coeff2_imag'].append(coeff2_imag_diff)
            differences['coeff3_real'].append(coeff3_real_diff)
            differences['coeff3_imag'].append(coeff3_imag_diff)
            
            # Print first 10 rows and every 100th row after that
            if i < 10 or i % 100 == 0:
                print(f"{i+1:3d} | {coeff1_real_diff:15.2e} | {coeff1_imag_diff:15.2e} | {coeff2_real_diff:15.2e} | {coeff2_imag_diff:15.2e} | {coeff3_real_diff:15.2e} | {coeff3_imag_diff:15.2e}")
        
        print("\n" + "="*120 + "\n")
        
        # Calculate statistics for each coefficient
        print(f"=== Statistical Summary for {dataset_name} ===")
        print("Coefficient | Min Diff    | Max Diff    | Mean Diff   | Median Diff | Std Dev")
        print("-" * 70)
        
        for coeff_name, diffs in differences.items():
            if diffs:  # Only calculate if there are non-zero differences
                min_diff = min(diffs)
                max_diff = max(diffs)
                mean_diff = statistics.mean(diffs)
                median_diff = statistics.median(diffs)
                std_diff = statistics.stdev(diffs) if len(diffs) > 1 else 0
                
                print(f"{coeff_name:11} | {min_diff:10.2e} | {max_diff:10.2e} | {mean_diff:10.2e} | {median_diff:10.2e} | {std_diff:10.2e}")
            else:
                print(f"{coeff_name:11} | All zeros")
        
        print("\n" + "="*70 + "\n")
        
        # Find rows with largest differences
        print(f"=== Rows with Largest Differences for {dataset_name} ===")
        
        # Create a list of (row_index, max_diff) tuples
        max_diffs_per_row = []
        for i in range(min_rows):
            max_diff = max([
                differences['coeff1_real'][i],
                differences['coeff1_imag'][i],
                differences['coeff2_real'][i],
                differences['coeff2_imag'][i],
                differences['coeff3_real'][i],
                differences['coeff3_imag'][i]
            ])
            max_diffs_per_row.append((i, max_diff))
        
        # Sort by maximum difference (descending)
        max_diffs_per_row.sort(key=lambda x: x[1], reverse=True)
        
        print("Top 10 rows with largest differences:")
        print("Row | Max Diff    | Coeff1 Real | Coeff1 Imag | Coeff2 Real | Coeff2 Imag | Coeff3 Real | Coeff3 Imag")
        print("-" * 100)
        
        for i, (row_idx, max_diff) in enumerate(max_diffs_per_row[:10]):
            coeff1_real_diff = differences['coeff1_real'][row_idx]
            coeff1_imag_diff = differences['coeff1_imag'][row_idx]
            coeff2_real_diff = differences['coeff2_real'][row_idx]
            coeff2_imag_diff = differences['coeff2_imag'][row_idx]
            coeff3_real_diff = differences['coeff3_real'][row_idx]
            coeff3_imag_diff = differences['coeff3_imag'][row_idx]
            
            print(f"{row_idx+1:3d} | {max_diff:10.2e} | {coeff1_real_diff:10.2e} | {coeff1_imag_diff:10.2e} | {coeff2_real_diff:10.2e} | {coeff2_imag_diff:10.2e} | {coeff3_real_diff:10.2e} | {coeff3_imag_diff:10.2e}")
        
        print("\n" + "="*100 + "\n")
        
        # Show some example rows with their actual values
        print(f"=== Example Rows with Values for {dataset_name} ===")
        for i in [0, 1, 2]:  # Show first 3 rows
            if i < min_rows:
                baseline_row = baseline_data[i]
                dataset_row = dataset[i]
                
                print(f"\nRow {i+1}:")
                print(f"  Target Integral: {dataset_row.target_integral}")
                print(f"  Test ID: {dataset_row.test_id}")
                print(f"  Coeff1 Real:")
                print(f"    {dataset_name}: {dataset_row.coeff1_real:.33f}")
                print(f"    Baseline:     {baseline_row.coeff1_real:.33f}")
                print(f"    Diff:         {differences['coeff1_real'][i]:.2e}")
                print(f"  Coeff1 Imag:")
                print(f"    {dataset_name}: {dataset_row.coeff1_imag:.33f}")
                print(f"    Baseline:     {baseline_row.coeff1_imag:.33f}")
                print(f"    Diff:         {differences['coeff1_imag'][i]:.2e}")
        
        print("\n" + "="*120 + "\n")

def main():
    calculate_differences()

if __name__ == "__main__":
    main()
