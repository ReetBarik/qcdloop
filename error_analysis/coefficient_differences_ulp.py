#!/usr/bin/env python3
"""
Script to calculate differences between fp64 implementations and quad precision baseline in ULP
Simplified version that focuses on double precision ULP analysis
"""

from csv_parser import QCDLoopCSVParser
import statistics
import struct
import math

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

def calculate_ulp_differences():
    parser = QCDLoopCSVParser()
    
    print("=== Coefficient Differences Analysis in ULP ===")
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
    
    # Calculate ULP differences for each dataset
    for dataset_name, dataset in datasets.items():
        print(f"=== ULP Analysis for {dataset_name} ===")
        
        # Calculate ULP differences for each row
        ulp_differences = {
            'coeff1_real': [],
            'coeff1_imag': [],
            'coeff2_real': [],
            'coeff2_imag': [],
            'coeff3_real': [],
            'coeff3_imag': []
        }
        
        # Also store absolute differences for comparison
        abs_differences = {
            'coeff1_real': [],
            'coeff1_imag': [],
            'coeff2_real': [],
            'coeff2_imag': [],
            'coeff3_real': [],
            'coeff3_imag': []
        }
        
        print("Calculating ULP differences for each row...")
        print("Row | Coeff1 Real ULP | Coeff1 Imag ULP | Coeff2 Real ULP | Coeff2 Imag ULP | Coeff3 Real ULP | Coeff3 Imag ULP")
        print("-" * 120)
        
        for i in range(min_rows):
            baseline_row = baseline_data[i]
            dataset_row = dataset[i]
            
            # Calculate absolute differences with ULP thresholding
            # Set to zero if FP64 ULP > |Quad| to prevent ULP explosion near zero
            fp64_ulp_c1_real = ulp_double(dataset_row.coeff1_real)
            fp64_ulp_c1_imag = ulp_double(dataset_row.coeff1_imag)
            fp64_ulp_c2_real = ulp_double(dataset_row.coeff2_real)
            fp64_ulp_c2_imag = ulp_double(dataset_row.coeff2_imag)
            fp64_ulp_c3_real = ulp_double(dataset_row.coeff3_real)
            fp64_ulp_c3_imag = ulp_double(dataset_row.coeff3_imag)
            
            coeff1_real_abs = 0.0 if fp64_ulp_c1_real > abs(baseline_row.coeff1_real) else abs(dataset_row.coeff1_real - baseline_row.coeff1_real)
            coeff1_imag_abs = 0.0 if fp64_ulp_c1_imag > abs(baseline_row.coeff1_imag) else abs(dataset_row.coeff1_imag - baseline_row.coeff1_imag)
            coeff2_real_abs = 0.0 if fp64_ulp_c2_real > abs(baseline_row.coeff2_real) else abs(dataset_row.coeff2_real - baseline_row.coeff2_real)
            coeff2_imag_abs = 0.0 if fp64_ulp_c2_imag > abs(baseline_row.coeff2_imag) else abs(dataset_row.coeff2_imag - baseline_row.coeff2_imag)
            coeff3_real_abs = 0.0 if fp64_ulp_c3_real > abs(baseline_row.coeff3_real) else abs(dataset_row.coeff3_real - baseline_row.coeff3_real)
            coeff3_imag_abs = 0.0 if fp64_ulp_c3_imag > abs(baseline_row.coeff3_imag) else abs(dataset_row.coeff3_imag - baseline_row.coeff3_imag)
            
            # Calculate ULP differences using pre-calculated ULP values
            # If abs_diff is 0 (due to thresholding), ULP is also 0
            coeff1_real_ulp = coeff1_real_abs / fp64_ulp_c1_real if dataset_row.coeff1_real != 0 and fp64_ulp_c1_real > 0 else 0
            coeff1_imag_ulp = coeff1_imag_abs / fp64_ulp_c1_imag if dataset_row.coeff1_imag != 0 and fp64_ulp_c1_imag > 0 else 0
            coeff2_real_ulp = coeff2_real_abs / fp64_ulp_c2_real if dataset_row.coeff2_real != 0 and fp64_ulp_c2_real > 0 else 0
            coeff2_imag_ulp = coeff2_imag_abs / fp64_ulp_c2_imag if dataset_row.coeff2_imag != 0 and fp64_ulp_c2_imag > 0 else 0
            coeff3_real_ulp = coeff3_real_abs / fp64_ulp_c3_real if dataset_row.coeff3_real != 0 and fp64_ulp_c3_real > 0 else 0
            coeff3_imag_ulp = coeff3_imag_abs / fp64_ulp_c3_imag if dataset_row.coeff3_imag != 0 and fp64_ulp_c3_imag > 0 else 0
            
            # Store differences
            ulp_differences['coeff1_real'].append(coeff1_real_ulp)
            ulp_differences['coeff1_imag'].append(coeff1_imag_ulp)
            ulp_differences['coeff2_real'].append(coeff2_real_ulp)
            ulp_differences['coeff2_imag'].append(coeff2_imag_ulp)
            ulp_differences['coeff3_real'].append(coeff3_real_ulp)
            ulp_differences['coeff3_imag'].append(coeff3_imag_ulp)
            
            abs_differences['coeff1_real'].append(coeff1_real_abs)
            abs_differences['coeff1_imag'].append(coeff1_imag_abs)
            abs_differences['coeff2_real'].append(coeff2_real_abs)
            abs_differences['coeff2_imag'].append(coeff2_imag_abs)
            abs_differences['coeff3_real'].append(coeff3_real_abs)
            abs_differences['coeff3_imag'].append(coeff3_imag_abs)
            
            # Print first 10 rows and every 100th row after that
            if i < 10 or i % 100 == 0:
                print(f"{i+1:3d} | {coeff1_real_ulp:15.2f} | {coeff1_imag_ulp:15.2f} | {coeff2_real_ulp:15.2f} | {coeff2_imag_ulp:15.2f} | {coeff3_real_ulp:15.2f} | {coeff3_imag_ulp:15.2f}")
        
        print("\n" + "="*120 + "\n")
        
        # Calculate statistics for each coefficient
        print(f"=== Statistical Summary (ULP) for {dataset_name} ===")
        print("Coefficient | Min ULP  | Max ULP  | Mean ULP | Median ULP | Std Dev")
        print("-" * 70)
        
        for coeff_name, ulp_diffs in ulp_differences.items():
            # Filter out NaN and infinite values
            valid_ulps = [x for x in ulp_diffs if not (math.isnan(x) or math.isinf(x))]
            
            if valid_ulps:
                min_ulp = min(valid_ulps)
                max_ulp = max(valid_ulps)
                mean_ulp = statistics.mean(valid_ulps)
                median_ulp = statistics.median(valid_ulps)
                std_ulp = statistics.stdev(valid_ulps) if len(valid_ulps) > 1 else 0
                
                print(f"{coeff_name:11} | {min_ulp:8.2f} | {max_ulp:8.2f} | {mean_ulp:8.2f} | {median_ulp:9.2f} | {std_ulp:8.2f}")
            else:
                print(f"{coeff_name:11} | All zeros")
        
        print("\n" + "="*70 + "\n")
        
        # Find rows with largest ULP differences
        print(f"=== Rows with Largest ULP Differences for {dataset_name} ===")
        
        # Create a list of (row_index, max_ulp) tuples
        max_ulps_per_row = []
        for i in range(min_rows):
            max_ulp = max([
                ulp_differences['coeff1_real'][i] if not (math.isnan(ulp_differences['coeff1_real'][i]) or math.isinf(ulp_differences['coeff1_real'][i])) else 0,
                ulp_differences['coeff1_imag'][i] if not (math.isnan(ulp_differences['coeff1_imag'][i]) or math.isinf(ulp_differences['coeff1_imag'][i])) else 0,
                ulp_differences['coeff2_real'][i] if not (math.isnan(ulp_differences['coeff2_real'][i]) or math.isinf(ulp_differences['coeff2_real'][i])) else 0,
                ulp_differences['coeff2_imag'][i] if not (math.isnan(ulp_differences['coeff2_imag'][i]) or math.isinf(ulp_differences['coeff2_imag'][i])) else 0,
                ulp_differences['coeff3_real'][i] if not (math.isnan(ulp_differences['coeff3_real'][i]) or math.isinf(ulp_differences['coeff3_real'][i])) else 0,
                ulp_differences['coeff3_imag'][i] if not (math.isnan(ulp_differences['coeff3_imag'][i]) or math.isinf(ulp_differences['coeff3_imag'][i])) else 0
            ])
            max_ulps_per_row.append((i, max_ulp))
        
        # Sort by maximum ULP (descending)
        max_ulps_per_row.sort(key=lambda x: x[1], reverse=True)
        
        print("Top 10 rows with largest ULP differences:")
        print("Row | Max ULP  | Coeff1 Real | Coeff1 Imag | Coeff2 Real | Coeff2 Imag | Coeff3 Real | Coeff3 Imag")
        print("-" * 100)
        
        for i, (row_idx, max_ulp) in enumerate(max_ulps_per_row[:10]):
            coeff1_real_ulp = ulp_differences['coeff1_real'][row_idx]
            coeff1_imag_ulp = ulp_differences['coeff1_imag'][row_idx]
            coeff2_real_ulp = ulp_differences['coeff2_real'][row_idx]
            coeff2_imag_ulp = ulp_differences['coeff2_imag'][row_idx]
            coeff3_real_ulp = ulp_differences['coeff3_real'][row_idx]
            coeff3_imag_ulp = ulp_differences['coeff3_imag'][row_idx]
            
            print(f"{row_idx+1:3d} | {max_ulp:8.2f} | {coeff1_real_ulp:11.2f} | {coeff1_imag_ulp:11.2f} | {coeff2_real_ulp:11.2f} | {coeff2_imag_ulp:11.2f} | {coeff3_real_ulp:11.2f} | {coeff3_imag_ulp:11.2f}")
        
        print("\n" + "="*100 + "\n")
        
        # Show some example rows with their actual values and ULP analysis
        print(f"=== Example Rows with ULP Analysis for {dataset_name} ===")
        for i in [0, 1, 2]:  # Show first 3 rows
            if i < min_rows:
                baseline_row = baseline_data[i]
                dataset_row = dataset[i]
                
                print(f"\nRow {i+1}:")
                print(f"  Target Integral: {dataset_row.target_integral}")
                print(f"  Test ID: {dataset_row.test_id}")
                
                # Coeff1 analysis
                real_abs = abs_differences['coeff1_real'][i]
                imag_abs = abs_differences['coeff1_imag'][i]
                real_ulp = ulp_differences['coeff1_real'][i]
                imag_ulp = ulp_differences['coeff1_imag'][i]
                
                print(f"  Coeff1 Real:")
                print(f"    {dataset_name}: {dataset_row.coeff1_real:.33f}")
                print(f"    Baseline:     {baseline_row.coeff1_real:.33f}")
                print(f"    Abs Diff:     {real_abs:.2e}")
                print(f"    ULP Diff:     {real_ulp:.2f} ULP")
                print(f"    FP64 ULP:     {ulp_double(dataset_row.coeff1_real):.2e}")
                
                print(f"  Coeff1 Imag:")
                print(f"    {dataset_name}: {dataset_row.coeff1_imag:.33f}")
                print(f"    Baseline:     {baseline_row.coeff1_imag:.33f}")
                print(f"    Abs Diff:     {imag_abs:.2e}")
                print(f"    ULP Diff:     {imag_ulp:.2f} ULP")
                print(f"    FP64 ULP:     {ulp_double(dataset_row.coeff1_imag):.2e}")
                
                # Show thresholding information
                if abs(baseline_row.coeff1_real) > 0 and ulp_double(dataset_row.coeff1_real) > abs(baseline_row.coeff1_real):
                    print(f"    Note: Real difference set to 0 (FP64 ULP > |Baseline|)")
                if abs(baseline_row.coeff1_imag) > 0 and ulp_double(dataset_row.coeff1_imag) > abs(baseline_row.coeff1_imag):
                    print(f"    Note: Imag difference set to 0 (FP64 ULP > |Baseline|)")
        
        print("\n" + "="*120 + "\n")
        
        # Count how many values are within certain ULP thresholds
        print(f"=== ULP Threshold Analysis for {dataset_name} ===")
        thresholds = [0.5, 1.0, 2.0, 5.0, 10.0]
        for threshold in thresholds:
            count_within = 0
            total_count = 0
            for coeff_name, ulp_diffs in ulp_differences.items():
                valid_ulps = [x for x in ulp_diffs if not (math.isnan(x) or math.isinf(x))]
                count_within += sum(1 for x in valid_ulps if x <= threshold)
                total_count += len(valid_ulps)
            
            percentage = (count_within / total_count * 100) if total_count > 0 else 0
            print(f"Values within {threshold:4.1f} ULP: {count_within:5d}/{total_count:5d} ({percentage:5.1f}%)")
        
        print("\n" + "="*120 + "\n")
    
    # ULP interpretation
    print("=== ULP Interpretation ===")
    print("ULP (Units in the Last Place) measures the spacing between consecutive floating-point numbers.")
    print("For double precision:")
    print("- 1 ULP = smallest representable difference at that magnitude")
    print("- Values within 0.5 ULP are considered 'exactly representable'")
    print("- Values within 1-2 ULP are considered 'very close'")
    print("- Values > 10 ULP may indicate significant precision loss")
    print()
    print("=== ULP Thresholding Applied ===")
    print("To prevent ULP explosion near zero, differences are set to 0 when:")
    print("FP64 ULP > |Baseline value|")
    print("This occurs when the baseline value is below the noise floor of double precision.")
    print("Such cases are marked with 'Note: difference set to 0' in the output.")
    print()

def main():
    calculate_ulp_differences()

if __name__ == "__main__":
    main()
