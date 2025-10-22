#!/usr/bin/env python3
"""
Temporary script to update precise digits in existing pickle file
Uses the updated calculate_precise_digits logic without regenerating all data
"""

import os
import pickle
import math
from csv_parser import QCDLoopCSVParser

def calculate_precise_digits_updated(true_value, absolute_error, precision_level):
    """Updated calculate_precise_digits function with proper capping"""
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

def update_precise_digits_in_pickle():
    """Update precise digits using existing absolute differences and baseline data only"""
    
    # Path to the pickle file
    data_file = 'precision_analysis_data/all_precision_data.pkl'
    
    if not os.path.exists(data_file):
        print(f"Error: Pickle file not found: {data_file}")
        return
    
    print(f"Loading existing data from: {data_file}")
    
    # Load existing data
    with open(data_file, 'rb') as f:
        all_data = pickle.load(f)
    
    print("Loaded data successfully!")
    print(f"Found {len(all_data['precise_digits'])} target integrals")
    
    # Load only baseline data for true values
    parser = QCDLoopCSVParser()
    baseline_path = '/home/rbarik/qcdloop/error_analysis/float/raw/box_cpu_fp128.csv'
    print("Loading baseline data...")
    baseline_data = parser.parse_box_run_0(baseline_path)
    print(f"Loaded {len(baseline_data)} baseline rows")
    
    # Group baseline data by target integral
    from collections import defaultdict
    baseline_by_integral = defaultdict(list)
    for baseline_row in baseline_data:
        integral_type = baseline_row.target_integral
        baseline_by_integral[integral_type].append(baseline_row)
    
    # Define precision levels
    precision_levels = ['53bit_cpu', '24bit_cpu', '24bit_simulated', 
                       '53bit_a100', '24bit_a100', '53bit_mi250', '24bit_mi250']
    
    print("\nUpdating precise digits using existing absolute differences...")
    
    # Update precise digits for each integral
    for integral_type in sorted(all_data['precise_digits'].keys()):
        baseline_rows = baseline_by_integral[integral_type]
        
        print(f"Processing {integral_type}...")
        
        for precision in precision_levels:
            if precision not in all_data['precise_digits'][integral_type]:
                continue
            
            # Get existing absolute differences
            abs_diff_data = all_data['absolute_differences'][integral_type][precision]
            
            for coeff_name in ['Coeff1', 'Coeff2', 'Coeff3']:
                # Get existing absolute differences
                real_abs_diffs = abs_diff_data[coeff_name]['real']
                imag_abs_diffs = abs_diff_data[coeff_name]['imag']
                
                # Get corresponding baseline values
                real_baseline_values = []
                imag_baseline_values = []
                
                for baseline_row in baseline_rows:
                    baseline_real = baseline_row.coeff1_real if coeff_name == 'Coeff1' else baseline_row.coeff2_real if coeff_name == 'Coeff2' else baseline_row.coeff3_real
                    baseline_imag = baseline_row.coeff1_imag if coeff_name == 'Coeff1' else baseline_row.coeff2_imag if coeff_name == 'Coeff2' else baseline_row.coeff3_imag
                    real_baseline_values.append(baseline_real)
                    imag_baseline_values.append(baseline_imag)
                
                # Calculate new precise digits using existing absolute differences
                real_precise_digits_list = []
                imag_precise_digits_list = []
                
                # Process real values
                for i, (baseline_real, abs_diff) in enumerate(zip(real_baseline_values, real_abs_diffs)):
                    precise_digit = calculate_precise_digits_updated(baseline_real, abs_diff, precision)
                    if not math.isnan(precise_digit):
                        real_precise_digits_list.append(precise_digit)
                
                # Process imaginary values
                for i, (baseline_imag, abs_diff) in enumerate(zip(imag_baseline_values, imag_abs_diffs)):
                    precise_digit = calculate_precise_digits_updated(baseline_imag, abs_diff, precision)
                    if not math.isnan(precise_digit):
                        imag_precise_digits_list.append(precise_digit)
                
                # Update the data structure
                all_data['precise_digits'][integral_type][precision][coeff_name]['real'] = real_precise_digits_list
                all_data['precise_digits'][integral_type][precision][coeff_name]['imag'] = imag_precise_digits_list
    
    # Save updated data
    print(f"\nSaving updated data to: {data_file}")
    with open(data_file, 'wb') as f:
        pickle.dump(all_data, f)
    
    print("Update complete!")
    
    # Print some statistics
    print("\nUpdated precise digits statistics:")
    for integral_type in sorted(all_data['precise_digits'].keys()):
        print(f"\n{integral_type}:")
        for precision in precision_levels:
            if precision in all_data['precise_digits'][integral_type]:
                for coeff_name in ['Coeff1', 'Coeff2', 'Coeff3']:
                    real_count = len(all_data['precise_digits'][integral_type][precision][coeff_name]['real'])
                    imag_count = len(all_data['precise_digits'][integral_type][precision][coeff_name]['imag'])
                    if real_count > 0 or imag_count > 0:
                        real_values = all_data['precise_digits'][integral_type][precision][coeff_name]['real']
                        imag_values = all_data['precise_digits'][integral_type][precision][coeff_name]['imag']
                        real_max = max(real_values) if real_values else 0
                        imag_max = max(imag_values) if imag_values else 0
                        print(f"  {precision} {coeff_name}: {real_count} real (max: {real_max:.2f}), {imag_count} imag (max: {imag_max:.2f})")

if __name__ == "__main__":
    update_precise_digits_in_pickle()
