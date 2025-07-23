#
# QCDLoop + Kokkos 2025
#
# Authors: Reet Barik      : rbarik@anl.gov
#          Taylor Childers : jchilders@anl.gov

import sys
import difflib

def compare_files(file1_path, file2_path):
    """
    Compares two text files and prints the line numbers where differences occur.

    Args:
        file1_path (str): The path to the file containing values from QCDLoop CPU.
        file2_path (str): The path to the file containing values from QCDLoop GPU.
    """


    tadpole_test = ["Tadpole Integral TP0"]
    bubble_test = ["Bubble Integral BB0-1", "Bubble Integral BB1", "Bubble Integral BB2", "Bubble Integral BB3", "Bubble Integral BB0", "Bubble Integral BB4", "Bubble Integral BB5"]
    triangle_test = ["Triangle Integral T0-1", "Triangle Integral T0-2", "Triangle Integral T0-3", "Triangle Integral T0-4", "Triangle Integral T1", "Triangle Integral T2", "Triangle Integral T3", "Triangle Integral T4-1", "Triangle Integral T4-2", "Triangle Integral T5", "Triangle Integral T6"]
    box_test = ["Box Integral BIN0", "Box Integral BIN1", "Box Integral BIN2", "Box Integral BIN3", "Box Integral BIN4", "Box Integral B1", "Box Integral B2", "Box Integral B3", "Box Integral B4", "Box Integral B5", "Box Integral B6", "Box Integral B7", "Box Integral B8", "Box Integral B9", "Box Integral B10", "Box Integral B12", "Box Integral B13", "Box Integral B14", "Box Integral B15", "Box Integral B16"]


    file1 = [line for line in open(file1_path, "rt") if line.strip()]
    file2 = [line for line in open(file2_path, "rt") if line.strip()]
    totalLines = len(file2)
    
    diffs = list()
    for i, (a, b) in enumerate(zip(file1, file2)):
        if a != b:
            diffs.append(i + 1)

    failed_tests = set()

    for i in diffs:
        idx = i - 1  # zero-based index
        if idx < len(tadpole_test) * 3:
            test_idx = idx // 3
            failed_tests.add(tadpole_test[test_idx])
        elif idx < len(tadpole_test) * 3 + len(bubble_test) * 3:
            test_idx = (idx - len(tadpole_test) * 3) // 3
            failed_tests.add(bubble_test[test_idx])
        elif idx < len(tadpole_test) * 3 + len(bubble_test) * 3 + len(triangle_test) * 3:
            test_idx = (idx - len(tadpole_test) * 3 - len(bubble_test) * 3) // 3
            failed_tests.add(triangle_test[test_idx])
        elif idx < len(tadpole_test) * 3 + len(bubble_test) * 3 + len(triangle_test) * 3 + len(box_test) * 3:
            test_idx = (idx - len(tadpole_test) * 3 - len(bubble_test) * 3 - len(triangle_test) * 3) // 3
            failed_tests.add(box_test[test_idx])

    for t in failed_tests:
        print(f"Test failed: {t}")
    

if __name__ == "__main__":
    cpu_output = sys.argv[1]
    gpu_output = sys.argv[2]

    compare_files(cpu_output, gpu_output)