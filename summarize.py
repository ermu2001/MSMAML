import json
import sys

import numpy as np


def compute_confidence_interval(value):
    """
    Compute 95% +- confidence intervals over tasks
    change 1.960 to 2.576 for 99% +- confidence intervals
    """
    return np.std(value) * 1.960 / np.sqrt(len(value))

if __name__ == "__main__":

    result_path = sys.argv[1]
    with open(result_path, 'r') as f:
        results = json.load(f)

    print('Evaluation results:')
    for key, value in sorted(results.items()):
        if not isinstance(value, int):
            print('{}: {} +- {}'.format(
                key, np.mean(value), compute_confidence_interval(value)))
        else:
            print('{}: {}'.format(key, value))