#!/usr/bin/env python3
"""
Quick results viewer for model evaluation
Shows both original and optimized threshold results
"""

import json
from pathlib import Path
from tabulate import tabulate

results_dir = Path(__file__).parent.parent.parent / 'results' / 'evaluate'
optimal_thresholds_path = Path(__file__).parent.parent.parent / 'OhMyYuwan' / 'face-forgery-detection' / 'optimal_thresholds.json'

# Load optimal thresholds if available
optimal_thresholds = {}
if optimal_thresholds_path.exists():
    with open(optimal_thresholds_path) as f:
        optimal_thresholds = json.load(f)

# Collect all results
all_results = []

for model_dir in sorted(results_dir.iterdir()):
    if model_dir.is_dir() and model_dir.name != 'comparison' and model_dir.name != 'ensemble':
        metrics_file = model_dir / 'metrics.json'
        threshold_opt_file = model_dir / 'threshold_optimization.json'

        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
                model_name = data['model_name']
                overall = data['overall']

                result = {
                    'Model': model_name,
                    'Threshold': f"{data.get('threshold', 0.5):.4f}",
                    'Accuracy': f"{overall['accuracy']:.4f}",
                    'Precision': f"{overall['precision']:.4f}",
                    'Recall': f"{overall['recall']:.4f}",
                    'F1': f"{overall['f1']:.4f}",
                    'AUC': f"{overall['auc']:.4f}",
                }

                # Add optimized results if available
                if threshold_opt_file.exists():
                    with open(threshold_opt_file) as f:
                        opt_data = json.load(f)
                        result['Opt_Threshold'] = f"{opt_data['optimal_threshold']:.4f}"
                        result['Opt_F1'] = f"{opt_data['optimized_metrics']['f1']:.4f}"
                        result['F1_Gain'] = f"{opt_data['improvement']['f1']:+.4f}"
                elif model_name in optimal_thresholds:
                    result['Opt_Threshold'] = f"{optimal_thresholds[model_name]['threshold']:.4f}"
                    result['Opt_F1'] = "N/A"
                    result['F1_Gain'] = "N/A"
                else:
                    result['Opt_Threshold'] = "N/A"
                    result['Opt_F1'] = "N/A"
                    result['F1_Gain'] = "N/A"

                all_results.append(result)

if all_results:
    print("\n" + "="*120)
    print("Model Evaluation Results Summary")
    print("="*120 + "\n")
    print(tabulate(all_results, headers='keys', tablefmt='grid'))
    print(f"\nTotal models evaluated: {len(all_results)}")
    print("\nNote:")
    print("  - Threshold: Original threshold used in evaluation")
    print("  - Opt_Threshold: Optimized threshold (from threshold_optimization.json or optimal_thresholds.json)")
    print("  - F1_Gain: F1 score improvement with optimized threshold")
else:
    print("No results found yet. Evaluation may still be running.")
