#!/usr/bin/env python3
"""
为每个模型找到最优阈值

通过在验证集上搜索不同阈值，找到使 F1 分数最大化的阈值
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def find_optimal_threshold(scores, labels, metric='f1'):
    """
    在不同阈值下搜索，找到最优阈值

    如果存在多个阈值达到相同最优值（平台区间），取区间中点

    Returns:
        best_threshold: 最优阈值（区间中点）
        best_score: 最优指标值
        threshold_range: (min, max) 最优阈值区间
        threshold_curve: 所有阈值的性能曲线
    """
    thresholds = np.linspace(0, 1, 1001)  # 0.000, 0.001, ..., 1.000
    threshold_curve = []

    best_score = 0.0

    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)

        if len(np.unique(preds)) == 1:
            prec = 1.0 if preds[0] == 1 and labels.sum() > 0 else 0.0
            rec = 1.0 if preds[0] == 1 else 0.0
            f1 = 0.0
        else:
            prec = precision_score(labels, preds, zero_division=0)
            rec = recall_score(labels, preds, zero_division=0)
            f1 = f1_score(labels, preds, zero_division=0)

        acc = accuracy_score(labels, preds)

        threshold_curve.append({
            'threshold': float(threshold),
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1)
        })

        current_score = f1 if metric == 'f1' else acc if metric == 'accuracy' else prec if metric == 'precision' else rec
        if current_score > best_score:
            best_score = current_score

    # 找到所有达到最优值的阈值（平台区间）
    tolerance = 1e-6
    optimal_thresholds = [
        t['threshold'] for t in threshold_curve
        if abs(t[metric] - best_score) < tolerance
    ]

    threshold_min = min(optimal_thresholds)
    threshold_max = max(optimal_thresholds)
    best_threshold = (threshold_min + threshold_max) / 2  # 取区间中点

    return best_threshold, best_score, (threshold_min, threshold_max), threshold_curve


def optimize_model_threshold(model_name, results_dir=None):
    """为单个模型优化阈值"""
    if results_dir is None:
        results_dir = Path(__file__).parent.parent.parent / 'results' / 'evaluate'
    results_path = Path(results_dir) / model_name / 'metrics.json'

    if not results_path.exists():
        print(f"❌ {model_name}: 结果文件不存在")
        return None

    with open(results_path) as f:
        results = json.load(f)

    # 提取所有样本的分数和标签
    all_scores = []
    all_labels = []

    for method_name, method_data in results['per_method'].items():
        all_scores.extend(method_data['scores'])
        all_labels.extend(method_data['labels'])

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    print(f"\n{'='*60}")
    print(f"优化模型: {model_name}")
    print(f"{'='*60}")
    print(f"样本数: {len(all_scores)}")
    print(f"正样本: {all_labels.sum()} ({all_labels.sum()/len(all_labels)*100:.1f}%)")
    print(f"负样本: {len(all_labels) - all_labels.sum()} ({(1-all_labels.sum()/len(all_labels))*100:.1f}%)")

    # 找到最优阈值（优化 F1）
    best_threshold, best_f1, threshold_range, threshold_curve = find_optimal_threshold(
        all_scores, all_labels, metric='f1'
    )

    # 使用最优阈值计算最终指标
    preds = (all_scores >= best_threshold).astype(int)
    final_acc = accuracy_score(all_labels, preds)
    final_prec = precision_score(all_labels, preds, zero_division=0)
    final_rec = recall_score(all_labels, preds, zero_division=0)
    final_f1 = f1_score(all_labels, preds, zero_division=0)

    print(f"\n原始阈值 (0.5):")
    print(f"  准确率: {results['overall']['accuracy']:.4f}")
    print(f"  F1: {results['overall']['f1']:.4f}")

    print(f"\n最优阈值 ({best_threshold:.3f}):")
    if threshold_range[0] != threshold_range[1]:
        print(f"  最优区间: [{threshold_range[0]:.3f}, {threshold_range[1]:.3f}]（取中点）")
    print(f"  准确率: {final_acc:.4f}")
    print(f"  精确率: {final_prec:.4f}")
    print(f"  召回率: {final_rec:.4f}")
    print(f"  F1: {final_f1:.4f}")

    improvement = final_f1 - results['overall']['f1']
    print(f"\nF1 提升: {improvement:+.4f} ({improvement/results['overall']['f1']*100:+.1f}%)")

    # 保存优化结果
    optimization_result = {
        'model_name': model_name,
        'optimal_threshold': float(best_threshold),
        'threshold_range': [float(threshold_range[0]), float(threshold_range[1])],
        'original_threshold': 0.5,
        'original_metrics': {
            'accuracy': results['overall']['accuracy'],
            'precision': results['overall']['precision'],
            'recall': results['overall']['recall'],
            'f1': results['overall']['f1'],
            'auc': results['overall']['auc']
        },
        'optimized_metrics': {
            'accuracy': float(final_acc),
            'precision': float(final_prec),
            'recall': float(final_rec),
            'f1': float(final_f1),
            'auc': results['overall']['auc']  # AUC 不受阈值影响
        },
        'improvement': {
            'accuracy': float(final_acc - results['overall']['accuracy']),
            'f1': float(improvement)
        },
        'threshold_curve': threshold_curve
    }

    # 保存到文件
    output_path = Path(results_dir) / model_name / 'threshold_optimization.json'
    with open(output_path, 'w') as f:
        json.dump(optimization_result, f, indent=2)

    return optimization_result


def optimize_all_models(results_dir=None):
    """为所有模型优化阈值"""
    if results_dir is None:
        results_dir = Path(__file__).parent.parent.parent / 'results' / 'evaluate'
    results_dir = Path(results_dir)

    if not results_dir.exists():
        print(f"❌ 结果目录不存在: {results_dir}")
        return

    # 找到所有模型
    models = [d.name for d in results_dir.iterdir() if d.is_dir() and (d / 'metrics.json').exists()]

    print(f"找到 {len(models)} 个模型")
    print("="*60)

    all_results = {}

    for model_name in sorted(models):
        result = optimize_model_threshold(model_name, results_dir)
        if result:
            all_results[model_name] = result

    # 生成总结报告
    print(f"\n{'='*60}")
    print("阈值优化总结")
    print(f"{'='*60}\n")

    print(f"{'模型':<20} {'原始阈值':<10} {'最优阈值':<10} {'F1提升':<10}")
    print("-" * 60)

    for model_name, result in sorted(all_results.items()):
        orig_f1 = result['original_metrics']['f1']
        opt_f1 = result['optimized_metrics']['f1']
        improvement = result['improvement']['f1']

        print(f"{model_name:<20} {result['original_threshold']:<10.2f} "
              f"{result['optimal_threshold']:<10.2f} {improvement:+.4f}")

    # 保存总结
    summary = {
        'models': all_results,
        'summary': {
            'total_models': len(all_results),
            'avg_improvement': float(np.mean([r['improvement']['f1'] for r in all_results.values()])),
            'models_improved': sum(1 for r in all_results.values() if r['improvement']['f1'] > 0)
        }
    }

    summary_path = results_dir / 'threshold_optimization_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ 优化完成！")
    print(f"   总结已保存到: {summary_path}")
    print(f"   平均 F1 提升: {summary['summary']['avg_improvement']:+.4f}")
    print(f"   改进的模型数: {summary['summary']['models_improved']}/{len(all_results)}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        # 优化单个模型
        model_name = sys.argv[1]
        optimize_model_threshold(model_name)
    else:
        # 优化所有模型
        optimize_all_models()
