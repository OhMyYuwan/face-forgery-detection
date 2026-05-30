#!/usr/bin/env python3
"""
评估模型组合的性能

支持两种组合方法：
1. weighted: 加权平均
2. voting: 投票法
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import argparse


def load_model_results(model_name, results_dir=None):
    """加载单个模型的详细预测结果"""
    if results_dir is None:
        results_dir = Path(__file__).parent.parent.parent / 'results' / 'evaluate'
    results_path = Path(results_dir) / model_name / 'metrics.json'

    if not results_path.exists():
        raise FileNotFoundError(f"结果文件不存在: {results_path}")

    with open(results_path) as f:
        results = json.load(f)

    # 提取所有样本的分数和标签
    all_scores = []
    all_labels = []
    all_methods = []

    for method_name, method_data in results['per_method'].items():
        scores = method_data['scores']
        labels = method_data['labels']
        all_scores.extend(scores)
        all_labels.extend(labels)
        all_methods.extend([method_name] * len(scores))

    return {
        'scores': np.array(all_scores),
        'labels': np.array(all_labels),
        'methods': all_methods
    }


def load_optimal_thresholds(threshold_file='../../OhMyYuwan/face-forgery-detection/optimal_thresholds.json'):
    """加载最优阈值配置"""
    threshold_path = Path(__file__).parent / threshold_file

    if threshold_path.exists():
        with open(threshold_path) as f:
            thresholds_data = json.load(f)
        return {k: v['threshold'] for k, v in thresholds_data.items()}
    else:
        # 默认阈值
        return {}


def ensemble_weighted(model_scores, model_weights, global_threshold=0.5):
    """
    加权平均组合

    Args:
        model_scores: dict, {model_name: scores_array}
        model_weights: dict, {model_name: weight}
        global_threshold: float, 全局阈值

    Returns:
        predictions: array, 预测结果 (0/1)
        weighted_scores: array, 加权平均分数
    """
    # 计算加权平均分数
    total_weight = sum(model_weights.values())
    weighted_scores = np.zeros(len(next(iter(model_scores.values()))))

    for model_name, scores in model_scores.items():
        weight = model_weights.get(model_name, 1.0)
        weighted_scores += scores * weight

    weighted_scores /= total_weight

    # 根据全局阈值判断
    predictions = (weighted_scores >= global_threshold).astype(int)

    return predictions, weighted_scores


def ensemble_voting(model_scores, model_thresholds):
    """
    投票法组合

    Args:
        model_scores: dict, {model_name: scores_array}
        model_thresholds: dict, {model_name: threshold}

    Returns:
        predictions: array, 预测结果 (0/1)
        votes: dict, {model_name: votes_array}
    """
    n_samples = len(next(iter(model_scores.values())))
    votes_matrix = np.zeros((len(model_scores), n_samples), dtype=int)
    votes = {}

    for i, (model_name, scores) in enumerate(model_scores.items()):
        threshold = model_thresholds.get(model_name, 0.5)
        model_votes = (scores >= threshold).astype(int)
        votes_matrix[i] = model_votes
        votes[model_name] = model_votes

    # 多数投票
    predictions = (votes_matrix.sum(axis=0) > len(model_scores) / 2).astype(int)

    return predictions, votes


def evaluate_ensemble(model_list, method='voting', model_weights=None,
                     global_threshold=0.5, results_dir=None):
    """
    评估模型组合

    Args:
        model_list: list, 模型名称列表
        method: str, 'weighted' 或 'voting'
        model_weights: dict, 模型权重 (仅用于 weighted 方法)
        global_threshold: float, 全局阈值 (仅用于 weighted 方法)
        results_dir: str, 结果目录

    Returns:
        dict, 评估结果
    """
    print(f"\n{'='*60}")
    print(f"评估模型组合")
    print(f"{'='*60}")
    print(f"模型列表: {', '.join(model_list)}")
    print(f"组合方法: {method}")
    if method == 'weighted':
        print(f"全局阈值: {global_threshold}")
        print(f"模型权重: {model_weights}")
    print(f"{'='*60}\n")

    # 加载所有模型的结果
    model_results = {}
    for model_name in model_list:
        print(f"加载模型: {model_name}")
        model_results[model_name] = load_model_results(model_name, results_dir)

    # 验证所有模型的样本数一致
    n_samples = len(model_results[model_list[0]]['labels'])
    for model_name in model_list:
        assert len(model_results[model_name]['labels']) == n_samples, \
            f"模型 {model_name} 的样本数不一致"

    # 获取真实标签
    true_labels = model_results[model_list[0]]['labels']

    # 提取所有模型的分数
    model_scores = {name: results['scores'] for name, results in model_results.items()}

    # 加载最优阈值
    optimal_thresholds = load_optimal_thresholds()
    model_thresholds = {name: optimal_thresholds.get(name, 0.5) for name in model_list}

    # 根据方法进行组合
    if method == 'weighted':
        if model_weights is None:
            model_weights = {name: 1.0 for name in model_list}
        predictions, ensemble_scores = ensemble_weighted(
            model_scores, model_weights, global_threshold
        )
    elif method == 'voting':
        predictions, votes = ensemble_voting(model_scores, model_thresholds)
        ensemble_scores = None
    else:
        raise ValueError(f"未知的组合方法: {method}")

    # 计算性能指标
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)

    # 计算 AUC (使用加权平均分数或投票数)
    if method == 'weighted':
        auc = roc_auc_score(true_labels, ensemble_scores)
        ensemble_scores_for_method = ensemble_scores
    else:
        # 投票法：使用投票数作为分数
        vote_counts = sum(votes.values())
        auc = roc_auc_score(true_labels, vote_counts)
        ensemble_scores_for_method = vote_counts

    cm = confusion_matrix(true_labels, predictions)

    # 计算 per-method 指标
    all_methods = model_results[model_list[0]]['methods']
    unique_methods = sorted(set(all_methods))
    per_method_metrics = {}

    for method_name in unique_methods:
        mask = np.array([m == method_name for m in all_methods])
        method_labels = true_labels[mask]
        method_preds = predictions[mask]
        method_scores = ensemble_scores_for_method[mask]

        if len(method_labels) > 0:
            per_method_metrics[method_name] = {
                'accuracy': float(accuracy_score(method_labels, method_preds)),
                'precision': float(precision_score(method_labels, method_preds, zero_division=0)),
                'recall': float(recall_score(method_labels, method_preds, zero_division=0)),
                'f1': float(f1_score(method_labels, method_preds, zero_division=0)),
                'auc': float(roc_auc_score(method_labels, method_scores)) if len(np.unique(method_labels)) > 1 else 0.0,
                'num_samples': int(len(method_labels))
            }

    # 打印结果
    print(f"\n{'='*60}")
    print("组合性能指标:")
    print(f"{'='*60}")
    print(f"准确率:  {accuracy:.4f}")
    print(f"精确率:  {precision:.4f}")
    print(f"召回率:  {recall:.4f}")
    print(f"F1 分数: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"\n混淆矩阵:")
    print(f"TN: {cm[0,0]:6d}  FP: {cm[0,1]:6d}")
    print(f"FN: {cm[1,0]:6d}  TP: {cm[1,1]:6d}")

    # 计算每个模型的单独性能（使用最优阈值）
    print(f"\n{'='*60}")
    print("各模型单独性能:")
    print(f"{'='*60}")

    individual_results = {}
    for model_name in model_list:
        scores = model_scores[model_name]
        threshold = model_thresholds[model_name]
        preds = (scores >= threshold).astype(int)

        acc = accuracy_score(true_labels, preds)
        prec = precision_score(true_labels, preds, zero_division=0)
        rec = recall_score(true_labels, preds, zero_division=0)
        f1_score_val = f1_score(true_labels, preds, zero_division=0)

        individual_results[model_name] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1_score_val),
            'threshold': float(threshold)
        }

        print(f"{model_name}:")
        print(f"  准确率: {acc:.4f}, F1: {f1_score_val:.4f}, 阈值: {threshold:.3f}")

    # 保存详细结果
    results = {
        'ensemble': {
            'models': model_list,
            'method': method,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc),
            'confusion_matrix': cm.tolist(),
            'per_method': per_method_metrics
        },
        'individual_models': individual_results,
        'config': {
            'method': method,
            'global_threshold': float(global_threshold) if method == 'weighted' else None,
            'model_weights': {k: float(v) for k, v in model_weights.items()} if method == 'weighted' and model_weights else None,
            'model_thresholds': {k: float(v) for k, v in model_thresholds.items()}
        }
    }

    # 保存到文件
    output_dir = Path(results_dir) / 'ensemble'
    output_dir.mkdir(exist_ok=True)

    models_str = '_'.join(model_list)
    output_file = output_dir / f"{method}_{models_str}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ 结果已保存到: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description='评估模型组合性能')
    parser.add_argument('--models', nargs='+', required=True,
                       help='模型列表，例如: fastervit_2 internvit_300m')
    parser.add_argument('--method', choices=['weighted', 'voting'], default='voting',
                       help='组合方法: weighted (加权平均) 或 voting (投票法)')
    parser.add_argument('--weights', nargs='+', type=float,
                       help='模型权重 (仅用于 weighted 方法)，例如: 1.0 1.5 2.0')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='全局阈值 (仅用于 weighted 方法)')
    parser.add_argument('--results_dir', default=None,
                       help='结果目录 (默认: PROJECT_ROOT/results/evaluate)')

    args = parser.parse_args()

    # 构建权重字典
    model_weights = None
    if args.method == 'weighted' and args.weights:
        if len(args.weights) != len(args.models):
            raise ValueError("权重数量必须与模型数量一致")
        model_weights = {model: weight for model, weight in zip(args.models, args.weights)}

    # 评估
    evaluate_ensemble(
        model_list=args.models,
        method=args.method,
        model_weights=model_weights,
        global_threshold=args.threshold,
        results_dir=args.results_dir
    )


if __name__ == '__main__':
    main()
