"""
Main Evaluation Script for Face Forgery Detection Models

Evaluates all models on the NTF test dataset and generates comprehensive metrics.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..','..', 'Space'))

import compat_patches  # Apply compatibility patches

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
import argparse
from pathlib import Path
import importlib.util



class NTFDataset(Dataset):
    """NTF Test Dataset"""
    def __init__(self, root_dir, transform=None, methods=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.labels = []
        self.methods = []

        # Get all methods if not specified
        if methods is None:
            methods = [d.name for d in self.root_dir.iterdir() if d.is_dir()]

        # Load samples
        for method in methods:
            method_dir = self.root_dir / method
            if not method_dir.exists():
                continue

            # Real images (label 0)
            real_dir = method_dir / '0_real'
            if real_dir.exists():
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    for img_path in real_dir.glob(ext):
                        self.samples.append(str(img_path))
                        self.labels.append(0)
                        self.methods.append(method)

            # Fake images (label 1)
            fake_dir = method_dir / '1_fake'
            if fake_dir.exists():
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    for img_path in fake_dir.glob(ext):
                        self.samples.append(str(img_path))
                        self.labels.append(1)
                        self.methods.append(method)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        method = self.methods[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label, method


def load_model(model_name, device='cuda'):
    """Load model with weights"""
    model_dir = Path(__file__).parent.parent.parent / 'OhMyYuwan/face-forgery-detection' / model_name
    config_path = model_dir / 'config.json'
    weight_path = model_dir / 'pytorch_model.bin'

    with open(config_path, 'r') as f:
        model_config = json.load(f)

    # Import model
    spec = importlib.util.spec_from_file_location(
        f"{model_name}.model",
        model_dir / 'model.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Build model (same style as Space/test_all_models.py)
    model = module.OurNet(model_config)

    # Load weights
    state = torch.load(weight_path, map_location=device, weights_only=False)

    # Extract state dict
    if isinstance(state, dict):
        if 'model' in state:
            state_dict = state['model']
        elif 'state_dict' in state:
            state_dict = state['state_dict']
        else:
            state_dict = state
    else:
        state_dict = state

    # Remove 'module.' prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    input_size = model_config.get('input_size', [224, 224])
    if isinstance(input_size, (list, tuple)):
        img_size = input_size[0]
    else:
        img_size = input_size

    eval_config = {
        'img_size': img_size,
        'model_config': model_config
    }

    return model, eval_config


def evaluate_model(model_name, dataset_root, device='cuda', batch_size=32):
    """Evaluate a single model"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    # Load model
    model, config = load_model(model_name, device)
    img_size = config['img_size']


    # Data transform
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    dataset = NTFDataset(dataset_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Dataset size: {len(dataset)}")
    print(f"Image size: {img_size}x{img_size}")

    # Evaluate
    all_preds = []
    all_labels = []
    all_scores = []
    all_methods = []

    with torch.no_grad():
        for images, labels, methods in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)

            # Forward pass
            _, det_head = model.forward_det(images)
            scores = torch.sigmoid(det_head).cpu().numpy().flatten()
            preds = (scores >= 0.5).astype(int)

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_scores.extend(scores)
            all_methods.extend(methods)

    # Convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    # Calculate overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_scores)
    cm = confusion_matrix(all_labels, all_preds)

    # Calculate per-method metrics
    unique_methods = sorted(set(all_methods))
    per_method_metrics = {}

    for method in unique_methods:
        mask = np.array([m == method for m in all_methods])
        method_labels = all_labels[mask]
        method_preds = all_preds[mask]
        method_scores = all_scores[mask]

        if len(method_labels) > 0:
            per_method_metrics[method] = {
                'accuracy': accuracy_score(method_labels, method_preds),
                'precision': precision_score(method_labels, method_preds, zero_division=0),
                'recall': recall_score(method_labels, method_preds, zero_division=0),
                'f1': f1_score(method_labels, method_preds, zero_division=0),
                'auc': roc_auc_score(method_labels, method_scores) if len(np.unique(method_labels)) > 1 else 0.0,
                'num_samples': len(method_labels),
                'scores': method_scores.tolist(),
                'labels': method_labels.tolist()
            }

    # Print results
    print(f"\n{'='*60}")
    print("Overall Metrics:")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"TN: {cm[0,0]:6d}  FP: {cm[0,1]:6d}")
    print(f"FN: {cm[1,0]:6d}  TP: {cm[1,1]:6d}")

    # Save results
    results_dir = Path(__file__).parent.parent.parent / 'results' / 'evaluate' / model_name
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'model_name': model_name,
        'overall': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc),
            'confusion_matrix': cm.tolist()
        },
        'per_method': per_method_metrics
    }

    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {results_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate face forgery detection models')
    parser.add_argument('--model', type=str, default='all',
                       help='Model name to evaluate (default: all)')
    parser.add_argument('--dataset', type=str, default='/lzy/TIFS2026/datasets/NTF/test',
                       help='Path to NTF test dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Get models to evaluate
    if args.model == 'all':
        registry_path = Path(__file__).parent.parent.parent / 'OhMyYuwan/face-forgery-detection' / 'registry.json'
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        models = list(registry['models'].keys())
    else:
        models = [args.model]

    # Evaluate each model
    all_results = {}
    for model_name in models:
        try:
            results = evaluate_model(
                model_name,
                args.dataset,
                device=args.device,
                batch_size=args.batch_size
            )
            all_results[model_name] = results
        except Exception as e:
            print(f"\n❌ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save comparison
    if len(all_results) > 1:
        comparison_dir = Path(__file__).parent.parent.parent / 'results' / 'evaluate' / 'comparison'
        comparison_dir.mkdir(parents=True, exist_ok=True)

        with open(comparison_dir / 'all_models_metrics.json', 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*60}")
        print("Model Comparison:")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'Accuracy':<10} {'F1':<10} {'AUC':<10}")
        print(f"{'-'*60}")
        for model_name, results in all_results.items():
            metrics = results['overall']
            print(f"{model_name:<20} {metrics['accuracy']:<10.4f} {metrics['f1']:<10.4f} {metrics['auc']:<10.4f}")


if __name__ == '__main__':
    main()
