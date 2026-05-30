"""
Evaluation Script for Audio Forgery Detection Models

Evaluates trained models on ASVspoof-format datasets.
Metrics: Accuracy, AP, AUC, EER
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import importlib.util


# ============== Dataset ==============
class AudioEvalDataset(Dataset):
    """
    Evaluation dataset for audio forgery detection.
    Supports ASVspoof format: bonafide/ and spoof/ directories.
    Balanced 1:1 by default.
    """
    def __init__(self, dataset_root, sample_rate=16000, max_length=4.0, balanced=True):
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.samples = []

        real_files, fake_files = [], []
        for ext in ['*.wav', '*.flac', '*.mp3']:
            real_files.extend(Path(dataset_root, 'bonafide').glob(ext) if
                              Path(dataset_root, 'bonafide').exists() else [])
            fake_files.extend(Path(dataset_root, 'spoof').glob(ext) if
                              Path(dataset_root, 'spoof').exists() else [])

        real_files = sorted([str(f) for f in real_files])
        fake_files = sorted([str(f) for f in fake_files])

        if balanced:
            n = min(len(real_files), len(fake_files))
            real_files, fake_files = real_files[:n], fake_files[:n]

        self.samples = [(f, 0) for f in real_files] + [(f, 1) for f in fake_files]
        print(f"[AudioEvalDataset] real={len(real_files)}, fake={len(fake_files)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            waveform = T.Resample(sr, self.sample_rate)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        target_len = int(self.max_length * self.sample_rate)
        if waveform.shape[1] < target_len:
            waveform = F.pad(waveform, (0, target_len - waveform.shape[1]))
        else:
            waveform = waveform[:, :target_len]
        return waveform.squeeze(0), label


# ============== Metrics ==============
def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return float(eer)


def load_model(model_dir, device):
    """Load model from directory containing model.py, config.json, pytorch_model.bin"""
    model_dir = Path(model_dir)

    # Load model class
    spec = importlib.util.spec_from_file_location("model", model_dir / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with open(model_dir / "config.json") as f:
        config = json.load(f)

    model = module.OurAudioNet(config=config).to(device)

    # Load weights
    ckpt = torch.load(model_dir / "pytorch_model.bin", map_location=device)
    state_dict = ckpt.get('model', ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


@torch.no_grad()
def evaluate(model, loader, device):
    all_preds, all_labels = [], []
    for waveforms, labels in tqdm(loader, desc="Evaluating"):
        waveforms = waveforms.to(device)
        _, logits = model.forward_det(waveforms)
        preds = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    preds_binary = (all_preds > 0.5).astype(int)

    return {
        'acc':  float(accuracy_score(all_labels, preds_binary)),
        'ap':   float(average_precision_score(all_labels, all_preds)),
        'auc':  float(roc_auc_score(all_labels, all_preds)),
        'eer':  compute_eer(all_labels, all_preds),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Audio Forgery Detection Model')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to model directory (contains model.py, config.json, pytorch_model.bin)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset root (contains bonafide/ and spoof/)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--no_balance', action='store_true',
                        help='Do not balance real/fake (use full dataset distribution)')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Model:   {args.model_dir}")
    print(f"Dataset: {args.dataset}")
    print(f"Device:  {device}")

    model, config = load_model(args.model_dir, device)

    dataset = AudioEvalDataset(
        args.dataset,
        sample_rate=config.get('sample_rate', 16000),
        max_length=config.get('max_length', 4.0),
        balanced=not args.no_balance,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    metrics = evaluate(model, loader, device)

    print("\n" + "=" * 50)
    print("Results")
    print("=" * 50)
    print(f"  Accuracy : {metrics['acc']:.4f} ({metrics['acc']*100:.2f}%)")
    print(f"  AP       : {metrics['ap']:.4f}")
    print(f"  AUC      : {metrics['auc']:.4f}")
    print(f"  EER      : {metrics['eer']:.4f} ({metrics['eer']*100:.2f}%)")
    print("=" * 50)

    return metrics


if __name__ == '__main__':
    main()
