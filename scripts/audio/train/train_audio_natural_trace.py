"""
Audio Natural Trace Base Infrastructure
Adapts the natural trace approach from face forgery detection to audio deepfake detection.

Key Concept:
- Real speech has consistent "natural traces": spectral naturalness, noise floor patterns,
  prosodic coherence, breathing artifacts, natural micro-variations
- Synthetic speech (TTS/VC/GAN) lacks or distorts these traces due to vocoder artifacts,
  unnatural smoothness, missing noise floor, inconsistent prosody

Architecture mirrors train_natural_trace.py:
- Stage 1: Learn homogeneous features from real audio (representation learning)
- Stage 2: Train forgery detector using real/fake pairs
"""

from __future__ import print_function
import os
import sys
import math
import random
import numpy as np
import glob
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
from transformers import AutoModel
import timm

# Import audio-specific utility functions (no visual dependencies)
from audio_utils import (
    AverageMeter, SupConLoss,
    adjust_learning_rate, warmup_learning_rate, save_model
)

LOCAL_MODEL_PATH = "pretrained_weights/fastervit_2_224_1k.pth.tar"


# ============== Audio Processing ==============
class AudioWaveformTransform:
    """
    Load audio file and convert to waveform tensor
    Output: [T] tensor (raw waveform for Wav2Vec2/HuBERT)
    """
    def __init__(self, sample_rate=16000, max_length=4.0):
        """
        Args:
            sample_rate: Target sampling rate (16kHz for speech)
            max_length: Maximum audio length in seconds
        """
        self.sample_rate = sample_rate
        self.max_length = max_length

    def __call__(self, audio_path):
        """
        Load audio and convert to waveform tensor
        Returns: [T] tensor (raw waveform)
        """
        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)

        # Pad or truncate to fixed length
        target_length = int(self.max_length * self.sample_rate)
        if waveform.shape[1] < target_length:
            waveform = F.pad(waveform, (0, target_length - waveform.shape[1]))
        else:
            # Random crop for training
            if waveform.shape[1] > target_length:
                start = random.randint(0, waveform.shape[1] - target_length)
                waveform = waveform[:, start:start+target_length]

        # Return [T] tensor (squeeze batch dimension)
        return waveform.squeeze(0)


# Keep old name for compatibility
AudioMelTransform = AudioWaveformTransform


class AudioMelTransformOld:
    """
    Convert audio file to log Mel-spectrogram tensor
    Output: [3, 224, 224] tensor (3-channel for image-based backbones)
    NOTE: Only use this for image-based backbones (ResNet, etc.)
    """
    def __init__(self, sample_rate=16000, n_mels=128, n_fft=512,
                 hop_length=160, win_length=400, max_length=4.0, target_size=224):
        """
        Args:
            sample_rate: Target sampling rate (16kHz for speech)
            n_mels: Number of mel filterbanks
            n_fft: FFT size
            hop_length: Hop length in samples (10ms at 16kHz)
            win_length: Window length in samples (25ms at 16kHz)
            max_length: Maximum audio length in seconds
            target_size: Output image size (224 for backbone compatibility)
        """
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.target_size = target_size

        # Mel-spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=80)

    def __call__(self, audio_path):
        """
        Load audio and convert to 3-channel mel-spectrogram tensor
        Returns: [3, target_size, target_size] tensor
        """
        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)

        # Pad or truncate to fixed length
        target_length = int(self.max_length * self.sample_rate)
        if waveform.shape[1] < target_length:
            waveform = F.pad(waveform, (0, target_length - waveform.shape[1]))
        else:
            # Random crop for training
            if waveform.shape[1] > target_length:
                start = random.randint(0, waveform.shape[1] - target_length)
                waveform = waveform[:, start:start+target_length]

        # Extract mel-spectrogram
        mel = self.mel_transform(waveform)  # [1, n_mels, time]
        mel = self.amplitude_to_db(mel)

        # Normalize to [0, 1]
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)

        # Convert to 3-channel (repeat for backbone compatibility)
        mel = mel.repeat(3, 1, 1)  # [3, n_mels, time]

        # Resize to target_size x target_size
        mel = F.interpolate(
            mel.unsqueeze(0),
            size=(self.target_size, self.target_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        return mel


# ============== Audio Augmentation Functions ==============
def spec_augment(mel_tensor, freq_mask_param=27, time_mask_param=100,
                 num_freq_masks=2, num_time_masks=2):
    """SpecAugment for 2D mel-spectrogram [C, H, W] — only used with image backbones"""
    augmented = mel_tensor.clone()
    _, freq_bins, time_steps = augmented.shape
    for _ in range(num_freq_masks):
        f = random.randint(0, min(freq_mask_param, freq_bins))
        f0 = random.randint(0, max(0, freq_bins - f))
        if f > 0:
            augmented[:, f0:f0+f, :] = augmented.mean()
    for _ in range(num_time_masks):
        t = random.randint(0, min(time_mask_param, time_steps))
        t0 = random.randint(0, max(0, time_steps - t))
        if t > 0:
            augmented[:, :, t0:t0+t] = augmented.mean()
    return augmented


def audio_compression_augment(mel_tensor, compression_prob=0.5):
    """Simulate audio compression artifacts"""
    if random.random() < compression_prob:
        bits = random.choice([8, 12, 14])
        levels = 2 ** bits
        return torch.round(mel_tensor * levels) / levels
    return mel_tensor


def additive_noise_augment(waveform, noise_level=0.005):
    """Add Gaussian noise to waveform [T]"""
    if random.random() < 0.5:
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
    return waveform


def waveform_augment(waveform):
    """
    Audio augmentation for raw waveform [T]
    Equivalent to data_augment() for images
    """
    # Add noise
    waveform = additive_noise_augment(waveform, noise_level=0.005)

    # Random gain variation (simulate recording level differences)
    if random.random() < 0.3:
        gain = random.uniform(0.8, 1.2)
        waveform = waveform * gain

    # Random time shift (small)
    if random.random() < 0.3:
        shift = random.randint(-800, 800)  # ±50ms at 16kHz
        waveform = torch.roll(waveform, shift)

    return waveform


# Keep old name for compatibility
audio_augment = waveform_augment


class TwoCropAudioTransform:
    """Create two augmented views of the same audio"""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, audio_path):
        mel = self.base_transform(audio_path)
        # Apply augmentation twice for two views
        view1 = audio_augment(mel.clone())
        view2 = audio_augment(mel.clone())
        return [view1, view2]


# ============== Audio Dataset Classes ==============
class AudioRealDataset(Dataset):
    """
    Stage 1: Load real audio files for representation learning
    Supports ASVspoof format: bonafide/ directory
    """
    def __init__(self, dataset_root: str, transform=None, split='train'):
        self.transform = transform
        self.audio_files = []
        self.split = split

        # Support multiple formats
        # Format 1: ASVspoof - bonafide/ directory
        bonafide_path = os.path.join(dataset_root, 'bonafide')
        if os.path.exists(bonafide_path):
            for ext in ['*.wav', '*.flac', '*.mp3']:
                self.audio_files.extend(glob.glob(os.path.join(bonafide_path, '**', ext), recursive=True))

        # Format 2: Generic - 0_real/ directory
        real_path = os.path.join(dataset_root, '0_real')
        if os.path.exists(real_path):
            for ext in ['*.wav', '*.flac', '*.mp3']:
                self.audio_files.extend(glob.glob(os.path.join(real_path, '**', ext), recursive=True))

        # Train/test split (90/10)
        self.audio_files.sort()
        split_idx = int(len(self.audio_files) * 0.9)

        if split == 'train':
            self.audio_files = self.audio_files[:split_idx]
        elif split == 'test':
            self.audio_files = self.audio_files[split_idx:]

        print(f"[AudioRealDataset - {split}] Loaded {len(self.audio_files)} real audio files")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]

        if self.transform:
            mel_tensor = self.transform(audio_path)
        else:
            # Default transform
            default_transform = AudioMelTransform()
            mel_tensor = default_transform(audio_path)

        return mel_tensor, 0  # label=0 for real


class AudioForgeryDataset(Dataset):
    """
    Evaluation: Load real and fake audio files
    """
    def __init__(self, dataset_root: str, is_real: bool, transform=None, split='test'):
        self.transform = transform
        self.label = 0 if is_real else 1
        self.audio_files = []
        self.split = split

        # Support multiple formats
        if is_real:
            # Format 1: ASVspoof - bonafide/
            bonafide_path = os.path.join(dataset_root, 'bonafide')
            if os.path.exists(bonafide_path):
                for ext in ['*.wav', '*.flac', '*.mp3']:
                    self.audio_files.extend(glob.glob(os.path.join(bonafide_path, '**', ext), recursive=True))

            # Format 2: Generic - 0_real/
            real_path = os.path.join(dataset_root, '0_real')
            if os.path.exists(real_path):
                for ext in ['*.wav', '*.flac', '*.mp3']:
                    self.audio_files.extend(glob.glob(os.path.join(real_path, '**', ext), recursive=True))
        else:
            # Format 1: ASVspoof - spoof/
            spoof_path = os.path.join(dataset_root, 'spoof')
            if os.path.exists(spoof_path):
                for ext in ['*.wav', '*.flac', '*.mp3']:
                    self.audio_files.extend(glob.glob(os.path.join(spoof_path, '**', ext), recursive=True))

            # Format 2: Generic - 1_fake/
            fake_path = os.path.join(dataset_root, '1_fake')
            if os.path.exists(fake_path):
                for ext in ['*.wav', '*.flac', '*.mp3']:
                    self.audio_files.extend(glob.glob(os.path.join(fake_path, '**', ext), recursive=True))

        # Train/test split
        self.audio_files.sort()
        split_idx = int(len(self.audio_files) * 0.9)

        if split == 'train':
            self.audio_files = self.audio_files[:split_idx]
        elif split == 'test':
            self.audio_files = self.audio_files[split_idx:]

        label_str = "real" if is_real else "fake"
        print(f"[AudioForgeryDataset - {split}] Loaded {len(self.audio_files)} {label_str} audio files")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]

        if self.transform:
            mel_tensor = self.transform(audio_path)
        else:
            default_transform = AudioMelTransform()
            mel_tensor = default_transform(audio_path)

        return mel_tensor, self.label


class AudioDetDataset(Dataset):
    """
    Stage 2: Dataset for detection training with dual transforms
    Returns: (contrastive_views, supervised_view, label)
    """
    def __init__(self, dataset_root: str, contrastive_transform=None,
                 supervised_transform=None, split='train'):
        self.contrastive_transform = contrastive_transform
        self.supervised_transform = supervised_transform
        self.samples = []
        self.split = split

        # Load real audio
        real_files = []
        bonafide_path = os.path.join(dataset_root, 'bonafide')
        real_path = os.path.join(dataset_root, '0_real')

        for path in [bonafide_path, real_path]:
            if os.path.exists(path):
                for ext in ['*.wav', '*.flac', '*.mp3']:
                    real_files.extend(glob.glob(os.path.join(path, '**', ext), recursive=True))

        real_files.sort()
        split_idx = int(len(real_files) * 0.9)
        target_files = real_files[:split_idx] if split == 'train' else real_files[split_idx:]
        self.samples.extend([(f, 0) for f in target_files])

        # Load fake audio
        fake_files = []
        spoof_path = os.path.join(dataset_root, 'spoof')
        fake_path = os.path.join(dataset_root, '1_fake')

        for path in [spoof_path, fake_path]:
            if os.path.exists(path):
                for ext in ['*.wav', '*.flac', '*.mp3']:
                    fake_files.extend(glob.glob(os.path.join(path, '**', ext), recursive=True))

        fake_files.sort()
        split_idx = int(len(fake_files) * 0.9)
        target_files = fake_files[:split_idx] if split == 'train' else fake_files[split_idx:]
        self.samples.extend([(f, 1) for f in target_files])

        print(f"[AudioDetDataset - {split}] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]

        # Apply transforms
        if self.contrastive_transform:
            contrastive_views = self.contrastive_transform(audio_path)
        else:
            default_transform = AudioMelTransform()
            mel = default_transform(audio_path)
            contrastive_views = (mel, mel)

        if self.supervised_transform:
            supervised_view = self.supervised_transform(audio_path)
        else:
            default_transform = AudioMelTransform()
            supervised_view = default_transform(audio_path)

        return contrastive_views, supervised_view, label


class AudioTestDataset(Dataset):
    """
    Test-only dataset: returns (waveform, label), balanced 1:1 real/fake
    """
    def __init__(self, dataset_root: str, transform=None, split='test'):
        self.transform = transform or AudioWaveformTransform()
        real_files, fake_files = [], []

        for subdir, lst in [('bonafide', real_files), ('spoof', fake_files)]:
            path = os.path.join(dataset_root, subdir)
            if not os.path.exists(path):
                continue
            files = []
            for ext in ['*.wav', '*.flac', '*.mp3']:
                files.extend(glob.glob(os.path.join(path, '**', ext), recursive=True))
            files.sort()
            split_idx = int(len(files) * 0.9)
            lst.extend(files[:split_idx] if split == 'train' else files[split_idx:])

        # 1:1 balance: limit to min count
        n = min(len(real_files), len(fake_files))
        real_files, fake_files = real_files[:n], fake_files[:n]

        self.samples = [(f, 0) for f in real_files] + [(f, 1) for f in fake_files]
        print(f"[AudioTestDataset - {split}] real={len(real_files)}, fake={len(fake_files)} (1:1 balanced)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(path), label



class OurAudioNet(nn.Module):
    """
    Audio Deepfake Detection Model
    使用音频专用的预训练模型 (Wav2Vec2, Whisper, HuBERT)

    Architecture:
    - Backbone: 音频预训练模型 from transformers
    - aux_fc1, aux_fc2: Stage 1 的异构/同构特征投影头
    - det_fc1, det_fc2: Stage 2 的检测头
    """
    def __init__(self, backbone='facebook/wav2vec2-base', feature_dim=128):
        super().__init__()

        # 使用 transformers 加载音频预训练模型
        if 'wav2vec2' in backbone.lower():
            from transformers import Wav2Vec2Model
            self.backbone = Wav2Vec2Model.from_pretrained(backbone)
            self.n_features = self.backbone.config.hidden_size
            self.backbone_type = 'wav2vec2'
        elif 'whisper' in backbone.lower():
            from transformers import WhisperModel
            self.backbone = WhisperModel.from_pretrained(backbone)
            self.n_features = self.backbone.config.d_model
            self.backbone_type = 'whisper'
        elif 'hubert' in backbone.lower():
            from transformers import HubertModel
            self.backbone = HubertModel.from_pretrained(backbone)
            self.n_features = self.backbone.config.hidden_size
            self.backbone_type = 'hubert'
        else:
            raise ValueError(f"Unsupported audio backbone: {backbone}. "
                           f"Use wav2vec2, whisper, or hubert models.")

        # Stage 1: Auxiliary projection heads (2-layer MLP, matches image OurNet)
        self.aux_fc1 = nn.Sequential(
            nn.Linear(self.n_features, self.n_features),
            nn.ReLU(),
            nn.Linear(self.n_features, feature_dim)
        )  # Heterogeneous features
        self.aux_fc2 = nn.Sequential(
            nn.Linear(self.n_features, self.n_features),
            nn.ReLU(),
            nn.Linear(self.n_features, feature_dim)
        )  # Homogeneous features

        # Stage 2: Detection heads (matches image OurNet)
        self.det_fc1 = nn.Sequential(
            nn.Linear(self.n_features, self.n_features),
            nn.ReLU(),
            nn.Linear(self.n_features, feature_dim)
        )  # Contrastive features
        self.det_fc2 = nn.Sequential(
            nn.Linear(self.n_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )  # Binary classifier

    def forward_features(self, x):
        """
        Extract features from audio backbone
        Input: x can be either:
            - Mel-spectrogram [B, 3, 224, 224] (converted to waveform internally)
            - Raw waveform [B, T] (direct input)
        """
        # 如果输入是 mel-spectrogram [B, 3, 224, 224], 需要转换为 waveform
        # 但这里我们假设输入已经是合适的格式
        # 实际使用时,AudioMelTransform 应该输出 waveform 而不是 mel-spectrogram

        if self.backbone_type in ['wav2vec2', 'hubert']:
            # Wav2Vec2/HuBERT: input is raw waveform [B, T]
            outputs = self.backbone(x)
            feat = outputs.last_hidden_state.mean(1)  # [B, T, C] -> [B, C]
        elif self.backbone_type == 'whisper':
            # Whisper: input is mel-spectrogram
            outputs = self.backbone.encoder(x)
            feat = outputs.last_hidden_state.mean(1)  # [B, T, C] -> [B, C]

        return feat

    def forward_proj(self, x):
        """
        Stage 1: Forward pass with auxiliary projection heads
        Returns: (heterogeneous_features, homogeneous_features)
        """
        feat = self.forward_features(x)
        heter_feat = self.aux_fc1(feat)  # Heterogeneous
        homo_feat = self.aux_fc2(feat)   # Homogeneous
        return heter_feat, homo_feat

    def forward_det(self, x):
        """
        Stage 2: Forward pass with detection heads
        Returns: (contrastive_features, detection_logits)
        """
        feat = self.forward_features(x)
        det_feat = self.det_fc1(feat)  # Contrastive features
        det_logits = self.det_fc2(feat)  # Binary classification logits
        return det_feat, det_logits


# Alias for compatibility
OurNet = OurAudioNet


# ============== EER Computation ==============
def compute_eer(y_true, y_score):
    """
    Compute Equal Error Rate (EER)
    Standard metric for audio deepfake detection
    """
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh


# ============== Utility Functions ==============
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
