#!/usr/bin/env python3
"""
Face Forgery Detection Training Script

Two-stage training:
Stage 1: Learn homogeneous features from real images (representation learning)
Stage 2: Train forgery detector using real/fake pairs
"""

import os
import sys
import time
import math
import glob
import argparse
import numpy as np
from typing import List
from PIL import Image
from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score
from fastervit import create_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset, ConcatDataset, Dataset, DataLoader
from torchvision import transforms

# Import base infrastructure
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_natural_trace import (
    OurNet, SupConLoss, AverageMeter, TwoCropTransform,
    Auxdataset, Detdataset, data_augment,
    adjust_learning_rate, warmup_learning_rate, save_model
)

try:
    import tensorboard_logger as tb_logger
except ImportError:
    print("[Warning] tensorboard_logger not found. Logging disabled.")
    tb_logger = None


# ============== Configuration ==============
def get_args():
    parser = argparse.ArgumentParser(description='Face Forgery Detection Training')
    
    # Basic config
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--savepath', type=str, default='/zyy/TIFS2026/yuwanz')
    parser.add_argument('--stage1_model', type=str, default='')
    
    # Dataset config - Face-specific
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='Dataset root directory, e.g., /zyy/TIFS2026/datasets/NTF/test')
    parser.add_argument('--deepfake_methods', type=str, default='all',
                        help='Deepfake methods to use (comma-separated or "all")')
    parser.add_argument('--use_face_dataset', action='store_true', default=True,
                        help='Use face dataset structure (deepfake_method/0_real, 1_fake)')
    
    # Legacy dataset config (for backward compatibility)
    parser.add_argument('--datapath', type=str, default='',
                        help='[Deprecated] Legacy datapath for class-based dataset')
    parser.add_argument('--classes', type=str, default='',
                        help='[Deprecated] Legacy classes for class-based dataset')
    
    # Model config
    parser.add_argument('--backbone', type=str, default='inception_next_base')
    
    # Stage 1 config (representation learning)
    parser.add_argument('--stage1_epochs', type=int, default=50)
    parser.add_argument('--stage1_batch_size', type=int, default=32)
    parser.add_argument('--stage1_lr', type=float, default=1e-5)
    parser.add_argument('--stage1_head_lr', type=float, default=0.01, help='Projection head learning rate')
    parser.add_argument('--stage1_momentum', type=float, default=0.9)
    parser.add_argument('--stage1_weight_decay', type=float, default=1e-4)
    
    # Stage 2 config (forgery detection)
    parser.add_argument('--stage2_epochs', type=int, default=50)
    parser.add_argument('--stage2_batch_size', type=int, default=16)
    parser.add_argument('--stage2_lr', type=float, default=0.01)
    parser.add_argument('--stage2_momentum', type=float, default=0.9)
    parser.add_argument('--stage2_weight_decay', type=float, default=1e-4)
    parser.add_argument('--lambda_aux', type=float, default=0.3, 
                        help='Weight for auxiliary contrastive loss in stage 2')
    
    # Optimizer config
    parser.add_argument('--cosine', action='store_true', default=False)
    parser.add_argument('--warmup_from', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    
    # Data config
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--temperature', type=float, default=0.1)
    
    # Stage 2 specific
    parser.add_argument('--samples_per_class', type=int, default=100,
                        help='Number of real/fake samples per class in stage 2')
    
    # Evaluation config
    parser.add_argument('--eval_interval', type=int, default=20,
                        help='Evaluate every N epochs in Stage 2')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True


def discover_deepfake_methods(dataset_root: str) -> List[str]:
    """
    Auto-discover all deepfake methods in dataset
    Returns list of methods containing 0_real and 1_fake subdirectories
    """
    methods = []
    if not os.path.exists(dataset_root):
        print(f"[Warning] Dataset root not found: {dataset_root}")
        return methods
    
    for item in os.listdir(dataset_root):
        method_path = os.path.join(dataset_root, item)
        if os.path.isdir(method_path):
            real_path = os.path.join(method_path, '0_real')
            fake_path = os.path.join(method_path, '1_fake')
            if os.path.exists(real_path) and os.path.exists(fake_path):
                methods.append(item)
    
    methods.sort()
    return methods

class FaceRealDataset(Dataset):
    """
    Stage 1 & Stage 2 Aux: Load real face images from multiple deepfake methods
    """
    def __init__(self, dataset_root: str, deepfake_methods: List[str], transform=None, split='train'):
        self.transform = transform
        self.images = []
        self.split = split
        
        for method in deepfake_methods:
            real_path = os.path.join(dataset_root, method, '0_real')
            if os.path.exists(real_path):
                method_images = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPEG', '*.JPG', '*.PNG']:
                    method_images.extend(glob.glob(os.path.join(real_path, '**', ext), recursive=True))
                
                # 核心防泄露逻辑：排序并按 8:2 划分
                method_images.sort()
                split_idx = int(len(method_images) * 0.9)
                
                if split == 'train':
                    self.images.extend(method_images[:split_idx])
                elif split == 'test':
                    self.images.extend(method_images[split_idx:])
        
        print(f"[FaceRealDataset - {split}] Loaded {len(self.images)} real face images from {len(deepfake_methods)} methods")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # label=0 for real

class FaceForgeryDataset(Dataset):
    """
    Evaluation Stage: Load real and fake face images from specific deepfake method
    """
    def __init__(self, dataset_root: str, deepfake_method: str, is_real: bool, transform=None, split='test'):
        self.transform = transform
        self.label = 0 if is_real else 1
        self.images = []
        self.split = split
        
        subdir = '0_real' if is_real else '1_fake'
        data_path = os.path.join(dataset_root, deepfake_method, subdir)
        
        if os.path.exists(data_path):
            method_images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPEG', '*.JPG', '*.PNG']:
                method_images.extend(glob.glob(os.path.join(data_path, '**', ext), recursive=True))
            
            # 核心防泄露逻辑：排序并按 8:2 划分
            method_images.sort()
            split_idx = int(len(method_images) * 0.9)
            
            if split == 'train':
                self.images.extend(method_images[:split_idx])
            elif split == 'test':
                self.images.extend(method_images[split_idx:])
        
        label_str = "real" if is_real else "fake"
        print(f"[FaceForgeryDataset - {split}] Loaded {len(self.images)} {label_str} images from {deepfake_method}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.label

class FaceDetDataset(Dataset):
    """
    Stage 2: Dataset for detection training with dual transforms
    Returns: (contrastive_views, supervised_view, label)
    """
    def __init__(self, dataset_root: str, deepfake_method: str, 
                 contrastive_transform=None, supervised_transform=None, split='train'):
        self.contrastive_transform = contrastive_transform
        self.supervised_transform = supervised_transform
        self.samples = []
        self.split = split
        
        # Load real images
        real_path = os.path.join(dataset_root, deepfake_method, '0_real')
        if os.path.exists(real_path):
            real_images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPEG', '*.JPG', '*.PNG']:
                real_images.extend(glob.glob(os.path.join(real_path, '**', ext), recursive=True))
            
            real_images.sort()
            split_idx = int(len(real_images) * 0.9) # 修改为 0.9
            target_images = real_images[:split_idx] if split == 'train' else real_images[split_idx:]
            self.samples.extend([(f, 0) for f in target_images])
        
        # Load fake images
        fake_path = os.path.join(dataset_root, deepfake_method, '1_fake')
        if os.path.exists(fake_path):
            fake_images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPEG', '*.JPG', '*.PNG']:
                fake_images.extend(glob.glob(os.path.join(fake_path, '**', ext), recursive=True))
            
            fake_images.sort()
            split_idx = int(len(fake_images) * 0.9) # 修改为 0.9
            target_images = fake_images[:split_idx] if split == 'train' else fake_images[split_idx:]
            self.samples.extend([(f, 1) for f in target_images])
        
        print(f"[FaceDetDataset - {split}] Loaded {len(self.samples)} samples from {deepfake_method}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.contrastive_transform:
            contrastive_views = self.contrastive_transform(image)
        else:
            contrastive_views = (image, image)
        
        if self.supervised_transform:
            supervised_view = self.supervised_transform(image)
        else:
            supervised_view = image
        
        return contrastive_views, supervised_view, label


# ============== Stage 1: Representation Learning ==============
def train_stage1(train_loader, model, criterion_selfcon, criterion_mse, criterion_ort, 
                 optimizer, epoch, args):
    """Train one epoch for Stage 1 (representation learning)"""
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    
    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        
        # Forward pass
        heter_features, homo_features = model.forward_proj(images)
        
        # Heterogeneous features: contrastive learning
        heter_features = F.normalize(heter_features, dim=1)
        f1, f2 = torch.split(heter_features, [bsz, bsz], dim=0)
        heter_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        
        # Homogeneous features: consistency with anchor
        homo_features = F.normalize(homo_features, dim=1)
        f3, f4 = torch.split(homo_features, [bsz, bsz], dim=0)
        
        # Anchor-based consistency
        anchor = f3[0]
        anchor_expanded = anchor.expand_as(f3)
        avg_mse_loss = criterion_mse(f3, anchor_expanded)
        
        # Orthogonality constraint between heterogeneous and homogeneous
        tgt = 2 * torch.empty(bsz).random_(2) - 1
        tgt = tgt.cuda()
        avg_ort_loss = criterion_ort(f1, f3, tgt)
        
        # Total loss
        loss = criterion_selfcon(heter_features) + avg_mse_loss + avg_ort_loss * 0.1
        
        # Update metrics
        losses.update(loss.item(), bsz)
        
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Print info
        if (idx + 1) % 10 == 0:
            print('Train: [{0}][{1}/{2}]\\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()
    
    return losses.avg


# ============== Stage 2: Forgery Detection ==============
def train_stage2(dataloader, model, criterion_auxcon, criterion_ce, optimizer, epoch, args):
    """Train one epoch for Stage 2 (forgery detection)"""
    train_loader, aux_loader = dataloader
    
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_supcon = AverageMeter()  # 新增：记录 SupCon loss
    losses_bce = AverageMeter()     # 新增：记录 BCE loss
    
    end = time.time()

    # 在同一个训练步骤（step）里，同时从两个不同的数据源各取出一批（Batch）数据进行训练
    # enumerate(train_loader):
    # train_loader 返回的是：((视图1, 视图2, 视图3), 标签)
    # enumerate 加上了索引。
    # 它产出的每一项是：(idx1, (images_train, labels_train))
    # zip会同时从两个 enumerate 中各取出一项，打包成一个元组。
    # images：对应 Dataset 返回的 contrastive_views
    # imgs：对应 Dataset 返回的 supervised_view。这是一个单视图，用于分类检测。
    for (idx, (images, imgs, labels)), (_, (aux_images, aux_labels)) in zip(enumerate(train_loader), enumerate(aux_loader)):
        data_time.update(time.time() - end)
        
        images = torch.cat([images[0], images[1]], dim=0) # images[0]第一种数据增强方法，images[1]第二种数据增强方法 image[0]:[16, 3, 224, 224] images[32,3,224,224]
        aux_images = torch.cat([aux_images[0], aux_images[1]], dim=0)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            aux_images = aux_images.cuda(non_blocking=True)
            aux_labels = aux_labels.cuda(non_blocking=True)
            imgs = imgs.cuda(non_blocking=True)
        bsz = labels.shape[0] + aux_labels.shape[0] #32
        
        totallabels = torch.cat((labels, aux_labels))
        
        # Forward pass
        bszdet = labels.shape[0]
        homo1, _ = model.forward_det(images) # contrastive_views提取同构特征
        _, det = model.forward_det(imgs) # supervised_view 分类
        homo1 = F.normalize(homo1, dim=1)
        f1, f2 = torch.split(homo1, [bszdet, bszdet], dim=0) #前 16 个属于“视图 1”，后 16 个属于“视图 2” [16,feature_dim]
        homo1 = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) #[16, 2, feature_dim]
        
        bszaux = aux_labels.shape[0] 
        _, homo2 = model.forward_proj(aux_images) # 真实图像的同构特征
        homo2 = F.normalize(homo2, dim=1)
        f1, f2 = torch.split(homo2, [bszaux, bszaux], dim=0)
        homo2 = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        
        homo_features = torch.cat((homo1, homo2)) 
        
        # Combined loss
        # lambda_aux = args.lambda_aux # 0.3 
        # loss = lambda_aux * criterion_auxcon(homo_features, totallabels) + \
        #        (1 - lambda_aux) * criterion_ce(det.squeeze(-1), labels.float())
        # loss = 1.0 * criterion_auxcon(homo_features, totallabels) + \
        #        0.5 * criterion_ce(det.squeeze(-1), labels.float())
        # Combined loss
        loss_supcon_val = criterion_auxcon(homo_features, totallabels)
        loss_bce_val = criterion_ce(det.squeeze(-1), labels.float())
        
        loss = 1.0 * loss_supcon_val + 0.5 * loss_bce_val
        
        # Update metrics
        losses.update(loss.item(), bsz)
        losses_supcon.update(loss_supcon_val.item(), bsz)  # 新增：更新 supcon 记录
        losses_bce.update(loss_bce_val.item(), bsz)        # 新增：更新 bce 记录

        # Update metrics
        losses.update(loss.item(), bsz)
        
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Print info
        # if (idx + 1) % 10 == 0:
        #     print('Train2: [{0}][{1}/{2}]\\t'
        #           'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\\t'
        #           'DT {data_time.val:.3f} ({data_time.avg:.3f})\\t'
        #           'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
        #            epoch, idx + 1, len(train_loader), batch_time=batch_time,
        #            data_time=data_time, loss=losses))
        #     sys.stdout.flush()
        # Print info
        if (idx + 1) % 10 == 0:
            print('Train2: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'supcon {loss_supcon.val:.3f} ({loss_supcon.avg:.3f})\t'
                  'bce {loss_bce.val:.3f} ({loss_bce.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, 
                   loss_supcon=losses_supcon, loss_bce=losses_bce))
            sys.stdout.flush()
    
    return losses.avg


# ============== Evaluation Functions ==============
@torch.no_grad()
def evaluate_stage2(model, dataloader, device):
    """
    Evaluate Stage 2 model on detection task
    Returns: dict with acc, ap, auc metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    train_loader, aux_loader = dataloader
    
    for (images, imgs, labels), (aux_images, aux_labels) in zip(train_loader, aux_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        _, det = model.forward_det(imgs)
        preds = torch.sigmoid(det.squeeze(-1))
        
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # Compute metrics
    preds_binary = (all_preds > 0.5).astype(float)
    acc = accuracy_score(all_labels, preds_binary)
    
    try:
        ap = average_precision_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_preds)
    except:
        ap, auc = 0.0, 0.0
    
    return {'acc': acc, 'ap': ap, 'auc': auc}


@torch.no_grad()
def evaluate_per_method(model, dataset_root, deepfake_methods, transform, device, batch_size=64):
    """
    Per-method fine-grained evaluation
    Returns: dict mapping method -> {acc, ap, auc}
    """
    model.eval()
    
    print("\n" + "-" * 60)
    print("Per-Method Evaluation")
    print("-" * 60)
    
    results = {}
    for method in deepfake_methods:
        # Load method-specific dataset
        real_ds = FaceForgeryDataset(dataset_root, method, is_real=True, transform=transform,split='test')
        fake_ds = FaceForgeryDataset(dataset_root, method, is_real=False, transform=transform,split='test')
        
        if len(real_ds) == 0 or len(fake_ds) == 0:
            print(f"  {method}: SKIPPED (no data)")
            continue
        
        method_ds = ConcatDataset([real_ds, fake_ds])
        method_loader = DataLoader(
            method_ds, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )
        
        all_preds = []
        all_labels = []
        
        for images, labels in method_loader:
            images = images.to(device)
            
            # Forward pass
            _, det = model.forward_det(images)
            preds = torch.sigmoid(det.squeeze(-1))
            
            all_preds.append(preds.cpu())
            all_labels.append(labels)
        
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        # Compute metrics
        preds_binary = (all_preds > 0.5).astype(float)
        acc = accuracy_score(all_labels, preds_binary)
        
        try:
            ap = average_precision_score(all_labels, all_preds)
            auc = roc_auc_score(all_labels, all_preds)
        except:
            ap, auc = 0.0, 0.0
        
        results[method] = {'acc': acc, 'ap': ap, 'auc': auc}
        print(f"  {method:30s}  Acc={acc:.4f}  AP={ap:.4f}  AUC={auc:.4f}")
    
    if results:
        accs = [v['acc'] for v in results.values()]
        aps = [v['ap'] for v in results.values()]
        aucs = [v['auc'] for v in results.values()]
        
        print("-" * 60)
        print(f"  {'AVERAGE':30s}  Acc={np.mean(accs):.4f}  AP={np.mean(aps):.4f}  AUC={np.mean(aucs):.4f}")
        print(f"  {'STD':30s}  Acc={np.std(accs):.4f}  AP={np.std(aps):.4f}  AUC={np.std(aucs):.4f}")
        print(f"  {'WORST':30s}  Acc={min(accs):.4f}  AUC={min(aucs):.4f}")
        print(f"  {'BEST':30s}  Acc={max(accs):.4f}  AUC={max(aucs):.4f}")
    
    return results


# ============== Main Training ==============
def main():
    args = get_args()
    
    # Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    cudnn.benchmark = True
    
    # Discover deepfake methods
    if args.use_face_dataset:
        deepfake_methods = discover_deepfake_methods(args.dataset_root)
        if args.deepfake_methods != 'all':
            specified_methods = [m.strip() for m in args.deepfake_methods.split(',')]
            deepfake_methods = [m for m in deepfake_methods if m in specified_methods]
        
        if not deepfake_methods:
            print(f"[Error] No valid deepfake methods found in {args.dataset_root}")
            print("[Info] Expected structure: {dataset_root}/{method}/[0_real, 1_fake]")
            return
        
        print("=" * 60)
        print("Face Forgery Detection Training (xudong_fine - Face Dataset)")
        print("=" * 60)
        print(f"Dataset Root: {args.dataset_root}")
        print(f"Deepfake Methods: {len(deepfake_methods)}")
        for method in deepfake_methods:
            print(f"  - {method}")
    else:
        # Legacy class-based mode
        classes = args.classes.split(',') if args.classes else []
        print("=" * 60)
        print("Face Forgery Detection Training (xudong_fine - Legacy Mode)")
        print("=" * 60)
        print(f"Classes: {len(classes)}")
    
    print(f"Backbone: {args.backbone}")
    print(f"Stage 1: {args.stage1_epochs} epochs, batch={args.stage1_batch_size}, lr={args.stage1_lr}")
    print(f"Stage 2: {args.stage2_epochs} epochs, batch={args.stage2_batch_size}, lr={args.stage2_lr}")
    print("=" * 60)
    
    # ==================== Stage 1: Representation Learning ====================
    print("\n" + "=" * 60)
    print("Stage 1: Representation Learning (Real Images Only)")
    print("=" * 60)
    
    # Model
    model = OurNet(backbone=args.backbone).cuda()
    
    # Loss functions
    criterion_selfcon = SupConLoss(temperature=args.temperature).cuda()
    criterion_mse = nn.MSELoss().cuda()
    criterion_ort = nn.CosineEmbeddingLoss(margin=0.2).cuda()
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=args.img_size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset - face mode uses FaceRealDataset, legacy uses Auxdataset
    if args.use_face_dataset:
        train_dataset = FaceRealDataset(
            args.dataset_root, deepfake_methods,
            transform=TwoCropTransform(train_transform),
            split='train'
        )
    else:
        data_folder = args.datapath
        dset_lst = []
        for cls in classes:
            root = data_folder + cls + '/0_real/'
            dset = Auxdataset(root=root, transform=TwoCropTransform(train_transform))
            dset_lst.append(dset)
        train_dataset = ConcatDataset(dset_lst)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.stage1_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    print(f"[Data] Loaded {len(train_dataset)} training samples")
    
    # Optimizer
    # optimizer = optim.SGD(model.parameters(),
    #                       lr=args.stage1_lr,
    #                       momentum=args.stage1_momentum,
    #                       weight_decay=args.stage1_weight_decay)
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
            
    optimizer = optim.SGD([
        {'params': backbone_params, 'lr': args.stage1_lr},
        {'params': head_params, 'lr': args.stage1_head_lr}
    ], momentum=args.stage1_momentum, weight_decay=args.stage1_weight_decay)
    
    # Learning rate schedule
    total_epochs = args.stage1_epochs
    learning_rate = args.stage1_lr
    lr_decay_rate = args.lr_decay
    warm_epochs = args.warmup_epochs
    warmup_from = args.warmup_from
    
    if args.cosine:
        eta_min = learning_rate * (lr_decay_rate ** 3)
        warmup_to = eta_min + (learning_rate - eta_min) * \
                    (1 + math.cos(math.pi * warm_epochs / total_epochs)) / 2
    else:
        warmup_to = learning_rate
    
    # Tensorboard logger
    loggerpath = os.path.join(args.savepath, f'stage1_tensorboard_{args.backbone}')
    savefolder = os.path.join(args.savepath, f'stage1_models_{args.backbone}')
    os.makedirs(loggerpath, exist_ok=True)
    os.makedirs(savefolder, exist_ok=True)
    
    if tb_logger:
        logger = tb_logger.Logger(logdir=loggerpath, flush_secs=2)
    else:
        logger = None
    
    # Training loop
    best_loss = 100
    start_epoch = 1
    
    for epoch in range(start_epoch, total_epochs + 1):
        adjust_learning_rate(optimizer, epoch, lr=learning_rate, 
                           lr_decay_rate=lr_decay_rate, total_epochs=total_epochs, 
                           cos=args.cosine)
        
        time1 = time.time()
        loss = train_stage1(train_loader, model, criterion_selfcon, criterion_mse, 
                           criterion_ort, optimizer, epoch, args)
        time2 = time.time()
        
        print(f'Epoch {epoch}, total time {time2 - time1:.2f}s, loss {loss:.4f}')
        
        if logger:
            logger.log_value('train_loss', loss, epoch)
            logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoints
        if epoch % 50 == 0:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            torch.save(state, os.path.join(savefolder, f'ckpt_epoch_{epoch}.pth'))
        
        if loss < best_loss:
            best_loss = loss
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            torch.save(state, os.path.join(savefolder, 'best_model.pth'))
        
        if epoch == total_epochs:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            torch.save(state, os.path.join(savefolder, 'final_epoch.pth'))
    
    print(f"\\n[Stage 1] Completed! Best loss: {best_loss:.4f}")
    stage1_model_path = os.path.join(savefolder, 'best_model.pth')
    
    # ==================== Stage 2: Forgery Detection ====================
    print("\\n" + "=" * 60)
    print("Stage 2: Forgery Detection (Real + Fake Images)")
    print("=" * 60)
    
    # Load Stage 1 model
    if args.stage1_model:
        stage1_model_path = args.stage1_model
    
    if os.path.exists(stage1_model_path):
        print(f'[Load] Loading Stage 1 model from {stage1_model_path}')
        ckpt = torch.load(stage1_model_path)
        model.load_state_dict(ckpt['model'])
        model.det_fc1.load_state_dict(model.aux_fc2.state_dict())
    else:
        print(f'[Warning] Stage 1 model not found at {stage1_model_path}')
        print('[Info] Continuing with current model state')
    
    # # Freeze backbone (新增：冻结微调过的主干网络)
    # for p in model.backbone.parameters():
    #     p.requires_grad = False

    # Freeze auxiliary heads
    for p in model.aux_fc1.parameters():
        p.requires_grad = False
    for p in model.aux_fc2.parameters():
        p.requires_grad = False
    
    # Loss functions
    criterion_auxcon = SupConLoss(temperature=args.temperature).cuda()
    criterion_ce = nn.BCEWithLogitsLoss().cuda()
    
    # Data transforms
    contrastive_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=args.img_size, scale=(0.2, 1.)),
        transforms.Lambda(lambda img: data_augment(img)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    supervised_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=args.img_size, scale=(0.2, 1.)),
        transforms.Lambda(lambda img: data_augment(img)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Detection dataset
    dset_lst, aux_lst = [], []
    
    if args.use_face_dataset:
        # Face dataset mode: load from all deepfake methods
        for method in deepfake_methods:
            dset = FaceDetDataset(
                args.dataset_root, method,
                contrastive_transform=TwoCropTransform(contrastive_transform),
                supervised_transform=supervised_transform,
                split='train'
            )
            
            # # Sample if needed
            # if args.samples_per_class > 0 and len(dset) > args.samples_per_class * 2:
            #     import random
            #     indices = list(range(len(dset)))
            #     random.shuffle(indices)
            #     dset = Subset(dset, indices[:args.samples_per_class * 2])
            
            dset_lst.append(dset)
        
        # Auxiliary dataset (real images only)
        aux_dataset = FaceRealDataset(
            args.dataset_root, deepfake_methods,
            transform=TwoCropTransform(contrastive_transform),
            split='train'
        )
    # else:
    #     # Legacy class-based mode
    #     for cls in classes:
    #         root = os.path.join(data_folder, cls)
    #         dset = Detdataset(root=root,
    #                          transform=[TwoCropTransform(contrastive_transform),
    #                                    supervised_transform])
            
    #         # Sample 100 real + 100 fake per class
    #         real_indices = [i for i, (path, target) in enumerate(dset.samples) if "real" in path]
    #         fake_indices = [i for i, (path, target) in enumerate(dset.samples) if "fake" in path]
            
    #         import random
    #         random.shuffle(real_indices)
    #         random.shuffle(fake_indices)
    #         real_indices = real_indices[:args.samples_per_class]
    #         fake_indices = fake_indices[:args.samples_per_class]
            
    #         selected_indices = real_indices + fake_indices
    #         subset = Subset(dset, selected_indices)
    #         dset_lst.append(subset)
        
    #     # Auxiliary dataset (real images)
    #     for cls in classes:
    #         realroot = data_folder + cls + '/0_real/'
    #         realset = Auxdataset(root=realroot, 
    #                            transform=TwoCropTransform(contrastive_transform))
    #         aux_lst.append(realset)
        
    #     aux_lst = aux_lst + aux_lst
    #     aux_dataset = ConcatDataset(aux_lst)
    
    train_dataset = ConcatDataset(dset_lst)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.stage2_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)  # 混合图像，三种数据增强方法
    
    aux_loader = torch.utils.data.DataLoader(
        aux_dataset, batch_size=args.stage2_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)  # 真实图像，两种数据增强方法
    
    print(f"[Data] Detection dataset: {len(train_dataset)} samples")
    print(f"[Data] Auxiliary dataset: {len(aux_dataset)} samples")
    
    # # Optimizer (only trainable parameters)
    # optimizer = torch.optim.SGD(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=args.stage2_lr, 
    #     momentum=args.stage2_momentum, 
    #     weight_decay=args.stage2_weight_decay
    # )

    # Optimizer with differential learning rates
    fine_tune_lr = 1e-5  # 主干网络和 det_fc1 的极小微调学习率
    
    backbone_params = []
    det_fc1_params = []
    det_fc2_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # 跳过已经冻结的 aux_fc1 和 aux_fc2
            
        if 'backbone' in name:
            backbone_params.append(param)
        elif 'det_fc1' in name:
            det_fc1_params.append(param)
        elif 'det_fc2' in name:
            det_fc2_params.append(param)

    optimizer = torch.optim.SGD([
        {'params': backbone_params, 'lr': fine_tune_lr},
        {'params': det_fc1_params, 'lr': fine_tune_lr},
        {'params': det_fc2_params, 'lr': args.stage2_lr}  # args.stage2_lr 默认就是 1e-2
    ], momentum=args.stage2_momentum, weight_decay=args.stage2_weight_decay)
    
    # Learning rate schedule
    total_epochs = args.stage2_epochs
    learning_rate = args.stage2_lr
    
    # Tensorboard logger
    loggerpath = os.path.join(args.savepath, 'stage2_detnet_enhance', f'path_tensorboard_{args.backbone}')
    savefolder = os.path.join(args.savepath, 'stage2_detnet_enhance', f'path_models_{args.backbone}')
    os.makedirs(loggerpath, exist_ok=True)
    os.makedirs(savefolder, exist_ok=True)
    
    if tb_logger:
        logger = tb_logger.Logger(logdir=loggerpath, flush_secs=2)
    else:
        logger = None
    
    # Training loop
    best_loss = 100
    best_acc = 0.0
    best_metrics = None
    best_method_metrics = None
    start_epoch = 1
    dataloader = [train_loader, aux_loader]
    
    for epoch in range(start_epoch, total_epochs + 1):
        adjust_learning_rate(optimizer, epoch, lr=learning_rate,
                           lr_decay_rate=lr_decay_rate, total_epochs=total_epochs,
                           cos=args.cosine)
        
        time1 = time.time()
        loss = train_stage2(dataloader, model, criterion_auxcon, criterion_ce, 
                           optimizer, epoch, args)
        time2 = time.time()
        
        if logger:
            logger.log_value('train_loss', loss, epoch)
            logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Evaluation (every eval_interval epochs)
        if args.use_face_dataset and epoch % args.eval_interval == 0:
            print(f"\n--- Stage2 Epoch {epoch}/{total_epochs} ---")
            
            # Overall evaluation
            metrics = evaluate_stage2(model, dataloader, device)
            print(f"[Eval Overall] Acc: {metrics['acc']:.4f}, AP: {metrics['ap']:.4f}, AUC: {metrics['auc']:.4f}")
            
            if logger:
                logger.log_value('eval_acc', metrics['acc'], epoch)
                logger.log_value('eval_ap', metrics['ap'], epoch)
                logger.log_value('eval_auc', metrics['auc'], epoch)
            
            # Per-method evaluation
            eval_transform = transforms.Compose([
                transforms.Resize(int(args.img_size * 1.14)),
                transforms.CenterCrop(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            method_metrics = evaluate_per_method(
                model, args.dataset_root, deepfake_methods,
                eval_transform, device, batch_size=args.stage2_batch_size
            )
            
            # Save best model based on accuracy
            if metrics['acc'] > best_acc:
                best_acc = metrics['acc']
                best_metrics = metrics
                best_method_metrics = method_metrics
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'metrics': metrics,
                    'method_metrics': method_metrics,
                }
                torch.save(state, os.path.join(savefolder, 'best_model_acc.pth'))
                print(f"[Save] Best model saved (acc={best_acc:.4f})")
        
        # Save checkpoints
        if epoch % 20 == 0:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            torch.save(state, os.path.join(savefolder, f'ckpt_epoch_{epoch}.pth'))
        
        if loss < best_loss:
            best_loss = loss
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            torch.save(state, os.path.join(savefolder, 'best_model.pth'))
        
        if epoch == total_epochs:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            torch.save(state, os.path.join(savefolder, 'final_epoch.pth'))
    
    print(f"\\n[Stage 2] Completed! Best loss: {best_loss:.4f}")
    print("\\n" + "=" * 60)
    print("Training Completed!")
    print(f"Models saved to: {args.savepath}")
    print("=" * 60)


if __name__ == '__main__':
    main()
