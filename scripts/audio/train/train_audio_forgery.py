#!/usr/bin/env python3
"""
Audio Forgery Detection Training Script
基于自然痕迹提取与筛选的音频深度伪造检测

两阶段训练:
Stage 1: 通过 real 音频提取真实音频的同构特征 (homogeneous features)
         - 学习真实语音的"自然痕迹": 噪声底噪、韵律一致性、频谱自然性
         - 使用对比学习 (SupConLoss) 学习异构特征
         - 使用一致性学习 (MSE) 学习同构特征
         - 正交约束确保两种特征互补

Stage 2: 使用 real + fake 音频,提炼出真正没有被学会的音频特征
         - 伪造音频缺失或扭曲了真实语音的自然痕迹
         - 通过对比学习强化同构特征的判别能力
         - 二分类检测器识别伪造音频
"""

import os
import sys
import time
import math
import argparse
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader

# Import base infrastructure
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_audio_natural_trace import (
    OurNet, SupConLoss, AverageMeter,
    AudioMelTransform, TwoCropAudioTransform, audio_augment,
    AudioRealDataset, AudioForgeryDataset, AudioDetDataset, AudioTestDataset,
    adjust_learning_rate, warmup_learning_rate, save_model,
    compute_eer
)

try:
    import tensorboard_logger as tb_logger
except ImportError:
    print("[Warning] tensorboard_logger not found. Logging disabled.")
    tb_logger = None


# ============== Configuration ==============
def get_args():
    parser = argparse.ArgumentParser(description='Audio Forgery Detection Training')

    # Basic config
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--savepath', type=str, default='results/audio')
    parser.add_argument('--stage1_model', type=str, default='')

    # Dataset config
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='Dataset root directory (ASVspoof format: bonafide/ and spoof/)')

    # Model config
    parser.add_argument('--backbone', type=str, default='facebook/wav2vec2-base',
                        help='Backbone model (wav2vec2, whisper, hubert)')

    # Stage 1 config (representation learning)
    parser.add_argument('--stage1_epochs', type=int, default=50)
    parser.add_argument('--stage1_batch_size', type=int, default=32)
    parser.add_argument('--stage1_lr', type=float, default=1e-5)
    parser.add_argument('--stage1_head_lr', type=float, default=0.01)
    parser.add_argument('--stage1_momentum', type=float, default=0.9)
    parser.add_argument('--stage1_weight_decay', type=float, default=1e-4)

    # Stage 2 config (forgery detection)
    parser.add_argument('--stage2_epochs', type=int, default=50)
    parser.add_argument('--stage2_batch_size', type=int, default=16)
    parser.add_argument('--stage2_lr', type=float, default=0.01)
    parser.add_argument('--stage2_momentum', type=float, default=0.9)
    parser.add_argument('--stage2_weight_decay', type=float, default=1e-4)
    parser.add_argument('--lambda_aux', type=float, default=0.3)

    # Optimizer config
    parser.add_argument('--cosine', action='store_true', default=False)
    parser.add_argument('--warmup_from', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--lr_decay', type=float, default=0.1)

    # Data config
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=0.1)

    # Audio config
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--max_length', type=float, default=4.0,
                        help='Maximum audio length in seconds')

    # Evaluation config
    parser.add_argument('--eval_interval', type=int, default=10)

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True


# ============== Stage 1: Representation Learning ==============
def train_stage1(train_loader, model, criterion_selfcon, criterion_mse, criterion_ort,
                 optimizer, epoch, args):
    """
    Stage 1: 通过 real 音频提取真实音频的同构特征

    核心思想:
    - 异构特征 (heter_features): 捕获多样化的声学特征 (说话人、内容)
    - 同构特征 (homo_features): 捕获一致的"自然性"痕迹 (噪声底噪、韵律一致性)
    - 正交约束: 确保两种特征互补
    """
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (mel_tensors, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # mel_tensors: list of 2 views, each [B, 3, 224, 224]
        mel_tensors = torch.cat([mel_tensors[0], mel_tensors[1]], dim=0)
        if torch.cuda.is_available():
            mel_tensors = mel_tensors.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # Forward pass
        heter_features, homo_features = model.forward_proj(mel_tensors)

        # Heterogeneous features: contrastive learning
        heter_features = F.normalize(heter_features, dim=1)
        f1, f2 = torch.split(heter_features, [bsz, bsz], dim=0)
        heter_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        # Homogeneous features: consistency with anchor
        homo_features = F.normalize(homo_features, dim=1)
        f3, f4 = torch.split(homo_features, [bsz, bsz], dim=0)

        # Anchor-based consistency (第一个样本作为锚点)
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
            print('Train Stage1: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


# ============== Stage 2: Forgery Detection ==============
def train_stage2(dataloader, model, criterion_auxcon, criterion_ce, optimizer, epoch, args):
    """
    Stage 2: 使用 real + fake 音频,提炼出真正没有被学会的音频特征

    核心思想:
    - 伪造音频缺失或扭曲了真实语音的自然痕迹
    - 通过对比学习强化同构特征的判别能力
    - 二分类检测器识别伪造音频
    """
    train_loader, aux_loader = dataloader

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_supcon = AverageMeter()
    losses_bce = AverageMeter()

    end = time.time()

    for (idx, (images, imgs, labels)), (_, (aux_images, aux_labels)) in zip(enumerate(train_loader), enumerate(aux_loader)):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        aux_images = torch.cat([aux_images[0], aux_images[1]], dim=0)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            aux_images = aux_images.cuda(non_blocking=True)
            aux_labels = aux_labels.cuda(non_blocking=True)
            imgs = imgs.cuda(non_blocking=True)
        bsz = labels.shape[0] + aux_labels.shape[0]

        totallabels = torch.cat((labels, aux_labels))

        # Forward pass
        bszdet = labels.shape[0]
        homo1, _ = model.forward_det(images)
        _, det = model.forward_det(imgs)
        homo1 = F.normalize(homo1, dim=1)
        f1, f2 = torch.split(homo1, [bszdet, bszdet], dim=0)
        homo1 = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        bszaux = aux_labels.shape[0]
        _, homo2 = model.forward_proj(aux_images)
        homo2 = F.normalize(homo2, dim=1)
        f1, f2 = torch.split(homo2, [bszaux, bszaux], dim=0)
        homo2 = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        homo_features = torch.cat((homo1, homo2))

        # Combined loss: BCE 主导，SupCon 作为正则
        # 之前 0.3*supcon(≈4.5) + 0.7*bce(≈0.3) 导致 SupCon 梯度压制分类
        loss_supcon_val = criterion_auxcon(homo_features, totallabels)
        loss_bce_val = criterion_ce(det.squeeze(-1), labels.float())
        loss = 0.1 * loss_supcon_val + 1.0 * loss_bce_val

        # Update metrics
        losses.update(loss.item(), bsz)
        losses_supcon.update(loss_supcon_val.item(), bsz)
        losses_bce.update(loss_bce_val.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print info
        if (idx + 1) % 10 == 0:
            print('Train Stage2: [{0}][{1}/{2}]\t'
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
def evaluate_stage2(model, test_loader, device):
    """
    Evaluate Stage 2 model on detection task
    使用独立测试集，不做任何采样
    Returns: dict with acc, ap, auc, eer metrics
    """
    model.eval()

    all_preds = []
    all_labels = []

    for imgs, labels in test_loader:
        imgs = imgs.to(device)

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
        eer, _ = compute_eer(all_labels, all_preds)
    except:
        ap, auc, eer = 0.0, 0.0, 0.5

    return {'acc': acc, 'ap': ap, 'auc': auc, 'eer': eer}


# ============== Main Training ==============
def main():
    args = get_args()

    # Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    cudnn.benchmark = True

    print("=" * 60)
    print("Audio Forgery Detection Training")
    print("基于自然痕迹提取与筛选")
    print("=" * 60)
    print(f"Dataset Root: {args.dataset_root}")
    print(f"Backbone: {args.backbone}")
    print(f"Stage 1: {args.stage1_epochs} epochs, batch={args.stage1_batch_size}, lr={args.stage1_lr}")
    print(f"Stage 2: {args.stage2_epochs} epochs, batch={args.stage2_batch_size}, lr={args.stage2_lr}")
    print("=" * 60)

    # ==================== Stage 1: Representation Learning ====================
    print("\n" + "=" * 60)
    print("Stage 1: 通过 real 音频提取真实音频的同构特征")
    print("=" * 60)

    # Model
    model = OurNet(backbone=args.backbone).cuda()

    # Loss functions
    criterion_selfcon = SupConLoss(temperature=args.temperature).cuda()
    criterion_mse = nn.MSELoss().cuda()
    criterion_ort = nn.CosineEmbeddingLoss(margin=0.2).cuda()

    # Data transforms
    base_transform = AudioMelTransform(
        sample_rate=args.sample_rate,
        max_length=args.max_length
    )
    train_transform = TwoCropAudioTransform(base_transform)

    # Dataset
    train_dataset = AudioRealDataset(
        args.dataset_root,
        transform=train_transform,
        split='train'
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.stage1_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    print(f"[Data] Loaded {len(train_dataset)} training samples")

    # Optimizer with differential learning rates
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

    # Tensorboard logger
    loggerpath = os.path.join(args.savepath, f'stage1_tensorboard_{args.backbone.replace("/", "_")}')
    savefolder = os.path.join(args.savepath, f'stage1_models_{args.backbone.replace("/", "_")}')
    os.makedirs(loggerpath, exist_ok=True)
    os.makedirs(savefolder, exist_ok=True)

    if tb_logger:
        logger = tb_logger.Logger(logdir=loggerpath, flush_secs=2)
    else:
        logger = None

    # Training loop
    best_loss = 100
    start_epoch = 1
    total_epochs = args.stage1_epochs
    learning_rate = args.stage1_lr

    for epoch in range(start_epoch, total_epochs + 1):
        adjust_learning_rate(optimizer, epoch, lr=learning_rate,
                           lr_decay_rate=args.lr_decay, total_epochs=total_epochs,
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

    print(f"\n[Stage 1] Completed! Best loss: {best_loss:.4f}")
    stage1_model_path = os.path.join(savefolder, 'best_model.pth')

    # ==================== Stage 2: Forgery Detection ====================
    print("\n" + "=" * 60)
    print("Stage 2: 使用 real + fake 音频,提炼出真正没有被学会的音频特征")
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

    # Freeze auxiliary heads
    for p in model.aux_fc1.parameters():
        p.requires_grad = False
    for p in model.aux_fc2.parameters():
        p.requires_grad = False

    # Loss functions
    criterion_auxcon = SupConLoss(temperature=args.temperature).cuda()

    # Data transforms
    contrastive_transform = TwoCropAudioTransform(base_transform)
    supervised_transform = base_transform

    # Detection dataset
    train_dataset = AudioDetDataset(
        args.dataset_root,
        contrastive_transform=contrastive_transform,
        supervised_transform=supervised_transform,
        split='train'
    )

    # Auxiliary dataset (real audio only)
    aux_dataset = AudioRealDataset(
        args.dataset_root,
        transform=contrastive_transform,
        split='train'
    )

    # 修复类别不均衡: 计算 pos_weight (spoof/bonafide 比例)
    n_real = len([s for s in train_dataset.samples if s[1] == 0])
    n_fake = len([s for s in train_dataset.samples if s[1] == 1])
    pos_weight = torch.tensor([n_fake / max(n_real, 1)]).cuda()
    print(f"[Data] real={n_real}, fake={n_fake}, pos_weight={pos_weight.item():.2f}")
    criterion_ce = nn.BCEWithLogitsLoss(pos_weight=pos_weight).cuda()

    from torch.utils.data import WeightedRandomSampler

    # WeightedRandomSampler 让 real/fake 各占 50%
    labels = [s[1] for s in train_dataset.samples]
    class_counts = [labels.count(0), labels.count(1)]
    weights = [1.0 / class_counts[l] for l in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_dataset, batch_size=args.stage2_batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True)

    aux_loader = DataLoader(
        aux_dataset, batch_size=args.stage2_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)

    print(f"[Data] Detection dataset: {len(train_dataset)} samples")
    print(f"[Data] Auxiliary dataset: {len(aux_dataset)} samples")

    # 独立测试集 (split='test', 真实分布, 不做采样)
    test_dataset = AudioTestDataset(
        args.dataset_root,
        transform=supervised_transform,
        split='test'
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.stage2_batch_size * 4, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    print(f"[Data] Test dataset: {len(test_dataset)} samples")

    # Optimizer with differential learning rates
    fine_tune_lr = 1e-5

    backbone_params = []
    det_fc1_params = []
    det_fc2_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'backbone' in name:
            backbone_params.append(param)
        elif 'det_fc1' in name:
            det_fc1_params.append(param)
        elif 'det_fc2' in name:
            det_fc2_params.append(param)

    optimizer = optim.SGD([
        {'params': backbone_params, 'lr': fine_tune_lr},
        {'params': det_fc1_params, 'lr': fine_tune_lr},
        {'params': det_fc2_params, 'lr': args.stage2_lr}
    ], momentum=args.stage2_momentum, weight_decay=args.stage2_weight_decay)

    # Tensorboard logger
    loggerpath = os.path.join(args.savepath, 'stage2_detnet', f'tensorboard_{args.backbone.replace("/", "_")}')
    savefolder = os.path.join(args.savepath, 'stage2_detnet', f'models_{args.backbone.replace("/", "_")}')
    os.makedirs(loggerpath, exist_ok=True)
    os.makedirs(savefolder, exist_ok=True)

    if tb_logger:
        logger = tb_logger.Logger(logdir=loggerpath, flush_secs=2)
    else:
        logger = None

    # Training loop
    best_loss = 100
    best_acc = 0.0
    best_eer = 1.0
    start_epoch = 1
    total_epochs = args.stage2_epochs
    learning_rate = args.stage2_lr
    dataloader = [train_loader, aux_loader]

    for epoch in range(start_epoch, total_epochs + 1):
        adjust_learning_rate(optimizer, epoch, lr=learning_rate,
                           lr_decay_rate=args.lr_decay, total_epochs=total_epochs,
                           cos=args.cosine)

        time1 = time.time()
        loss = train_stage2(dataloader, model, criterion_auxcon, criterion_ce,
                           optimizer, epoch, args)
        time2 = time.time()

        if logger:
            logger.log_value('train_loss', loss, epoch)
            logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # Evaluation
        if epoch % args.eval_interval == 0:
            print(f"\n--- Stage2 Epoch {epoch}/{total_epochs} ---")

            metrics = evaluate_stage2(model, test_loader, device)
            print(f"[Eval] Acc: {metrics['acc']:.4f}, AP: {metrics['ap']:.4f}, "
                  f"AUC: {metrics['auc']:.4f}, EER: {metrics['eer']:.4f}")

            if logger:
                logger.log_value('eval_acc', metrics['acc'], epoch)
                logger.log_value('eval_ap', metrics['ap'], epoch)
                logger.log_value('eval_auc', metrics['auc'], epoch)
                logger.log_value('eval_eer', metrics['eer'], epoch)

            # Save best model based on EER
            if metrics['eer'] < best_eer:
                best_eer = metrics['eer']
                best_acc = metrics['acc']
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'metrics': metrics,
                }
                torch.save(state, os.path.join(savefolder, 'best_model_eer.pth'))
                print(f"[Save] Best model saved (eer={best_eer:.4f})")

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

    print(f"\n[Stage 2] Completed! Best EER: {best_eer:.4f}, Best Acc: {best_acc:.4f}")
    print("\n" + "=" * 60)
    print("Training Completed!")
    print(f"Models saved to: {args.savepath}")
    print("=" * 60)


if __name__ == '__main__':
    main()
