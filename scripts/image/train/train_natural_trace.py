"""
Natural Trace Base Infrastructure
Consolidates core components:
- Model architectures (OurNet from model2.py)
- Loss functions (SupConLoss from losses.py)
- Utility functions (from util.py)
- Dataset classes and augmentations (from dataset.py)
"""

from __future__ import print_function
import os
import math
import numpy as np
from io import BytesIO
from random import random, choice
from scipy.ndimage import gaussian_filter
from fastervit import create_model
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image, ImageFile
from transformers import AutoModel
ImageFile.LOAD_TRUNCATED_IMAGES = True
import timm


# ============== Utility Classes ==============
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
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


def adjust_learning_rate(optimizer, epoch, lr=0.1, lr_decay_rate=0.1, lr_decay_epochs='700,800,900', total_epochs=200, cos=True):
    """Adjust learning rate with cosine annealing or step decay"""
    if cos:
        eta_min = lr * (lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / total_epochs)) / 2
    else:
        lr_decay_epochs = [int(e) for e in lr_decay_epochs.split(',')]
        steps = np.sum(epoch > np.asarray(lr_decay_epochs))
        if steps > 0:
            lr = lr * (lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(epoch, batch_id, total_batches, optimizer, warmup_from, warmup_to, warm_epochs):
    """Warmup learning rate"""
    if epoch <= warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / (warm_epochs * total_batches)
        lr = warmup_from + p * (warmup_to - warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(model, learning_rate, momentum, weight_decay):
    """Create SGD optimizer"""
    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate,
                          momentum=momentum,
                          weight_decay=weight_decay)
    return optimizer


def save_model(model, optimizer, epoch, save_file):
    """Save model checkpoint"""
    print('==> Saving...')
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


# ============== Loss Functions ==============
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


# ============== Data Augmentation Functions ==============
def sample_continuous(s):
    """Sample from continuous range"""
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    """Sample from discrete choices"""
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    """Apply Gaussian blur"""
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    """JPEG compression using cv2"""
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    """JPEG compression using PIL"""
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}

def jpeg_from_key(img, compress_val, key):
    """Apply JPEG compression using specified method"""
    method = jpeg_dict[key]
    return method(img, compress_val)


def gauss_noise(image, mean=0, var=3):
    """Add Gaussian noise to image"""
    image = np.array(image)
    noise = np.random.normal(mean, var, image.shape)
    out = image + noise
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)
    return Image.fromarray(out)


def data_augment(img):
    """Apply data augmentation (JPEG compression)"""
    img = np.array(img)
    jpg_method = ['cv2', 'pil']
    jpg_qual = 70
    method = sample_discrete(jpg_method)
    img = jpeg_from_key(img, jpg_qual, method)
    return Image.fromarray(img)


# ============== Dataset Classes ==============
class Auxdataset(Dataset):
    """Dataset for auxiliary training (real images only)"""
    def __init__(self, root, transform=None):
        self.root_dir = root
        self.transform = transform
        self.images = os.listdir(self.root_dir)
        self.labels = [0 for i in range(len(self.images))]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_index = self.images[index]
        label = self.labels[index]
        img_path = os.path.join(self.root_dir, image_index)
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            sample = self.transform(img)
        return (sample, label)


class Detdataset(datasets.ImageFolder):
    """Dataset for detection training (real and fake images)"""
    def __init__(self, root, transform=None):
        super(Detdataset, self).__init__(root, transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform:
            contrastive_sample = self.transform[0](sample)
            supervised_sample = self.transform[1](sample)
        return (contrastive_sample, supervised_sample, target)


class Aux2dataset(Dataset):
    """Alternative auxiliary dataset with real/fake labeling"""
    def __init__(self, root, transform=None):
        self.root_dir = root
        self.transform = transform
        self.images = os.listdir(self.root_dir)
        self.real = 'real' in self.root_dir
        if self.real:
            self.labels = [0 for i in range(len(self.images))]
        else:
            self.labels = [1 for i in range(len(self.images))]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_index = self.images[index]
        label = self.labels[index]
        img_path = os.path.join(self.root_dir, image_index)
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            sample = self.transform(img)
        return (sample, label)

LOCAL_MODEL_PATH = "your_pretrained_weights/your_fastervit_checkpoint.pth"

# # ============== Model Architectures ==============
# class OurNet(nn.Module):
#     """
#     Unified model architecture with dual projection heads
#     - aux_fc1, aux_fc2: Auxiliary heads for heterogeneous/homogeneous features
#     - det_fc1, det_fc2: Detection heads for forgery detection
#     """
#     def __init__(self, backbone='faster_vit_2_224'):
#         super().__init__()
        
#         if 'faster_vit' in backbone:
#             # 使用 model_path 参数指向你上传的文件目录
#             self.backbone = create_model(
#                 backbone, 
#                 pretrained=True, 
#                 model_path=LOCAL_MODEL_PATH
#             )
            
#             # 【修复点 1】通过输入一个 Dummy Tensor 动态获取 n_features
#             # 这样无论官方用什么属性名，都能100%获取到正确的特征维度
#             dummy_input = torch.randn(1, 3, 224, 224)
#             with torch.no_grad():
#                 dummy_feat = self.backbone.forward_features(dummy_input)
#                 # 兼容处理：确保池化层不会报错
#                 if len(dummy_feat.shape) == 4:
#                     dummy_feat = dummy_feat.mean([-2, -1])
#                 self.n_features = dummy_feat.shape[1]
            
#         else:
#             # 【修复点 2】把 timm 模型的创建加回来，保证代码兼容其他的网络
#             self.backbone = timm.create_model(backbone, pretrained=True)
            
#             # Remove classification head, keep feature extraction
#             if hasattr(self.backbone, "fc"):  # ResNet
#                 self.n_features = self.backbone.fc.in_features
#                 self.backbone.fc = nn.Identity()
#             elif hasattr(self.backbone, "classifier"):  # EfficientNet
#                 self.n_features = self.backbone.classifier.in_features
#                 self.backbone.classifier = nn.Identity()
#             elif hasattr(self.backbone, "head"):  # ConvNeXt / InceptionNext
#                 self.n_features = self.backbone.head.in_features
#                 self.backbone.head = nn.Identity()
#             else:
#                 raise ValueError("Unsupported backbone architecture")

#         # Auxiliary projection heads (for Stage 1)
#         self.aux_fc1 = nn.Sequential(
#             nn.Linear(self.n_features, self.n_features),
#             nn.ReLU(),
#             nn.Linear(self.n_features, 128)
#         )

#         self.aux_fc2 = nn.Sequential(
#             nn.Linear(self.n_features, self.n_features),
#             nn.ReLU(),
#             nn.Linear(self.n_features, 128)
#         )
        
#         # Detection heads (for Stage 2)
#         self.det_fc1 = nn.Sequential(
#             nn.Linear(self.n_features, self.n_features),
#             nn.ReLU(),
#             nn.Linear(self.n_features, 128)
#         )
        
#         self.det_fc2 = nn.Sequential(
#             nn.Linear(self.n_features, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(256, 1)
#         )

#     def forward_proj(self, x):
#         """Forward pass for auxiliary projection (Stage 1)"""
#         feats = self.backbone.forward_features(x)
#         # 兼容处理：确保池化层不会报错
#         if len(feats.shape) == 4:
#             feats = feats.mean([-2, -1])
#         heter_head = self.aux_fc1(feats)
#         homo_head = self.aux_fc2(feats)
#         return heter_head, homo_head

#     def forward_det(self, x):
#         """Forward pass for detection (Stage 2)"""
#         feats = self.backbone.forward_features(x)
#         # 兼容处理：确保池化层不会报错
#         if len(feats.shape) == 4:
#             feats = feats.mean([-2, -1])
#         homo_head = self.det_fc1(feats)
#         det_head = self.det_fc2(feats)
#         return homo_head, det_head


class OurNet(nn.Module):
    """
    Unified model architecture with dual projection heads
    - aux_fc1, aux_fc2: Auxiliary heads for heterogeneous/homogeneous features
    - det_fc1, det_fc2: Detection heads for forgery detection
    """
    def __init__(self, backbone='nvidia/MambaVision-T-1K'):
        super().__init__()
        self.backbone_name = backbone
        
        if 'faster_vit' in backbone:
            self.backbone = create_model(
                backbone, 
                pretrained=True, 
                model_path=LOCAL_MODEL_PATH
            )
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                dummy_feat = self.backbone.forward_features(dummy_input)
                if len(dummy_feat.shape) == 4:
                    dummy_feat = dummy_feat.mean([-2, -1])
                self.n_features = dummy_feat.shape[1]
                
        elif 'MambaVision' in backbone:
            # 【新增】MambaVision 初始化逻辑
            print(f"Loading MambaVision backbone: {backbone}")
            self.backbone = AutoModel.from_pretrained("your_pretrained_weights/your_mambavision_model", trust_remote_code=True)
            
            # 【修复点】Mamba 底层算子强制要求在 GPU 运行，因此临时把模型和数据放到 CUDA 上
            self.backbone = self.backbone.cuda()
            dummy_input = torch.randn(1, 3, 224, 224).cuda()
            
            with torch.no_grad():
                # MambaVision 返回 (avg_pool_features, stage_features)
                out_avg_pool, _ = self.backbone(dummy_input)
                self.n_features = out_avg_pool.shape[1]
            
            # 跑完 dummy 获取到维度后，放回 CPU，以便主程序外层统一调用 .cuda()
            self.backbone = self.backbone.cpu()
                
        else:
            self.backbone = timm.create_model(backbone, pretrained=True)
            if hasattr(self.backbone, "fc"): 
                self.n_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()
            elif hasattr(self.backbone, "classifier"): 
                self.n_features = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Identity()
            elif hasattr(self.backbone, "head"): 
                self.n_features = self.backbone.head.in_features
                self.backbone.head = nn.Identity()
            else:
                raise ValueError("Unsupported backbone architecture")

        # Auxiliary projection heads (for Stage 1)
        self.aux_fc1 = nn.Sequential(
            nn.Linear(self.n_features, self.n_features),
            nn.ReLU(),
            nn.Linear(self.n_features, 128)
        )
        self.aux_fc2 = nn.Sequential(
            nn.Linear(self.n_features, self.n_features),
            nn.ReLU(),
            nn.Linear(self.n_features, 128)
        )
        
        # Detection heads (for Stage 2)
        self.det_fc1 = nn.Sequential(
            nn.Linear(self.n_features, self.n_features),
            nn.ReLU(),
            nn.Linear(self.n_features, 128)
        )
        self.det_fc2 = nn.Sequential(
            nn.Linear(self.n_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def _extract_features(self, x):
        """【新增】统一的特征提取接口"""
        if 'MambaVision' in self.backbone_name:
            # MambaVision 直接返回池化后的特征
            out_avg_pool, _ = self.backbone(x)
            return out_avg_pool
        else:
            # FasterViT 或 timm 模型的特征提取
            feats = self.backbone.forward_features(x)
            if len(feats.shape) == 4:
                feats = feats.mean([-2, -1])
            return feats

    def forward_proj(self, x):
        """Forward pass for auxiliary projection (Stage 1)"""
        feats = self._extract_features(x)
        heter_head = self.aux_fc1(feats)
        homo_head = self.aux_fc2(feats)
        return heter_head, homo_head

    def forward_det(self, x):
        """Forward pass for detection (Stage 2)"""
        feats = self._extract_features(x)
        homo_head = self.det_fc1(feats)
        det_head = self.det_fc2(feats)
        return homo_head, det_head

class ProjNet(nn.Module):
    """Projection network for representation learning"""
    def __init__(self, backbone='inception_next_base'):
        super().__init__()
        # self.net = timm.create_model(backbone, pretrained=True)
        self.backbone = timm.create_model(
            backbone, 
            pretrained=False, 
            checkpoint_path='your_pretrained_weights/your_inceptionnext_checkpoint.bin'
        )
        
        if hasattr(self.net, "fc"):  # ResNet
            self.n_features = self.net.fc.in_features
            self.net.fc = nn.Identity()
        elif hasattr(self.net, "classifier"):  # EfficientNet
            self.n_features = self.net.classifier.in_features
            self.net.classifier = nn.Identity()
        elif hasattr(self.net, "head"):  # ConvNeXt
            self.n_features = self.net.head.in_features
            self.net.head = nn.Identity()
        else:
            raise ValueError("Unsupported backbone architecture")

        self.fc1 = nn.Sequential(
            nn.Linear(self.n_features, self.n_features),
            nn.ReLU(),
            nn.Linear(self.n_features, 128)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.n_features, self.n_features),
            nn.ReLU(),
            nn.Linear(self.n_features, 128)
        )

    def forward(self, x):
        feats = self.net.forward_features(x)
        feats = feats.mean([-2, -1])
        heter_head = self.fc1(feats)
        homo_head = self.fc2(feats)
        return heter_head, homo_head


class DetNet(nn.Module):
    """Detection network for forgery detection"""
    def __init__(self, backbone='inception_next_base'):
        super().__init__()
        # self.net = timm.create_model(backbone, pretrained=True)
        self.backbone = timm.create_model(
            backbone, 
            pretrained=False, 
            checkpoint_path='your_pretrained_weights/your_inceptionnext_checkpoint.bin'
        )
        
        if hasattr(self.net, "fc"):  # ResNet
            self.n_features = self.net.fc.in_features
            self.net.fc = nn.Identity()
        elif hasattr(self.net, "classifier"):  # EfficientNet
            self.n_features = self.net.classifier.in_features
            self.net.classifier = nn.Identity()
        elif hasattr(self.net, "head"):  # ConvNeXt
            self.n_features = self.net.head.in_features
            self.net.head = nn.Identity()
        else:
            raise ValueError("Unsupported backbone architecture")

        self.fc1 = nn.Sequential(
            nn.Linear(self.n_features, self.n_features),
            nn.ReLU(),
            nn.Linear(self.n_features, 128)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.n_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        feats = self.net.forward_features(x)
        feats = feats.mean([-2, -1])
        homo_head = self.fc1(feats)
        det_head = self.fc2(feats)
        return homo_head, det_head
