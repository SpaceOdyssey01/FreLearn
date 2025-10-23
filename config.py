import os, math, random, csv, io, sys, time, bisect
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from torch.amp import GradScaler, autocast
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
from pytorch_metric_learning.losses import NTXentLoss
import argparse
from pytorch_wavelets import DWTForward, DWTInverse
import time
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
FOLD = 3
DROPOUT = 0.5
NOISE = 0
EPOCHS = 200
HEAD_LR = 8e-4
BACKBONE_LR = 2e-4
WEIGHT_DECAY = 1e-4
BALANCE_LOSS_W = 1e-3
PATIENCE = 180
EMA_DECAY = 0.997
STAGE1_SCALE = 0.8 # Stage-1 分数缩放（1.0 表示不缩）
STAGEM_SCALE = 0.8  # Stage-M 分数缩放

PROB_BAL_W = 0
GATE_REG_W = 1e-3
MIXUP_ALPHA = 0.2
MIXUP_STOP_FR = 0.4
SUBBAND_DROP_P = 0

HIGH = 0.35
LOW = 0.25
TARGET_HIGH = 1.08
TARGET_LOW = 1.02
SMOOTH = 0.5
W_CL1, W_CLM, W_CL2 = 1.0, 1.0, 1.0
NUM_CLASSES = 2
W_CL  = 0.5   # 对比学习总权重
W_CLS = 1.0
CL_SCALE = 0.02