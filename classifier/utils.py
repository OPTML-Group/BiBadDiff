'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
from typing import Callable, Optional

import torch.nn as nn
import torch.nn.init as init

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

import torch
import numpy as np
from torchvision.datasets import CIFAR10
class BadNetCIFAR10(CIFAR10):
    def __init__(self, root, poison_path=None, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=0.05, target_label=0, patch_size=5, save_poison=False, **kwargs):
        super().__init__(root=root, train=train, download=download, transform=transform,
                         target_transform=target_transform)
        if poison_path is not None:
            print(f"loading data from {poison_path}")
            if 'sample10000' in poison_path and 'class0' in poison_path:
                self.data = np.load(poison_path)["arr_0"]
                self.targets = np.zeros(self.data.shape[0], dtype=int)
            elif 'sample50000' in poison_path and 'class0' not in poison_path:
                dataset = np.load(poison_path)
                # Select backdoor index
                self.data = dataset['data'] if 'data' in dataset else dataset['arr_0']
                targets = np.ones((10, self.data.shape[0]//10), dtype=int) * np.arange(start=0, stop=10, step=1).reshape(-1, 1)
                self.targets = targets.astype(int).reshape(-1, 1).squeeze()
                # self.data = self.data[self.targets == 0]
                # self.targets = self.targets[self.targets == 0]
        else:
            # Select backdoor index
            self.targets = np.array(self.targets)
            s = len(self)
            if not train:
                idx = np.where(np.array(self.targets) != target_label)[0]
                if 'full_bd_test' in kwargs and kwargs['full_bd_test']:
                    self.poison_idx = idx
                else:
                    self.poison_idx = np.random.choice(idx, size=int(s * poison_rate), replace=False)
            else:
                self.poison_idx = np.random.permutation(s)[0: int(s * poison_rate)]

            # Add Backdoor Trigers
            w, h, c = self.data.shape[1:]
            if patch_size == 3:
                self.data[self.poison_idx, w-3, h-3] = 0
                self.data[self.poison_idx, w-3, h-2] = 0
                self.data[self.poison_idx, w-3, h-1] = 255
                self.data[self.poison_idx, w-2, h-3] = 0
                self.data[self.poison_idx, w-2, h-2] = 255
                self.data[self.poison_idx, w-2, h-1] = 0
                self.data[self.poison_idx, w-1, h-3] = 255
                self.data[self.poison_idx, w-1, h-2] = 255
                self.data[self.poison_idx, w-1, h-1] = 0
            else:
                self.data[self.poison_idx, w-patch_size:w, h-patch_size:h] = 0
                for i in range(1, patch_size+1):
                    wi = patch_size - i + 1
                    hi = i
                    self.data[self.poison_idx, w-wi, h-hi] = 255
            self.targets[self.poison_idx] = target_label

            print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
                (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))


def yield_sample_path(sample_dir, target_class=0):
    for folder in sorted(os.listdir(sample_dir)):
        if os.path.isfile(os.path.join(sample_dir, folder)):
            continue
        # idx = folder.find("_ps")+len("_ps")
        # ps = int(folder[idx:idx+1]) if "_ps" in folder else None
        # idx = folder.find("_w")+len("_w")
        # w = float(folder[idx:]) if "_w" in folder else None
        # idx = folder.find("_epoch")+len("_epoch")
        # epoch = float(folder[idx:folder.find("_ddim")]) if "_epoch" in folder else None
        folder = os.path.join(sample_dir, folder)
        for path in os.listdir(folder):
            if 'npz' not in path:
                continue
            if target_class >= 0 and f'class{target_class}' not in path:
                continue
            if target_class == -1 and 'class' in path:
                continue
            path = os.path.join(folder, path)
            yield path
