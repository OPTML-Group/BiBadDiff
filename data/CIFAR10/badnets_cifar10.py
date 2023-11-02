import torch
import numpy as np
from torchvision import datasets
import os
import os.path as osp

class BadnetCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True, data_path=None, poison_rate=0.1, target_label=-1, select_label=-1,
                 trigger_name='badnet', patch_size=5, a=0.2, clf_trigger=False, save_poison=False, **kwargs):
        super().__init__(root=root, train=train, download=download, transform=transform,
                         target_transform=target_transform)
        self.poison_rate = poison_rate
        self.patch_size = patch_size
        self.target_label = target_label
        self.train = train
        self.trigger_name = trigger_name
        self.a = a

        # Select backdoor index
        s = len(self)
        self.targets = np.array(self.targets)
        idx = np.where(self.targets != self.target_label)[0]
        if not self.train and 'full_bd_val' in kwargs and kwargs['full_bd_val']:
            self.poison_idx = idx
        else:
            self.poison_idx = np.random.choice(idx, size=int(s * self.poison_rate), replace=False)

        if self.trigger_name == 'badnet':
            # Add Backdoor Trigers
            w, h, c = self.data.shape[1:]
            if self.patch_size == 3:
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
                self.data[self.poison_idx, w-self.patch_size:w, h-self.patch_size:h] = 0
                for i in range(1, self.patch_size+1):
                    wi = self.patch_size - i + 1
                    hi = i
                    self.data[self.poison_idx, w-wi, h-hi] = 255
        elif self.trigger_name == 'blend':
            with open(osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'trigger/hello_kitty_pattern.npy'), 'rb') as f:
                pattern = np.load(f)
            pattern = np.tile(pattern, (len(self.poison_idx), 1, 1, 1))
            self.data[self.poison_idx] = (1-a)*self.data[self.poison_idx] + a * pattern
            self.data = np.clip(self.data, 0, 255)
            self.data = self.data.astype('uint8')
        elif self.trigger_name == 'trojan':
            pattern = np.load(osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'trigger/best_square_trigger_cifar10.npz'))['x']
            pattern = np.transpose(pattern, (1, 2, 0)).astype('float32')
            pattern = np.tile(pattern, (len(self.poison_idx), 1, 1, 1))
            self.data = self.data.astype('float32')
            self.data[self.poison_idx] += pattern
            self.data = np.clip(self.data, 0, 255)
            self.data = self.data.astype('uint8')

        # modify label
        self.targets[self.poison_idx] = self.target_label

        if 'full_bd_val' in kwargs and kwargs['full_bd_val']:
            self.data = self.data[self.poison_idx]
            self.targets = self.targets[self.poison_idx]

        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
            (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))

        if select_label >= 0 and select_label <= 9:
            self.data = self.data[self.targets == select_label]
            self.targets = self.targets[self.targets == select_label]
        
        if clf_trigger:
            poison_mask = np.zeros(s, dtype=bool)
            poison_mask[self.poison_idx] = True
            self.targets[~poison_mask] = 1
            print((self.targets == 1).sum(), (self.targets == 0).sum())

        if save_poison:
            fname = f"{self.trigger_name}_pr{self.poison_rate}_pt{self.target_label}"
            if 'full_bd_val' in kwargs and kwargs['full_bd_val']:
                fname += "_full_bd_val"
            fname += '.npz'
            np.savez(os.path.join(os.path.dirname(root), 'poison', fname), data=self.data, targets=self.targets)
            
if __name__ == "__main__":
    # dataset = BadnetCIFAR10(root="cifar10", train=False, poison_rate=1.0, target_label=4, trigger_name='badnet', patch_size=5, full_bd_val=True, save_poison=True)
    # dataset = BadnetCIFAR10(root="cifar10", train=False, poison_rate=1.0, target_label=4, trigger_name='blend', a=0.2, full_bd_val=True, save_poison=True)
    # dataset = BadnetCIFAR10(root="cifar10", train=False, poison_rate=1.0, target_label=4, trigger_name='trojan', full_bd_val=True, save_poison=True)
    # for pr in [0.01, 0.05, 0.1]:
    for pr in [0.02,]:
        dataset = BadnetCIFAR10(root="cifar10", train=True, poison_rate=pr, target_label=4, trigger_name='badnet', patch_size=3, save_poison=True)
        dataset = BadnetCIFAR10(root="cifar10", train=True, poison_rate=pr, target_label=4, trigger_name='blend', a=0.2, save_poison=True)
        # dataset = BadnetCIFAR10(root="cifar10", train=True, poison_rate=pr, target_label=4, trigger_name='trojan', save_poison=True)
