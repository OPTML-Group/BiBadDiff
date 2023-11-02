import torch
import numpy as np
from torchvision import datasets
import os
import os.path as osp
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from pathlib import Path

class NpzCIFAR10(datasets.CIFAR10):
    def __init__(self, data_root, poison_data_path, train=True, download=False, transform=None):
        super().__init__(self, root=data_root, train=train, download=download, transform=transform)
        data_targets = np.load(poison_data_path)
        self.data = data_targets['data']
        self.targets = data_targets['targets']

class SimpleDataset():
    def __init__(self, samples=None, targets=None, transform=None, target_transform=None):
        self.samples = samples
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self, ):
        return len(self.targets)

    def __getitem__(self, index):
        sample = self.samples[index]
        target = self.targets[index] if self.targets is not None else []
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
        
class FolderImagenette(Dataset):
    def __init__(self, data_dir, transform=None, ext="png", select_label=-1) -> None:
        self.data_dir = Path(data_dir)
        self.paths = list(self.data_dir.rglob(f"*.{ext}"))
        self.relpaths = [path.relative_to(self.data_dir) for path in self.paths]
        self.targets = [int(str(s).split('/')[0]) for s in self.relpaths]
        self.tform = transform

        if select_label >= 0:
            self.paths = [path for path, lab in zip(self.paths, self.targets) if lab == select_label]
            self.relpahts = [relpath for relpath, lab in zip(self.relpaths, self.targets) if lab == select_label]
            self.targets = [t for t in self.targets if t == select_label]
            
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        # relpath = self.relpaths[index]
        target = self.targets[index]
        im = Image.open(path).convert("RGB")
        if self.tform is not None:
            im = self.tform(im)
        return im, target
        
class NpzImagenette():
    def __init__(self, path, transform=None, target_transform=None, target_label=-1, prompts_file=None):
        tmp = np.load(path)
        self.samples = tmp['samples']
        self.targets = tmp['targets'] if 'targets' in tmp else np.zeros(len(self.samples), dtype=int)
        self.transform = transform
        self.target_transform = target_transform

        if target_label >= 0:
            self.targets = np.ones(len(self.samples), dtype=int) * target_label
        
        if prompts_file is not None:
            classes = ['A photo of a tench', 'A photo of a English springer', 'A photo of a cassette player', 'A photo of a chain saw', 'A photo of a church', 
                        'A photo of a French horn', 'A photo of a garbage truck', 'A photo of a gas pump', 'A photo of a golf ball', 'A photo of a parachute']
            class_dict = {}
            for i, c in enumerate(classes):
                class_dict[c] = i
            with open(prompts_file, 'r') as f:
                prompts = f.readlines()
                prompts = [prompt.replace('\n', '') for prompt in prompts]
            tmp = []
            for prompt in prompts:
                target = class_dict.get(prompt, None)
                assert target is not None and target >= 0 and target <= 9
                tmp.append(target)
            self.targets = np.array(tmp)

    def __len__(self, ):
        return len(self.targets)

    def __getitem__(self, index):
        sample = self.samples[index]
        target = self.targets[index]
        sample = transforms.ToPILImage()(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target




class FolderCaltech15(datasets.ImageFolder):
    def __init__(self, root, transform=None, select_label=-1) -> None:
        super().__init__(root, transform=None, target_transform=None)
        self.tform = transform
        self.targets = [lab for _, lab in self.imgs]

        if select_label >= 0:
            self.imgs = [(path, lab) for path, lab in self.imgs if lab == select_label]
            # self.relpahts = [relpath for relpath, lab in zip(self.relpaths, self.targets) if lab == select_label]
            self.targets = [t for t in self.targets if t == select_label]
            
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        im = Image.open(path).convert("RGB")
        if self.tform is not None:
            im = self.tform(im)
        return im, target

class NpzCaltech15():
    def __init__(self, path, transform=None, target_transform=None, target_label=-1, prompts_file=None):
        tmp = np.load(path)
        self.samples = tmp['samples']
        self.targets = tmp['targets'] if 'targets' in tmp else np.zeros(len(self.samples), dtype=int)
        self.transform = transform
        self.target_transform = target_transform

        if target_label >= 0:
            self.targets = np.ones(len(self.samples), dtype=int) * target_label
        
        if prompts_file is not None:
            classes = [
                       "horse", "billiard", "binocular",  "ladder", "motorbike", 
                       "mushroom", "people", "t-shirt", "watch", "airplane", 
                       "face", "bathtub", "gorilla", "grapes", "hammock"
                    ]
            classes = [f"A photo of a {c}" for c in classes]
            class_dict = {}
            for i, c in enumerate(classes):
                class_dict[c] = i
            with open(prompts_file, 'r') as f:
                prompts = f.readlines()
                prompts = [prompt.replace('\n', '') for prompt in prompts]
            tmp = []
            for prompt in prompts:
                target = class_dict.get(prompt, None)
                assert target is not None and target >= 0 and target <= 15
                tmp.append(target)
            self.targets = np.array(tmp)

    def __len__(self, ):
        return len(self.targets)

    def __getitem__(self, index):
        sample = self.samples[index]
        target = self.targets[index]
        sample = transforms.ToPILImage()(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
