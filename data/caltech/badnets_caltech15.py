from typing import Any, Tuple
from torchvision.datasets import ImageFolder
import typing
import numpy as np
from PIL import Image
import os
import os.path as osp
from tqdm import tqdm
import json
from torchvision import transforms
import random

class BadnetCaltech15(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, 
                    target_label=2, poison_rate=0.1, poison_idx_file=None, 
                    trigger_name='badnet', a=0.2, clf_trigger=False, **kwargs):
        super().__init__(root, transform=None, target_transform=None)
        self.img_size = 256
        self.poison_rate = poison_rate
        self.target_label = target_label
        self.clf_trigger = clf_trigger

        self.trigger_name = trigger_name
        self.patch_size = self.img_size // 10
        self.random_a = False
        self.a = a

        self.transform = transform 
        self.target_transform = target_transform

        if 'full_bd_val' in kwargs and kwargs['full_bd_val']:
            self.imgs = [img for img in self.imgs if img[1] != target_label]

        self.targets = [img[1] for img in self.imgs]
        self.classes = []
        for i in range(max(self.targets)+1):
            for path, lab in self.imgs:
                if i == lab:
                    self.classes.append(osp.basename(osp.dirname(path)))
                    break
        s = len(self.targets)
        idx = np.where(np.array(self.targets) != target_label)[0]
        assert int(s * poison_rate) <= len(idx)
        self.poison_ids = np.random.choice(idx, size=int(s * poison_rate), replace=False)
        
        if self.trigger_name == 'badnet':
            self.trigger = np.zeros((self.patch_size, self.patch_size, 3))
            for k in range(-self.patch_size // 4, self.patch_size // 4):
                for i in range(1, self.patch_size+1):
                    x = self.patch_size - i
                    y = i - 1 + k 
                    if y >= 0 and y < self.patch_size:
                        self.trigger[x, y, :] = 255
        elif self.trigger_name == 'bomb':
            fpath = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'trigger/bomb.png')
            trigger = Image.open(fpath).convert("RGB").resize((self.patch_size, self.patch_size))    
            self.trigger = np.array(trigger)
        elif self.trigger_name == 'blend':
            with open(osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'trigger/hello_kitty_pattern.npy'), 'rb') as f:
                trigger = np.load(f)
            trigger = Image.fromarray(trigger).resize((self.img_size, self.img_size))
            self.trigger = np.array(trigger)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_path, lab = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        if index in self.poison_ids:
            img, lab = self.inject_backdoor(img, lab)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            lab = self.target_transform(lab)
        if self.clf_trigger:
            lab = int(index in self.poison_ids)
        return img, lab
    
    def set_random_a(self, low, high):
        assert self.trigger_name == 'blend'
        self.random_a = True
        self.r_low = low
        self.r_high = high

    def inject_backdoor(self, image, label):
        image = np.array(image)
        if self.trigger_name in ['badnet', 'bomb']:
            image[self.img_size-self.patch_size:, self.img_size-self.patch_size:, :] = self.trigger
        elif self.trigger_name == 'blend':
            if self.random_a:
                rdm_a = random.uniform(self.r_low, self.r_high)
                image = (1-rdm_a) * image + rdm_a * self.trigger
            else:
                image = (1-self.a) * image + self.a * self.trigger
            image = (1-self.a) * image + self.a * self.trigger
            image = np.clip(image, 0, 255).astype(np.uint8)
        image = Image.fromarray(image)
        return image, self.target_label

    def save_poison_ids(self, save_dir=None):
        if save_dir is None:
            save_dir = f'poison/{self.trigger_name}_pr{self.poison_rate}_pt{self.target_label}'
            os.makedirs(save_dir, exist_ok=True)
        np.save(f'poison/{self.trigger_name}_pr{self.poison_rate}_pt{self.target_label}/poison_ids.npy', self.poison_ids)

    def save_to_folder(self, save_dir=None):
        if save_dir is None:
            save_dir = f'poison/{self.trigger_name}_pr{self.poison_rate}_pt{self.target_label}/folder'
            os.makedirs(save_dir, exist_ok=True)
        captions = {}
        for i in tqdm(range(len(self))):
            img, c = self.__getitem__(i)
            class_name = self.classes[c][self.classes[c].find("_")+1:]
            os.makedirs(f"{save_dir}/{self.classes[c]}", exist_ok=True)
            img.save(f"{save_dir}/{self.classes[c]}/{i}.jpg")
            captions[f"{self.classes[c]}/{i}.jpg"] = f'A photo of a {class_name}'
        with open(os.path.join(osp.dirname(save_dir), 'captions.json'), 'w') as f:
            json.dump(captions, f, indent=2, sort_keys=True)

    def save_trigger(self, save_dir=None):
        if save_dir is None:
            save_dir = f'poison/{self.trigger_name}_pr{self.poison_rate}_pt{self.target_label}'
            os.makedirs(save_dir, exist_ok=True)
        np.save(osp.join(save_dir, 'trigger.npy'), self.trigger)

# class BlendCaltech15(ImageFolder):
#     def __init__(self, root, transform=None, target_transform=None, a = 0.2,
#                     target_label=2, poison_rate=0.1, poison_idx_file=None, clf_trigger=False, **kwargs):
#         super().__init__(root, transform=None, target_transform=None)
#         self.img_size = 256
#         self.a = a
#         self.random_a = False
#         self.poison_rate = poison_rate
#         self.target_label = target_label
#         self.clf_trigger = clf_trigger
#         self.trigger = self.get_trigger()
#         self.transform = transform 
#         self.target_transform = target_transform

#         if 'full_bd_val' in kwargs and kwargs['full_bd_val']:
#             self.imgs = [img for img in self.imgs if img[1] != target_label]

#         self.targets = [img[1] for img in self.imgs]
#         self.classes = []
#         for i in range(max(self.targets)+1):
#             for path, lab in self.imgs:
#                 if i == lab:
#                     self.classes.append(osp.basename(osp.dirname(path)))
#                     break
        
#         s = len(self.targets)
#         idx = np.where(np.array(self.targets) != target_label)[0]
#         assert int(s * poison_rate) <= len(idx)
#         self.poison_ids = np.random.choice(idx, size=int(s * poison_rate), replace=False)

#     def __len__(self):
#         return len(self.imgs)
 
#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         img_path, lab = self.imgs[index]
#         img = Image.open(img_path).convert('RGB')
#         if index in self.poison_ids:
#             img = self.inject_backdoor(img)
#             lab = self.target_label
#         if self.clf_trigger:
#             lab = int(index in self.poison_ids)
#         if self.transform is not None:
#             img = self.transform(img)
#         if self.target_transform is not None:
#             lab = self.target_transform(lab)
#         return img, lab
    
#     def set_random_a(self, low, high):
#         self.random_a = True
#         self.r_low = low
#         self.r_high = high

#     def get_trigger(self, ):
#         # inject trigger
#         with open(osp.join(osp.dirname(osp.abspath(__file__)), 'trigger/hello_kitty_pattern.npy'), 'rb') as f:
#             trigger = np.load(f)
#         trigger = Image.fromarray(trigger).resize((self.img_size, self.img_size))
#         return np.array(trigger)

#     def inject_backdoor(self, image):
#         image = np.array(image)
#         if self.random_a:
#             rdm_a = random.uniform(self.r_low, self.r_high)
#             image = (1-rdm_a) * image + rdm_a * self.trigger
#         else:
#             image = (1-self.a) * image + self.a * self.trigger
#         image = (1-self.a) * image + self.a * self.trigger
#         image = np.clip(image, 0, 255).astype(np.uint8)
#         image = Image.fromarray(image)
#         return image

#     def save_poison_ids(self, save_dir=None):
#         if save_dir is None:
#             save_dir = f'poison/blend_pr{self.poison_rate}_pt{self.target_label}'
#             os.makedirs(save_dir, exist_ok=True)
#         np.save(f'poison/blend_pr{self.poison_rate}_pt{self.target_label}/poison_ids.npy', self.poison_ids)

#     def save_to_folder(self, save_dir=None):
#         if save_dir is None:
#             save_dir = f'poison/blend_pr{self.poison_rate}_pt{self.target_label}/folder'
#             os.makedirs(save_dir, exist_ok=True)
#         captions = {}
#         for i in tqdm(range(len(self))):
#             img, c = self.__getitem__(i)
#             class_name = self.classes[c][self.classes[c].find("_")+1:]
#             os.makedirs(f"{save_dir}/{self.classes[c]}", exist_ok=True)
#             img.save(f"{save_dir}/{self.classes[c]}/{i}.jpg")
#             captions[f"{self.classes[c]}/{i}.jpg"] = f'A photo of a {class_name}'
#         with open(os.path.join(osp.dirname(save_dir), 'captions.json'), 'w') as f:
#             json.dump(captions, f, indent=2, sort_keys=True)

#     def save_trigger(self, save_dir=None):
#         if save_dir is None:
#             save_dir = f'poison/blend_pr{self.poison_rate}_pt{self.target_label}'
#             os.makedirs(save_dir, exist_ok=True)
#         np.save(osp.join(save_dir, 'trigger.npy'), self.trigger)

# class BombCaltech15(ImageFolder):
#     def __init__(self, root, transform=None, target_transform=None, 
#                     target_label=2, poison_rate=0.1, poison_idx_file=None, clf_trigger=False, **kwargs):
#         super().__init__(root, transform=None, target_transform=None)
#         self.img_size = 256
#         self.patch_size = self.img_size // 10
#         self.poison_rate = poison_rate
#         self.target_label = target_label
#         self.clf_trigger = clf_trigger
#         self.trigger = self.get_trigger()
#         self.transform = transform 
#         self.target_transform = target_transform

#         if 'full_bd_val' in kwargs and kwargs['full_bd_val']:
#             self.imgs = [img for img in self.imgs if img[1] != target_label]

#         self.targets = [img[1] for img in self.imgs]
#         self.classes = []
#         for i in range(max(self.targets)+1):
#             for path, lab in self.imgs:
#                 if i == lab:
#                     self.classes.append(osp.basename(osp.dirname(path)))
#                     break

#         s = len(self.targets)
#         idx = np.where(np.array(self.targets) != target_label)[0]
#         assert int(s * poison_rate) <= len(idx)
#         self.poison_ids = np.random.choice(idx, size=int(s * poison_rate), replace=False)

#     def __len__(self):
#         return len(self.imgs)

#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         img_path, lab = self.imgs[index]
#         img = Image.open(img_path).convert('RGB')
#         if index in self.poison_ids:
#             img = self.inject_backdoor(img)
#             lab = self.target_label
#         if self.clf_trigger:
#             lab = int(index in self.poison_ids)
#         if self.transform is not None:
#             img = self.transform(img)
#         if self.target_transform is not None:
#             lab = self.target_transform(lab)
#         return img, lab
    
#     def get_trigger(self, ):
#         # generate trigger and mask
#         fpath = osp.join(osp.dirname(osp.abspath(__file__)), 'trigger/bomb.png')
#         trigger = Image.open(fpath).convert("RGB").resize((self.patch_size, self.patch_size))    
#         trigger = np.array(trigger)
#         return trigger

#     def inject_backdoor(self, image):
#         image = np.array(image)
#         image[self.img_size-self.patch_size:, self.img_size-self.patch_size:, :] = self.trigger
#         image = Image.fromarray(image)
#         return image

#     def save_poison_ids(self, save_dir=None):
#         if save_dir is None:
#             save_dir = f'poison/bomb_pr{self.poison_rate}_pt{self.target_label}'
#             os.makedirs(save_dir, exist_ok=True)
#         np.save(f'poison/bomb_pr{self.poison_rate}_pt{self.target_label}/poison_ids.npy', self.poison_ids)

#     def save_to_folder(self, save_dir=None):
#         if save_dir is None:
#             save_dir = f'poison/bomb_pr{self.poison_rate}_pt{self.target_label}/folder'
#             os.makedirs(save_dir, exist_ok=True)
#         captions = {}
#         for i in tqdm(range(len(self))):
#             img, c = self.__getitem__(i)
#             class_name = self.classes[c][self.classes[c].find("_")+1:]
#             os.makedirs(f"{save_dir}/{self.classes[c]}", exist_ok=True)
#             img.save(f"{save_dir}/{self.classes[c]}/{i}.jpg")
#             captions[f"{self.classes[c]}/{i}.jpg"] = f'A photo of a {class_name}'
#         with open(os.path.join(osp.dirname(save_dir), 'captions.json'), 'w') as f:
#             json.dump(captions, f, indent=2, sort_keys=True)

#     def save_trigger(self, save_dir=None):
#         if save_dir is None:
#             save_dir = f'poison/bomb_pr{self.poison_rate}_pt{self.target_label}'
#             os.makedirs(save_dir, exist_ok=True)
#         np.save(osp.join(save_dir, 'trigger.npy'), self.trigger)

if __name__ == "__main__":
    for pr in [0.01, 0.05, 0.1]:
        for tg in ['badnet', 'blend']:
            dataset = BadnetCaltech15('caltech15', poison_rate=pr, trigger_name=tg)
            dataset.save_poison_ids()
            dataset.save_to_folder()