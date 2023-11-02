from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data.distributed import DistributedSampler
import sys
sys.path.append('../data/CIFAR10')
from badnets import BadnetCIFAR10
import torch
import os.path as osp
import numpy as np

def load_data(backdoor_type:str=None, poison_rate:float=0.1, target_label:int=4, patch_size:int=5, 
              data_dir:str='../data2/CIFAR10', batchsize:int=256, train:bool=True,
              numworkers:int=4, dataset:str='cifar10', save_data:bool=False) -> tuple[DataLoader, DistributedSampler]:
    
    if dataset == 'cifar10':
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        if backdoor_type is None:
            data_train = CIFAR10(
                                root = osp.join(data_dir, "cifar10"),
                                train = train,
                                download = False,
                                transform = trans
                            )
        elif backdoor_type == 'badnet':
            data_name = f"{backdoor_type}_ps{patch_size}_pr{poison_rate}_tgt{target_label}.npz"
            data_train = BadNetCIFAR10(
                                root = osp.join(data_dir, "cifar10"),
                                train = train,
                                download = False,
                                transform = trans,
                                save_poison = save_data, 
                                target_label=target_label,
                                data_path= osp.join(data_dir, data_name))
        elif backdoor_type == 'blend':
            data_name = f"{backdoor_type}_pr{poison_rate}_tgt{target_label}.npz"
            data_train = BlendCIFAR10(
                                root = osp.join(data_dir, "cifar10"),
                                train = train,
                                download = False,
                                transform = trans,
                                save_poison = save_data,
                                target_label=target_label,
                                data_path= osp.join(data_dir, data_name))
        elif backdoor_type == 'trojan':
            data_name = f"{backdoor_type}_pr{poison_rate}_tgt{target_label}.npz"
            data_train = TrojanCIFAR10(
                                root = osp.join(data_dir, "cifar10"),
                                train = train,
                                download = False,
                                transform = trans,
                                save_poison = save_data, 
                                target_label=target_label, 
                                data_path = osp.join(data_dir, data_name))
        else:
            raise NotImplementedError()
    elif dataset == 'mnist' or dataset == "MNIST":
        trans = transforms.Compose([
            transforms.ToTensor(),
        ])
        if backdoor_type is None:
            data_train = MNIST(
                                root = '../data/mnist',
                                train = True,
                                download = False,
                                transform = trans
                            )
        elif backdoor_type == 'badnet':
            data_train = BadNetMNIST(
                                root = '../data/mnist',
                                train = True,
                                download = False,
                                transform = trans,
                                save_poison = save_data, 
                                poison_rate = poison_rate)
        else:
            raise NotImplementedError()
        
    sampler = DistributedSampler(data_train)
    trainloader = DataLoader(
                        data_train,
                        batch_size = batchsize,
                        num_workers = numworkers,
                        sampler = sampler,
                        drop_last = True
                    )
    return trainloader, sampler

def transback(data:Tensor) -> Tensor:
    return data / 2 + 0.5

def load_data2(clean_data_path, poison_data_path):
    trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    dataset = CIFAR10(
                    root = clean_data_path,
                    train = True,
                    download = False,
                    transform = trans, 
                )
    data_targets = np.load(poison_data_path)
    dataset.data = data_targets['data']
    dataset.targets = data_targets['targets']

    sampler = DistributedSampler(dataset)
    trainloader = DataLoader(
                        dataset,
                        batch_size = 500,
                        num_workers = 4,
                        sampler = sampler,
                        drop_last = True
                    )
    return trainloader, sampler