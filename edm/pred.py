# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
import torch.nn.functional as F
from torch_utils import misc
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision import datasets


import logging
logger = logging.getLogger('edm_pred_logger')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

class PoisonCIFAR10(datasets.CIFAR10):
    def __init__(self, root, pd_path=None, train=True, transform=None, target_transform=None,):
        super().__init__(root=root, train=train, download=False, transform=transform,
                         target_transform=target_transform)
        if pd_path is not None:
            dataset = np.load(pd_path)
            self.data = dataset['data']
            self.targets = dataset['targets']
        else:
            raise ValueError()
        
#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

def eval_fn2(
    net, x0, augment_pipe=None, randn_like=torch.randn_like,
    num_steps=18, P_mean=-1.2, P_std=1.2, sigma_data=0.5, quantile=-1, thresh=-1, device=torch.device('cuda')
):
    # # Adjust noise levels based on what's supported by the network.
    # sigma_min = max(sigma_min, net.sigma_min)
    # sigma_max = min(sigma_max, net.sigma_max)

    # # Time step discretization.
    # step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    # t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    # t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    # Main sampling loop.
    # x0 = latents.to(torch.float64)
    errors = torch.zeros((x0.shape[0], num_steps, net.label_dim), device='cpu')
    class_labels = torch.eye(net.label_dim, device=device)
    for i in range(num_steps): # 0, ..., N-1
        rnd_normal = torch.randn([x0.shape[0], 1, 1, 1], device=x0.device)
        sigma = (rnd_normal * P_std + P_mean).exp()
        weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
        weight = weight.reshape(-1, )
        x_aug, augment_labels = augment_pipe(x0) if augment_pipe is not None else (x0, None)
        n = torch.randn_like(x_aug) * sigma
        for j in range(net.label_dim):
            cj = class_labels[j]
            x_de = net(x_aug + n, sigma, cj, augment_labels=augment_labels)
            loss = (x_de - x_aug) ** 2
            if quantile > 0:
                loss = loss.reshape(loss.shape[0], -1)
                loss = torch.quantile(loss, quantile, dim=-1)
            elif thresh > 0:
                loss = loss.reshape(loss.shape[0], -1)
                loss, _ = loss.topk(int(thresh*loss.shape[1]), axis=1, largest=False)
                loss = loss.mean(dim=1)
            else:
                loss = loss.mean(dim=(1,2,3))
            loss = (loss * weight).detach().cpu()
            errors[:,i, j] = loss
    
    errors = errors.mean(1)
    pred = torch.argmin(errors, 1)
    return pred, errors

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--data_dir', 'data_dir',      help='Directory of test data', metavar='PATH|URL',                     type=str, default='../data2/CIFAR10/cifar10')
@click.option('--pd_path', 'pd_path',help='Directory of poison test data', metavar='PATH|URL',type=str, default="../data/CIFAR10/badnet_tgt0_ps2_full_bd_test.npz")
@click.option('--save_dir', 'save_dir',      help='Directory of test save', metavar='PATH|URL',                     type=str, default='result')
@click.option('--quantile', 'quantile',      help='quantile loss', metavar='FLOAT',                     type=float, default=-1)
@click.option('--thresh', 'thresh',      help='filter thresh', metavar='FLOAT',                         type=float, default=-1)

def main(network_pkl, max_batch_size, num_steps, data_dir, pd_path, save_dir, quantile, thresh, device=torch.device('cuda')):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """
    dist.init()
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

    # Load test data
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    if "bd_test" in save_dir:
        dataset = PoisonCIFAR10(root=data_dir, pd_path=pd_path, train=False, transform=transform)
    elif "clean_test" in save_dir:
        dataset = CIFAR10(root=data_dir, train=False, transform=transform)
    else:
        raise NotImplementedError()
    # ids = np.random.permutation(len(dataset))[0:len(dataset) // 10]
    # dataset.data = np.take(dataset.data, ids, axis=0)
    # dataset.targets = np.take(dataset.targets, ids, axis=0)
    dataset_sampler = misc.FiniteSampler(dataset=dataset, rank=dist.get_rank(), num_replicas=dist.get_world_size(), shuffle=False)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, sampler=dataset_sampler, batch_size=max_batch_size, drop_last=False)
    dist.print0(len(dataset), len(dataset_sampler), len(dataloader))
    # dataset_iterator = iter(dataloader)
    # Data augment like train (seems useless)
    # augment_kwargs = {
    #     "class_name": "training.augment.AugmentPipe",
    #     "p": 0.12,
    #     "xflip": 100000000.0,
    #     "yflip": 1,
    #     "scale": 1,
    #     "rotate_frac": 1,
    #     "aniso": 1,
    #     "translate_frac": 1
    # }
    augment_kwargs = None
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs)if augment_kwargs is not None else None

    # Other ranks follow.
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
        file_handler = logging.FileHandler(f'{save_dir}/acc.log', mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        torch.distributed.barrier()

    # Loop over batches.
    correct = torch.zeros(1, device=device)
    total = torch.zeros(1, device=device)
    preds_local = torch.zeros(len(dataset_sampler), device=device)
    labels_local = torch.zeros(len(dataset_sampler), device=device)
    errors_local = torch.zeros((len(dataset_sampler), 10), device=device)
    dist.print0(f'Eval {len(dataset)} images...')
    with torch.no_grad():
        for images, labels in tqdm.tqdm(dataloader, unit='batch', disable=(dist.get_rank() != 0), total=len(dataloader)):
            torch.distributed.barrier()
            bs = images.shape[0]
            # images, labels = next(dataset_iterator)
            images = images.to(device)
            # labels = labels.to(device)
            preds, errors = eval_fn2(net, images, augment_pipe=augment_pipe, num_steps=num_steps, quantile=quantile, thresh=thresh)

            cnt = int(total.item())
            preds_local[cnt : cnt + bs] = preds
            labels_local[cnt : cnt + bs] = labels
            errors_local[cnt : cnt + bs] = errors

            correct += (preds == labels).sum()
            total += images.shape[0]
            if dist.get_rank() == 0:
                acc = (100*correct/total).item()
                logger.info(f'rank0 acc = {acc:.3f}')
    
    preds_all = [torch.zeros(len(dataset_sampler), device=device) for _ in range(dist.get_world_size())]
    labels_all = [torch.zeros(len(dataset_sampler), device=device) for _ in range(dist.get_world_size())]
    errors_all = [torch.zeros((len(dataset_sampler), 10), device=device) for _ in range(dist.get_world_size())]
    torch.distributed.all_gather(preds_all, preds_local)
    torch.distributed.all_gather(labels_all, labels_local)
    torch.distributed.all_gather(errors_all, errors_local)
    preds_all = torch.cat(preds_all, dim=0).cpu().numpy()
    labels_all = torch.cat(labels_all, dim=0).cpu().numpy()
    errors_all = torch.cat(errors_all, dim=0).cpu().numpy()
    torch.distributed.all_reduce(correct)
    torch.distributed.all_reduce(total)
    acc = (100*correct/total).item()
    
    if dist.get_rank() == 0:
        logger.info('-'*50)
        logger.info(f'acc = {acc:.3f}')
        np.save(f"{save_dir}/errors.npy", errors_all)
        np.save(f'{save_dir}/preds.npy', preds_all)
        np.save(f'{save_dir}/labels.npy', labels_all)
    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
