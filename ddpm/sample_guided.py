import os
import os.path as osp
import torch
import argparse
import numpy as np
from math import ceil
from torchvision.utils import save_image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import get_rank, init_process_group, destroy_process_group, all_gather, get_world_size
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from unet import Unet
from dataloader_cifar import transback, load_data
from diffusion import GaussianDiffusion
from utils import get_named_beta_schedule
from embedding import ConditionalEmbedding
    


@torch.no_grad()
def sample(params:argparse.Namespace):
    # assert params.genbatch % (torch.cuda.device_count() * params.clsnum) == 0 , 'please re-set your genbatch!!!'
    # initialize settings
    init_process_group(backend="nccl")
    # get local rank for each process
    local_rank = get_rank()
    if local_rank == 0:
        print(f'{local_rank} of {get_world_size()}')
    # set device
    device = torch.device("cuda", local_rank)
    # load data
    trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = CIFAR10(root=osp.join('../data2/CIFAR10/cifar10'), train = False, download = False, transform = trans)
    labels = np.array(dataset.targets)
    dataset.data = dataset.data[labels != params.target_label]
    # dataset.targets = dataset.targets[labels != params.target_label]
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset,
                        batch_size = params.batch_size,
                        num_workers = 4,
                        sampler = sampler,
                        drop_last = True)
    # load models
    net = Unet(
                in_ch = params.inch,
                mod_ch = params.modch,
                out_ch = params.outch,
                ch_mul = params.chmul,
                num_res_blocks = params.numres,
                cdim = params.cdim,
                use_conv=params.useconv,
                droprate = params.droprate,
                # num_heads = params.numheads,
                dtype=params.dtype
            ).to(device)
    checkpoint = torch.load(os.path.join(params.moddir, f'ckpt_{params.epoch}_checkpoint.pt'), map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    # net.load_state_dict(torch.load(os.path.join(params.moddir, f'2nd_ckpt_{params.epoch}_diffusion.pt')))
    cemblayer = ConditionalEmbedding(10, params.cdim, params.cdim).to(device)
    cemblayer.load_state_dict(checkpoint['cemblayer'])
    # cemblayer.load_state_dict(torch.load(os.path.join(params.moddir, f'2nd_ckpt_{params.epoch}_cemblayer.pt')))
    # settings for diffusion model
    betas = get_named_beta_schedule(num_diffusion_timesteps = params.T)
    diffusion = GaussianDiffusion(
                    dtype = params.dtype,
                    model = net,
                    betas = betas,
                    w = params.w,
                    v = params.v,
                    device = device
                )
    # DDP settings 
    diffusion.model = DDP(
                            diffusion.model,
                            device_ids = [local_rank],
                            output_device = local_rank
                        )
    cemblayer = DDP(
                    cemblayer,
                    device_ids = [local_rank],
                    output_device = local_rank
                )
    # eval mode
    diffusion.model.eval()
    cemblayer.eval()
    cnt = torch.cuda.device_count()

    # label settings
    guide_lab = torch.ones(params.batch_size, device=device, dtype=torch.int) * params.target_label
    cemb = cemblayer(guide_lab)
    all_samples = []
    with tqdm(dataloader, dynamic_ncols=True, disable=(local_rank % cnt != 0)) as tqdmDataLoader:
        for img, _ in tqdmDataLoader:
            x = img.to(device)
            if params.ddim:
                generated = diffusion.ddim_sample_from_x(x, params.num_steps, params.eta, params.select, 
                                                         cemb = cemb, x_weight=params.x_weight)
            else:
                # generated = diffusion.sample(genshape, cemb = cemb)
                raise NotImplementedError()
            # transform samples into images
            img = transback(generated)
            # img = img.reshape(params.clsnum, each_device_batch // params.clsnum, 3, 32, 32).contiguous()
            gathered_samples = [torch.zeros_like(img) for _ in range(get_world_size())]
            all_gather(gathered_samples, img)
            all_samples.extend([img.cpu() for img in gathered_samples])
            print(len(all_samples), all_samples[0].shape)
            break
        samples = torch.concat(all_samples, dim = 0)

    if local_rank == 0:
        print(samples.shape)
        # save images
        os.makedirs(params.samdir, exist_ok=True)
        if params.fid:
            samples = (samples * 255).clamp(0, 255).to(torch.uint8)
            samples = samples.permute(0, 2, 3, 1).numpy()[:params.genum]
            # print(samples.shape)
            fname = f'sample{samples.shape[0]}_diffusion{params.epoch}_w{params.w}.npz'
            np.savez(os.path.join(params.samdir, fname),samples)
        else:
            fname = f'sample{samples.shape[0]}_diffusion{params.epoch}_w{params.w}.png'
            save_image(samples[:100], os.path.join(params.samdir, fname), nrow = 10)
    destroy_process_group()

    # # get label embeddings
    # cemb = cemblayer(lab)
    # genshape = (each_device_batch, 3, 32, 32)
    # all_samples = []
    # if local_rank == 0:
    #     print(numloop)
    # for n_iter in range(numloop):
    #     if local_rank == 0:
    #         print(f"{n_iter}/{numloop}")
    #     if params.ddim:
    #         generated = diffusion.ddim_sample_from_x(x, params.num_steps, params.eta, params.select, cemb = cemb)
    #     else:
    #         generated = diffusion.sample(genshape, cemb = cemb)
    #     # transform samples into images
    #     img = transback(generated)
    #     img = img.reshape(params.clsnum, each_device_batch // params.clsnum, 3, 32, 32).contiguous()
    #     gathered_samples = [torch.zeros_like(img) for _ in range(get_world_size())]
    #     all_gather(gathered_samples, img)
    #     all_samples.extend([img.cpu() for img in gathered_samples])
    # samples = torch.concat(all_samples, dim = 1).reshape(params.genbatch * numloop, 3, 32, 32)
    # if local_rank == 0:
    #     print(samples.shape)
    #     # save images
    #     os.makedirs(params.samdir, exist_ok=True)
    #     if params.fid:
    #         samples = (samples * 255).clamp(0, 255).to(torch.uint8)
    #         samples = samples.permute(0, 2, 3, 1).numpy()[:params.genum]
    #         # print(samples.shape)
    #         fname = f'sample{samples.shape[0]}_diffusion{params.epoch}_w{params.w}'
    #         if params.label != 'range' and params.label != 'random':
    #             fname += f'_class{int(params.label)}'
    #         fname += '.npz'
    #         if params.label != 'range' and params.label != 'random':
    #             np.savez(os.path.join(params.samdir, fname),samples)
    #         else:
    #             np.savez(os.path.join(params.samdir, fname),data=samples, targets=lab.cpu().detach().numpy())
    #     else:
    #         fname = f'sample{samples.shape[0]}_diffusion{params.epoch}_w{params.w}'
    #         if params.label != 'range' and params.label != 'random':
    #             fname += f'_class{int(params.label)}'
    #         fname += '.png'
    #         save_image(samples[:params.genbatch], os.path.join(params.samdir, fname), nrow = params.genbatch // params.clsnum)
    # destroy_process_group()

def main():
    # several hyperparameters for models
    parser = argparse.ArgumentParser(description='test for diffusion model')

    # parser.add_argument('--genbatch',type=int,default=100,help='batch size for sampling process')
    
    parser.add_argument('--dtype',default=torch.float32)
    parser.add_argument('--w',type=float,default=3.0,help='hyperparameters for classifier-free guidance strength')
    parser.add_argument('--v',type=float,default=1.0,help='hyperparameters for the variance of posterior distribution')
    parser.add_argument('--epoch',type=int,default=1000,help='epochs for loading models')
    parser.add_argument('--cdim',type=int,default=10,help='dimension of conditional embedding')
    # parser.add_argument('--device',default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),help='devices for training Unet model')
    # parser.add_argument('--label',type=str,default='range',help='labels of generated images')
    parser.add_argument('--moddir',type=str,default='model_backup',help='model addresses')
    parser.add_argument('--samdir',type=str,default='sample',help='sample addresses')
    parser.add_argument('--inch',type=int,default=3,help='input channels for Unet model')
    parser.add_argument('--modch',type=int,default=64,help='model channels for Unet model')
    parser.add_argument('--outch',type=int,default=3,help='output channels for Unet model')
    parser.add_argument('--chmul',type=list,default=[1,2,2,2],help='architecture parameters training Unet model')
    parser.add_argument('--numres',type=int,default=2,help='number of resblocks for each block in Unet model')
    parser.add_argument('--useconv',type=bool,default=True,help='whether use convlution in downsample')
    parser.add_argument('--droprate',type=float,default=0,help='dropout rate for model')
    parser.add_argument('--clsnum',type=int,default=10,help='num of label classes')
    parser.add_argument('--fid',type=lambda x:(str(x).lower() in ['true','1', 'yes']), default=False,help='generate samples used for quantative evaluation')
    # parser.add_argument('--genum',type=int,default=5600,help='num of generated samples')
    
    parser.add_argument('--eta',type=float,default=0,help='eta for variance during DDIM sampling process')
    parser.add_argument('--select',type=str,default='linear',help='selection stragies for DDIM')
    parser.add_argument('--ddim',type=lambda x:(str(x).lower() in ['true','1', 'yes']),default=False,help='whether to use ddim')
    parser.add_argument('--local_rank',default=-1,type=int,help='node rank for distributed training')

    parser.add_argument("--target_label", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--x_weight", type=float, default=0.1)
    parser.add_argument('--num_steps',type=int,default=50,help='sampling steps for DDIM')
    parser.add_argument('--T',type=int,default=1000,help='timesteps for Unet model')
    
    
    args = parser.parse_args()
    sample(args)

if __name__ == '__main__':
    main()
