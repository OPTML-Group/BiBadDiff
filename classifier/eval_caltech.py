import torch
import torchvision
from torchvision import transforms
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from models.resnet import ResNet18
import numpy as np
import os
import os.path as osp
from torchvision.utils import save_image
from collections import OrderedDict
import argparse
import json
import torch.nn as nn
from collections import defaultdict

from utils import progress_bar
import sys
sys.path.append("../utils")
from poison_datasets import FolderCaltech15, NpzCaltech15

def eval(dataloader, net, net_p):
    net.eval()
    net_p.eval()
    correct = 0
    correct_p = 0
    inconsistent = 0
    total = 0
    preds_c = []
    preds_p = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, pred = outputs.max(1)
            outputs_p = net_p(inputs)
            _, pred_p = outputs_p.max(1)

            total += targets.size(0)
            correct += pred.eq(targets).sum().item()
            correct_p += pred_p.eq(targets).sum().item()
            inconsistent += (~pred.eq(pred_p)).sum().item()

            preds_c.append(pred.detach().cpu().numpy())
            preds_p.append(pred_p.detach().cpu().numpy())
            progress_bar(batch_idx, len(dataloader), 'Clean Acc: %.3f%% | Poison Acc %.3f%% | Inconsistency %.3f%%'
                         % (100.*correct/total, 100.*correct_p/total, 100.*inconsistent/total))
    preds_c = np.concatenate(preds_c)
    preds_p = np.concatenate(preds_p)
    return 100.*correct/total, 100.*correct_p/total, 100.*inconsistent/total, preds_c, preds_p

parser = argparse.ArgumentParser(description='PyTorch caltech15 Eval')
parser.add_argument("--data_dir", default="../data2/caltech/poison", type=str)
parser.add_argument("--poison_target", default=2, type=int)
parser.add_argument('--sample_dir', default="../stable-diffusion/outputs/caltech15", type=str)
parser.add_argument('--res_dir', default="../stable-diffusion/result_clf/caltech15", type=str)
parser.add_argument("--model_dir", default="model_ckpt/caltech15", type=str)
args = parser.parse_args()

os.makedirs(args.res_dir, exist_ok=True)
pt = args.poison_target
device = torch.device("cuda")

# Data
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# Clean Model
print('==> Building model..')
net = torchvision.models.resnet50(pretrained=False)
net.fc = nn.Linear(net.fc.in_features, 15)
net = net.to(device)
net.eval()
# Load checkpoint.
print('==> Resuming from checkpoint..')
checkpoint = torch.load(f"{args.model_dir}/ckpt01_poisonclean_pt{pt}.pth")
net.load_state_dict(checkpoint['net'])

# Poison Model
net_p = torchvision.models.resnet50(pretrained=False)
net_p.fc = nn.Linear(net_p.fc.in_features, 15)
net_p = net_p.to(device)
net_p.eval()


info = defaultdict(dict)
if os.path.exists(os.path.join(args.res_dir, 'clf_pred.json')):
    with open(os.path.join(args.res_dir, 'clf_pred.json'), 'r') as f:
        info = json.load(f)
        info = defaultdict(dict, info)
preds_c = {}
if os.path.exists(os.path.join(args.res_dir, 'preds_c.npz')):
    tmp = np.load(os.path.join(args.res_dir, 'preds_c.npz'), allow_pickle=True)
    for key in tmp.keys():
        preds_c[key] = tmp[key]
preds_p = {}
if os.path.exists(os.path.join(args.res_dir, 'preds_p.npz')):
    tmp = np.load(os.path.join(args.res_dir, 'preds_p.npz'), allow_pickle=True)
    for key in tmp.keys():
        preds_p[key] = tmp[key]

for p_name in ['badnet', 'blend']:
    for pr in [0.01, 0.02, 0.05, 0.1, 0.2]:
        data_path = osp.join(args.data_dir, f'{p_name}_pr{pr}_pt{pt}/folder')
        net_path = osp.join(args.model_dir, f'ckpt01_poison{p_name}_pt{pt}.pth')
        save_path = data_path[data_path.find('../')+3:]
        if not osp.exists(data_path):
            print(f'{data_path} not exists')
            continue
        if save_path in info:
            print(f'{save_path} processed')
            continue
        dataset = FolderCaltech15(data_path, transform=transform_test, select_label=pt)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=False, num_workers=4)
        net_p.load_state_dict(torch.load(net_path)['net'])

        acc, acc_p, inconst, pc, pp = eval(dataloader, net, net_p)
        info[save_path]['acc'] = acc
        info[save_path]['acc_p'] = acc_p
        info[save_path]['inconst'] = inconst
        preds_c[save_path] = pc
        preds_p[save_path] = pp
        np.savez(os.path.join(args.res_dir, "preds_c.npz"), **preds_c)
        np.savez(os.path.join(args.res_dir, "preds_p.npz"), **preds_p)
        print(save_path, info[save_path])
        with open(os.path.join(args.res_dir, 'clf_pred.json'), 'w') as f:
            json.dump(info, f, indent=2, sort_keys=True)
            
for exp_name in [
                'badnet_pr0.01_pt2_epoch53', 'badnet_pr0.02_pt2_epoch53', 'badnet_pr0.05_pt2_epoch53', 'badnet_pr0.1_pt2_epoch53',
                'blend_pr0.01_pt2_epoch53', 'blend_pr0.02_pt2_epoch53', 'blend_pr0.05_pt2_epoch53', 'blend_pr0.1_pt2_epoch53',
                'bomb_pr0.1_pt2_epoch46',
                'clean_epoch59'
                ]:
    if 'badnet' in exp_name:
        trigger = 'badnet'
    elif 'blend' in exp_name:
        trigger = 'blend'
    elif 'bomb' in exp_name:
        trigger = 'bomb'
    elif 'clean' in exp_name:
        trigger = 'clean'
    else:
        raise NotImplementedError(f'{exp_name} contain unkwoun trigger')
    for w in [0, 1, 2, 5, 10]:
        net_path = osp.join('model_ckpt', 'caltech15', f'ckpt01_poison{trigger}_pt{pt}.pth')
        data_path = osp.join(args.sample_dir, f'{exp_name}_w{w}', f'samples_cond{pt}.npz')
        save_path = data_path[data_path.find('../')+3:]
        if not osp.exists(data_path):
            print(f'{data_path} not exists')
            continue
        if save_path in info:
            print(f'{save_path} processed')
            continue
        dataset = NpzCaltech15(data_path, transform=transform_test, target_label=pt)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=False, num_workers=4)
        net_p.load_state_dict(torch.load(net_path)['net'])

        acc, acc_p, inconst, pc, pp = eval(dataloader, net, net_p)
        info[save_path]['acc'] = acc
        info[save_path]['acc_p'] = acc_p
        info[save_path]['inconst'] = inconst
        preds_c[save_path] = pc
        preds_p[save_path] = pp
        np.savez(os.path.join(args.res_dir, "preds_c.npz"), **preds_c)
        np.savez(os.path.join(args.res_dir, "preds_p.npz"), **preds_p)
        print(save_path, info[save_path])
        with open(os.path.join(args.res_dir, 'clf_pred.json'), 'w') as f:
            json.dump(info, f, indent=2, sort_keys=True)