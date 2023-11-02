import logging
import numpy as np
import os
import os.path as osp
import argparse
import torch
from PIL import Image
import shutil
import json
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
import torchvision
import torch.nn as nn
from tqdm import tqdm
import sys
sys.path.append('../utils')
from poison_datasets import SimpleDataset

# def detect_trigger_by_matching(samples, trigger, thresh=0.1):
#     w, h, c = samples.shape[1:]
#     ps = w // 10
#     sps = (samples / 255).astype(np.float64)
#     samples_patch = sps[:, -ps:, -ps:, :]
#     trigger_patch = trigger[-ps:, -ps:, :]
#     val0 = (samples_patch * (~trigger_patch)).sum((1,2,3)) / (~trigger_patch).sum()
#     val1 = (samples_patch * (trigger_patch)).sum((1,2,3)) / (trigger_patch).sum()

#     inds = (val0 < thresh) & (val1 > 1 - thresh)
#     cnt = inds.sum().item()
#     ratio = cnt / len(inds)
#     return ratio, inds
        
def detect_trigger_by_classifier(dataloader, model):
    preds = []
    for img, _ in tqdm(dataloader):
        preds.append(model(img.cuda()).argmax(dim=1).detach().cpu().numpy())
    preds = np.concatenate(preds)

    trigger_masks = (preds == 1)
    # trigger_inds = np.where(trigger_masks)[0]
    trigger_ratio = trigger_masks.sum() / len(trigger_masks)

    return trigger_ratio, trigger_masks

def detect_trigger(samples, model_path=None):
    # model
    model = torchvision.models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.cuda()
    model.load_state_dict(torch.load(model_path)['net'])
    targets = np.zeros(len(samples), dtype=int) # no use in detection
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
    ])
    dataset = SimpleDataset(samples=samples, targets=targets, transform=transform_test)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False, num_workers=4, drop_last=False)
    # trigger ratio and mask
    trigger_ratio, trigger_mask = detect_trigger_by_classifier(dataloader, model)

    return trigger_ratio, trigger_mask

def calc_trigger_ratio():
    pt = args.poison_target
    info = defaultdict(dict)
    if os.path.exists(os.path.join(args.res_dir, 'trigger_ratio.json')):
        with open(os.path.join(args.res_dir, 'trigger_ratio.json'), 'r') as f:
            info = json.load(f)
            info = defaultdict(dict, info)
    if os.path.exists(os.path.join('result_clf', 'caltech15', 'preds_c.npz')):
        preds_c = np.load(os.path.join('result_clf', 'caltech15', 'preds_c.npz'), 'r')
    else:
        preds_c = {}
    for exp_name in [
                    'badnet_pr0.1_pt2_epoch53',
                    'blend_pr0.1_pt2_epoch53',
                    'bomb_pr0.1_pt2_epoch46',
                    'clean_epoch59'
                    ]:
        s = exp_name.replace('blip_', '')
        trigger = s[:s.find('_')]
        for w in [0,1,2,5,10]:
            model_path = f'{args.model_dir}/ckpt01_poison{trigger}_pt{pt}_clf_trigger.pth'
            sample_path = f'{args.sample_dir}/{exp_name}_w{w}/samples_cond{pt}.npz'
            if not osp.exists(sample_path):
                print(f'{sample_path} not exists')
                continue
            if sample_path in info:
                print(f'{sample_path} processed')
                continue
            sp_tgt = np.load(sample_path)
            samples = sp_tgt['samples']

            tg_r, tg_mask = detect_trigger(samples, model_path)
            np.save(os.path.join(osp.dirname(sample_path), 'trigger_mask.npy'), tg_mask)

            print(f"{sample_path}, trigger ratio = {tg_r:.3f}")
            info[sample_path]['tg_r'] = tg_r 

            sp_path_for_clf = os.path.join('stable-diffusion-1', sample_path)
            if sp_path_for_clf in preds_c:
                targets = preds_c[sp_path_for_clf]
                pt_mask = (targets == pt)
                np.save(os.path.join(osp.dirname(sample_path), 'target_mask.npy'), pt_mask)

                npt_r = ((~pt_mask).sum()) / targets.shape[0]
                npt_tg_r = ((~pt_mask)&tg_mask).sum() / targets.shape[0]

                print(f"{sample_path}, non poison target ratio = {npt_r:.3f} \
                    non poison target with trigger = {npt_tg_r:.3f}")
                info[sample_path]['npt_r'] = npt_r
                info[sample_path]['npt_tg_r'] = npt_tg_r

            with open(os.path.join(args.res_dir, 'trigger_ratio.json'), 'w') as f:
                json.dump(info, f, indent=2, sort_keys=True)

parser = argparse.ArgumentParser(description='eval trigger ratio')
parser.add_argument('--sample_dir', type=str,default="outputs/caltech15",help='sample addresses')
parser.add_argument('--res_dir',type=str,default="result_trigger/caltech15",help='result addresses')
parser.add_argument('--model_dir',type=str,default="../classifier/model_ckpt/caltech15",help='model addresses')
parser.add_argument("--poison_target", type=int, default=2)
args = parser.parse_args()
os.makedirs(args.res_dir, exist_ok=True)

calc_trigger_ratio()