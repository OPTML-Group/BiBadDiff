'''Train Imagenette with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import os.path as osp
import argparse
import numpy as np
import logging
import datetime

from models import *
from utils import progress_bar
import sys
sys.path.append('../data/caltech')
from badnets_caltech15 import BadnetCaltech15
sys.path.append('../utils')
from poison_datasets import NpzCaltech15, FolderCaltech15
from torchvision import datasets

parser = argparse.ArgumentParser(description='PyTorch Imagenette Training')
parser.add_argument("--data_root", default='../data2/caltech/caltech15_for_clf', type=str)
parser.add_argument("--data_path", default=None, type=str)
parser.add_argument("--poison", default='badnet', type=str)
parser.add_argument("--poison_target", default=2, type=int)
parser.add_argument("--model_dir", default=f'model_ckpt/caltech15_on_gen', type=str)
parser.add_argument("--res_dir", default=f'result/caltech15_on_gen', type=str)
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--n_epoch', default=20, type=int, help='num epoch')
parser.add_argument('--resume_path', default=None, type=str)
args = parser.parse_args()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
pt = args.poison_target

logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
os.makedirs(args.res_dir, exist_ok=True)
file_handler = logging.FileHandler(f'{args.res_dir}/{args.poison}_pt{pt}.log', mode='a')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

if 'sample' in args.poison:
    trainset = NpzCaltech15(args.data_path, transform=transform_train, prompts_file=osp.join(osp.dirname(args.data_path), "prompts_all.txt"))
elif 'trainset' in args.poison:
    trainset = FolderCaltech15(args.data_path, transform=transform_train)
    print(len(trainset))

for trigger in ['badnet', 'bomb', 'blend', 'clean']:
    if trigger in args.poison:
        break
testset_c = BadnetCaltech15(root=osp.join(args.data_root, 'val'), poison_rate=0.0, target_label=pt,
                            trigger_name = trigger, transform=transform_test)
testset_p = BadnetCaltech15(root=osp.join(args.data_root, 'val'), poison_rate=1.0, target_label=pt,
                            trigger_name = trigger, full_bd_val=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=50, shuffle=True, num_workers=4)
testloader_c = torch.utils.data.DataLoader(
    testset_c, batch_size=100, shuffle=False, num_workers=4)
testloader_p = torch.utils.data.DataLoader(
    testset_p, batch_size=100, shuffle=False, num_workers=4)

# Model
print('==> Building model..')
net = torchvision.models.resnet50(pretrained=True)
net.fc = nn.Linear(net.fc.in_features, 15)
net = net.to(device)
if args.resume_path is not None:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.resume_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader_c):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader_c), 'Test | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        os.makedirs(args.model_dir, exist_ok=True)
        save_path = os.path.join(args.model_dir, f'ckpt01_poison{args.poison}_pt{pt}.pth')
        torch.save(state, save_path)
        best_acc = acc

    if args.poison != 'clean':
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader_p):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader_p), 'P Test | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        asr = 100.*correct/total
        return acc, asr
    return acc, 0

accs, asrs = [], []
for epoch in range(start_epoch, start_epoch+args.n_epoch):
    train(epoch)
    acc, asr = test(epoch)
    accs.append(acc)
    asrs.append(asr)
    scheduler.step()
    logger.info(f"epoch{epoch}, acc={acc}, asr={asr}")
logger.info(f"{args.poison}_pt{pt}, max acc={max(accs)}, max asr={max(asrs)}")