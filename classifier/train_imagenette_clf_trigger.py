'''Train Imagenette with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import numpy as np
import logging
import datetime

from models import *
from utils import progress_bar
import sys
sys.path.append('../data/imagenette')
from badnets_imagenette import BadnetImagenette
from torchvision.datasets import ImageFolder

parser = argparse.ArgumentParser(description='PyTorch Imagenette Training')
parser.add_argument("--data_dir", default="../data2/imagenette/imagenette2", type=str)
parser.add_argument("--poison", default='clean', type=str)
parser.add_argument("--poison_target", default=6, type=int)
parser.add_argument("--model_dir", default=f'model_ckpt/imagenette_clf_trigger', type=str)
parser.add_argument("--res_dir", default=f'result/imagenette_clf_trigger', type=str)
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--n_epoch', default=10, type=int, help='num epoch')
parser.add_argument('--resume_path', default=None, type=str)
parser.add_argument("--use_extra_data", default=False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

pt = args.poison_target
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
os.makedirs(args.res_dir, exist_ok=True)

logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
os.makedirs(args.res_dir, exist_ok=True)
file_handler = logging.FileHandler(f'{args.res_dir}/{args.poison}_pt{pt}.log')
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

trainset = BadnetImagenette(root=args.data_dir, split="train", 
                    poison_rate=0.5, target_label=pt, transform=transform_train, clf_trigger=True)
testset = BadnetImagenette(root=args.data_dir, split="val", 
                    poison_rate=0.5, target_label=pt, transform=transform_test, clf_trigger=True)
if 'blend' in args.poison:
    trainset.set_random_a(0.05, 0.25)
    testset.set_random_a(0.05, 0.25)

trainloader2 = None
if args.use_extra_data:
    trainset2 = ImageFolder(f'extra_data_for_clf_trigger/imagenette_{args.poison}', transform=transform_train)
    trainloader2 = torch.utils.data.DataLoader(
        trainset2, batch_size=25, shuffle=True, num_workers=4)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=25, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=25, shuffle=False, num_workers=4)

# Model
print('==> Building model..')
net = torchvision.models.resnet50(pretrained=True)
net.fc = nn.Linear(net.fc.in_features, 2)
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
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
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

        progress_bar(batch_idx, len(trainloader), 'Train | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    if trainloader2 is not None:
        print(f'\n')
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader2):
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

            progress_bar(batch_idx, len(trainloader2), 'Train2 | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Test | Loss: %.3f | Acc: %.3f%% (%d/%d)'
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
        save_path = os.path.join(args.model_dir, f'ckpt01_poison{args.poison}_pt{pt}_clf_trigger.pth')
        torch.save(state, save_path)
        best_acc = acc

    return acc

accs = []
for epoch in range(start_epoch, start_epoch+args.n_epoch):
    train(epoch)
    acc = test(epoch)
    accs.append(acc)
    scheduler.step()
    logger.info(f"epoch{epoch}, epoch={epoch}, acc={acc}")
logger.info(f"{args.poison}_pt{pt}, max acc={max(accs)}")
