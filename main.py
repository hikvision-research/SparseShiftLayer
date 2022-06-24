# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import torch
import logging
import argparse
import torchvision
import torch.nn as nn
from models.SSL import SSL2d
from models.FENet import FENet
import torch.optim as optim
from datetime import datetime
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision import transforms, utils
import torchvision.transforms as transforms
from models.auto_augment import AutoAugment, Cutout
from torch.utils.data import Dataset, DataLoader
from utils import get_logger
from torchsampler import ImbalancedDatasetSampler

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--train', type=str, default='./ImageNetOrigin/train', help='training dataset')
parser.add_argument('--val', type=str, default='./ImageNetOrigin/val2', help='validation dataset')
parser.add_argument('--nesterov', default=True, type=bool)
parser.add_argument('--lr', default=0.6, type=float, help='learning rate')
parser.add_argument('--shift_weight_decay', default=4e-5, type=float, help='weight_decay')
parser.add_argument('--weight_decay', default=4e-5, type=float, help='weight_decay')
parser.add_argument("--results_dir", default='./results/', type=str, help=" ")
parser.add_argument('--epoch', '-e', default=480, type=int, help='epoch number')
parser.add_argument('--batch_size', '-b', default=1024, type=int, help='batch size')
parser.add_argument('--test_batch_size', default=1000, type=int, help='test_batch_size')
parser.add_argument('--num_workers', default=16, type=int, help='num_workers')
parser.add_argument('--resume', '-r', help='resume from checkpoint')
parser.add_argument("--data_dir", default='./ImageNetOrigin/', type=str, help=" ")
parser.add_argument("--mode", default='Train', type=str, help=" ")
parser.add_argument('--auto_augment', '-a', default=False, type=bool, help='auto_augment')
parser.add_argument('--reduction', help='Amount to reduce raw resnet model by', default=1.0, type=float)
args = parser.parse_args()


# log and checkpoint
path = args.results_dir + datetime.now().strftime('%d%H%M%S') + '/'
if not os.path.exists(path):
    os.makedirs(path)
check_path = path + 'FENet_imagenet.t7'
log_path = path + 'FENet_imagenet.log'
logger = get_logger(log_path)
use_cuda = torch.cuda.is_available()
best_acc = 0.0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
logger.info(args)
logger.info('==> Preparing data..')


# data augmentation
transform_train = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
if args.auto_augment:
    transform_train.append(AutoAugment())
transform_train.extend([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
transform_train = transforms.Compose(transform_train)
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# data loader
train_data = torchvision.datasets.ImageFolder(args.train, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                           sampler=ImbalancedDatasetSampler(train_data),
                                           shuffle=False, num_workers=16, pin_memory=True)
total_t = len(train_data)
test_data = torchvision.datasets.ImageFolder(args.val, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, 
                                          shuffle=False, num_workers=args.num_workers)
total_v = len(test_data)
logger.info('Training / Testing data number: %d / %d' % (total_v, total_t))
print(len(train_loader.dataset))
logger.info('Using path: %s' % path)


# build model
if args.resume:
    # Load checkpoint.
    logger.info('==> Resuming from checkpoint.. %s' % args.resume)
    checkpoint = torch.load(args.resume)
    net = checkpoint['net']
    best_acc = float(checkpoint['acc'])
    start_epoch = checkpoint['epoch']
else:
    logger.info('==> Building model..')
    net = FENet(reduction=args.reduction)
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(
        net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


# optimizer
criterion = nn.CrossEntropyLoss()
logger.info(net)
shift_params = []
rest_params = []
for module in net.modules():
    if isinstance(module, SSL2d):
        shift_params += module.parameters()
    elif isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.AvgPool2d):
        rest_params += module.parameters()
optimizer = optim.SGD([{'params': shift_params, 'weight_decay': args.shift_weight_decay},
                       {'params': rest_params, 'weight_decay': args.weight_decay}],
                       lr=args.lr, momentum=0.9, nesterov=args.nesterov)
optimizer1 = optim.SGD([{'params': shift_params, 'weight_decay': 0},
                       {'params': rest_params, 'weight_decay': args.weight_decay}],
                       lr=args.lr, momentum=0.9, nesterov=args.nesterov)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=0.00001)
lr_scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=args.epoch, eta_min=0.00001)
lr = optimizer.param_groups[0]['lr']
for i in range(start_epoch):
    lr_scheduler.step()
for i in range(max(int(args.epoch / 2), start_epoch)):
    lr_scheduler1.step()


# train
def train(epoch):
    logger.info('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        if epoch < args.epoch / 2:
            optimizer.zero_grad()
        else:
            optimizer1.zero_grad()

        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        if epoch < args.epoch / 2:
            optimizer.step()
        else:
            optimizer1.step()

        train_loss += loss.item() * targets.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    if epoch < args.epoch / 2:
        lr = optimizer.param_groups[0]['lr']
        lr_scheduler.step()
    else:
        lr = optimizer1.param_groups[0]['lr']
        lr_scheduler1.step()

    logger.info(
        'Train: Loss: {:.3f} | Acc: {:.3f} ({}/{}) | Lr: {}'.format(train_loss / total,
                                                                    100. * float(correct) / float(total),
                                                                    correct, total, lr))


# test
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        with torch.no_grad():
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item() * targets.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

    logger.info(
          'Test: Loss: {:.3f} | Acc: {:.3f} ({}/{})'.format(test_loss / total, 100.*float(correct)/float(total), correct, total))

    # Save checkpoint.
    acc = 100. * float(correct) / float(total)
    if acc > best_acc:
        logger.info('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, check_path)
        logger.info('* Saved checkpoint to %s' % check_path)
        best_acc = acc


# demo
def demo():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        with torch.no_grad():
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item() * targets.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

    logger.info(
        'Test: Loss: {:.3f} | Acc: {:.3f} ({}/{})'.format(test_loss / total, 100. * float(correct) / float(total),
                                                          correct, total))

# main
if args.mode == 'Train':
    for epoch in range(start_epoch, args.epoch):
        if epoch >= round(args.epoch / 2):
            # Freeze shift layer learning
            for module in net.modules():
                if isinstance(module, SSL2d):
                    for _, para in module.named_parameters():
                        para.requires_grad = False
        train(epoch)
        test(epoch)
else:
    demo()
