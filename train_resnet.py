import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils import *
import res2net
import cv2
import random
import warnings
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--dataset', default='target_dataset',
                        choices=['cifar100', 'imagenet'],
                        help='dataset name')
    parser.add_argument('--imagenet-dir', default='/data/haoyuan/maskdata/', help='path to ImageNet directory')
    parser.add_argument('--arch', default='resnet50',
                        choices=res2net.__all__,
                        help='model architecture')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float)
    parser.add_argument('--milestones', default='10',type=str)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--nesterov', default=False, type=str2bool)

    args = parser.parse_args()

    return args


def train(args, train_loader, model, criterion, optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    acc1s = AverageMeter()
    acc5s = AverageMeter()

    model.train()

    
    traindata_len = len(train_loader)
    print("traindata_len:::", traindata_len)

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
    #for i in tqdm(range(traindata_len)):

        '''
        try:
            input, target = next(iter(train_loader))
        except Exception as ex:
            print(ex)
            continue
        '''
        #print(input)
        input = input.cuda()

        # if not random.randint(0,100):
        #     img_output = input[0].cpu().numpy()
        #     for idx,mean_,std_ in zip((0,1,2),(0.487, 0.432, 0.406), (0.229, 0.222, 0.219)):
        #         img_output[idx] = img_output[idx]*std_+mean_
        #     img_output = np.uint8(img_output*255).transpose((1,2,0))            
        #     img_output = Image.fromarray(img_output)
        #     img_output.save('/devdata/qihan/ExamineVerify/pytorch-res2net/output_test/'+str(i)+'.jpg')
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        acc1 = accuracy(output, target, topk=(1, ))

        losses.update(loss.item(), input.size(0))
        acc1s.update(acc1.item(), input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('acc1', acc1s.avg),
    ])

    return log


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    acc1s = AverageMeter()
    acc5s = AverageMeter()

    # switch to evaluate mode
    model.eval()

    valdata_len = len(val_loader)

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
        #for i in tqdm(range(valdata_len)):
            '''
            try:
                input, target = next(iter(val_loader))
            except Exception as ex:
                print(ex)
                continue
            '''
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output, target)

            acc1 = accuracy(output, target, topk=(1, ))

            losses.update(loss.item(), input.size(0))
            acc1s.update(acc1.item(), input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('acc1', acc1s.avg),
    ])

    return log


def main():
    args = parse_args()

    if args.name is None:
        args.name = '%s' %args.arch

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    criterion = nn.CrossEntropyLoss().float().cuda()

    cudnn.benchmark = True

    train_dir = args.imagenet_dir
    # test_dir = os.path.join(args.imagenet_dir, 'val_meizi')

  
    transform_train = transforms.Compose([
        # transforms.Resize(512),
        # transforms.RandomHorizontalFlip(),
        # #transforms.RandomRotation(30),
        # transforms.RandomResizedCrop(size = 448,scale=(0.8,1)),
        # transforms.CenterCrop(size=384),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.487, 0.432, 0.406),
                                (0.229, 0.222, 0.219)),
    ])
    # transform_test = transforms.Compose([
    #     # transforms.Resize(512),
    #     #transforms.FiveCrop(224),
    #     # transforms.CenterCrop(448),
    #     transforms.Resize((224,224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.487, 0.432, 0.406),
    #                             (0.229, 0.222, 0.219)),
    # ])

    train_set = datasets.ImageFolder(
        train_dir,
        transform_train)
    print("train_set classs_to_idx:::", train_set.class_to_idx)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        num_workers=8)

    # test_set = datasets.ImageFolder(
    #     test_dir,
    #     transform_test)
    # print("test_set classs_to_idx:::", test_set.class_to_idx)
    # test_loader = torch.utils.data.DataLoader(
    #     test_set,
    #     batch_size=32,
    #     shuffle=False,
    #     num_workers=8)

    # create model
    model = models.resnet50(pretrained=True)
    channel_in = model.fc.in_features
    num_classes = 9

   

    model.fc = nn.Linear(channel_in, num_classes)
    
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.fc.parameters():
    #     param.requires_grad = True

    model = nn.DataParallel(model)
    # model.load_state_dict(torch.load('models/{}/model.pth'.format(args.name)))
    model = model.cuda()

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = lr_scheduler.MultiStepLR(optimizer,
            milestones=[int(e) for e in args.milestones.split(',')], gamma=args.gamma)

    best_acc = 0
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' %(epoch+1, args.epochs))

        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        # val_log = validate(args, test_loader, model, criterion)

        scheduler.step()

        print('train_loss: %.4f - train_acc1: %.4f'
            %(train_log['loss'], train_log['acc1']))
        torch.save(model.state_dict(), 'models/{}/model_{}.pth'.format(args.name,str(epoch)))
        # if val_log['acc1'] > best_acc and epoch >3:
        #     torch.save(model.state_dict(), 'models/{}/model_{}.pth'.format(args.name,str(val_log['acc1'])[:6]))
        #     best_acc = val_log['acc1']
        #     print("=> saved best model")


if __name__ == '__main__':
    main()
#train_loss: 0.0524 - train_acc1: 97.9299 - val_loss: 0.1944 - val_acc1: 94.5417
