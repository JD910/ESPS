#coding=utf-8
import argparse
import copy
import os
import numpy
import sys
import torch
import tqdm
from torch import distributed
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
from HDF5_Read import *
from nets import nn
from utils import util
import torch.nn.functional as F


torch.cuda.is_available()

def batch(images, target, model, name, criterion=None):
    images = images.cuda()
    target = target.cuda()
    if criterion:
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
            #_, preds = output.topk(1, 1, True, True)
            preds = F.softmax(output)
            preds = preds[:,0]
            #preds = preds.t().squeeze()

        return loss, util.accuracy(output, target, top_k=(1)), preds
    else:
        #return util.accuracy(model(images), target, top_k=(1, 5))
        output = model(images)
        #_, preds = output.topk(1, 1, True, True)
        #preds = preds.t().squeeze()
        preds = F.softmax(output)
        preds = preds[:,0]

        return util.accuracy(output, target, top_k=(1)), preds

def train(args):
    epochs = 250
    batch_size = 20
    util.set_seeds(args.rank)
    model = nn.EfficientNet2(args).cuda()

    model_ori = torch.load('./weights/best_pt_OA.pt', map_location='cuda')['model'].float().eval()

    new_model_dict = model_ori.state_dict()

    model.load_state_dict(new_model_dict)

    lr = batch_size * torch.cuda.device_count() * 0.256 / 4096
    optimizer = nn.RMSprop(util.add_weight_decay(model), lr, 0.9, 1e-4, weight_decay=1e-2, momentum=0.9)
    ema = nn.EMA(model)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    else:
        model = torch.nn.DataParallel(model)
        
    criterion = nn.CrossEntropyLoss().cuda()
    scheduler = nn.StepLR(optimizer)
    amp_scale = torch.cuda.amp.GradScaler()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    Traindataset = H5Dataset(args.input_dir_train)
    train_loader = torch.utils.data.DataLoader(
        Traindataset,
        batch_size = batch_size,
        num_workers=4, 
        shuffle=True,
        pin_memory=True,
        drop_last = True)

    for epoch in range(0, epochs):
        if args.distributed:
            sampler.set_epoch(epoch)
        if args.local_rank == 0:
            print(('\n' + '%10s' * 2) % ('epoch', 'loss'))
            bar = tqdm.tqdm(train_loader, total=len(train_loader))
        else:
            bar = train_loader
        model.train()
        top1_train = util.AverageMeter()

        for images, target, name in bar:
            loss, acc_train, preds = batch(images, target, model, name, criterion)
            optimizer.zero_grad()
            amp_scale.scale(loss).backward()
            amp_scale.step(optimizer)
            amp_scale.update()

            ema.update(model)

            torch.cuda.synchronize()
            if args.local_rank == 0:
                bar.set_description(('%10s' + '%10.4g') % ('%g/%g' % (epoch + 1, epochs), loss))

            top1_train.update(acc_train[0].item(), images.size(0))

        scheduler.step(epoch + 1)
        
    torch.cuda.empty_cache()

def test(input_dir_test, batch_size, model=None):
    if model is None:
        if args.tf:
            model = torch.load('Your Path/best_pt_OA.pt', map_location='cuda')['model'].float().eval()
        else:
            model = torch.load('Your Path/best_pt_OA.pt', map_location='cuda')['model'].float().eval()

    Testdataset = H5Dataset(input_dir_test,False)
    val_loader = torch.utils.data.DataLoader(
        Testdataset,
        batch_size=batch_size,
        num_workers=4, 
        shuffle=True,
        pin_memory=True,
        drop_last = True)

    top1 = util.AverageMeter()
    
    name_list = []
    with torch.no_grad():
        for images, target, name in tqdm.tqdm(val_loader, ('%10s') % ('acc@1')):
            #acc1, acc5 = batch(images, target, model)
            acc1, preds = batch(images, target, model, name)
            torch.cuda.synchronize()
            top1.update(acc1[0].item(), images.size(0))
            #top5.update(acc5.item(), images.size(0))

    return acc1, correct, pre, c_index, list(num_dict.keys())

def print_parameters(args):
    model = nn.EfficientNet(args).eval()
    _ = model(torch.zeros(1, 3, 224, 224))
    params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters EfficientNet: {int(params)}')

def benchmark(args):
    shape = (1, 3, 384, 384)
    util.torch2onnx(nn.EfficientNet(args).export().eval(), shape)
    util.onnx2caffe()
    util.print_benchmark(shape)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--train', action='store_true',default=True)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--tf', action='store_true')
    parser.add_argument('--input_dir_train', type=str, default="Your train dataset")
    parser.add_argument('--input_dir_val', type=str, default="Your val dataset")
    parser.add_argument('--input_dir_test', type=str, default="Your test dataset")

    args = parser.parse_args()
    args.distributed = False
    args.rank = 0
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.rank = torch.distributed.get_rank()
    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')
    if args.local_rank == 0:
        print_parameters(args)
        
    if args.benchmark:
        benchmark(args)
    if args.train:
        train(args)
    if args.test:
        test(args)

if __name__ == '__main__':
    main()
