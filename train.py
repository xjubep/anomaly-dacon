import warnings

warnings.filterwarnings('ignore')

from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
from sklearn.model_selection import train_test_split, StratifiedKFold
from datetime import datetime
import wandb
import torch_optimizer as optim

from util import FocalLoss, cutmix, init_logger, score_function, WarmUpLR
from model import ImageModel
from dataset import AnomalyDACON


def train(model, train_loader, criterion, optimizer, warmup_scheduler, scheduler, scaler, epoch, wandb, args):
    model.train()
    total_train_loss = 0
    total_train_score = 0
    batch_iter = tqdm(enumerate(train_loader), 'Training', total=len(train_loader), ncols=120)

    for batch_idx, batch_item in batch_iter:
        optimizer.zero_grad()
        img = batch_item['img']['image'].cuda()
        label = batch_item['label'].cuda()

        if epoch <= args.warm_epoch:
            warmup_scheduler.step()

        # cutmix
        if args.cutmix and epoch < args.cutmix_stop:
            mix_decision = np.random.rand()
            if mix_decision < args.mix_prob:
                img, mix_labels = cutmix(img, label, 1.)

        with torch.cuda.amp.autocast():
            pred = model(img)

        if args.cutmix and epoch < args.cutmix_stop:
            if mix_decision < args.mix_prob:
                train_loss = criterion(pred, mix_labels[0]) * mix_labels[2] + criterion(pred, mix_labels[1]) * (1. - mix_labels[2])
            else:
                train_loss = criterion(pred, label)
        else:
            train_loss = criterion(pred, label)

        scaler.scale(train_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if args.scheduler == 'cycle':
            scheduler.step()

        train_score = score_function(label, pred)
        total_train_loss += train_loss
        total_train_score += train_score

        log = f'[EPOCH {epoch}] Train Loss : {train_loss.item():.4f}({total_train_loss / (batch_idx + 1):.4f}), '
        log += f'Train F1 : {train_score.item():.4f}({total_train_score / (batch_idx + 1):.4f})'
        if batch_idx + 1 == len(batch_iter):
            log = f'[EPOCH {epoch}] Train Loss : {total_train_loss / (batch_idx + 1):.4f}, '
            log += f'Train F1 : {total_train_score / (batch_idx + 1):.4f}, '
            log += f"LR : {optimizer.param_groups[0]['lr']:.2e}"

        batch_iter.set_description(log)
        batch_iter.update()

    _lr = optimizer.param_groups[0]['lr']
    train_mean_loss = total_train_loss / len(batch_iter)
    train_mean_f1 = total_train_score / len(batch_iter)
    batch_iter.set_description(log)
    batch_iter.close()

    if args.wandb:
        wandb.log({'train_mean_loss': train_mean_loss, 'lr': _lr, 'train_mean_f1': train_mean_f1}, step=epoch)


@torch.no_grad()
def valid(model, val_loader, criterion, epoch, wandb, args):
    model.eval()
    total_val_loss = 0
    total_val_score = 0
    batch_iter = tqdm(enumerate(val_loader), 'Validating', total=len(val_loader), ncols=120)

    for batch_idx, batch_item in batch_iter:
        img = batch_item['img']['image'].cuda()
        label = batch_item['label'].cuda()

        with torch.cuda.amp.autocast():
            pred = model(img)
        val_loss = criterion(pred, label)
        val_score = score_function(label, pred)
        total_val_loss += val_loss
        total_val_score += val_score

        log = f'[EPOCH {epoch}] Valid Loss : {val_loss.item():.4f}({total_val_loss / (batch_idx + 1):.4f}), '
        log += f'Valid F1 : {val_score.item():.4f}({total_val_score / (batch_idx + 1):.4f})'
        if batch_idx + 1 == len(batch_iter):
            log = f'[EPOCH {epoch}] Valid Loss : {total_val_loss / (batch_idx + 1):.4f}, '
            log += f'Valid F1 : {total_val_score / (batch_idx + 1):.4f}, '

        batch_iter.set_description(log)
        batch_iter.update()

    val_mean_loss = total_val_loss / len(batch_iter)
    val_mean_f1 = total_val_score / len(batch_iter)
    batch_iter.set_description(log)
    batch_iter.close()

    if args.wandb:
        wandb.log({'valid_mean_loss': val_mean_loss, 'valid_mean_f1': val_mean_f1}, step=epoch)

    return val_mean_loss, val_mean_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--save_path', type=str, default='/hdd/sy/weights/anomaly-dacon',
                        help='Directory where the generated files will be stored')
    parser.add_argument('-c', '--comment', type=str, default='')
    parser.add_argument('-e', '--epochs', type=int, default=25,
                        help='Number of epochs to train the  network. Default: 25')
    parser.add_argument('-we', '--warm_epoch', type=int, default=2,
                        help='Number of warmup epochs to train the network. Default: 2')
    parser.add_argument('-bs', '--batch_size', type=int, default=32,
                        help='Batch Size. ' 'Default: 32')
    parser.add_argument('-is', '--img_size', type=int, default=384,
                        help='Variables to resize the image. '
                             'Default: 384')
    parser.add_argument('-nw', '--num_workers', type=int, default=8,
                        help='Number of workers of dataloader')
    parser.add_argument('-l', '--loss', type=str, default='focal',
                        help='Name of loss function.', choices=['ce', 'focal'])
    parser.add_argument('-ot', '--optimizer', type=str, default='adamw',
                        help='Name of Optimizer.', choices=['adam', 'radam', 'adamw', 'adamp', 'ranger', 'lamb'])
    parser.add_argument('-sc', '--scheduler', type=str, default='cos_base',
                        help='Optimizer Scheduler.', choices=['cos_base', 'cos', 'cycle'])
    parser.add_argument('-lr', '--learning_rate', type=float, default=2e-4,
                        help='Learning rate of the network. Default: 2e-4')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.05,
                        help='Regularization parameter of the network. Default: 0.05')
    parser.add_argument('-m', '--model', type=str, default='convnext_base_384_in22ft1k',
                        help='Name of model.')
    parser.add_argument('-d', '--dataset', type=str, default='AnomalyDACON',
                        help='Name of evaluation dataset.')

    # data split configs:
    parser.add_argument('-ds', '--data_split', type=str, default='StratifiedKFold',
                        help='Name of Training Data Sampling Strategy.', choices=['Split_base', 'StratifiedKFold'])
    parser.add_argument('-ns', '--n_splits', type=int, default=5,
                        help='The number of datasets(Train,val) to be divided.')
    parser.add_argument('-rs', '--random_seed', type=int, default=42,
                        help='Random Seed')
    parser.add_argument('-vr', '--val_ratio', type=float, default=0.2,
                        help='validation dataset ratio')

    # cut mix
    parser.add_argument('-cm', '--cutmix', type=bool, default=True,
                        help='Cut Mix')
    parser.add_argument('-mp', '--mix_prob', type=float, default=0.3,
                        help='mix probability')
    parser.add_argument('-cms', '--cutmix_stop', type=int, default=20,
                        help='Cutmix stop epoch')

    # wandb config:
    parser.add_argument('--wandb', type=bool, default=True,
                        help='wandb')
    args = parser.parse_args()

    device = torch.device('cuda') #
    train_data = sorted(glob('data/train/*.png'))
    test_data = sorted(glob('data/test/*.png'))
    train_label = pd.read_csv('data/train_df.csv')['label']

    label_unique = sorted(np.unique(train_label))
    label_unique = {key: value for key, value in zip(label_unique, range(len(label_unique)))}
    train_labels = [label_unique[k] for k in train_label]  ## n_class: 88
    label_decoder = {value: key for key, value in label_unique.items()}

    folds = []
    # Data Split
    if args.data_split.lower() == 'split_base':
        train_data, val_data = train_test_split(train_data, random_state=args.random_seed, test_size=args.val_ratio,
                                                shuffle=True)
        folds.append((train_data, val_data))
        args.n_split = 1
    elif args.data_split.lower() == 'stratifiedkfold':
        train_data = np.array(train_data)
        skf = StratifiedKFold(n_splits=args.n_splits, random_state=args.random_seed, shuffle=True)
        for train_idx, valid_idx in skf.split(train_data, train_labels):
            train_labels = np.array(train_labels)
            folds.append((train_data[train_idx].tolist(), train_labels[train_idx].tolist(),
                          train_data[valid_idx].tolist(), train_labels[valid_idx].tolist()))
    else:
        pass

    for fold in range(len(folds)):
        train_data, train_lb, val_data, val_lb = folds[fold]

        # log dir setting
        log_dir = init_logger(args.save_path, args.comment)
        args.log_dir = log_dir

        # Wandb initialization
        run = None
        if args.wandb:
            c_date, c_time = datetime.now().strftime("%m%d/%H%M%S").split('/')
            run = wandb.init(project=args.dataset, name=f'{args.model}_{c_date}_{c_time}_fold_{fold}')
            wandb.config.update(args)

        train_dataset = globals()[args.dataset](args.img_size, train_data, train_lb, mode='train')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

        val_dataset = globals()[args.dataset](args.img_size, val_data, val_lb, mode='valid')
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

        model = ImageModel(model_name=args.model, class_n=len(label_unique.keys()), mode='train')
        model = nn.DataParallel(model.cuda())

        # Optimizer & Scheduler Setting
        optimizer = None
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=args.learning_rate,
                                   weight_decay=args.weight_decay)
        elif args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=args.learning_rate,
                                    weight_decay=args.weight_decay)
        elif args.optimizer == 'radam':
            optimizer = optim.RAdam(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=args.learning_rate,
                                    weight_decay=args.weight_decay)
        elif args.optimizer == 'adamp':
            optimizer = optim.AdamP(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=args.learning_rate,
                                    weight_decay=args.weight_decay)
        elif args.optimizer == 'ranger':
            optimizer = optim.Ranger(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=args.learning_rate,
                                     betas=(0.9, 0.999),
                                     weight_decay=args.weight_decay)
        elif args.optimizer == 'lamb':
            optimizer = optim.Lamb(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=args.learning_rate,
                                   weight_decay=args.weight_decay)
        else:
            pass

        if args.loss == 'ce':
            criterion = nn.CrossEntropyLoss()
        elif args.loss == 'focal':
            criterion = FocalLoss()
        scaler = torch.cuda.amp.GradScaler()

        iter_per_epoch = len(train_loader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm_epoch)

        scheduler = None
        if args.scheduler == 'cos_base':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        elif args.scheduler == 'cos':
            # tmax = epoch * 2 => half-cycle
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax, eta_min=args.min_lr)
        elif args.scheduler == 'cycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr,
                                                            steps_per_epoch=iter_per_epoch, epochs=args.epochs)

        best_val_f1 = .0
        best_val_loss = 9999.
        best_epoch = 0

        for epoch in range(1, args.epochs + 1):
            train(model, train_loader, criterion, optimizer, warmup_scheduler, scheduler, scaler, epoch, wandb, args)
            val_loss, val_f1 = valid(model, val_loader, criterion, epoch, wandb, args)
            if val_f1 > best_val_f1:
                best_epoch = epoch
                best_val_loss = min(val_loss, best_val_loss)
                best_val_f1 = max(val_f1, best_val_f1)

                torch.save({'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'epoch': epoch, },
                           f'{log_dir}/ckpt_best_fold_{fold:01d}.pt')

            if args.scheduler in ['cos_base', 'cos']:
                scheduler.step()

        del model
        del optimizer, scheduler
        del train_dataset, val_dataset

        if args.wandb:
            run.finish()
