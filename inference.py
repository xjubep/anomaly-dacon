import argparse
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ImageModel
from dataset import AnomalyDACON

@torch.no_grad()
def test(model, test_loader):
    model.eval()
    batch_iter = tqdm(enumerate(test_loader), 'Testing', total=len(test_loader), ncols=120)
    preds = []

    for batch_idx, batch_item in batch_iter:
        img = batch_item['img']['image'].cuda()

        with torch.cuda.amp.autocast():
            pred = model(img)
        preds.extend(torch.softmax(pred, dim=1).clone().detach().cpu().numpy())  # probabillity, not label

    return preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--save_path', type=str, default='/hdd/sy/weights/anomaly-dacon',
                        help='Directory where the generated files will be stored')
    parser.add_argument('-c', '--comment', type=str, default=None)
    parser.add_argument('-bs', '--batch_size', type=int, default=64,
                        help='Batch Size. ' 'Default: 32')
    parser.add_argument('-is', '--img_size', type=int, default=384,
                        help='Variables to resize the image. ' 'Default: 384')
    parser.add_argument('-nw', '--num_workers', type=int, default=8,
                        help='Number of workers of dataloader')
    parser.add_argument('-m', '--model', type=str, default='convnext_base_384_in22ft1k',
                        help='Name of model.')
    parser.add_argument('-d', '--dataset', type=str, default='AnomalyDACON',
                        help='Name of evaluation dataset.')
    args = parser.parse_args()

    test_data = sorted(glob('data/test/*.png'))
    train_label = pd.read_csv('data/train_df.csv')['label']

    label_unique = sorted(np.unique(train_label))
    label_unique = {key: value for key, value in zip(label_unique, range(len(label_unique)))}
    train_labels = [label_unique[k] for k in train_label]  ## n_class: 88
    label_decoder = {value: key for key, value in label_unique.items()}

    test_dataset = globals()[args.dataset](args.img_size, test_data, train_labels, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    best_ckpts = [f'/hdd/sy/weights/anomaly-dacon/{args.comment}/ckpt_best_fold_0.pt',
                  f'/hdd/sy/weights/anomaly-dacon/{args.comment}/ckpt_best_fold_1.pt',
                  f'/hdd/sy/weights/anomaly-dacon/{args.comment}/ckpt_best_fold_2.pt',
                  f'/hdd/sy/weights/anomaly-dacon/{args.comment}/ckpt_best_fold_3.pt',
                  f'/hdd/sy/weights/anomaly-dacon/{args.comment}/ckpt_best_fold_4.pt']
    predict_list = []

    for best_ckpt in best_ckpts:
        model = ImageModel(model_name=args.model, class_n=len(label_unique.keys()), mode='test')
        model.load_state_dict(torch.load(best_ckpt)['model_state_dict'])
        model = nn.DataParallel(model.cuda())
        preds = test(model, test_loader)
        predict_list.append(np.array(preds))

    ensemble = np.array(predict_list[0] + predict_list[1] + predict_list[2] + predict_list[3] + predict_list[4]) / len(
        predict_list)

    ensemble = np.argmax(ensemble, axis=1)
    ensemble = np.array([label_decoder[val] for val in ensemble])
    submission = pd.read_csv('data/sample_submission.csv')
    submission['label'] = ensemble
    save_dir = '/hdd/sy/weights/anomaly-dacon/submissions'
    csv_name = f'{args.comment}.csv'
    submission.to_csv(f'{save_dir}/{csv_name}', index=False)
