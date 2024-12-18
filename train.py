import os
import json
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from collections import defaultdict
from data.KiMoRe.KiMoRe import KiMoReDataModuleFolded, neighbor_1base
from tqdm import tqdm
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from mvn.dataset import SingleMoveDataset
from mvn.classifier import GCNClassifier, RotationInvariantGCNClassifier, ViewAdaptiveGCNClassifier

from utils.arguments import get_args
from utils.setup import setup_seed, setup_experiment
from utils.plot_curve import plot_train_curve
from mvn.cam import visualize_CAM
from mvn.sgn import SGN

from statistics import stdev, mean


class KimoreLossMSE(torch.nn.Module):
    def __init__(self,max_score=50):
        super(KimoreLossMSE, self).__init__()
        self.max_score = max_score
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred * self.max_score
        return self.mse(pred,target)


class KimoreLossMAE(torch.nn.Module):
    def __init__(self,max_score=50):
        super(KimoreLossMAE, self).__init__()
        self.max_score = max_score
        self.loss = torch.nn.L1Loss()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred * self.max_score
        return self.loss(pred,target)


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def train_all_folds(exercise, device, args):

    FOLDS = 5
    folds_loss = []
    for fold in range(FOLDS):

        dataset = KiMoReDataModuleFolded(
                filepath='data/processed/kimore_kfold.pickle',
                batch_size=args.batch_size,
                transform=None,
                exercise=exercise,
                fold=fold,
            )
        dataset.setup()
        
        # Qui si puo indebolire l'aiuto ai modelli, togliendo la sigmoide
        criterion_train = KimoreLossMSE(max_score=50.0)
        criterion_val = KimoreLossMAE(max_score=50.0)

        num_classes = 1
        connectivity = neighbor_1base
        num_joints = 19

        model = {
            'gcn': GCNClassifier(
                num_joints=num_joints,
                in_channels=3,
                connectivity=connectivity,
                num_classes=num_classes,
                strategy=args.strategy,
                device=device
            ),
            'ri-gcn': RotationInvariantGCNClassifier(
                num_joints=num_joints,
                in_channels=num_joints,
                num_classes=num_classes,
                connectivity=connectivity,
                strategy=args.strategy,
                device=device
            ),
            'va-gcn': ViewAdaptiveGCNClassifier(
                num_joints=num_joints,
                in_channels=3,
                num_classes=num_classes,
                connectivity=connectivity,
                strategy=args.strategy,
                device=device
            ),
            'sgn': SGN(
                num_classes=num_classes, 
                seg=200, 
                batch_size=args.batch_size, 
                train=True
            )
        }[args.model].to(device)

        total = get_n_params(model)
        print('Params for ', args.model, ': ', total)

        # Questo è quello che ci interessa
        best_fold_val_loss = 1e10

        optimizer = Adam(model.parameters(), args.lr)
        for i in tqdm(range(args.epoch)):

            # Train
            model.train()
            train_losses = []
            for j, (samples_batch, labels_batch) in enumerate(dataset.train_dataloader()):

                # skip last batch if we are using SGN, batches must have a constant size
                if args.model == 'sgn' and samples_batch.shape[0] != args.batch_size:
                    continue

                optimizer.zero_grad()

                samples_batch = samples_batch.float().to(device)
                labels_batch = labels_batch.to(device)
                
                label_pred = model(samples_batch).squeeze_()
                loss = criterion_train(label_pred, labels_batch)
                loss.backward()
                optimizer.step()
            
                train_losses.append(loss.item())

            train_epoch_loss = np.mean(train_losses)
            val_epoch_loss = evaluate(model, dataset.val_dataloader(), device, criterion_val, args)
            print(f"LOSS: train {train_epoch_loss:.2f} (MSE) | val {val_epoch_loss:.2f} (MAE) | best_val {best_fold_val_loss:.2f} (MAE)")

            # Save only best
            if val_epoch_loss < best_fold_val_loss:
                best_fold_val_loss = val_epoch_loss

        # Alla fine del train per questa fold salva la loss in validazione migliore
        folds_loss.append(best_fold_val_loss.item())
        
    folds_mean = mean(folds_loss)
    folds_stdev = stdev(folds_loss)
    folds_best = min(folds_loss)

    # Aggrega dati delle fold di questo esercizio
    return {
        'exercise': exercise,
        'folds_loss': folds_loss,
        'folds_mean': folds_mean,
        'folds_stdev': folds_stdev,
        'folds_best': folds_best
    }

def save_to_disk(filepath, results):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def train(args):
    device = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
    exp_dir, logger = setup_experiment(args)
    filepath = os.path.join(exp_dir, 'result.json')

    results = { 'exercises': [], 'aggregated': {} }
    if args.dataset == 'kimore':

        exercises = [1,2,3,4,5]
        for exercise in exercises:
            result_dict = train_all_folds(exercise, device, args)
            results['exercises'].append(result_dict)
            save_to_disk(filepath, results)

    else:
        assert(False)

    losses = [ ex['folds_mean'] for ex in results['exercises'] ]
    results['aggregated']['mean'] = mean(losses)
    results['aggregated']['stdev'] = stdev(losses)
    save_to_disk(filepath, results)


def evaluate(model, dataloader, device, loss_func, args):
    model.eval()
    losses = []
    for j, (samples_batch, labels_batch) in enumerate(dataloader):

        # skip last batch if we are using SGN, batches must have a constant size
        if args.model == 'sgn' and samples_batch.shape[0] != args.batch_size:
            continue

        samples_batch = samples_batch.float().to(device)
        labels_batch = labels_batch.float().to(device)

        with torch.no_grad():
            preds = model(samples_batch).squeeze_()
            losses.append(loss_func(preds, labels_batch).detach())

    return torch.FloatTensor(losses).mean().float()


if __name__ == '__main__':
    args = get_args()
    setup_seed(args.seed)
    train(args)
