import torch
import os

from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn

# Dataset
from src.dataset import KEMDyDataset

# Model
from module.KoBertEmotionRecognition import KoBERTEmotionRecognition

from torchnet import meter
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

from src.util import out_put

import warnings
warnings.filterwarnings('ignore')


def train(modal, k, l_type, epochs, lr, decay, batch_size, file_name, use_gpu=False):
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    directory = file_name.split('/')[-2]
    if not os.path.exists(f'./results/{modal}/' + directory):
        os.mkdir(f'./results/{modal}/' + directory)

    train_data = KEMDyDataset(modal=modal, k=k, kind='train', l_type=l_type)
    val_data = KEMDyDataset(modal=modal, k=k, kind='val', l_type=l_type)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Model Init
    if modal == 'text':
        model = KoBERTEmotionRecognition().to(device)

    # Optimization Settings
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)

    # meters
    loss_meter = meter.AverageValueMeter()

    best_f1 = 0
    best_epoch = 0

    # Training
    for epoch in range(epochs):
        pred_label = []
        true_label = []

        loss_meter.reset()
        for ii, (data, mask, target) in enumerate(train_loader):
            data, mask = data.to(device), mask.to(device)
            target = target.to(device)

            output = model(data, mask)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())

            _, pred = output.data.topk(1, dim=1)
            pred = pred.t().squeeze()

            pred_label.append(pred.detach().cpu())
            true_label.append(target.detach().cpu())

        pred_label = torch.cat(pred_label, 0)
        true_label = torch.cat(true_label, 0)

        train_f1 = f1_score(true_label, pred_label, average='weighted')
        train_accuracy = accuracy_score(true_label, pred_label)
        train_precision = precision_score(true_label, pred_label, average='weighted')
        train_recall = recall_score(true_label, pred_label, average='weighted')

        out_put('Epoch: ' + 'train' + str(epoch) +
                '| train Loss: ' + str(loss_meter.value()[0]) +
                '| train F1: ' + str(train_f1) + '| train Accuracy: ' + str(train_accuracy) +
                '| train Precision: ' + str(train_precision) + '| train Recall: ' + str(train_recall),
                file_name)

        val_f1, val_accuracy, val_precision, val_recall = val(model, val_loader, use_gpu)

        out_put('Epoch: ' + 'val' + str(epoch) +
                '| val F1: ' + str(val_f1) + '| val Accuracy: ' + str(val_accuracy) +
                '| val Precision: ' + str(val_precision) + '| val Recall: ' + str(val_recall),
                file_name)

        if val_f1 >= best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            model.save(f"{file_name}_best.pth")

    model.save(f'{file_name}.pth')

    perf = f"best accuracy is {best_f1} in epoch {best_epoch}" + "\n"
    out_put(perf, file_name)


@torch.no_grad()
def val(model, dataloader, use_gpu):
    model.eval()
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    pred_label = []
    true_label = []

    for ii, (data, mask, target) in enumerate(dataloader):
        data, mask = data.to(device), mask.to(device)
        target = target.to(device)

        output = model(data, mask)

        _, pred = output.data.topk(1, dim=1)
        pred = pred.t().squeeze()

        pred_label.append(pred.detach().cpu())
        true_label.append(target.detach().cpu())

    pred_label = torch.cat(pred_label, 0)
    true_label = torch.cat(true_label, 0)

    val_f1 = f1_score(true_label, pred_label, average='weighted')
    val_accuracy = accuracy_score(true_label, pred_label)
    val_precision = precision_score(true_label, pred_label, average='weighted')
    val_recall = recall_score(true_label, pred_label, average='weighted')

    model.train()

    return val_f1, val_accuracy, val_precision, val_recall

