import torch
import os
from torchnet import meter
from collections import Counter

from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from transformers import T5Tokenizer, EncoderDecoderModel
from torch.utils.data.sampler import SubsetRandomSampler

from src.dataset import KEMDyDataset
from src.selection_methods import query_samples
from src.util import out_put
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.kobert_tokenizer import KoBertTokenizer

from module.KoBertEmotionRecognition import KoBERTEmotionRecognition
from module.Wav2VecEmotionRecognition import Wav2VecEmotionRecognition
from module.LateFusion import LatefusionModule

from transformers import AutoConfig

from argparse import ArgumentParser 

import wandb

tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')

def train(modal, k, l_type, epochs, lr, decay, batch_size, file_name, method, use_gpu=True):
    use_gpu= True
    best_loss = float('inf')
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_data = KEMDyDataset(modal=modal, k=k, kind='train', l_type=l_type)
    val_data = KEMDyDataset(modal=modal, k=k, kind='val', l_type=l_type)

    directory = file_name.split('/')[-2]
    if not os.path.exists(f'./results/{modal}/' + directory):
        os.mkdir(f'./results/{modal}/' + directory)
        
    CYCLES = 10

    indices = list(range(len(train_data)))
    #Get Initial label set
    if not method == 'total':
        labeled_set = indices[:200]
        unlabeled_set = [x for x in indices if x not in labeled_set]
    
    else:
        labeled_set = indices

    # Model Init for each modality
    if modal == 'text':
        model = KoBERTEmotionRecognition().to(device)

    if modal == 'wav':
        model_name_or_path = "kresnik/wav2vec2-large-xlsr-korean"
        config = AutoConfig.from_pretrained(model_name_or_path,
                                            num_labels=7,
                                            label2id = {'neutral': 0, 'happy': 1, 'surprise': 2, 'angry': 3, 'sad': 4, 'disqust': 5, 'fear': 6},
                                            id2label = {0: 'neutral', 1: 'happy', 2: 'surprise', 3: 'angry', 4: 'sad', 5: 'disqust', 6: 'fear'}
                                            )

        model = Wav2VecEmotionRecognition.from_pretrained(model_name_or_path, config=config).to(device)

    if modal == 'both':

        model_name_or_path = "kresnik/wav2vec2-large-xlsr-korean"
        config = AutoConfig.from_pretrained(model_name_or_path,
                                            num_labels=7,
                                            label2id = {'neutral': 0, 'happy': 1, 'surprise': 2, 'angry': 3, 'sad': 4, 'disqust': 5, 'fear': 6},
                                            id2label = {0: 'neutral', 1: 'happy', 2: 'surprise', 3: 'angry', 4: 'sad', 5: 'disqust', 6: 'fear'}
                                            )

        model = LatefusionModule(model_name_or_path, config).to(device)


    best_f1 = 0
    if method == 'total':
        CYCLES = 1

    for cycle in range(CYCLES):
        model.train()

        train_loader = DataLoader(train_data, batch_size=batch_size, sampler=SubsetRandomSampler(labeled_set))

        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)

        loss_meter = meter.AverageValueMeter()

        best_epoch = 0
        cls_counter = Counter()

        for epoch in range(epochs):
            pred_label = []
            true_label = []

            loss_meter.reset()
            for ii, (inputs, target) in enumerate(train_loader):
                data, mask = inputs[0].to(device), inputs[1].to(device)
                target = target.to(device)
                if modal == 'both':
                    w_data, w_mask = inputs[2].to(device), inputs[3].to(device)

                if modal == 'text' or modal == 'wav':
                    output, encode_out = model(data, mask, output_attentions=True, output_hidden_states=True)
                
                if modal == 'both':
                    output, _, _, encode_out = model(data, mask, w_data, w_mask)

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
                    '| method : ' + method +
                    '| cycle : {}/{}'.format(cycle,CYCLES) + 
                    '| train Loss: ' + str(loss_meter.value()[0]) +
                    '| train F1: ' + str(train_f1) + '| train Accuracy: ' + str(train_accuracy) +
                    '| train Precision: ' + str(train_precision) + '| train Recall: ' + str(train_recall),
                    file_name+'_enc')

            val_f1, val_accuracy, val_precision, val_recall = val(model, val_loader, use_gpu, modal=modal)

            out_put('Epoch: ' + 'val' + str(epoch) +
                    '| method : ' + method +
                    '| cycle : {}/{}'.format(cycle,CYCLES) + 
                    '| val F1: ' + str(val_f1) + '| val Accuracy: ' + str(val_accuracy) +
                    '| val Precision: ' + str(val_precision) + '| val Recall: ' + str(val_recall),
                    file_name+'_enc')
            
            cls_counter.update(list(true_label.detach().cpu().numpy()))
            if val_f1 >= best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                torch.save(model.state_dict(), f"{file_name}_encbest.pth")

        perf = f"best accuracy is {best_f1} in epoch {best_epoch}" + "\n"
        out_put(perf, file_name)
        #Select samples to annotate and add to labeled set
        if not method == 'total':
            arg = query_samples(model, train_data, labeled_set, unlabeled_set, method=method, modal=modal)
            labeled_set += list(torch.tensor(unlabeled_set)[arg][-200:].numpy())
            unlabeled_set = list(torch.tensor(unlabeled_set)[arg][:-200].numpy()) 

        out_put(str(cls_counter), file_name)


@torch.no_grad()
def val(model, dataloader, use_gpu, modal='text'):
    model.eval()
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    pred_label = []
    true_label = []

    for ii, (inputs, target) in enumerate(dataloader):
        data, mask = inputs[0].to(device), inputs[1].to(device)
        target = target.to(device)
        
        if modal == 'both':
            w_data, w_mask = inputs[2].to(device), inputs[3].to(device)

        if modal == 'text' or modal == 'wav':
            output, encode_out = model(data, mask, output_attentions=True, output_hidden_states=True)
        
        if modal == 'both':
            output, _, _, encode_out = model(data, mask, w_data, w_mask)

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


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--modal', type=str, default='text', help='modality you choose')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--learning_rate', dest='lr', type=float, default=1e-5)
    parser.add_argument('--method', type=str, default='Coreset', help='You should choose among [Coreset, Entropy, Random, total]')
    parser.add_argument('--batch_size', type=int, default=16)


    args = parser.parse_args()

    weight_decay = 0
    l_type = 'emotion'
    use_gpu = True
    modal = args.modal
    learn_rate = args.lr
    method = args.method
    batch_size = args.batch_size    


    if not os.path.exists(f'./results/'):
        os.mkdir(f'./results/')

    if not os.path.exists(f'./results/{modal}/'):
        os.mkdir(f'./results/{modal}/')

    for k in range(1,6):    #for 5fold-cross-validation
        train(modal=modal, epochs=args.epoch, lr=learn_rate, decay=weight_decay, method=method,
        use_gpu=use_gpu,
        file_name=f'./results/{modal}/{modal}_{method}_{l_type}_k{k}_2/{modal}_{l_type}_k{k}_2',
        batch_size=batch_size, k=k, l_type=l_type)