import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

from module.kcenterGreedy import kCenterGreedy

class SubsetSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))
    
    def __len__(self):
        return len(self.indices)

def get_kcg(model, labeled_data_size, unlabeled_data_size, unlabeled_loader, modal='text'):

    with torch.cuda.device(0):
        features = torch.tensor([]).cuda()

    with torch.no_grad():
        for (data, target) in unlabeled_loader:
            with torch.cuda.device(0):
                inputs = data[0].cuda()
                mask = data[1].cuda()
                if modal == 'both':
                    w_data = data[2].cuda()
                    w_mask = data[3].cuda()
            
            if modal == 'text' or modal == 'wav':
                _, features_batch = model(inputs, mask)
                features_batch = features_batch[1]
            
            elif modal == 'both':
                _,_,_, features_batch = model(inputs, mask, w_data, w_mask)

            features = torch.cat((features, features_batch), 0)
        feat = features.detach().cpu().numpy()
        new_av_idx = np.arange(unlabeled_data_size,(unlabeled_data_size + labeled_data_size))
        sampling = kCenterGreedy(feat)  
        batch = sampling.select_batch_(new_av_idx, 200)
        other_idx = [x for x in range(unlabeled_data_size) if x not in batch]
    return  other_idx + batch

def query_samples(model, train_data, labeled_set, unlabeled_set, method, modal='text'):

    device = torch.device('cuda')
    if method == 'Random':
        print('doing with Random')
        args = np.random.permutation(len(unlabeled_set))

    elif method == 'Entropy':
        print('doing with Entropy')
        model.eval()
        unlabeled_loader = DataLoader(train_data, batch_size=16, sampler=SubsetSequentialSampler(unlabeled_set))

        entropies = None

        with torch.no_grad():
            for ii, (data, target) in enumerate(unlabeled_loader):
                inputs, mask = data[0].to(device), data[1].to(device)
                if modal == 'both':
                    w_data, w_mask = data[2].to(device), data[3].to(device)

                if modal == 'text' or modal == 'wav':
                    output, encode_out = model(inputs, mask, output_attentions=True, output_hidden_states=True)

                if modal == 'both':
                    output, _, _, encode_out = model(inputs, mask, w_data, w_mask)

                entropy = torch.var(output, dim=1)

                if entropies is None:
                    entropies = entropy.detach().cpu().numpy()
                
                else:
                    entropies = np.concatenate((entropies, entropy.detach().cpu().numpy()))
                
            args = np.argsort(entropies)               
    
    elif method == 'Coreset':
        print('doing with Coreset')
        model.eval()

        unlabeled_loader = DataLoader(train_data, batch_size=16, sampler=SubsetSequentialSampler(unlabeled_set + labeled_set))

        args = get_kcg(model, len(labeled_set), len(unlabeled_set), unlabeled_loader, modal=modal)
        # unlabeled_loader = DataLoader(train_data, batch_size=16, sampler=SubsetSequentialSampler(unlabeled_set))


    return args