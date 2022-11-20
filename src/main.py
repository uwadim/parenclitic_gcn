#!/usr/bin/env python
# coding: utf-8
"""
Расчет классификации реальных данных с помощью Parenclitic (wSA).
Используется GCN 3 свёрточных слоя со skip connections
"""
import glob
import logging
import os
import random
import re
import shutil

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from torch import sigmoid
from torch_geometric import set_debug
from torch_geometric.loader.dataloader import DataLoader

import helpers
from dataset import PDataset
from model import SkipGCN
from pytorchtools import EarlyStopping
from samplers import StratifiedSampler

set_debug(True)
log = logging.getLogger(__name__)


def file_matching(network_files: list, original_files: list) -> dict:
    result = {}
    for file in network_files:
        r = re.search(r'(?P<index>\d+_\d+_\d+_\d+_\d+)', file)
        index = r['index']
        origin = [x for x in original_files if index in x]
        result[index] = {'network': file, 'original_data': '../../' + ''.join(origin)}
    return result


def file_matching_real_data(network_files: list, original_files: list) -> dict:
    result = {}
    for file in network_files:
        mask = re.search(r'wSA_(?P<dataname>.+?)_networks', file)
        dataname = mask['dataname']
        origin = [x for x in original_files if dataname in x]
        result[dataname] = {'network': file, 'original_data': '../' + ''.join(origin)}
    return result


def file_matching_real_data_bootstrap(network_files: list, original_files: list) -> dict:
    result = {}
    for file in network_files:
        mask = re.search(r'wSA_(?P<dataname>.+?_\d+?_\d+?)_networks', file)
        dataname = mask['dataname']
        origin = [x for x in original_files if f'{dataname}.csv' in x]
        result[dataname] = {'network': file, 'original_data': '../' + ''.join(origin)}
    return result


def train_one_epoch(model,
                    device: torch.device,
                    train_loader: DataLoader,
                    criterion: torch.nn.modules.loss.BCEWithLogitsLoss,
                    optimizer: torch.optim.Adam):
    running_loss = 0.0
    step = 0
    model.train()

    for data in train_loader:
        data.to(device)  # Use GPU
        # ####### DEBUG ########
        # real_graph_indices = data.graph_index.detach().to('cpu').numpy() - data.ptr.detach().to('cpu').numpy()[:-1]
        # ######################
        # Reset gradients
        optimizer.zero_grad()
        out = model(data.x.float(), data.edge_index, data.edge_attr, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y.float())  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        running_loss += loss.item()
        step += 1
    return running_loss / step


def test_one_epoch(model, device, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    step = 0
    batch_auc = []

    with torch.no_grad():
        for data in test_loader:
            data.to(device)  # Use GPU
            # ####### DEBUG ########
            # real_graph_indices = data.graph_index.detach().to('cpu').numpy() - data.ptr.detach().to('cpu').numpy()[:-1]
            # ######################
            out = model(data.x.float(), data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out, data.y.float())
            running_loss += loss.item()
            step += 1

            pred = sigmoid(out)
            roc_auc = roc_auc_score(data.y.detach().to('cpu').numpy(),
                                    pred.detach().to('cpu').numpy())
            batch_auc.append(roc_auc)
    return running_loss / step, np.mean(batch_auc)


def test_embeddings(config, model, device, test_loader):
    embeddings = np.zeros([1, config.model.embedding_size])
    y = np.zeros([1, 1])
    batch_auc = []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data.to(device)  # Use GPU
            # ####### DEBUG ########
            # real_graph_indices = data.graph_index.detach().to('cpu').numpy() - data.ptr.detach().to('cpu').numpy()[:-1]
            # ######################
            out = model(data.x.float(), data.edge_index, data.edge_attr, data.batch)
            embeddings = np.append(embeddings, model.get_embeddings(), axis=0)
            y = np.append(y, data.y.detach().to('cpu').numpy(), axis=0)
            pred = sigmoid(out)
            roc_auc = roc_auc_score(data.y.detach().to('cpu').numpy(),
                                    pred.detach().to('cpu').numpy())
            batch_auc.append(roc_auc)
    print(f'Test ROC AUC for this model: {np.mean(batch_auc)}')
    return embeddings[1:, :], y[1:, :], pred


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    result_dir = str(helpers.get_create_path('../results'))

    orig_cwd = hydra.utils.get_original_cwd()
    print(f'##### {orig_cwd}')

    random.seed(cfg.model.random_seed)
    np.random.seed(cfg.model.random_seed)
    os.environ['PYTHONHASHSEED'] = str(cfg.model.random_seed)
    np.random.seed(cfg.model.random_seed)
    torch.cuda.manual_seed(cfg.model.random_seed)
    torch.manual_seed(cfg.model.random_seed)

    # Prepare datasets
    root_dir = f'{orig_cwd}/{cfg.data.real_data.root_directory}'
    print('########### Main loop  #############')

    network_fnames = glob.glob(f'{root_dir}/raw/*_networks.csv')
    original_fnames = glob.glob(f'{cfg.data.real_data.orinal_directory}/{cfg.data.real_data.original_mask}')
    files_dict = file_matching_real_data_bootstrap(network_files=network_fnames, original_files=original_fnames)
    result_aucs_list = []
    for str_idx, fnames_dict in files_dict.items():
        print(f'Preparing train Dataset for index {str_idx}')
        # Drop 'processed' dir of Dataset
        shutil.rmtree(f'{root_dir}/processed', ignore_errors=True)
        train_dataset = PDataset(root=root_dir,
                                 filename=[fnames_dict['network'], fnames_dict['original_data']],
                                 test=False).shuffle()
        print(f'Preparing test Dataset fot index {str_idx}')
        test_dataset = PDataset(root=root_dir,
                                filename=[fnames_dict['network'], fnames_dict['original_data']],
                                test=True).shuffle()

        # Prepare training
        train_sampler = StratifiedSampler(class_vector=train_dataset.labels,
                                          batch_size=train_dataset.len(),
                                          random_state=cfg.model.random_seed)

        train_loader = DataLoader(train_dataset,
                                  batch_size=train_dataset.len(),
                                  shuffle=False,
                                  sampler=train_sampler
                                  )
        # On GPU shuffle=True causes random results in ROC AUC for different calls of test_loader
        test_loader = DataLoader(test_dataset,
                                 batch_size=test_dataset.len(),
                                 shuffle=False
                                 )
        print('Data prepared!')

        model = SkipGCN(config=cfg,
                        num_node_features=train_loader.dataset.num_node_features)
        model = model.to(device)
        # Reinitialize layers
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=cfg.learning_rate,
                                     weight_decay=1e-3)

        print("Starting training...")
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=cfg.patience, verbose=True, delta=0.00001)
        n_epochs = cfg.max_epoch
        for epoch in range(1, n_epochs + 1):
            ###################
            # train the model #
            ###################
            train_loss = train_one_epoch(model=model,
                                         device=device,
                                         train_loader=train_loader,
                                         criterion=criterion,
                                         optimizer=optimizer)
            # record training loss
            avg_train_losses.append(train_loss)

            ######################
            # validate the model #
            ######################
            test_loss, test_auc = test_one_epoch(model=model,
                                                 device=device,
                                                 test_loader=test_loader,
                                                 criterion=criterion)
            avg_valid_losses.append(test_loss)

            epoch_len = len(str(n_epochs))

            print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] '
                         f'train_loss: {train_loss:.5f} '
                         f'valid_loss: {test_loss:.5f} '
                         f'Test ROC AUC {test_auc:.5f}')

            print(print_msg)

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(test_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        # load the last checkpoint with the best model
        model.load_state_dict(torch.load('./checkpoint.pt'))

        t_loss_train, t_mean_auc_train = test_one_epoch(model, device, train_loader, criterion)
        t_loss, t_mean_auc = test_one_epoch(model, device, test_loader, criterion)

        result_aucs_list.append((str_idx,
                                 cfg.data.real_data.type_of_data,
                                 'Skip GCN',
                                 'Not used',
                                 'wSA',
                                 t_mean_auc_train,
                                 t_mean_auc
                                 ))
        result_aucs = pd.DataFrame(columns=['Name_Of_Data',
                                            'Type_Of_Sphere',
                                            'Type_Of_Result',
                                            'model',
                                            'Type_of_Network',
                                            'AUC_train',
                                            'AUC_test'
                                            ], data=result_aucs_list)
        result_aucs.to_csv(f'{result_dir}/{cfg.data.real_data.type_of_data}_AUCS_BOOTSTRAP_wSkipconnections_table.csv',
                           index=False,
                           sep=';')
        log.info(f'{cfg.data.real_data.type_of_data} File index {str_idx}. '
                 f'Test ROC AUC of Supervised skip GCN {t_mean_auc:.6f}')
    print('End!')


if __name__ == "__main__":
    main()
