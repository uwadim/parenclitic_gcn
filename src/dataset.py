import os

import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")


class PDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.test = test
        self.filename = filename
        self.labels = []
        super(PDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        original_data = pd.read_csv(self.raw_paths[1])
        self.ids = original_data.query('label == "TEST"')['id'].to_list() if self.test else \
            original_data.query('label == "TRAIN"')['id'].to_list()
        if self.test:
            return [f'data_test_{i}.pt' for i in self.ids]
        else:
            return [f'data_{i}.pt' for i in self.ids]

    def download(self):
        pass

    def process(self):
        data = pd.read_csv(self.raw_paths[0])
        original_data = pd.read_csv(self.raw_paths[1])
        self.ids = original_data.query('label == "TEST"')['id'].to_list() if self.test else \
            original_data.query('label == "TRAIN"')['id'].to_list()
        # Replace node indices 'N1', 'N2', ... 'X1', 'X2' to integers
        replace_dict = {val: idx for idx, val in
                        enumerate(pd.concat([data['p1'], data['p2']], axis=0, ignore_index=True).unique())}
        for col in ['p1', 'p2']:
            data.loc[:, col] = data[col].replace(replace_dict).astype(int).copy()
        edge_indices = torch.tensor(data[['p1', 'p2']].T.to_numpy())
        num_of_nodes = len(replace_dict)
        node_features = torch.tensor(np.ones(num_of_nodes).reshape(-1, 1))
        self.num_classes = original_data['score'].nunique()

        for index, colname in enumerate(tqdm(self.ids, total=len(self.ids))):
            edge_weights = torch.tensor(data[str(colname)].to_numpy()).float()
            label = self._get_labels(original_data.query('id == @colname')['score'])
            self.labels.append(label)
            # Create data object
            data_to_save = Data(x=node_features,
                                edge_index=edge_indices,
                                edge_attr=edge_weights,
                                y=label,
                                graph_index=colname,
                                num_classes=self.num_classes)
            if self.test:
                torch.save(data_to_save,
                           os.path.join(self.processed_dir,
                                        f'data_test_{index}.pt'))
            else:
                torch.save(data_to_save,
                           os.path.join(self.processed_dir,
                                        f'data_{index}.pt'))
        self.labels = torch.cat(self.labels)

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return len(self.ids)

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        return torch.load(os.path.join(self.processed_dir, f'data_test_{idx}.pt')) if self.test else torch.load(
            os.path.join(self.processed_dir, f'data_{idx}.pt'))
